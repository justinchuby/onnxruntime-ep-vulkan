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

### [SUMMARY] 2026-08-02 (Switch's round through §8.9.18): four ABI-mirror insertions, GEMV kernel identity, `PROVEN-ELSEWHERE` proposed-withdrawn-replaced, and the layout guard going compile-time

- **Switch's round:** `--reprove` had been silently shrinking the ledger while printing `PASS` — fixed, `write_ledger()` now refuses any shrink and names the dropped keys. `NO-SUBJECT-WITNESS` was already reachable via three existing guards; both shapes planted in the lane with healthy controls. Stale input cache ruled out for all 95 entries (`compute_calls` now recorded and enforced at 1). **The real find:** `a52024f` inserted `device_losses` mid-struct without bumping `COUNTERS_ABI_VERSION`; three ctypes mirrors silently read every field below it eight bytes off (`dispatches_executed` read `device_losses`, always a plausible 0). ABI bumped to 4, mirrors repaired, lane changed to `struct_size` equality (not `>=`) since an append and an insertion are indistinguishable to a `>=` reader. Filed but not fixed: three hand-maintained ctypes mirrors of one C ABI is the standing defect.
- **§8.9.16, second evidence list:** `Add-i32`/`Mul-i32` declined via a hand-written `EXERCISED` list consulted *inside* the claim predicate, before any proof key existed — unfixable by proof by construction, a gate satisfiable only by hand. Deleted; replaced with `only_loadable_variants` derived from SPIR-V capability requirements (`Int64`/`shaderInt64`). Op suite 11 red → 6 red, ledger 95 → 97.
- **Session 28, GEMV kernel identity:** `ONNXRUNTIME_EP_VULKAN_GEMV_PACKED` selected a kernel with no artifact recording whether it fired. Added JSON-only `pipeline_variants`/`gemv_packed_spec_constant`, recorded from the effective (not requested) shader/spec pair. Falsifier 5/5 both devices. **Found while validating:** `898a2ba` repeated the exact same mid-struct-insertion defect a third time (three fields this time), caught by the same equality guard — not Mouse's diff. Built `counters_abi.py` (derives the mirror from `counters.rs` directly) but did not wire it into the three call sites, routing the work to Trinity instead — judged in the next session to have been the wrong call.
- **"The mirrors are gone" session:** self-corrected the previous routing decision — "a generator that co-exists with the thing it replaces is a fourth mirror." Deleted all three ctypes mirrors for real; layout discipline is now a compile-time `const _` assertion via a FNV-1a hash over `name:offset:size` registered in `COUNTERS_LAYOUT_REGISTRY`, verified by replaying the exact `898a2ba` insertion and getting `error[E0080]`. Also found the build error additionally required `E0063` (missing-field) fixes in five places — appends were already compiler-checked; only the version *numbering* wasn't. `PROVEN-ELSEWHERE` observed in both polarities on real hardware (`device0`=ALL-PROVEN, `device1`=PROVEN-ELSEWHERE-PRESENT, 355 claims, per-form `proved-on`/`running-on` disclosure).
- **`PROVEN-ELSEWHERE` withdrawn, then replaced:** Fact Checker's audit found model-level ULP evidence cannot promote unexercised per-form keys — the premise the implementation rested on. Replaced with a four-state `device_state` classifier (`PROVEN`/`DEVICE-UNATTRIBUTED`/`PROVEN-ON-ANOTHER-DEVICE`/`UNPROVEN`) keyed on the device *name* read off the run, not the selector ordinal — validated by the very run that would have produced a false-positive match had it been keyed on the ordinal (`DEVICE=0` opened a different physical GPU than expected). `parse_ledger` was found faulting all 103 proofs on one stale entry; split into whole-file vs per-entry fault tracking.
- **§8.9.18 alignment:** Morpheus upheld the refutation and withdrew his own promotion paragraph — `PROVEN-ELSEWHERE` keeps disclosure, loses promotion. The layout guard fired correctly on a pure rename (`device_mismatch_*` → `proven_elsewhere_*`, identical offsets/sizes) because the hash covers name too, demonstrating the mechanism working on a case that wasn't about layout. Fault-scope boundary corrected per Morpheus's rule (scope is set by what you cannot locate, not by severity) — unparseable lines moved back to whole-file faults.
- 📌 Team update (2026-08-02T02:03:46-07:00, from Scribe): Morpheus's R12 fourth generalisation — for a test result, the frame is the binary that ran it — drawn partly from two of Mouse's self-caught near-misses this session (a shared-worktree build linking a sibling's in-flight file; `Copy-Item` preserving `LastWriteTime` letting cargo silently re-run a mutated binary after a restore).
- 📌 Team update (2026-08-02T14:42:30-07:00, from Switch/Mouse): the re-proof-path fix above (`shader_digest`, `--reprove`, `STALE-SHADER`/`NO-SUBJECT-WITNESS`) landed in response to Switch's finding that the ledger could not be invalidated by changing its own subject; all 74 entries re-proved under the new scheme.

<!-- SUMMARIZED by Scribe 2026-08-03T19-55-00-07-00 -- entries from Switch's round (2026-08-02) through §8.9.18 alignment condensed above; full text lives in git history at 8566ce4 and earlier -->

---

- 📌 Team update (2026-08-02T22:37:04-07:00): Link's `link-ledger-toolchain-not-device` finding — it's your ledger, and Morpheus's ruling is pending. `registry::shader_digest_for` hashes SPIR-V bytes, so Ubuntu's `glslc` faults all 74 entries with no kernel change; meanwhile `"device": "device0"` is recorded on 74 of 75 entries and no predicate reads it. Morpheus ruled a three-state remedy (`PROVEN`/`PROVEN-ELSEWHERE`/`UNPROVEN`, plus `SUBJECT-CHANGED` vs `TOOLCHAIN-CHANGED` demotion) — decided by Link, Morpheus. (Superseded by the device-state classifier and §8.9.19/§8.9.22 work below.)
- 📌 Team update (2026-08-03T04-55-00-07-00): Link found the eleven Linux `cargo test --lib` failures are a representational difference in bindgen typing (MSVC `c_int` vs. GCC `c_uint`), not a signedness bug. More load-bearing: Ubuntu shaderc 2023.8 vs. Windows SDK v2026.2 compile different SPIR-V bytes from identical GLSL, faulting the ledger on Linux — this is the keying decision resolved by §8.9.19's two-digest split below. — decided by Link
- 📌 Team update (2026-08-03T04-55-00-07-00): Trinity measured that at the final RMSNorm, Vulkan is bit-exact against a float64 reference while ORT's CPU EP is the side carrying 1 ULP of error — the residual criterion 10 flags is not evidence of a Vulkan defect. — decided by Trinity

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

📌 Team update (2026-08-03T10-35-00-07-00): Switch found `gen_proof_ledger.py --check` checks the file against itself, not against the running build — the subject comparison only happens at runtime against *this build's* embedded digests, and the ledger is `include_str!`'d (`registry.rs:1890`), so `--reprove` has no effect until a rebuild. The coordinator has quoted `--check` as merge evidence six times; you own the repair. — decided by Switch

📌 Team update (2026-08-03T10-35-00-07-00): Rai opened RAI-012 🟡 — a decline message names the wrong subject (verified in-tree at 0× true-cause WARN against 42× a message false in both clauses, `ledger_faults: 97`). You are the named owner. — decided by Rai
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

## §8.10 — coverage measured from two models: gpt-oss-20b was at 1 claimed node of 374 (2026-08-03)
The task was "raise op coverage, cheaply, via the templates". I started from evidence rather than
a list, and the evidence said the cheapest coverage available was not a new op at all.
**The registry census first, because I have miscounted this three times on the record.**
`epctl --dump-capabilities --json`: **91 rows / 73 kernel-carrying (46 `live` + 27 `ready`) /
18 staged**. Morpheus's artifact-derived 91/71/20 is the same file two rows later — two rows moved
staged->live since. Nothing here is derived by reading registry source.
**The instrument.** `rust/tools/probe_model_op_census.py` joins the ONNX graph
(`load_external_data=False`, so gpt-oss-20b's 11.8 GB of external data never loads) against
`ONNXRUNTIME_EP_VULKAN_CLAIM_LOG` from a real session and reports claimed / declined /
no-decision per op type with decline codes and proof keys. It re-derives nothing — registry,
ledger lookup and shape classifier all stay in the DLL, same rule as `--check` asking the
artifact. No claim log is `ERROR(instrument)`, never a prediction.
**Criterion, stated before I picked:** expected coverage gain = nodes in a *real shipped model*
that decline today and would claim after the change, weighted by whether it unblocks a model
class, divided by kernel cost. Forms already backed by a generated template variant — proof-only
work — rank above new kernels. That criterion is what produced the pick, not a taxonomy.
**Phi-3.5 read 355/366, which everyone including me has been quoting. gpt-oss-20b read 1/374.**
Nobody had asked a second model. 292 of its 370 declines were `[unproven]` — not missing kernels.
Every one was the `runtime-extent` sibling of a form the ledger already proved `static`:
`Add/f16` (72), `Cast/f16>f32` (49), `Cast/f32>f16` (49), `MatMulNBits/f16+zero_points` (73),
`SkipSimplifiedLayerNorm/f32` both arities (48), `SimplifiedLayerNormalization/f32` (1). All 36
`ew_cast_*` modules were already generated. **The ledger read full while a whole model declined**,
which is correct behaviour and precisely why it was invisible. Standing rule now written into
`ledger_case_models.py`: prove both shape classes of a module in the same run.
**Then the generator went red and found a broken commitment.** `cast_f16_to_f32_dyn` failed with
`Unsupported("Cast output has no element type at compile time")` -> `EP_FAIL` -> BROKEN COMMITMENT.
`ep::tensor_desc` returns `None` for an edge with any symbolic dim and drops the **dtype** with the
shape. `claim::cast` reads the destination off the live `OrtValueInfo`, so `Compile` is fine;
`templates::ew_cast` read it off `OutRef.desc`, which the Compute-time dynamic re-translate rebuilds
from `NodeDesc`. So every dynamic `Cast` was claimed and then failed — 98 real nodes. **A Cast's
destination type is a node attribute, and a node attribute does not stop existing because an extent
is unknown.** `cast_destination()` prefers the resolved edge, falls back to `to`, and **refuses**
when they disagree; resolving that would put an unannounced reinterpretation of the model inside a
dispatch. I did not repair `tensor_desc` — handing every handler a desc with unusable shapes is a
worse trade than one helper.
**No grandfathering.** `proof_attempts.jsonl` had 106 MATCHed keys against the ledger's 103; the
difference was exactly `Cast f32>bool`, `Cast f32>i32`, `Cast i32>f32`, static — proven once and
lost. I re-measured them through `--append` rather than pasting them forward, the same rule I held
myself to on `--backfill-frame`.
### The delta, as a measurement
| model | before | after |
|---|---|---|
| gpt-oss-20b | **1 / 374 claimed** | **293 / 374** |
| Phi-3.5-mini-int4 | 355 / 366 | 355 / 366 |
Ledger **103 -> 115**, digest `94d994ba54821056` -> `0eef01359c467110`, 12/12 MATCH. **Frame the
new entries were proven under:** debug `onnxruntime_vulkan_ep.dll`, device0 = NVIDIA GeForce RTX
4060 Laptop GPU, ORT 1.28.0, `toolchain = shaderc v2026.2 v2026.2`, tolerance `rtol=1e-3,
atol=1e-5`, and — unlike the 103 older entries — a **recorded `spec_digest`**. No tolerance was
widened and no proof skipped; the 12 entries went through the ordinary three-pass path.
**What it unblocks, honestly.** 293 nodes on one model, which is the difference between gpt-oss-20b
being unrunnable on this EP and being mostly-claimed. It unblocks **no new op**: zero kernels were
written. What still blocks gpt-oss-20b is `QMoE` x24 (staged), `Reshape` x24 (unregistered) and
`GroupQueryAttention` x24 — the last declining on `[attribute]` with an **8-input** signature
(`f16,f16,f16,i32,i32,f16,f16,f16`) against the proven 7-input form. That is attention sinks, a
kernel-semantics question, and it is the largest single remaining item on either model.
**`shaderInt64`: stopped, as instructed.** Every remaining kernel-carrying Phi-3.5 decline is an
i64 form — `Cast/i64>i32` (both shape classes), `Sub/i64`, `Greater/i64`, `Gather/i64,i64>i64`.
Their SPIR-V declares `OpCapability Int64`, absent from `ENGINE_ENABLED_CAPABILITIES`, so they
decline `[dtype]`, not `[unproven]`. That extends Tank's "4 of 5 declines are one device feature"
to the full 8: **5 of 8 are one feature**; the other 3 are `Shape`, `ReduceSum`, `If`, unregistered,
one node each. I did not enable it and did not work around it.
### What surprised me
**The cheapest coverage in the project was not an op.** I expected to be picking between contrib
ops. The answer was a schema axis: the ledger had proven the `static` arm of nearly everything and
the `runtime-extent` arm of almost nothing, and Phi-3.5 is static-friendly enough that no
instrument we had could see it. The apparatus was not wrong — `unproven_decline_forms` named these
exactly — it was that nobody had pointed it at a second model.
**A form can be proven and still be lost.** Three Cast forms had MATCH rows and no ledger entry.
Whatever dropped them was silent, and only comparing the attempt log to the ledger showed it. I did
not find the cause and am not claiming one.
**`--reprove`'s refusal-before-measuring caught me being sloppy** exactly once: I regenerated the
ledger and re-ran a probe without rebuilding, and `include_str!` meant the EP was still answering
from 103 entries. The gate I wrote yesterday to protect somebody else caught me today.
### What my verification established, and what it did not
**Established:** `cargo test --lib` 540 passed / 0 failed; clippy `--all-targets -D warnings`
clean; `counters_abi.py --check` PASS with `ONNXRUNTIME_VULKAN_EP_LIB` **set** (`ledger_entries =
115`); `gen_proof_ledger.py --check` PASS, 115 entries, digest `0eef01359c467110`, every entry's
`shader_digest` agreeing with the built DLL. `tests/ops` **3 failed / 655 passed / 33 skipped /
3 xfailed** here, and the **same 3** — `test_criterion10`, `test_harness_census`
(`+ counters.rs::unprovable_decline_forms`, byte-identical drift text), `test_kv_device_residency`
— fail on a clean `85c57b2` worktree I built and ran separately. **My delta on `tests/ops` is
zero.** I ran only those three files on the baseline; the full baseline suite ran >25 minutes with
no output and I stopped it rather than report a number I did not watch finish, so I am **not**
claiming a full-suite before-count and I am not contradicting Trinity's 8 at `3bac325`.
**Did not:** anything about the **release** artifact — everything here is the debug build, and the
release DLL in this worktree is stale. Anything about another device or another driver. Anything
about `spec_digest` for the 103 older entries; they remain `SPEC-UNRECORDED`. Anything about
whether the 293 newly-claimed gpt-oss-20b nodes produce *correct output on that model* — the ledger
proves the **forms** against reference on small cases, and I did not run gpt-oss-20b end-to-end.
And `probe_model_op_census.py` is a **new instrument reporting on itself**: its only independent
corroboration is that it reproduced Phi-3.5's independently-known 355/366 without being told it.
**One test is intermittently red and I did not fix it.**
`registry::tests::the_digest_pair_separates_a_second_compiler_from_a_second_kernel` failed once in
eight full-suite runs and passes in isolation. Hypothesis only — cross-test device-state
interference through `state_for`'s device predicate. Unconfirmed, unexplained, reported rather than
buried.

## §8.9.22 — the portable digest was a checkout fingerprint (2026-08-03, later)
Link took a fresh Linux `.so` to my two-digest schema and got **0 of 103 agreeing** where Windows
got 102, with his probe showing **both** digests moved on all 103 and the build naming its own
toolchain `UNKNOWN`. He filed two defects. There were three, and the third is the one that made the
other two fixes matter.
**Defect 3, which nobody had named: `source_digest` hashed line endings.**
`build.rs::source_digest_for` hashed `fs::read` output raw. `core.autocrlf=true`, no
`.gitattributes` rule for `*.comp`/`*.glsl`, so this checkout is CRLF (I counted: `gqa_f16.comp`
CR=423 LF=423) and Linux checks out the same blob as LF. **The witness I added to be
toolchain-independent was less portable than the SPIR-V digest it exists to be more portable
than** — glslc emits byte-identical bytes from both. I had excluded the absolute `-I` path so the
digest would not be a machine fingerprint, and then hashed bytes a git option chose. Same mistake,
one level down. A digest that needs a version-control setting to be comparable is a machine
fingerprint with extra steps.
**Defect 1: `PROVEN-ELSEWHERE{toolchain}` could not be reached in the Python classifier.** Its
condition is "SPIR-V differs, source same"; `check_against_build` tested `spirv_digest` first and
unconditionally, so a condition tested earlier subsumed it and every Linux entry rendered as
`SUBJECT-CHANGED` -- not the wrong token but the **strongest available accusation**, telling a user
their source moved when their compiler moved. `registry.rs::subject_verdict` had the table right;
the gate that mirrors it did not, which makes it a second disagreeing opinion about the same fact.
It now mirrors the whole table in the same order, and FAILs on exactly what the runtime declines.
**Defect 2: `cmd_check` computed the explanation and threw it away.** Notes printed only after the
PASS line, four lines past the failure branch's `return 1`. In a Linux run where all 103 failures
had a toolchain cause, the word "toolchain" appeared nowhere; on PASS, where nothing needs
explaining, it printed in full. Present exactly when not needed, absent exactly when needed.
**Also: the eighth uninvoked accessor was mine.** `counters::unprovable_decline_forms` returned
`Vec<String>` via `unwrap_or_default()`, so a poisoned lock read as **"nothing is unprovable"** --
mis-scoring in the dead direction -- and had no caller because the emitter read the static. Now
`Option<Vec<String>>`, emitter goes through `forms_json`. That greened
`tests/ops/test_harness_census.py::test_census_baseline_has_no_drift`, one of the three reds I
reported as pre-existing this morning.
### The measurement, and its negative control
`rust/tools/probe_source_digest_eol.py`: two arms, predictions written first, tree restored and
rebuilt afterwards. **215/215 modules moved under the old rule; 0/215 under the new; P1 (SPIR-V
identical across arms) PASS in both.** I ran the negative control by reverting `normalize_shader_text`
and rebuilding, because P2 passing on a tree I had just fixed proves nothing about whether P2 can
fail. It can: it went `FAIL (215 moved)`. That reproduces Link's Linux reading on Windows without a
Linux box, and it is the strongest thing I have -- I still cannot see his machine.
`probe_ledger_subject_check.py` 5 arms -> 7 (8 assertions), **8/8**. Arm 6 plants a moved
`spirv_digest` with a **correct** `source_digest` and requires `PROVEN-ELSEWHERE{toolchain}` in a run
that FAILS for an unrelated reason -- one arm falsifying defects 1 and 2 together. Arm 7 moves both
and requires `SUBJECT-CHANGED`, so arm 6 cannot be satisfied by a classifier that merely stopped
failing. Arm 2 (Switch's real case) still fails and now says "both witnesses moved", which is the
sentence I wanted: the classifier got more precise, not looser.
`--rewitness-source`: 115 re-witnessed, 0 skipped. It carries `--backfill-frame`'s SPIR-V-equality
refusal **plus** a toolchain-equality refusal, because identical SPIR-V from two compilers does not
establish identical source. Ledger digest `0eef01359c467110` -> `f3bb172dffd6be28`; entry count
unchanged at 115; **no proof was re-asserted, only the source witness recomputed under the
corrected rule.**
### Coverage, re-measured under the new build
gpt-oss-20b **293 / 374 claimed**, unchanged from this morning -- the digest change and the
re-witness moved nothing, which is what I wanted to know. The 78 declines are unchanged in
composition: GQA x24 `[attribute]` (8-input, attention sinks), `Reshape` x24 `[not-registered]`,
QMoE x24 `[staged]`, and 5 `shaderInt64` forms plus `ReduceSum`/`Shape`. **I added no new op this
round and that is a real cost of taking the Linux defects first**, which was the right order:
coverage measured by an instrument that misattributes every entry on the second platform is
coverage nobody can bank.
### What surprised me
**The `UNKNOWN` toolchain is the more interesting half and I could not diagnose it.** I made the
capture read stderr and tolerate a non-zero exit, warned at build time, and added a test that fails
a build which embeds modules and cannot name its compiler -- but those are a gate and two guesses,
not a diagnosis. I have no Linux machine and did not pretend otherwise.
**A row can be unreachable three different ways and I have now shipped all three.** Unobservable
because every entry was device-unattributed (SOURCE-COSMETIC, two rounds ago); mis-scored by a
positional split (`clear_session_devices`, last round); and now unsatisfiable because an earlier
test subsumes it. The common shape is a counter whose only possible value is zero, and I have not
yet found a way to see it except by someone running the thing on a second machine.
**Deleting an accepted red removes the check, not the acceptance** -- Link found that I did exactly
that with his `how_to_remove_an_entry`, and the screen went from ruling on 8 subjects to 5 and
printed PASS over a red it had stopped looking at. `BUILD_SKIPPED=1; exit 0` reproduced inside the
tool built to make it impossible. I have not closed any red by deletion this round.
### What my verification established, and what it did not
**Established:** `cargo test --lib` **545 passed / 0 failed / 4 ignored**; clippy `--all-targets -D
warnings` clean; `counters_abi.py --check` PASS with `ONNXRUNTIME_VULKAN_EP_LIB` **set**
(`ledger_entries = 115`); `gen_proof_ledger.py --check` PASS, 115 entr(ies), digest
`f3bb172dffd6be28`, arithmetic **115 = 115 identical + 0 + 0 + 0 + 0 + 0**; the eol probe 3/3 with a
negative control that failed 215/215; the subject falsifier 8/8; `audit_instruments` census clean
(8 -> 8 subjects, no drift); gpt-oss-20b census 293/374 reproduced.
**Did not:** anything about Linux. Every arm above ran on Windows against the **debug** build; the
release artifact in this worktree is stale and nothing here speaks for it. **The claim that these
three repairs turn Linux's 0/103 into 103/103 is a PREDICTION, not a measurement** -- what I
measured is that the same *cause* (differing line endings, identical SPIR-V) now produces the
correct verdict here. Link's `.so` has to be rebuilt and re-checked before anyone quotes a Linux
number, and its `source_digest` values will only agree once his build carries this `build.rs`.
The 103 older entries remain `SPEC-UNRECORDED`. And the `UNKNOWN` toolchain cause is undiagnosed.

📌 Team update (2026-08-03T19:55:00-07:00): the deleted-proof incident — Tank found three proofs a merge deleted (proven at 26fd93f, absent from main; the removal happened inside merge commit eb84364 and history simplification hid it from the file's own log), re-proved, ledger 103→106. The deleting merge was the coordinator's, and the op suite was the only instrument that saw it — a git log-visible loss that no ledger check, census, or CI lane flagged on its own. Bears on your ledger-mechanics work: a merge can silently remove entries the same way --reprove and mid-struct insertions have silently corrupted state before. — decided by Scribe, from Tank's finding
📌 Team update (2026-08-03T19:55:00-07:00): Switch's refutation — "Phi-3.5 has never been a valid proof subject." Withholding one form and withholding nine from the graph produce the identical refusal (Shape, ReduceSum, If have no Vulkan handler), so no re-proof run on Phi-3.5 alone can distinguish a broken form from a form the model never reaches. Bears on your gpt-oss-20b second-model work and any future ledger claim that cites Phi-3.5 as the exercised subject. — decided by Switch
---

## §8.11 — the first non-LLM model, `Conv`, and an invariant for proofs that go missing (2026-08-03)

**Round brief:** coverage. Derive state from artifacts, state the criterion before selecting, use
templates, prove everything through the normal path, report the delta as a measurement. The
thesis under test: *the apparatus should make the next twenty kernels far cheaper than the first
twelve.*

### The state, derived from artifacts
`epctl --dump-capabilities --json` -> **91 rows / 73 kernel-carrying (46 live + 27 ready) / 18
staged**. Morpheus's 91/71/20 is the same file two rows earlier; two moved staged -> live. My own
count of this figure had been wrong three times on the record, so I stopped quoting and started
dumping.

### The census that no instrument could have produced
Two LLMs were the whole evidence base and **neither contains a convolution**. I fetched
MobileNetV2-12 (provenance + sha256 in `bench/results/model_provenance.json`) and censused it:
**0 / 105 claimed.** Zero. `Conv` x52 `[not-registered]`, and no `Gemm`, `MatMul`, `Softmax`,
`Transpose`, `Concat`, `Slice`, `Reduce*`, or pooling anywhere in the registry. **No non-LLM model
could run at all.** That is the 164-vs-12 shader ratio measured on a graph rather than counted.

45 of the 104 declines were `[unproven]` on ops we ship **live** — `Clip` and `Add` at
`runtime-extent`, forms no LLM case had ever produced at f32. §8.10's pattern one level out:
point the census at a new model *class* and it finds compiled variants nobody had proven.

### The criterion, and the op it rejected
Stated first: *nodes in a real shipped model that decline today and would claim after the change,
weighted by whether it unblocks a model class rather than completing a taxonomy, divided by
kernel cost; proof-only forms rank above new kernels.*

`Reshape` had 24 real declining nodes and opens the shape-family template — better on the count.
**All 24 are `Add -> Reshape -> QMoE`**, and QMoE is staged, so claiming it moves the island
boundary one node and unblocks nothing. That also re-tested my own `26fd93f` ruling, whose stated
falsifier was "no model contains one" — **the falsifier fired** (24 now exist) and the conclusion
survived for a reason the ruling never named. Recorded both halves.

### What I built and what it cost
One shader (`conv_f32.comp`, grouped as the general case so depthwise is free), one `op_table!`
row, four named declines. **It compiled on the first attempt** — the only build error in the whole
op was a missing `#[derive(Debug)]` on a struct a test called `.unwrap_err()` on.

Six proofs, all through the normal path, no flag disabled, stock `--rtol 1e-3 --atol 1e-5`:
the full `{bias, no bias} x {static, runtime-extent}` cross product for `Conv` (which is the
*entire* key space — arity and `shape_class` are the only components a `Conv` varies in), plus
`add_f32_dyn` and `clip_f32_dyn`. Ledger **115 -> 121**, `--check` PASS,
`6 identical + 115 SOURCE-COSMETIC + 0` everything else.

### The tolerance clause finally binding on something
`_models.py` has reserved the accumulating ops since M0: *derive per vendor, do not guess, do not
copy from fp32 elementwise*. `Conv` is the first accumulating op to land. `probe_conv_tolerance.py`
measured worst `max_rel 1.858e-4 / max_abs 5.722e-6` over twelve cases on the 4060, and `FP32_CONV`
is pinned at `rtol 1e-3 / atol 1e-5` — **exactly the ledger's own defaults**, so the conformance
gate cannot be quoted as looser than the proof gate.

**The probe's first run printed `0.000e+00` twelve times.** ORT had said `Unknown Provider Type:
VulkanExecutionProvider`, fallen back, and compared the CPU against itself. I nearly pinned a
tolerance to a number produced by an instrument that observed nothing. It now registers the EP and
refuses to print for any case the EP did not claim.

### The invariant, built before any op
A merge deleted three proofs and `--check` said PASS, because it asks whether every entry agrees
with the build and never whether one went *missing*. New check:

    {keys ever MATCHed in proof_attempts.jsonl} - {ledger keys} - {retired keys} == empty

It works only because `proof_attempts.jsonl` is append-only and merges **union** it: the two files
fail differently, which is the sole reason one can check the other. Falsified 6/6 by
`probe_ledger_loss.py`, whose third arm **replays the real `eb84364` incident from git** and names
exactly the three lost `Cast` keys.

That arm is also what caught a bug in my own instrument: `def f(path=ATTEMPTS)` binds at *import*
time, so the probe's patching did nothing and the function read today's files while claiming to
replay history. **2/6 on the first run.** A Python default argument is a snapshot; an instrument
whose inputs are snapshots cannot be pointed at history.

### The delta
| | before | after |
|---|---|---|
| registry rows / kernel-carrying | 91 / 73 | 92 / 74 |
| ledger entries | 115 | 121 |
| **MobileNetV2-12** | **0 / 105** | **97 / 105** |
| gpt-oss-20b | 293 / 374 | 293 / 374 |
| Phi-3.5-mini | 355 / 366 | 355 / 366 |
| `tests/ops` | 3 failed / 655 passed | 3 failed / 672 passed |
| `cargo test --lib` | 545 | 551 |

**What it unblocks: one new model class.** The remaining 7 are a single
`Shape->Gather->Unsqueeze->Concat->Reshape` classifier tail plus `GlobalAveragePool` and `Gemm` —
one contiguous island, not seven holes. Those two are the next picks by the same criterion.

### On the thesis
It holds, with an amendment. The kernel was cheap. The expensive parts were **choosing** the op
(which needed a third model that did not exist in the cache), deriving a tolerance the file forbade
guessing, and testing the attribute axes a proof key does not carry. Two of those three are
one-time costs for the *category* of accumulating spatial ops. **The cost moved from the kernel to
the selection** — and selection is a census, an instrument the project already owns.

### What my verification established, and what it did not
**Established:** on this Windows box, debug build, RTX 4060 Laptop GPU — `cargo test --lib`
**551 passed / 0 failed / 4 ignored**; clippy `--all-targets -D warnings` clean; `counters_abi.py
--check` PASS with `ONNXRUNTIME_VULKAN_EP_LIB` **set**; `gen_proof_ledger.py --check` PASS, 121
entries, digest `56c90131c5952a0d`, arithmetic **121 = 6 identical + 115 SOURCE-COSMETIC + 0 + 0 +
0 + 0**, loss invariant **0 missing / 0 retired**; `probe_ledger_loss.py` **6/6**;
`probe_phi35_claim_reading.py` unchanged at `claimed_nodes 355 / unproven_declines 5 /
islands_offered 1`; `tests/ops` **3 failed / 672 passed** with the same three declared reds
(`test_criterion10`, `test_census_baseline_has_no_drift` — bench-only drift, not mine —,
`test_kv_device_residency`); MobileNetV2 **0/105 -> 97/105** from two censuses of the same file
against two builds.

**Did not:** anything about **f16 convolution** — `Conv` declines it and no f16 vision graph has
been censused, so the packed-`uint` argument for a second module is reasoned, not measured.
Anything about **`auto_pad`** beyond that we decline it; I did not verify ORT's optimizers actually
rewrite it to explicit pads on producers other than this one. Anything about **any vendor but
NVIDIA** — the pinned `FP32_CONV` is one GPU's accumulation order and `_models.py` says per vendor.
Anything about **performance**: `conv_f32.comp` is a direct convolution with no tiling and no 1x1
fast path, and I took no timings, by instruction. Anything about **release builds or Linux**.
And the four `Conv` ledger entries establish **nothing** about `group`, `strides`, `dilations` or
`pads` — those are not key components; `tests/ops/test_conv.py` is the only thing that speaks for
them, on twelve combinations, which is not the space.

**Untouched, as instructed:** `shaderInt64`. It still blocks 5 of Phi-3.5's 8 declines and 4 of
gpt-oss's, and none of the ops I added needed it.
