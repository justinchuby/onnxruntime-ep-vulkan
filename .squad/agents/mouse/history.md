# Mouse (Op-Coverage) -- history.md

<!-- CONDENSED-AT: 31f277464ec6a2d7497b63a0384d9015557e05fe -->

## Learnings

### [SUMMARY] Turns 1-26 (2026-07-28-2026-08-01): registry/claim discipline, producer truth, opset windows, zero-logit fix, island wiring, RAI-011, gate-arm attribution

- Registry: 174 standard-domain rows + `com.microsoft` rows; claim log is JSONL; coverage always quoted as `(claimed_coverage, island_count, largest_island_flops)`, never bare percentage. Producer truth: authoritative producer is `onnxruntime/mobius@87fd878`; the emitted model file is the fact, not builder intent. Opset windows are part of claim logic (`ONNX_OPSET_LAST_RELEASED=26`, `REGISTERED=27`); `Attention` closes at opset 24 (input 6 changes semantics).
- Evidence discipline: `Live | Staged(reason)`; `EXERCISED` is the only positive-evidence list; template similarity ruled insufficient for `Sub/Mul/Div/Pow`; a mechanism in a file but not a call graph counts absent until run. Unconditional 4-float push-constant tail unlocked 7 elementwise ops; `Clip` still declines (runtime bounds). `MatMulNBits` shipped Live all-`M` fp32/fp16 off a CPU oracle.
- Shape/dtype: dynamic-shape Phi-3.5 was 356/363; static-gating staged kernels unlocked 0. Runtime-extent gating made 227 nodes predicate-clean, 161 immediately claimable -> first real model run. fp16 elementwise needed `only_proved_dtypes` (not `only_f32`) and packed integer half I/O; Intel exposed an odd-tail/subword bug the 4060 tolerated -> decline ORT subword tensors unless 4-byte-safe.
- **R9 zero-logit (Session 20):** 161 `MatMulNBits` dispatched, `compute_failures=0`, all-zero logits -- green suite falsified. Root cause: dynamic path built a 4-binding descriptor for a 5-binding fp16 kernel, output slot never bound. Fixed via `ShapeOnlyRecorder` preserving `k.bindings`. From here every claim must name the instrument that would go red if false; `model_output_equivalence` mandatory beside counters.
- **Session 23-24:** temp-token infra (`gpu_temps`/`free_all`) added; claiming 64 SkipNorm nodes moved islands 257->321 (node count != island-removal evidence). Multi-node dispatch fixed via island-wide name tokens (`compute_calls = islands x inferences` became the red falsifier for unwired partitioning). Last 10 nodes: only `SimplifiedLayerNormalization`+embed `Gather` claimed; final state **355 claimed / 1 island / 8 declines / 0 cuts**, boundary cost 399,376B upload + 457,344B readback.
- **Session 25-26, RAI-011 + gate arms:** `GetCapability` was short-circuiting to `Verdict::Claim` on single-cluster Phi-3.5, gate unreachable. Fixed: `partition::gate_islands` sole entry point, `GateOutcome::SoleIslandOverride`. Real defect found: `ep.rs`'s boundary-cost estimator counted internal island edges as boundary and substituted 128 for every unknown dim -> **89,199,100,032B estimated vs 856,720B measured (104,116x miss)**. Anchor exemption is the deciding term (shipping keeps the island; disabling it flips to `TRANSFER_DOMINATED`).
- Wall-clock figures withdrawn pending certified device-clock evidence (Intel later ruled permanently uncertifiable on this hardware); only counts/bytes/certified-companion-clock figures are quotable. 85.9% of inference wall-time is non-GPU (recording 68.3%, fence-wait 16.3%, kernels 14.1%) -> GPU-kernel tuning deprioritized behind command-buffer recording.

### [SUMMARY] 2026-08-01T21:15-2026-08-02 (pre-Switch's round): criterion 11 closes then reopens, ledger 9->95, roofline split, reprove-path defects

- Ledger wired end-to-end; two self-caught defects: `CLAIM_UNPROVEN` split keys on `,` (keys contain commas, moved to `;`); `ProofKey::validate` only checked for `/`, now requires `::` + 5 `/` + no empty component. Estimator miss (104,116x) split: internal-edge miscount fixed; `slot_bytes` substituting 128 for unknown runtime-extent dims left open and self-disclosed (~16,268x residual).
- **Criterion 11 reopened:** Morpheus ruled the ledger was scaffolding indistinguishable from the claim table. Repaired: every entry carries `claimed_nodes`/`dispatches_executed`/`worst_rel`; header-vs-file digest refusal; `LedgerLookup::{Hit,KeyAbsent,Faulted,NeverAttempted}` (`LEDGER-FAULTED` outranks `KEY-ABSENT`). Economics bound converted to an assertion that survives a 16,268x adversarial inflation, with a standing falsifier (128 under-counts at long prefill).
- Ledger populated 9->73: op suite 154 failed -> 37 failed; GQA entered `DIVERGENT worst_rel=16.726` (matches pre-existing xfail); non-MATCH verdicts go to `evidence/proof_attempts.jsonl`, never the ledger. Phi-3.5 runtime-extent 0->323/363 (GQA was the real remaining gap); `session.disable_cpu_ep_fallback` wired into `prove()` to distinguish ORT's own refusal from an instrument error.
- **Roofline split:** CPU byte/FLOP share is a curve vs context (0.07%/0%->73.77%/30.2%), not a flat "~3%/~30%" figure; claiming GQA (323->355 nodes, 33 islands->1) is what collapses CPU share. `ep.rs`'s FLOP estimator reported a constant 16.58% at every context (a node-count ratio in FLOP's clothes).
- **No re-proof path (Switch's find):** `--append` silently skipped already-claimed forms; GQA's ledger entry outlived two same-day shader rewrites uncaught. Fixed: per-entry `shader_digest` (FNV-1a/64 over dispatched SPIR-V), `--reprove`, demotion tokens `STALE-SHADER`/`NO-SUBJECT-WITNESS`; all 74 entries re-proved, key set byte-identical. Criterion 10 first reading: `DISAGREE` 65/65 -- a self-caught wrong-denominator tolerance-ratio bug had reported false 24.6x errors; honest ratio shows a smooth crossing (layer 30 pass at 0.85, layer 31 fail at 1.17).
- Staged-op sweep: 21 promoted, 3 refused; harness hole fixed (bool-returning ops need a non-degenerate-reference guard, mutation-tested); two ledger-mechanics defects found by breaking things on purpose (non-`--append` silently discarding 94/95 entries while `--check` PASSed on the empty result; `--reprove` never re-measuring against a healthy ledger). Op-suite reds 43->18.
- Team update (2026-08-02T02:03:46, Scribe): Morpheus's R12 "frame is the binary that ran it" drew partly on two of Mouse's self-caught near-misses (shared-worktree build linking a sibling's in-flight file; `Copy-Item` preserving `LastWriteTime` letting cargo silently re-run a mutated binary).
- Team update (2026-08-02T14:42:30, Switch/Mouse): the re-proof-path fix landed in response to Switch finding the ledger could not be invalidated by changing its own subject.

### [SUMMARY] 2026-08-02 (Switch's round through 8.9.18): ABI-mirror insertions, GEMV kernel identity, PROVEN-ELSEWHERE, compile-time layout guard

- `--reprove` had been silently shrinking the ledger while printing PASS -- fixed, `write_ledger()` refuses any shrink and names dropped keys. **Real find:** `a52024f` inserted `device_losses` mid-struct without bumping `COUNTERS_ABI_VERSION`; three ctypes mirrors silently read every field below it 8 bytes off (`dispatches_executed` read `device_losses`, always a plausible 0). ABI bumped to 4, mirrors repaired, lane changed to `struct_size` equality (not `>=`, since append and insertion are indistinguishable to a `>=` reader). Three hand-maintained ctypes mirrors of one C ABI filed as a standing defect (not fixed).
- Second hand-written `EXERCISED` evidence list (`Add-i32`/`Mul-i32`, consulted inside the claim predicate before any proof key existed) deleted; replaced by `only_loadable_variants` derived from SPIR-V capability requirements. Op suite 11red->6red, ledger 95->97.
- GEMV kernel identity: no artifact recorded whether the packed kernel fired; added JSON-only `pipeline_variants`/`gemv_packed_spec_constant` from the effective (not requested) shader/spec pair, falsifier 5/5 both devices. Found while validating: the mid-struct-insertion defect recurred a third time.

<!-- SUMMARIZED by Scribe 2026-08-04T20:25:00-07:00 -- entries from Round 10 §8.9.19 through the metadata-variant/blind_axes round (2026-08-02 through 2026-08-04, pre-MatMul) condensed below; full text lives in git history (blob 474e653c7cd77c39c6a4afefcd89ef6a41789957) -->

### [SUMMARY] §8.9.19 through the metadata-variant round (2026-08-02–2026-08-04): digest identity, PROVEN-ELSEWHERE, gpt-oss-20b census, Conv/MobileNetV2, BERT first census, and the composite-key defect
- **8.9.19-21:** one proof key now carries two digests (SPIR-V + host-side numeric params); dispatch-time frame witness closed a re-prove-against-stale-binary gap; `--check` widened to verify against the live DLL, not just its own prior output.
- **8.10/8.9.22:** gpt-oss-20b first census, 1->293/374 (stale gating, not a real gap); "portable digest" was a checkout/line-ending fingerprint, not content -- repaired to normalized-content hash.
- **8.11/8.12:** first Conv kernel landed, MobileNetV2 0->97/105; loss invariant formalized as a standing regression gate. BERT first census 473->480/1167, `MatMul` (95 nodes) flagged largest unclaimed op; proof key could not distinguish grouped-vs-dense `Conv` (open question for Morpheus); alloc/free-asymmetry panic found+fixed (root `allocator.rs` asymmetry still open/unowned).
- **Round 2026-08-04 (metadata-variant defect, `#form` reversal, `blind_axes`, two censuses):** fixed the `metadata` placeholder ledger-key defect for all 7 affected row families -- every such row dispatches exactly one shader, 1:1 under `<prefix>_<dtype>`; the composite-dispatch escape hatch was never actually used. Morpheus's ruling (`cpg = c / pc.group` means grouped is the general form, dense is `group=1` inside it -- no separate dense branch) reversed Mouse's own prior `form.rs`, deleted. Latent defect found while fixing the reported one: `variant_stem()` returned the whole variant component, so `@sel`-/`#form`-suffixed keys fell into a permissive unknown-stem branch -- Mouse's own prior `#form` suffix had silently broken this. Census after: MobileNetV2 97->98/105 (form-collapse only, no new kernel), BERT full census 480/1167. `ONNXRUNTIME_EP_VULKAN_DEVICE=1` opens Intel Iris Xe correctly (`PROVEN-ELSEWHERE{device}`) but establishes device identity only, not `FP32_CONV` tolerance on Intel (opened, not measured). `--check` still does not verify ledger-key mintability (flagged again as Tank's ABI-addition to make).
- Team update (2026-08-04T12:25:00-07:00, decided by Rai/Link): the field-level-reversion class (a `source_digest` moved inside a surviving ledger entry, with all count-based screens reading clean) happened twice; `ci/check_ledger_census.py`'s frame-witness arm now requires any deliberate `source_digest` move be declared, else `FAIL(condition=undeclared_witness_move)`.

## Round 2026-08-04 (later still) — `MatMul` shipped, and the census number turned out not to be the one that matters
### [SUMMARY] Round 2026-08-04 (later still) — `MatMul` shipped: `claimed_nodes` is not `dispatches_executed`, and the rank-0 vacuity defect
Registered `MatMul` (95 nodes, largest single BERT unlock). Result: `claimed_nodes` 480->481, but
`dispatches_executed` only 3->4 -- BERT claims 481/1274 rows at `GetCapability` and executes four
nodes; the partitioner's net-benefit gate sees 146 clusters and retains 4. **`claimed_nodes` is not
what executes, and never was** -- last round's "fragmentation, not coverage" verdict on
MobileNetV2 was right about cause, wrong about significance: fragmentation is the whole gap.
`probe_island_counterfactual.py` (first pass, optimistic baseline) ranked picks by retained-island
delta: `Reshape` +167 > `MatMul` +135 > `Concat` +110, with `MatMul`+`Reshape` = 738 nodes in 17
islands vs either alone worth almost nothing -- picks made in pairs from here (later refined, see
below). Own falsifier fired: BERT's 59 `Reshape` nodes carry the attention tensors themselves, so
the real distinction is not metadata-vs-data but **whether the op's output is the tensor or a
description of it**. Latent instrument defect found and named: ORT returns rank 0 for both a
genuine scalar and an unresolved rank; `classify_shapes` returns `Static` because "all dims
non-negative" is vacuously true over an empty list -- **724 of BERT's 1274 claim rows read
`static` having read nothing.** Repaired for `MatMul` only (schema admits no rank-0 operand there);
declined to build a general per-op minimum-rank table uninstructed. Deleted own second,
materially-wrong shape-inference implementation rather than reconciling two readings. `Gemm`
transpose extension ruled in favour but untested by BERT (`transB=0` proven at `transB=0`); stands
on the CI suite alone, unfalsified, pending Morpheus. Mintability gap judged: can wait behind
`Reshape`, not behind a second model.

### [SUMMARY] Round 2026-08-04 (later still) — `Reshape` shipped, proven, claims zero nodes; corrects last round's island count
Registered `Reshape` (predicted +167 top-ranked unlock). Ledger 131->133 (133 identical, 0
missing, 43 retired, 133 mintable); `cargo test --lib` 590->607/0. **Cost stated first:**
`bind_aliased_output` only fires for external plan I/O, not interior island edges, so aliasing
`Reshape` is an engine change (two live tensors, one allocation, independent lifetimes vs a
generation-stamped quarantine-on-free allocator) not an op change -- Switch's KV-disjointness
argument does not transfer. **Prediction published before the build (`a842d57`) missed in both
directions:** predicted 3 claimed nodes, got 0 -- the 3 named were i64 token-id reshapes declining
`[dtype]` (read `output_shapes`, not `input_dtypes`); two predicted-decline nodes passed the first
gate because the vacuous rank-0 defect (named last round) was reproduced in new code one round
later. **Structurally blocked in two places:** no inferred output rank (58/71 BERT `Reshape` take
their shape from a runtime `Cast`/`Concat`/`Shape` chain; `read_const_i64` returns `None` in all
four tree implementations); no output descriptor for a free axis (`[-1,4]` claimed then
`dispatch_ort` refused "output has no declared shape" at `Compute()` -- a broken commitment, not a
decline; standing rule: a gate must refuse what translate cannot *complete*, not what it might
dislike; `Slice`'s starts/ends carry the same trap). Rank-0 discriminator settled by arithmetic
(a rank-0 input holds exactly one element) rather than a minimum-rank table. **`dispatches_executed`
now the headlined metric**, `claimed_nodes` labelled "(upper bound)": BERT 481 claimed / **4**
executed / 4 islands; MobileNetV2 98/97/1; Phi-3.5 355/355/1; gpt-oss-20b not measurable (ORT
1.28.0 CPU `QMoE` refuses session creation, reported `ERROR(instrument)` not zero). **Corrects last
round's headline:** the counterfactual instrument ranked op *types*, the EP claims *nodes* --
re-run with a per-node gated baseline gives BERT `Reshape` optimistic +78/gated **+3**, `MatMul`
optimistic +128/gated **+0**, `Concat` +0/+0; `MatMul`+`Reshape` cumulative optimistic +184 in 27
islands, **gated +3 in 30 islands** -- "738 nodes in 17 islands" was the optimistic column of an
optimistic-baseline instrument. **There is no next op-registration pick on BERT**: the gated
ceiling over every unregistered op is three nodes; the real blocker is that ORT infers no ranks
through its `Shape`/`Cast`/`Concat` chain (724/1274 rows read `static` from nothing; 94 `MatMul` +
53 `Reshape` decline `[unknown-rank]`; `check_shape`'s `[unknown-rank]` branch is dead code on this
model class) -- one problem, three symptoms, not another kernel. Old `Reshape`/`Flatten` decline
ruling reversed by its own named falsifier (BERT has 71 `Reshape`, now `claim=True`, green vs CPU
oracle); `Flatten` stays declined, now measured: BERT 71/0, MobileNetV2 1/0, Phi-3.5 0/0.

### What my verification established, and what it did not
**Established** (RTX 4060 Laptop GPU device 0, Windows, debug, ORT 1.28.0): `cargo test --lib`
607/0; `cargo clippy` clean; **`cargo fmt --check` 0 diffs — RED when first run, in my own new
file, two hunks**; `gen_proof_ledger.py --check` subject arithmetic `133 = 133 identical + 0
SOURCE-COSMETIC + 0 PROVEN-ELSEWHERE + 0 SUBJECT-CHANGED + 0 SUBJECT-INDETERMINATE + 0
no-module-in-build`, loss invariant `176 ever MATCHed / 133 in ledger / 0 missing / 43 retired`,
mintability 133/133 and 43/43; `counters_abi.py --check` PASS; `probe_model_output_agreement.py`
on BERT **AGREE, 0/3 outputs disagree** (max_rel 1.1e-06). `pytest tests/ops`: **5 failed / 861
passed / 34 skipped / 3 xfailed** — 2 `FAIL(condition)` are pre-existing reds (`criterion10`
DIVERGENT, `kv_device_residency` CUDA binding), 3 are `ERROR(instrument)` from `device_losses=1`
in the phi35 lane (suite itself declares this not evidence about the EP).
**Reported because otherwise invisible:** the *first* full-suite run after this change reported
**44 failures in 51 minutes** — a device-loss cascade, not a regression; the immediately following
identical run gave the 5 above in 12:45. I did not
discover this by reasoning; I re-ran because the number was implausible. A single suite run on
this box is not a gate.
**NOT established / not run:** gpt-oss-20b entirely unmeasured (no session on this ORT build).
No f16 `Reshape` anywhere — the row is F32-capped deliberately. **No second device**: Intel Iris
Xe (device 1) never run, so nothing here bears on a UMA allocator or another driver's `ew_cast`
codegen. No release build. **No CI run** — Link owns CI and it has been red on `main` for a dozen
pushes; a green local gate is not a green CI. `DEVICE_MEMORY` and `KV_ARENA` never enabled, per
instruction, so nothing here bears on Switch's `ctx-4096` `ep_inter_76` failure. `allowzero=1` is
**declined, not handled** — no graph exists to test it on and an attribute claimed-but-untested is
the `Gemm` transpose mistake. The 68% claim-log name-match rate on BERT is a real limit on the
counterfactual table and biases every delta *downward*; the tool prints it.
