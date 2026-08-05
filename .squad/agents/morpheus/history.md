# Morpheus (Lead-Architect) — history.md

<!-- CONDENSED-AT: 7f0f3dc7fc3e9182fd992deffd71eb62ae010038 -->

## Learnings

## [SUMMARY] Compressed entries

<!-- SUMMARIZED by Scribe 2026-08-02T15:26:05.559743, re-condensed 2026-08-04T20:25:00-07:00 -->

- **[SUMMARY] 2026-08-02T01:42:02-04:30:29: closing then reopening criterion 10 (2026-08-02)** — Restored a merge-dropped sentence from the criterion-11 ruling; declined a rule for the coordinator's "four union-defect" misattribution (R13 unchanged). Closed criterion 10 (both devices MATCH/AGREE, argmax 30751 all runs), falsifier written in advance ("hardened because about to pass" vs "narrowed because it just failed" — refused both). **Reopened three hours later on the same artifact**: the closure certified the *symptom* (cross-run divergence) gone, not the *defect* (zero-init reads from unwritten KV outputs) — `test_phi35.py` Guard 1 already documents unwritten-descriptor outputs reading all-zero on Intel/NVIDIA, applied to output 0 (which has an oracle) but not the 50 reopened KV outputs, so a *clean* arena reads stably-wrong and green. Own R11-obligation-1 error caught same session: read `outputs_compared: 65` (a cross-run count) as 65 oracle comparisons. Discharge redefined as four arms: all-65 oracle with per-output tolerances + two extent keys; a wrong-and-stable planted all-zero control; a non-triviality guard both sides; existing attribution re-emitted not re-argued. Content verification by the ruling party shown weaker than adversarial review (coordinator's Devil's-Advocate framing surfaced the error).
- **[SUMMARY] Sessions 1–22: design, OQ rulings, contrib admission, M0 assessment, R6/R7/R8, §8.8 (2026-07-28–2026-07-30)** — DESIGN.md/README.md produced; capability baseline Vulkan ≥1.1 + sync2 + subgroup_size_control; milestones M0–M3; `com.microsoft` contrib domain admitted.
- **[SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)** — Round 4 cross-agent: worktree layout (`squad/switch` etc.) and inbox portability constraint across git worktrees.

### [SUMMARY] 2026-08-02 to 2026-08-03T11:56 rulings: f32-accumulation/ULP unit, ledger fails-open across devices, three failed figures, cost-argument refutation, key/frame/subject schema, three-clause device-capability test, register-count refutation

- **f32-accumulation ruling:** checked kernels before ruling on economics — every f16 kernel already accumulates in fp32 (`gqa_f16.comp`, layer-norm, GEMV); the question arrived with an embedded false factual claim. Residual is ULPs, not an "accumulation curve" — 64/65 outputs are small integer multiples of the fp16 ULP; rising absolute residual with depth is a scale artifact, not a correctness one. Predicted flat 1-3 ULP across 32 layers, on record before measuring.
- **Ledger fails open across devices:** Link found the EP prints "proven on device0" then claims the form on DEVICE=1 — `LedgerEntry.device` is recorded but read by zero predicates (a field no predicate reads is a comment with a schema, not a guard). Ruled PROVEN / PROVEN-ELSEWHERE / UNPROVEN, naming the silent extrapolation rather than costing it. Toolchain digest ruled over-broad-not-fabricated; demotion splits SUBJECT-CHANGED / TOOLCHAIN-CHANGED, neither claimable.
- **Three figures failed the check (2026-08-02T23:40):** `device0` in `gen_proof_ledger.py` is a selector ordinal, not a device identity. `parse_ledger` did the opposite of its own doc comment (faults the *whole* ledger on any one fault instead of demoting the stale entry). Weight-read-amplification "1.000000 exactly" refuted as `x/x` by definition — an instrument that cannot go red; "bit-identical to CPU EP" also false (62/65, 0.0625 max abs diff). ULP prediction refuted as measured: median 1 (flat, as predicted) but three outputs (logits=12, last layer K/V=4) step — a located defect at the head, not a curve.
- **Fact Checker refutation (2026-08-03T00:20):** the "cheap ULP invariant promotes PROVEN-ELSEWHERE" cost argument was wrong — the ULP series is model-level (indexed by output), no function maps output ordinal to proof key, and `ProofKey::from_node` cannot return evidence about one path for another; withdrew the paragraph, kept the state. Conceded the earlier "six declines" self-count measured numbering, not register growth — handed the tally to Fact Checker permanently. `parse_ledger` fault-scope ruled per-entry.
- Team updates (2026-08-03T04-55): Link — 11 Linux test failures are a bindgen int-width difference, not signedness, but Ubuntu vs Windows SDK shaderc compile different SPIR-V from identical GLSL (faults 74/75 ledger entries). Trinity — at the final RMSNorm, Vulkan is bit-exact against float64 and ORT's CPU EP carries the 1-ULP error; criterion 10's residual is not evidence Vulkan is the imprecise side.
- **Schema ruling (2026-08-03T05:05):** Link's real Linux case forced: **key = form only; frame = device/driver/ort_build/toolchain/tolerance; subject = what code was proven.** Look up by key, compare frame after — a frame mismatch was indistinguishable from key-absence, one `continue` was the whole Linux blocker. Two digests (SPIR-V + source) needed because no single hash is sensitive to the kernel and blind to the compiler; their disagreement is the instrument (nothing moved / SOURCE-COSMETIC / compiler-changed / kernel-changed). Discharges itself: grants Linux enough claim to run the per-form op suite, no model run needed.
- **Three-clause device-capability test (2026-08-03T11:32):** Tank's and Mouse's opposite defaults (claim-when-unsure vs decline-when-unsure) are R13 resolved by an aggregate-vs-itemised distinction, not permissiveness. Test: resolved-before-first-claim, session-immutable, passed-in-as-a-value — settles both in opposite directions correctly. Found §7.5's barrier-contract rule already generalizes here (unindexed by question, not found until now). Declined a sixth "too clean" number but kept the trigger (`UNIFORM(n, verdict)`).
- **Register-count refutation (2026-08-03T11:56):** the "under-numbered" self-diagnosis was itself wrong — independent count found §8.9.x cited 339 times (80 in `registry.rs`) vs ~1,337 for R1-R13; two namespaces were conflated, nobody was lost. 5 of 8 declines fell because others were already using the unminted principle (Trinity, Mouse) — "did I mint a number" and "did the project acquire a binding obligation" differ. Retired the self-scoring tally permanently. Endorsed Switch's tolerance-budget filing: a max-ULP criterion ranks fp16 GPU (337,178 ULP) worse than every int8 lane — a defect in the observable, independent of shipping; `NO_ULP_BAND_ADMITS_INT8` re-verified to survive the spacing-floor repair. int8 KV admission left OPEN pending Trinity's MEASUREMENT-class byte figure.

### [SUMMARY] 2026-08-04T06:40:00: the pre-Conv op count, the metadata variant that denies its own shader, and the self-witness bound

**Wrong count located in the row above the one being fixed.** Op count was reported 91/73 (kernel-carrying); actual is 92/74 (46 `live` + 28 `ready` + 18 `staged`, `epctl --dump-capabilities --json`) — Mouse's own delta table already carried 91/73 as its *before* column, so the stale figure was one row removed from the correction itself. Put provenance in the paragraph, not a footnote.
**`Conv`'s push constants ruled expressions, not paths:** `group`/`strides`/`dilations`/`pads` are uniform in one code path (`conv_f32.comp` branches on none), so the key omitting them is true; the owed remedy is disclosure (`blind_axes` on the row + a CI-time-suite clause), not ~52 new proofs. Separately found all four `Conv` keys render variant as `metadata` ("no shader") while the same entries list `"shaders": ["conv_f32"]` — the key denies a shader the subject has, making the variant component constant across every future `Conv` form and `form_is_provable` short-circuit permissive on this non-composite row (§8.9.21's loud-default test failing past its own reasoning).
**THE SELF-WITNESS BOUND (for Rai's residual):** an instrument reports the last event on its own side of a boundary; a positive reading is a fact about the attempt, never the arrival, and elaborating the instrument does not help since the elaboration runs on the same side. Two opposite-sign specimens: a canary that cannot fail, a `write()` that cannot distinguish success from silent loss — both R9 arriving through the *observer* rather than the observable.
**The stale-citation class is R13, nearly minted as new:** a reference resolving to a plausible non-referent is the defaulting read, but a version has an order an absence does not, so the loud-default remedy cannot reach it. Remedy already exists one floor down: a citation is a proof key with no subject digest — cite a state, not a path.
Carry forward: `blind_axes` is Mouse's, blocking before a second `Conv` variant; `REACHED_USER` should read `WRITE_SUCCEEDED`; 102/121 entries specialisation-unrecorded (§8.9.19/§8.9.21 debt, still unowned, third session naming it); §8.9.22 bought criterion 10 nothing (the test it was built for, in the uncomfortable direction) — preferred the other result, which is why the test was written first.

## 2026-08-04T08:50:00-07:00 — I was handed a true figure and a false inference, and the false half was mine to catch in one line of algebra

**The unsatisfiability claim is refuted by arithmetic that needed no run, and I nearly did not
check it because the shape was so persuasive.** *An unsatisfiable criterion is the dual of an
unfalsifiable one* is a good sentence, it is true, and I have been demoting the other half of that
pair all week — which is exactly why it walked straight past me. `np.allclose` is a **sum**,
`atol + rtol·|b|`, and the finding quoted one term of it. And it divided by the spacing at the
**tensor maximum** while the predicate evaluates **per element**. Two errors, same sentence, both
in the direction of the relaxation.

**The bound settles it for every fp16 tensor this project will ever compare.** `ulp(b) ≤ |b|·2⁻¹⁰`,
so `allowance/ulp(b) ≥ rtol·2¹⁰ = 20.48`, independent of magnitude. Swept the whole normal range:
minimum **20.48000 at |b|=32768**, exactly where the algebra puts it. **The corollary is the part
that inverts the reading and it is the thing I am most pleased to have found:** layer 31's key and
value do not fail sub-step. They fail by **more than twenty representable fp16 steps at their own
magnitudes**. The only thing that made them look sub-step was a step size borrowed from a value
~500× larger.

**And it is §8.9.22 with the sign reversed, on the same instrument, four days apart.** There the
denominator collapsed and made a sound residual look catastrophic. Here it inflates and makes a
real residual look like nothing. **I have now watched one construction fail in both directions**,
which is a better argument for the rule than either specimen alone — it is about the construction,
not about the tensor.

**The narrowing was hiding in the true observation, not in the false one.** The two-mechanism
finding is correct. Splitting the verdict on it would move outputs 63 and 64 out of
`OUTSIDE_TOLERANCE` with **no element moving** — admitting two of the three, in the round the
mechanism was found, on exactly the outputs that would go green. My own §8.9.22 test caught it
instantly. *A taxonomy bolted to the front of a relaxation is still a relaxation.* Worth noting
that the false premise was harmless — it collapses on contact — and the true one was the dangerous
one.

**Applied the test to my own ruling and stated the result in the weak form.** §8.9.24 admits
nothing **because it moves nothing**. That is a cheaper clean bill than §8.9.22's, which was at
risk and passed, and I said so rather than taking the credit.

**What caught my error was already in the artifact before I opened it.** Trinity turned "the
predicate does not read a ULP" from an observation into an **assertion scored against a
hand-written `allclose`**, on every per-output row. A reader arguing from a ULP figure to a verdict
now has the predicate printed in the row they are reading. **That is the loud form, built by
someone else, working on me.**

**The oracle question is now blocking and the ordering is the whole ruling.** At the final RMSNorm
we are bit-exact against float64 and ORT's CPU EP carries the 1 ULP, and nobody has asked which
side is wrong on 0, 63, 64. If it is the reference, every tolerance argument made first was an
argument about the wrong question — **and it would have been made in the direction of loosening,
using the reference's own error as the budget.** Costs a legitimate relaxation its place in the
queue; costs an illegitimate one its only route.

**THE MOVER IS NOT THE MEASURER, and I numbered someone else's sentence rather than my own.**
Trinity wrote it before anyone ruled it and has obeyed it three rounds, including at the moment it
would have turned two of three green. §8.9.18 part 2 says a sentence obeyed as binding is numbered
or withdrawn, so it is numbered. **I recorded the declines and did not score them** — a tally of
her declines has the identical defect as the tally of mine I retired four days ago, and the
temptation to build it was real.

**The second item is a class this project has not had, and the specimen is my prose.** §6.5
obligation 3 *predicts* `alloc_device_frame = SPLIT-DEVICE`; it was read back as a reading. Both
polarities actually read `SHARED`; the discriminant is `alloc_device_frame_allocator_index`, `1`
refusing / `0` passing. **Every prior class here is a failure of resolution** — R13's absent
referent, the amendment's sentinel, §8.9.23(6)'s wrong version. **This one resolves perfectly.**
What changes in transit is the **modality**, and the better-maintained the document the more
convincing the substitution. No number: the remedy is Switch's provenance classes plus one token,
`PREDICTION`.

**The part that lands on me.** I wrote that obligation in the **bare present indicative** — *"a run
with two devices reports `SPLIT-DEVICE`"* — which is an instruction and a description in the same
tense. **You cannot put the whole burden of telling a prediction from a reading on the reader when
the author wrote them identically.** Drafting rule: normative clauses take an explicit modal. Owed
at the next §6.5 edit.

**Carry forward**
- Fifth time this week a rule already in the record answered a question I was treating as new
  (§7.5, §5.4.1(a), the specialisation debt twice, Switch's provenance classes). I have now said
  *the register is under-indexed by question* three times and built nothing. **Named Fact Checker
  as owner of a question→ruling index rather than naming my intention again** — that is the
  decision-versus-mechanism gap I keep grading other people on, closed on my own side for once.
- No tolerance motion on criterion 10 until the float64 side-of-error answer exists. Trinity.
- `allowance_in_ulps_at_scale` companion field is owed in the comparator. Trinity.
- §6.5 obligation 3's tense. Mine, non-blocking.
- Still open and still unowned from yesterday: 102/121 specialisation-unrecorded; `REACHED_USER`
  → `WRITE_SUCCEEDED`; `metadata` variant on `Conv` (Mouse, blocking); int8 KV admission (mine,
  waiting on a MEASUREMENT-class byte figure).
- **The thing to watch:** the false premise was harmless and the true observation was the
  dangerous one. I should stop treating "is this claim true?" as the screening question for a
  criterion motion. The screening question is *what does it admit?*

📌 Team update (2026-08-04T12:25:00-07:00): the field-level reversion class has now happened twice
(Link's 115-of-121 `source_digest` restoration, then a second reversion inside a `squad/mouse`
merge), and every count-based screen read clean both times because the loss was a field inside a
surviving entry, not a missing entry — a screen that counts entries cannot see it. — decided by
Rai, Link

📌 

📌 Team update (2026-08-04T20-25-00-07-00): Mouse's `claimed_nodes` != `dispatches_executed` -- BERT claims 481 of 1274 rows at `GetCapability` but the partitioner's net-benefit gate retains only 4; every coverage figure quoted against `claimed_nodes` alone (including prior island/counterfactual rankings) is affected. `dispatches_executed` is the honest metric going forward. -- decided by Mouse

📌 Team update (2026-08-04T20-25-00-07-00): Trinity's model-scale float64 oracle finding -- at model scale both EPs are ~6x further from true than from each other (common error, not opposed) -- this bounds what an AGREE between the two EPs can mean: consistency-with-each-other is not evidence of correctness against the reals. -- decided by Trinity
