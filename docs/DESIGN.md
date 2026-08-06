# onnxruntime-ep-vulkan — Architecture Design

**Status:** v0 architecture of record — accepted for M0/M1 implementation. **§7 (Vulkan baseline) is frozen.**
**Date:** 2026-07-28T17:59:54-07:00 · **Last revised:** 2026-08-05T03:15:00-07:00 (**RULINGS §8.9.26 AND §8.9.27 — ANCHOR PHRASES: "A GATE WHOSE TRIGGER AND WHOSE REMEDY ARE KEYED ON DIFFERENT VARIABLES IS UNSATISFIABLE BY CONSTRUCTION", "AN INERT MECHANISM — THE REGISTER WORKED; THE READER DID NOT EXIST", "NAME THE READER AND NAME THE ACTION IT TAKES ON RED, BEFORE THE MECHANISM SHIPS."** **§8.9.26:** `decisions.md` stands at 67,623 bytes against a 51,200-byte Tier-2 gate with **zero entries age-eligible**, because the trigger is keyed on **size** and the only remedy is keyed on **age** — on a project writing faster than its own retention window the condition is not unmet but **unmeetable**, and Scribe has been left recording the breach as a judgement two rounds running. **This is §8.9.22/§8.9.24's demoted criteria with the sign reversed: not a check that can never go red, but a remedy that can never apply.** Rejected by name, each with what it admits: **velocity-proportional age tiers** and **size-triggered oldest-N** both make the newest round archivable at sufficient velocity — a remedy keyed on **rank** or **rate** can always reach a round that is still load-bearing; **"advisory when no remedy exists"** admits every breach and mints the inert-mechanism defect as policy; **"the threshold is wrong"** is true but insufficient, since it reproduces the standoff at the new number. **RULING: eligibility is re-keyed from age to supersession.** Three clauses, **modality declared `SUFFICIENT`** per §8.9.25 — **(a) SUPERSEDED** by a later, active, differently-authored entry that names it; **(b) DISCHARGED**, with the closing entry named and not itself; **(c) AGED**, the 30/7-day windows unchanged so nothing eligible before becomes ineligible now. Age was always a proxy for *is this content still doing work?*; supersession and discharge are **positive facts asserted by a later named entry**, and a remedy keyed on a positive fact **cannot reach a live round by construction**. **The safety condition is RESOLUTION, NOT SILENCE:** an archived entry's citations are unharmed if they still resolve, so the obligation moves onto the archivist — the `ARCHIVAL POINTER` must record each archived entry's heading, archive path, and **every token it mints**, or archival manufactures R13. A citation screen is retained **as a veto, never as a grant**. Demonstrated rather than asserted: Round 10's *"six declines"* entry is SUPERSEDED by name in Round 12 and is eligible; the other four are cited by live work (`counters_abi` 105 hits, `pipeline_variants` 35) and are not. **It does not clear 51,200 this round, and that residue is the ruling's product** — with eligibility keyed on supersession, a file over the threshold is a **true report of undischarged-obligation backlog**, and the correct response is to close obligations, not to cut the record. The gate becomes **diagnostic** instead of unsatisfiable. **AND THE CLASS UNDERNEATH IS NAMED: AN INERT MECHANISM.** Four specimens — the archive gate (fires, is read, hands its reader no permitted action), `ci/check_open_reds.py` (red and unread), `ci/check_flake_witness.py` (wired at four call sites in `ci.yml`, never handed a real log), and `LedgerEntry.device` (recorded, read by **zero** predicates — *"a field no predicate reads is a comment with a schema, not a guard"*, my own Round-9 prose, the class never named). **It is distinct and the discriminator is mechanical:** an unfalsifiable observable is a defect in the **predicate**, R13 in **resolution**, the self-witness bound in the **observer's position** — an inert mechanism has a sound predicate, resolving referents and the right vantage, **and its verdict changes nothing**. The proof of distinctness is that both CI specimens **ship negative controls** (`negative_control_open_reds.py`, `negative_control_flake_witness.py`): a negative control demonstrates a check *can* go red, which refutes "unfalsifiable" and **is silent on "inert"**. Two forms — **INERT/UNREAD** (no consumer) and **INERT/UNACTIONABLE** (consumer with no permitted action). **Minting rule, binding on every new mechanism: name the reader and name the action it takes on red, before the mechanism ships** — an inert check is worse than no check, because it consumes the attention a real one would have had and reports coverage it does not provide. Sixth occurrence of *a rule already in the register answering a question treated as new*, and the first where the earlier statement was my own unnumbered prose. **§8.9.27:** the two items §8.9.25 left open are **one item**. **`witnessed_at` is WITHDRAWN BEFORE IT SHIPS and Mouse is not needed** — §8.9.23 already ruled this exact question for `Conv` (disclosure + a CI-time suite), and both halves exist for `Gemm` today (`OpSpec::blind_axes` declaring `alpha`/`beta`/`transA`/`transB`, rendered by `disclosure::blind_axes_clause`; and `tests/ops/test_gemm_and_pool.py`, which says in its own docstring that it *is* the suite that disclosure points at, running `transA`/`transB`/`transAB`/`alpha_only`/`beta_only`/`negative_alpha` and every legal `C` broadcast). **The disqualifying ground is that `witnessed_at` as specified is an INERT/UNREAD mechanism** — defined non-key, so no predicate may read it, which is `LedgerEntry.device` again; and if a predicate ever does read it, a non-key field has become a key component through the back door and `Gemm`'s key silently reacquires the axes §8.9.23 ruled out. **Inert if it stays honest, unsound if it becomes useful.** **The dependency is the finding:** that closure rests on a suite, and a suite is a discharge only if it runs or says truthfully why it did not — `test_transb_is_a_transpose_and_not_a_relabelling`, **the only test in the tree that can tell a transpose from a relabelling** (`B.T` and `B` are the same multiset), takes no `require_vulkan` and reports `skip("EP did not claim the transB identity case")` **on a run where no EP existed**. Five call sites share the shape (three in `test_gemm_and_pool.py`, two in `test_conv.py`); the correct form exists twice in the tree already (`probe_conv_tolerance.py` raises `ERROR(instrument)` on the identical condition; `test_no_cpu_fallback.py` carries the fixture on every EP-probing test), and the `declines` family is **out of scope** because `assert_vulkan_does_not_claim` documents its own vacuity — a disclosed vacuous pass is not an undisclosed false skip. **Owner: TRINITY**, on the harness ground — `require_vulkan` is her fixture — **and the deliverable is not five decorators** but a screen that makes the shape unrepresentable: *a test in `tests/ops/` calling `is_vulkan_claimed` without requesting `require_vulkan` is a CI failure*, reader `lane_checks_suite`, action on red the test is fixed or the exemption written down. **§8.9.27(1) is CLOSED conditional on (2) landing, and reopens if it does not.** Both rulings lower no threshold, relax no gate and cannot turn a red row green; §8.9.26's exposed clause is (3)(b), whose falsifier is recorded in advance — if a discharge-under-(b) ever archives a live obligation, **(b) is withdrawn and (a) stands alone**.) · **Date:** 2026-07-28T17:59:54-07:00 · **Last revised:** 2026-08-04T19:40:00-07:00 (**RULING §8.9.25 — ANCHOR PHRASES: "AN AGREEMENT BOUNDS ONLY THE DIFFERENCE, NEVER THE DISTANCE FROM TRUTH", "A CLOSE CONDITION DECLARES ITS OWN MODALITY — SUFFICIENT OR UNBLOCKING — OR IT IS READ AS SUFFICIENT", "A DIRECTION THAT EXISTS FOR ONE OUTPUT IS NOT A BUDGET FOR THREE." CRITERION 10 STAYS OPEN AND `DIVERGENT`; THE ORACLE ANSWER IS THE REASON IT MAY NOT BE LOOSENED, NOT A LICENCE TO LOOSEN IT.** **(1) THE BLOCKING CONDITION OF §8.9.24(4) IS DISCHARGED AND DISCHARGING IT GRANTS NOTHING.** Trinity's model-scale float64 oracle answers *which side is further from true* for all three failing outputs, layer-at-a-time over 355 nodes and 32 proven-live layers, both devices, both reference variants, with the chain reading **initialisers and `input_ids` only** so that neither EP appears in its own derivation. The answer is **not uniform**: output 0 (logits) has a direction and it is **`cpu`** — ORT's own CPU EP, unanimous on five discriminators, both variants, both devices, **83 vs 70 element-ULP from true**; outputs 63 and 64 have **`direction: null`**, discriminators conflicting within a variant and the variants disagreeing across it. **This is exactly the shape a loosening argument wants and it is refused.** A tolerance motion built on it would admit **three** outputs on a direction that exists for **one**, and on the other two it would have to *default* — §8.9.21's loud-default failing permissive, on precisely the outputs that would go green. A verdict structure keyed on direction is §8.9.24(2) with a new taxonomy bolted to the front: it moves outputs out of `OUTSIDE_TOLERANCE` **with no element moving**. `atol` and `rtol` do not move — fifth round — and neither does the predicate, the unit or the verdict structure. **(2) THE DEPTH SERIES I ASKED FOR EXISTS AND IT RETURNED THE BRANCH THAT CONVICTS.** `bench/results/criterion10-dev{0,1}.json` carries `kv_depth_curve` over all 32 layers: median 0–2 ULP for layers 0–30, `kv_depth_largest_step` **1.0**, and `kv_depth_exceedances` = **layer 31, key and value, 4.0 ULP** — and layer 31's key and value **are outputs 63 and 64**. My own 2026-08-02 sentence reads *"flat ⇒ no accumulation defect; a step at layer L ⇒ a real defect, localised."* **The instrument I demanded, in the unit I demanded, located the defect at the one layer whose two outputs are two of the three that fail.** **AND MY CLOSE CONDITION WAS WRITTEN IN A MODALITY IT DOES NOT DECLARE.** *"It closes when the ULP series exists and is either flat or has a located step"* reads as **sufficient**; it was meant as the discharge of a blocking objection about the unit. Read as sufficient it closes criterion 10 today — on an artifact whose own `verdict` is `DIVERGENT` and whose `per_run_comparison` is `[DISAGREE, DISAGREE, DISAGREE]`, i.e. it closes a criterion whose text requires `MATCH` on a series that is not the criterion. **RULING: it was UNBLOCKING, not SUFFICIENT.** Drafting rule, binding this document and every criterion row in it: **a close condition declares `SUFFICIENT` or `UNBLOCKING`; an undeclared one is read as sufficient and therefore permissive.** This is §8.9.24(6)'s modality substitution one day later, same author, in the **deontic** direction instead of the tense one — *a condition that removes an objection* read as *a condition that grants a pass* — and it takes no number, because the remedy is the identical explicit modal. **(3) AN AGREEMENT BOUNDS ONLY THE DIFFERENCE.** Both EPs sit far further from a weight-only float64 reference than they sit from each other — their error is largely **common**, not opposed — **and I re-derived the ratio per output rather than quoting the model-scale figure, which does not survive: logits 70/83 vs 12 apart (5.8–6.9×), `present.31.key` 12/12 vs 4 (3.0×), `present.31.value` 6/7 vs 4 (1.5–1.75×).** "Roughly 6×" is a fact about the logits and is **not** a fact about the model; the ratio is quoted per output or not at all. **RULING: `comparison = AGREE` in `compare_all_outputs_to_cpu` means CONSISTENT-WITH-CPU and never CORRECT**, and a correctness claim resting on an AGREE carries that output's measured common-error ratio or is unquotable — a `means` string on the comparison itself, the mechanism `output_coverage.means` already uses. Owner: Trinity. **AND THE ASYMMETRY, WHICH IS THE LOAD-BEARING HALF AND CLOSES THE DOOR THIS RULING WOULD OTHERWISE OPEN:** a shared error **cancels in a difference**. `|a − b|` is untouched by any error both sides carry, so it remains an exact lower bound on the differential error. **The oracle weakens an AGREE and leaves a DISAGREE exactly where it was**, and criterion 10's red stands on the half of the picture the common-error finding does not reach. The common error is also **not** a defect in either EP: one graph, one set of weights, one storage format — two implementations of an fp16 graph are *expected* to share most of their distance from the reals, and what the finding bounds is how much correctness evidence **any** two-implementation agreement can ever carry. **(4) A QUOTABLE DIRECTION BESIDE A FLAG THAT DENIES IT.** The chain's roll-up is right — `direction: null` with a note. One level down, `by_reference_variant.f64.which_is_further_from_true` still reads `"vulkan"` on output 63 while `discriminators_conflict` is `true` in the same object, with the caveat carried in prose. **A record can default loudly at the top and quietly one level down, and the level a reader quotes from is the deepest one that answers their question.** The field carries the absence, not the value: `null` whenever the discriminators conflict, the conflicting verdicts staying in `verdict_by_discriminator` where they already are. Owner: Trinity; changes no verdict. **(5) `Gemm`'s TRANSPOSES STAY OUT OF THE KEY, AND WHAT IS OWED IS A RECORD, NOT A KEY COMPONENT.** §8.9.23 applies verbatim — `gemm_f32.comp` selects the index with a ternary on a push constant. **The axis is not untested**: `tests/ops/test_gemm_and_pool.py` runs `transB`, `transAB`, `wide_k` and `mobilenet_head` through `m.check`, which asserts the claim before comparing, and `ledger_case_models.py` mints `gemm_f32_transb{,_nobias,_dyn}`. **What is blind is the entry, not the axis**: those cases mint the same key as their `transB=0` siblings by design, so a reader cannot distinguish a key proven at both values of a blind axis from one proven at one. Remedy — **a non-key `witnessed_at` field on the ledger entry**, recording per declared blind axis the values its contributing cases carried; not a key component, which would re-split the space §8.9.23 deliberately merged. Owner: Mouse. **The `[rank]` decline is not about `transB`** — `translate` requires `a.rank() == 2` and the same head form at rank 2 is claimed in-lane, so reading the decline as "the axis cannot be exercised" mislocates a fact about `A` onto an attribute of `B`. **And the one I found by running the tests rather than reading them:** `test_transb_is_a_transpose_and_not_a_relabelling` — the only test in the tree that can tell a transpose from a *relabelling* — takes no `require_vulkan` fixture, so with no EP loadable it falls back to CPU and reports `pytest.skip("EP did not claim the transB identity case")`. **The EP was not present; the skip reason names a refusal that never happened.** Two terminal states, one token, and the token names the one that would be a finding. The R13 lane summary caught it (`LANE FAILURE (fallback log): 1`) — the lane is not silent, the test's own reason string is what lies. Fix: take the fixture like its fifteen neighbours. Owner: Mouse. No number: this is *A PREDICTION IS NOT A READING* in a third costume — a diagnostic that resolves perfectly to the wrong one of two branches — and the remedy is the same, name the branch you observed. **(6) THE README PROVENANCE FILING IS REFUTED, AND THE REFUTATION IS THE BETTER FINDING.** The filing says *"94 rows, of which 76 carry a kernel — read from `epctl --dump-capabilities --json`"* sources two numbers to a command yielding one, the dump having no kernel field. I ran it: 94 rows, `status` ∈ {`live` 46, `ready` 30, `staged` 18} — **and the row's boolean `live` is `true` on exactly 46 + 30 = 76 rows and `false` on all 18 staged.** The dump does carry the kernel fact. It carries it under the name `live`. **The citation is sound and the schema is not: one JSON row spells the noun `live` twice with two denotations** — `status == "live"` (46, the deprecated `OpStatus::Live` alias that grants nothing) and `live == true` (76, "this row has a kernel"). A reader checking the 76 reads the field named `live` and gets 46; the filing's own footnote records the first arm of that collision happening to its author without recognising it as the second. **RULING: the boolean is renamed `has_kernel`; `status` keeps its three tokens.** Owner: Tank for `epctl`, Mouse for the registry serialiser. Until it lands both this document and `README.md` state the derivation in the sentence — *76 = rows with `live == true`, which is not `status == "live"` (46)* — because **a true sentence whose check requires knowing that one word means two things is checkable only by someone who already knows the answer.** **A noun retired in prose and left in a schema is retired in the one place nobody quotes from.** §0's own count is corrected here from 92/74/46-28-18 to **94/76/46-30-18** — the fifth wrong reading of this integer on the record, moved by the `MatMul` and `Gemm` rows exactly as the last correction predicted, and it does not stop being wrong until it stops being written by hand. **(7) WHAT THIS RULING ADMITS, IN THE WEAK FORM AGAIN.** No predicate, threshold, tolerance or verdict changes; criterion 10 is `DIVERGENT` before and after; the three items opened — `means` on the comparison, `witnessed_at` on the ledger entry, `has_kernel` on the dump — are additive and none can turn a red row green. **It admits nothing because it moves nothing.** The one place it could have admitted something is (2), where the permissive reading of my own sentence closes criterion 10 today; I record that I found that reading in my own prose before anyone offered it to me, and that finding it is not the same as being immune to it.) · **Last revised:** 2026-08-04T08:50:00-07:00 (**RULING §8.9.24 — ANCHOR PHRASE: "A TOLERANCE IS UNSATISFIABLE ONLY WHERE ITS ALLOWANCE FALLS BELOW THE SPACING AT THE POINT THE PREDICATE IS EVALUATED." CRITERION 10's TOLERANCE IS SATISFIABLE EVERYWHERE IN fp16 WITH A FACTOR OF 20 TO SPARE; `atol` DOES NOT MOVE, THE VERDICT DOES NOT SPLIT BY MECHANISM, AND THE ORACLE QUESTION IS NOW BLOCKING.** **(1) THE UNSATISFIABILITY FINDING IS REFUTED BY ARITHMETIC THAT NEEDED NO RUN, AND IT FAILS TWICE IN ONE SENTENCE.** `np.allclose` tests `|a − b| ≤ atol + rtol·|b|`. The claim *"`atol=1e-3` is 0.128 ULP-at-scale on the logits — the tolerance demands finer than fp16 can express"* **(a)** quotes one term of a two-term sum and drops the term that dominates everywhere it matters, and **(b)** divides by the spacing at the **tensor maximum** while the predicate is evaluated **per element** — a numerator and a denominator taken at two different points. Full allowance at each tensor's own scale: **33.628 / 29.796 / 32.404 ULP-at-scale** on logits / `present.31.key` / `present.31.value`, i.e. **263× / 116× / 506×** the quoted figure. **The bound that settles it for every fp16 tensor this project will ever compare:** for a normal `b`, `ulp(b) ≤ |b|·2⁻¹⁰`, so `allowance/ulp(b) ≥ rtol·2¹⁰ = 20.48` **independent of magnitude, tensor and scale** — swept over the whole fp16 normal range the minimum is **20.48000 at |b| = 32768**, exactly where the algebra puts it; on subnormals `ulp = 2⁻²⁴` and the `atol` term alone gives **≥ 16,777 ULP**. **Corollary, and it inverts the reading of the failures: a failing element is one whose residual exceeded an allowance already ≥ 20.48 element-ULPs wide, so layer 31's key and value do not fail sub-step — they fail by more than twenty representable fp16 steps at their own magnitudes**, and the only thing that made them look sub-step was a step size borrowed from a value ~500× larger. **This is §8.9.22 with the sign reversed on the same instrument four days apart** — there the denominator collapsed and made a sound residual look catastrophic; here it inflates and makes a real residual look like nothing, which is the strongest available argument that the rule is about the construction and not about the tensor. `atol` does not move, `rtol` does not move, criterion 10 stays `DIVERGENT`. **The mirror-image principle is nonetheless affirmed: an unsatisfiable criterion is the exact dual of an unfalsifiable one and is demoted for the same reason — it cannot be met however correct the implementation is. The rule earns its number by being the cheap test that tells the two apart.** **(2) THE VERDICT DOES NOT SPLIT BY MECHANISM, AND THAT IS WHERE THE NARROWING WAS HIDING — IN THE TRUE OBSERVATION, NOT THE FALSE ONE.** The three failures really are two mechanisms — 6,056 elements on the logits is a bulk residual, 16 and 2 elements at layer 31 are a tail — and that is a fact about **causes**. **A verdict is a fact about a predicate; a mechanism is a fact about a cause. Both are reported and only one is a verdict.** Splitting would move outputs 63 and 64 out of `OUTSIDE_TOLERANCE` with **no element moving**: it admits two of the three failures, in the round the mechanism was found, on exactly the outputs that would go green — **a narrowing with a taxonomy bolted to the front of it**, failing §8.9.22's own test that *a change which makes nothing pass that did not pass before is a repair*. **Applied to this ruling, in the weak form deliberately: §8.9.24 admits nothing because it moves nothing** — a cheaper clean bill than §8.9.22's, which was at risk and passed. **(3) THE `ULP-at-scale` STATISTIC IS FENCED, NOT WITHDRAWN.** It soundly answers *"is this residual large relative to the tensor?"*. It may not stand beside a predicate it does not participate in, in a unit that predicate does not use. **Any row reporting a residual in `ULP-at-scale` also reports the allowance `atol + rtol·|b|` in the same unit and the failing set on the element basis**; `failing_residual_within_one_ulp_at_scale` may not appear without a companion `allowance_in_ulps_at_scale`. Owner: Trinity, in the comparator that already carries `verdict_predicate` — **and `verdict_predicate` is why this round produced a refutation instead of a relaxation: the mechanism that caught the error was in the artifact before the ruling opened it.** **(4) THE ORACLE QUESTION IS BLOCKING AND THE ORDERING IS THE RULING.** No motion to change criterion 10's tolerance, unit, predicate or verdict structure is entertained until outputs 0, 63 and 64 have a **float64 answer to which side is wrong**. At the final RMSNorm **Vulkan is bit-exact and ORT's CPU EP carries the 1 ULP**, and `divergent` has never once been asked *which* of the two is wrong. **If the reference is the wrong side, the correct remedy is a different oracle and every tolerance argument made first was an argument about the wrong question — and it would have been made in the direction of loosening, using the reference's own error as the budget.** Owner: Trinity. **(5) THE MOVER IS NOT THE MEASURER — ratified, because it is already being obeyed.** *A motion to change a criterion is not authored by the party whose run produced the failure it would relieve, and not in the round in which that failure was produced.* Both clauses are required: **the second alone permits laundering by delay, the first alone permits laundering by proxy.** Trinity wrote this sentence herself before anyone ruled it and has declined to move `atol` for three consecutive rounds, including once at the precise moment it would have turned two of three outputs green; §8.9.18 part 2 says a sentence obeyed as binding is numbered or withdrawn. **The declines are recorded and not scored** — a tally of another agent's declines carries the identical defect as the tally of my own that §8.9.18 part 4 retired. **(6) A PREDICTION IS NOT A READING — A NEW CLASS, RECORDED WITHOUT A NUMBER, AND THE SPECIMEN IS MY OWN PROSE.** The Intel arena refusal was first explained by citing `alloc_device_frame = SPLIT-DEVICE`; **§6.5 obligation 3 *predicts* exactly that token** — *"a run with two devices reports `SPLIT-DEVICE` on the transfer accounting"* — and the prediction was read back as a reading. One arm per process with a fresh counters file shows the frame is **`SHARED` in both polarities**; the real discriminant is one field over, `alloc_device_frame_allocator_index` = **`1` on the refusing arm and `0` on the passing arm** while the session's own device list puts the device at index `0` in both (`bench/results/arena_refusal-dev1-{noenv,pinned}-arena.json`). **A right headline with a wrong cause bolted to it.** **It is not a class we have: R13's referent is absent, R13-amendment-1's lookup resolves to a sentinel, §8.9.23(6)'s citation resolves to the wrong version — all three are failures of resolution. This one resolves perfectly.** The document is present, current, correctly cited and **true**; what changes in transit is the **modality** — *what the mechanism will report* becomes *what it reported* — and no lookup fails, nothing defaults, nothing is stale, and there is no silence for a loud default to fill. **The better-maintained the document, the more convincing the substitution.** **No number is owed because the register individuates by remedy and the remedy is already in the tree:** Switch's provenance classes gain a third token — **`MEASUREMENT` / `MODEL` / `PREDICTION`** — a `PREDICTION` may not be quoted where a `MEASUREMENT` is expected, and a claim resting on one **names the artifact that would have carried the measurement and reports it absent**. Numbering follows citation; the count is Fact Checker's. **And the half that lands on the author rather than the reader: §6.5 obligation 3 is written in the bare present indicative, which is an instruction and a description in the same tense. The obligation to distinguish a prediction from a reading cannot rest entirely on the reader when the author wrote them identically.** Drafting rule, binding this document first: **a normative clause about what an instrument must report takes an explicit modal — `must report`, `is required to report` — never the bare present indicative.** Owed at the next §6.5 edit; not blocking, because the artifact-side remedy does not depend on it.) · **Date:** 2026-07-28T17:59:54-07:00 · **Last revised:** 2026-08-04T06:40:00-07:00 (**RULING §8.9.23 — ANCHOR PHRASE: "A KEY NAMES THE PATH; THE CLAIM MUST NAME THE DOMAIN." THE `Conv` ATTRIBUTES DO NOT BELONG IN THE KEY, THE `Conv` KEY IS NEVERTHELESS FALSE TODAY, AND TWO CLASSES ARE SETTLED — ONE NEW, ONE NOT NEW.** **(1) MOUSE ASKED "SCHEMA CHANGE (YOURS) OR KEY COMPONENT (MINE)"; THE ANSWER IS NEITHER, AND THE THING THAT IS OWED IS A DISCLOSURE.** `group`, `strides`, `dilations` and `pads` are **push-constant values read by one uniform code path**: `rust/shaders/glsl/conv_f32.comp` computes `cpg = pc.c / pc.group`, `mpg = pc.m / pc.group`, indexes with `pc.stride_h`/`pc.pad_h`/`pc.dil_h`, and **branches on none of them** — grouped is the general form, dense is `group == 1`, depthwise is `group == C`, and `ops/conv.rs::translate` emits one `KernelRequest` with `spec_constants: vec![CONV_LOCAL_SIZE]` for all of them. Under §8.7's expression-vs-path distinction these are **expressions**. Two `Conv` nodes differing only in them are dispatched by the same module with the same bindings, which is exactly and only what a shared key asserts, **so the key is true**. Adding them would make it false in the other direction: it would assert that a stride-2 `Conv` runs different code from a stride-1 `Conv`, and it would demand a separate proof for most of MobileNetV2's 52 nodes. **A key that over-separates loses the one property a key exists for — "proof of one is proof of the other."** The test that settles this class in general, and it is not new, it is §8.9.21's frame test applied to attributes: **an attribute belongs in the key iff it selects the emitted code (a spec constant, a module stem, a binding arity); an attribute that only parameterises arithmetic inside one code path belongs to the case, not to the form.** `IsInf(detect_negative)` passes and is in a key; `Conv(group)` fails and must not be. **(2) WHAT IS OWED IS THAT THE CLAIM PUBLISHES WHAT THE KEY IS BLIND TO, AND THIS CLOSES RAI'S 🟡 WITH THE SAME MECHANISM.** `disclosure::disclose_claimed_forms`'s `Proven` arm prints the raw `ProofKey` and the artifact and nothing about the key's **domain**; a user who does not already know the schema reads "`Conv` proven" as "`Conv` is correct". **A proof key that a reader cannot tell the blindness of will eventually be read as a claim it never made — and the remedy is not to widen the key, it is to ship its blind list beside it.** Mechanism, and it is deliberately the cheapest one that cannot drift: a `blind_axes: &'static [&'static str]` on the row in `registry::OpSpec` — a registry field, not a ledger schema change, therefore Mouse's — rendered into the disclosure line for any row that declares one, together with the sentence that the axes are spoken for by a **CI-time** suite (`tests/ops/test_conv.py`, twelve combinations) and by nothing that ran in the reader's session. **(3) FOUND WHILE RULING, LARGER THAN THE QUESTION ASKED, AND BLOCKING: `Conv`'s KEY SAYS IT HAS NO SHADER.** All four entries render the variant component as the literal `metadata`, whose documented meaning in `registry::variant_key` is *"this row has no shader"*, while the **same entries** record `"shaders": ["conv_f32"]` with a real `shader_digest` and `source_digest`. Cause: `ops/conv.rs` registers `kernel!(None)`, so `spec.kernel.stem(F32)` is `None` and the sentinel is taken; the module is chosen inside `translate`, where the key does not look. **The subject knows the shader; the key denies it exists.** Three consequences, in order of when they bite: it is **false now**; the variant component is **constant across every present and future `Conv` form**, so the one mechanism that would separate a specialised grouped or depthwise kernel from the general one is inert on precisely the op that will acquire one first; and `registry::form_is_provable` short-circuits on `variant_is_generated("metadata")`, so **`Conv`'s provability is answered without ever consulting `conv_f32`** — in a shaderless build its key still reads provable. That is the **loud-default test (§8.9.21 part 3) failing in the permissive direction**, on a row that is not composite, which is the case that ruling did not anticipate. Repair, Mouse, before any second `Conv` kernel variant exists: **the variant component must be named by the code that dispatches**, not by a kernel table the row does not populate. Do **not** repair by widening the `metadata` sentinel's meaning — that is the cheapest satisfaction and it hides the same blindness one level further in. **(4) WHAT THE KEY IS STILL BLIND TO AFTER ALL OF THAT, STATED BECAUSE I HAVE TWICE FOUND THAT THE CHEAPEST SATISFACTION OF A DEFINITION IS THE ONE THAT HIDES ITS OWN BLINDNESS:** `group`/`strides`/`dilations`/`pads` by ruling; `auto_pad` and spatial rank ≠ 2 **not at all**, because both are declined at claim time and a declined node reaches no key — so the key's silence there is correct and unreadable as such; accumulation order across `cpg`, which is the axis the derived `FP32_CONV` tolerance absorbs on **one vendor** (`docs/OP_COVERAGE.md` §13.9.5); and — the one that will bite — **the blind list is written by hand, so it is blind to any axis nobody listed.** That is the same defect the shader-variant list already carries in `ops/elementwise.rs` (*"a list nobody can falsify is the next thing to go wrong"*), and it is accepted here **explicitly** rather than discovered later. **(5) THE SELF-WITNESS BOUND — ANCHOR PHRASE: "AN INSTRUMENT REPORTS THE LAST EVENT ON ITS OWN SIDE OF THE BOUNDARY."** Rai discharged RAI-013 by measurement — the default-arm disclosure reaches the console by a **direct stderr write that bypasses ORT's threshold entirely**, both devices, both polarities, with a working negative arm — and named the residual as the same class as this session's canary-token unobservability. It is, and here is the general form, which is not about stderr: **where the property of interest lies past a boundary the process cannot cross, an instrument can report only its own last action before the boundary; the positive reading is a fact about the attempt, never about the arrival, and no elaboration of the instrument moves the boundary, because the elaboration runs on this side too.** It has two arms of opposite sign and both are live in this repository today. **Arm (a), the guaranteed antecedent:** a check whose triggering condition is *made true by the context in which it runs* carries no information about the thing it warns of — a canary that can only be read by the reader whose absence it detects. **Arm (b), the unavailable observation:** `session_disclosure_info_reach: REACHED_USER` is produced by a `write()` that returns success identically for a terminal, a log file and a pipe with no reader. Arm (a) is a check that cannot fail; arm (b) is a check that cannot distinguish; **both are instruments whose output is a function of the apparatus and not of the subject**, which is R9's test arriving through the observer rather than through the observable. **Disposition is disclosure, never repair — and disclosure means the token names our side of the boundary.** The §0.2 line is landed here. `REACHED_USER` should read `WRITE_SUCCEEDED`; that is a one-token change, it is owed to whoever owns `disclosure.rs`, and it is **not** blocking, because Rai's residual is correctly a bound and not a violation. **(6) A TRUE CLAIM ON A STALE CITATION IS R13, NOT A NEW CLASS — AND THE REMEDY R13 CARRIES DOES NOT REACH IT.** Rai verified RAI-012 fixed by rebuilding under WSL and running the real suite (42 declines, all the corrected message, zero old-message instances) while the three artifact files cited as evidence remain **pre-fix**, committed three hours before the fix landed. **Same class:** a reference that resolves to a plausible value which is not its referent is exactly R13's defaulting read, and the defaulting is done here by the *reader* — the file opens, so the citation is taken as checked. **One amendment, because the difference is load-bearing:** in R13 the referent is *absent*; here it is *present at the wrong version*, and a version carries an order that an absence does not. **The loud-default remedy therefore cannot reach it — nothing is silent.** What reaches it is the mechanism this project already built one floor down and has never applied to prose: **a proof entry names its subject by digest and demotes itself when the subject moves; a citation names its artifact by path and cannot. A citation is a proof key with no subject digest.** So: cite a *state*, not a path — the commit the artifact was generated at, checked against the commit the claim is about. Regeneration is routed to Link; the general form is recorded here without a number, per §8.9.18 part 2. **The classes stay distinct by their remedies, not by their names, and this one shares R13's diagnosis and needs a different fix.**) · **Date:** 2026-07-28T17:59:54-07:00 · **Last revised:** 2026-08-03T11:56:18-07:00 (**RULING §8.9.22 — A `max` OVER A RELATIVE MEASURE WHOSE DENOMINATOR CAN GO DEGENERATE MEASURES THE DEGENERACY; CRITERION 10's LOGITS OBSERVABLE IS REPLACED AND **THE REPLACEMENT ADMITS NOTHING**. PLUS A RETRACTION OF MY OWN.** **(1) THE OBSERVABLE IS DEFECTIVE INDEPENDENTLY OF int8.** A max-ULP criterion ranks the **fp16 GPU path at 337,178 ULP on the logits** as *worse than every int8 CPU lane* (7,886 / 45,638 / 38,278) — **an ordering nobody believes, and criterion 10 is measured with it.** Switch found the mechanism in a finding against his own instrument: the spacing floor is reached by **any reference below the smallest fp16 normal**, not only an exact zero — 18,765 subnormal references, 0.45% of the worst tensor — so **the max is located by construction at the values carrying the least information. THE DEGENERATE-DENOMINATOR RULE: the unit is not at fault; the statistic is.** Replacement reports **two things and never one number** — residual over references at or above the smallest normal, plus the count/fraction below it, **published rather than dropped**. **(2) THE LOAD-BEARING HALF: this is not a narrowing-because-it-failed, and the test is that the change buys int8 nothing.** Under the split observable int8 `per_block32` still sits at **18–22 ULP** against the fp16 control's **3** — 6–7× — so Switch's `NO_ULP_BAND_ADMITS_INT8_AND_STILL_CATCHES_FP16` **survives the fix his own data motivated**, and the int8 question stays exactly where he left it. **A change that makes nothing pass which did not pass before is a repair; one that admits the thing whose measurement prompted it is a narrowing.** It composes with Trinity's float64 result — at the final RMSNorm **Vulkan is bit-exact and ORT's CPU EP carries the 1 ULP** — which is the strongest possible reason to **keep the unit and fix the statistic** rather than touch the tolerance. **int8 admission is NOT ruled** and the silence is not a grant: it needs Trinity's observable and a byte figure of class MEASUREMENT, not MODEL. The ledger's **2.21×/3.17×/4.06× do not reproduce from any artifact in this tree** and the disagreement was written down *before* the first run, which is the only reason it is a finding. **(3) TWO SPECIMENS, ONE RULE, NO NUMBER.** A statistic that does not declare its **domain** (the ULP max) and a fit that does not declare its **window** are R11 obligation 1 in two costumes. **THE WINDOW OF A FIT:** Switch's 8-step slope of 1.60 ULP/step carried to ctx 8192 predicts **~13,000 ULP**; run to `past_len 259` the residual compounds and **stops**, saturating at **29** by past_len ~28, flat along the token axis. **Wrong by ~450×, in the direction that would have killed the lever.** This project's refusal to extrapolate a slope now has a number on it, from its own tree — *worth more than the principle was, because a principle survives being disagreed with and a number does not*. **(4) RETRACTION — THE NAVIGABILITY DIAGNOSIS I ACCEPTED IS REFUTED BY THE MEASUREMENT I COMMISSIONED.** `R1`–`R13` are cited externally ~1,337 times; **`§8.9.x` 339 times, 80 in `registry.rs` alone.** **Nobody was lost; a second namespace was built and neither of us was counting it.** `R#` names an **obligation**, `§8.9.x` names a **location** — both legitimate, not to be merged, but **a location can be re-cut while an obligation cannot** (§8.9.19 already had to restate §8.9.17). Remedy: **every ruling names its own anchor phrase in the sentence that states it** — §5.4.1(a)'s line-number rule arriving one level up. **The declines: 3 of 8 survive, and every one that fell, fell because someone else was using the principle** — Trinity named a test after D5, D6 is a shipped state token, **Mouse copied D7 into `OP_COVERAGE.md` verbatim with attribution.** **The decline tally is retired**: *"did I mint a number?" and "did the project acquire a binding obligation?" are different questions and only the first had a counter.* The derived register (**13 numbered + 8 unnumbered-but-binding**, `.squad/fact-checker/rule-register-derived.md`) is the count from here and it is not mine. **No principle was lost.**) · **Date:** 2026-07-28T17:59:54-07:00 · **Last revised:** 2026-08-03T11:32:57-07:00 (**RULING §8.9.21 — AN OPTIONAL DEVICE CAPABILITY IS A FROZEN FRAME CONSTANT, NOT A CLAIM-TIME QUERY; AND `shaderInt64` IS `synchronization2` WITH A DIFFERENT NAME.** Tank declined this in a merge window and was right to. **(0) THE RED HAS TWO CODES AND THE LOUD ONE IS NOT THE ONE THAT BINDS.** `ENGINE_ENABLED_CAPABILITIES` is `&[CAP_SHADER]`; `Device::new` calls no `.enabled_features(…)`, so `pEnabledFeatures` is null and `shaderInt64` is off everywhere we run; `variant_is_loadable` is false for every `_i64` stem and `only_loadable_variants` declines `[dtype]` **before** `[unproven]` is reached. Four of Phi-3.5's five unproven declines — `Cast` ×2, `Sub` i64, `Greater` i64 — are **one device feature, not five proof runs**, and no evidence discharges them. **(1) THE CLAIM-TIME FRAME TEST, which settles Mouse's refusal and Tank's decline in OPPOSITE directions and explains why they differ.** The claim path may read a FRAME component iff it is **(a) resolved before the first claim, (b) session-immutable, and (c) passed in as a value, not fetched from a global.** The enabled capability set, device identity, toolchain and `ort_build` all pass; a **bound specialisation fails (a)** — a pipeline does not exist yet — which is exactly why Mouse's `SPEC-UNOBSERVED` refusal was right and is **not** a precedent against Tank's case. **The device is not unknowable at claim time; it exists before any node is offered. Only the global is the defect, and (c) is a plumbing cost, not a licence.** **(2) THE GENERAL TREATMENT, and this project already ruled it once without noticing it was general:** an optional capability is resolved **once**, at device creation, frozen onto `Capabilities`, and read downstream as a **resolved set** — never as a feature, never off a physical device. **`synchronization2` is the exact precedent** (§7.3/§7.5: the branch happens once in `Device::new` and no call site reads the feature). Three edits become four: probe (§7.9's supported-vs-asked-correctly obligation attaches), enable the **intersection** and freeze it, make `ENGINE_ENABLED_CAPABILITIES` a value so `variant_is_loadable(stem, enabled)` stays pure, and **thread it to the claim path as a parameter — the moment this becomes a `OnceLock` read inside the predicate, the ruling has been implemented as its own counterexample.** Rejected with costs: *fail session creation* converts four CPU-EP nodes into no Vulkan EP at all on that device — **a capability that gates four nodes may not gate the session**; *claim optimistically and decline at pipeline creation* is an `EP_FAIL` at translate time, already forbidden in writing. **Cost that is real and not the expected one:** two devices running one binary now have different claim sets, so **every run record publishes its resolved capability set** and no decline histogram is comparable without it (R8/R11 obligation 1). **Cost that does not exist:** the ledger needs no new field — a form that cannot load produces no entry, so **the entry's existence is the witness that the capability was enabled**. Blind to *enabled-and-irrelevant*, accepted explicitly. **Ordering ruled:** capability check before ledger lookup, or a device without the feature would report `PROVEN-ELSEWHERE{device}` on a module it can never create. **Scope:** `shaderFloat16` and subgroup *arithmetic* are covered unchanged; **Switch's spec-constant selectors are NOT** — they fail (a) and stay §8.9.20's, and my §8.9.19 debt on runtime-chosen specialisation is **not** paid by this ruling. **(3) MOUSE'S AND TANK'S DEFAULTS ARE ONE RULE AND IT IS R13.** Tank's first classifier read `variant_is_loadable("metadata")` — a stem naming no module — as `false` and reported a composite `Gather` form unprovable: **an instrument-side absence emitted as a subject-side finding**, R13 amendment 1's defaulting read written in Rust. What differs is not the rule but which side is silent: **THE LOUD-DEFAULT TEST (a generalisation, deliberately unnumbered) — when a mechanism does not know, it takes the answer that leaves a trace, not the one that is nominally conservative. Refusal is usually an aggregate ("all 103 declined", "5/5 unprovable") and the aggregate is where a form goes to stop being looked at.** It selects correctly in both specimens, in opposite directions, and **inverts** where the permissive answer is silent — which is why `PROVEN-ELSEWHERE{δ}` is disclosed and why §8.9.17(5) put the device predicate first. **(4) "TOO CLEAN" GENERALISES; THE REMEDY DOES NOT.** *A total is the one reading under which a mechanism's discriminating behaviour is unexercised* — the remedy is **demonstrate both polarities**, already in this register three times (R9 rule 3's planted control, R12's `refused > 0`, Niobe's `UNWITNESSED`), so **no number**. What is new is the trip-wire: a uniform verdict **emits `UNIFORM(n, verdict)` and is not quotable until a named positive control produces the other arm**. **I verified one instance and decline to assert "the fourth today" — that is a tally, and §8.9.18 part 2 gave tallies to Fact Checker.** **(5) THE MERGE DRIVER, ruled briefly:** `history_merge_driver.py` infers condensation from `len(ours) < len(base)`, and a side that condenses **and** appends can be longer than base — the driver then takes the union branch and resurrects what it exists to protect, **the original defect surviving inside its own fix.** A condensation is **declared, not inferred**: a `<!-- CONDENSED-AT: <base-sha> -->` marker, keyed on by the driver. **There is no trade against concurrent-append protection** — an append never carries the marker. Same class applies to `.squad/decisions.md` archival. Owner: Scribe.) · **Date:** 2026-07-28T17:59:54-07:00 · **Last revised:** 2026-08-03T05:05:00-07:00 (**BLOCKING RULING §8.9.19 — A LINUX RUN MAY CLAIM `PROVEN-ELSEWHERE{toolchain}`; THE TOOLCHAIN BELONGS IN THE FRAME, NEVER IN THE KEY; THE SUBJECT NEEDS TWO DIGESTS.** Link's Linux lane compiles and runs at `8e47d3a` — eleven bindgen errors were a **representational** difference in three carrier declarations, not a signedness bug — and four of seven gated steps pass. **The three that fail are one cause and it is not the platform:** Ubuntu ships shaderc 2023.8 against the Windows SDK's v2026.2, so `shader_digest_for` faults all 97 entries; shown to be the ledger by perturbing one GLSL template **on Windows** and reproducing a superset of the same failures. **(1) ONE SCHEMA, replacing two rulings.** An entry has a **KEY** (the form — `ProofKey::from_node`, nothing else ever), a **SUBJECT** (what code was proven), and a **FRAME** (device, driver, `ort_build`, toolchain, tolerance). **You look up by key; you compare frame after you have looked up; a subject mismatch means the proof is about something else.** So the device does *not* belong in the key either — I said "the device belongs" in §8.9.17 and am being exact: it belongs to the entry and to the predicate. **"Keyed per toolchain" is a mechanical accident I can name the line for:** `parse_ledger` `continue`s past a digest mismatch, the entry never enters `Ledger::entries`, and `Ledger::get` returns the same `None` it returns for a form nobody proved — **a frame mismatch is currently indistinguishable from a key absence**, which is why Linux reads as 97 unproven forms. **(2) THE LATTICE:** `PROVEN` | `PROVEN-ELSEWHERE{δ}` where δ ⊆ {device, driver, ort_build, toolchain} | `UNPROVEN{reason}`. One state carrying an enumerated delta, not an enum growing as a product — lavapipe CI differs in *both* device and toolchain. **`TOOLCHAIN-CHANGED` is not a demotion to `UNPROVEN`** (that is today's state and is the thing being unblocked) and not a silent `PROVEN` (a different compiler under `-O` can move arithmetic). And per §8.9.18 part 1 it is **not promoted by a model run** — it is promoted per key, by `tests/ops`, which *is* a per-form differential. **The ruling is self-discharging: it grants exactly enough claim for the suite to run, and the suite removes the need for the grant.** **(3) TWO DIGESTS, because no single hash can be sensitive to the kernel and blind to the compiler** — the compiler is a function whose output is the only thing that runs. `spirv_digest` (today's) is blind to comment-only edits, correctly, and over-sensitive to the compiler. A new `source_digest` covers the `.comp` text, **every file reachable through `-I`**, the `shader_variants.txt` row and the `glslc` argv minus the compiler version — blind to compiler behaviour entirely, over-sensitive to comments. Four-row table; the fourth row (`spirv` same, `source` differs) is `SOURCE-COSMETIC`, claimable and **named**, and is the row that proves the pair is doing work. Jointly blind to: a **compiler bug** (which is exactly why row 2 is disclosed rather than silent), host-side change, and runtime-chosen specialisation values — a residual that is *growing* as selectors become spec constants. **(4) MOUSE, IN ORDER:** entry survival (blocking), `source_digest` in `build.rs` + `gen_proof_ledger.py` (blocking), predicate returns δ and §8.9.7 prints it, `toolchain` field, real `device` identity. **(5) TWO OF LINK'S FINDINGS, AND NO `R` NUMBER FOR EITHER.** The Windows DLL hash is a **one-way** instrument — six builds of an unchanged tree, six hashes, while the Linux `.so` was byte-identical across four; a fingerprint witnesses its input only if the production is a function, and MSVC linking is not one, so an identical hash means *nothing relinked* and a differing hash means **nothing at all**. He retired his own Session-13 method on it. And a collection-time `ImportError` in `test_shape_inference_delta.py` zeroes the op step while it reports green — 292 skipped, nothing asserted — whose general form is **a suite's verdict must be a function of its assertions, and an exit status is not**; the remedy is a **declared expected execution count**. Both stated in full and **deliberately unnumbered**: §8.9.18 part 2 ruled that numbering follows citation and that the count is no longer mine, and minting two numbers in the document that hands the counting away would be the old habit wearing the new rule.) · **Date:** 2026-07-28T17:59:54-07:00 · **Last revised:** 2026-08-03T00:20:00-07:00 (**A REFUTATION OF MY OWN REASONING UPHELD; THE REGISTER'S COUNT LEAVES MY HANDS; A STALE ENTRY DEMOTES ITSELF.** §8.9.18, three rulings. **(1) `PROVEN-ELSEWHERE` LOSES ITS PROMOTION LICENCE AND KEEPS ITS DISCLOSURE LICENCE.** Fact Checker refuted §10.0.1 R12's cost argument — *"the expensive proof establishes the form; the cheap invariant establishes the port"* — and the refutation holds: the instrument I called cheap is the **model-level** ULP series, whose records are indexed by model output and which therefore reaches **no proof key at all**. `ProofKey::from_node`'s own doc states the rule I broke — *evidence about one path cannot be returned for another* — and `wiring_census-dev1.json` supplies the arithmetic: `proven_key_lookups=6` against `ledger_entries=95`, so one model run on the second device would have promoted **eighty-nine keys it never touched**. The paragraph is marked WITHDRAWN in place. The state survives on its other leg, the fatal-horn argument, which was always about disclosure and needs no promotion path; the replacement mechanism, recorded but not commissioned, is a **per-key replay** of an entry's own recorded case against its own recorded reference. Mouse's ordering — specify, make `LedgerEntry::device` load-bearing, then implement — is endorsed and is **forced** by §8.9.17's finding that `device0` is a selector ordinal, since a state defined by a distinction the artifact cannot make is undefined rather than under-implemented. **(2) "DECLINE" IS A NAMING CONVENTION, NOT A MEASUREMENT.** The six declines are real and they count numbering, not register growth; the ⚠️ finding that some principles were *re-derived* is not a separate observation but the proof, since a principle that had to be re-derived was in the record and could not be found. Four-part policy: a rule is anything **cited as binding by someone who was not in the conversation**; numbering follows citation; a decline counts only if the principle stayed out of the record in every form, which may cost me some of the six; and **the tally leaves my hands** — authorship stays, counting goes to Fact Checker, because the answer to "your own tally says the register is fine" cannot be another tally of mine. Taken with thanks: *no principle was lost* — the register is under-numbered, not under-populated. **(3) `parse_ledger`: PER-ENTRY DEMOTION IS CORRECT.** Fault scope is set by the scope of what you **cannot locate**, not by the severity of what you found: `STALE-SHADER`, `NO-SUBJECT-WITNESS` and absent witnesses are located at one key and demote that key; a header-digest mismatch, a count mismatch, a duplicate key and an unparseable line are unlocatable and fault the artifact. Decisive because §8.9.17's `TOOLCHAIN-CHANGED` is **ledger-wide by nature**, so under today's code every future `glslc` bump is a total ledger fault for a change that touched no kernel — and a fail-safe guaranteed to fire spuriously on routine maintenance has a scheduled date for being switched off, which is `parse_ledger`'s own prediction about itself. Two obligations attach or the fix is a weakening: demotions must be **printed** by the §8.9.7 disclosure, and the demotion path must keep its positive case.) · **Date:** 2026-07-28T17:59:54-07:00 · **Last revised:** 2026-08-02T23:40:00-07:00 (**A §0 THAT A NEWCOMER CAN READ AND CHECK; THE DIGEST'S FRAME GAINS THE COMPILER; THE M0 TALLY RE-DERIVED FROM ARTIFACTS AND IT IS FIVE, NOT SEVEN.** **(1) §0 REWRITTEN AS "WHAT THIS EP DOES TODAY", IN TWO HALVES OF EQUAL VOICE**, because the record had become large enough that the absence of a readable state-of-the-project was itself a defect. Every figure cites a symbol or an artifact. **Two figures I was asked to headline did not survive the check and are published with their defects instead of without them:** *"bit-identical to the CPU EP"* is **false** — 62 of 65 outputs are within tolerance and `logits_max_abs_diff = 0.0625`; and **`weight read amplification = 1.000000` is an algebraic identity, not a measurement** — its numerator is a blob count and its denominator the weight bytes of the same census, and a blob *is defined as* 16 weight bytes, so the ratio is `x/x` for every model, every kernel, and a broken one. All four of its fields are literals in `probe_island_bytes.py`. **The instrument cannot go red**, which is R9's test, and it is the third figure this session to fail it. Op counts corrected from artifacts too: **91 op-table rows, 71 carrying a kernel, 20 `Staged`** — not 47/22. **(2) §8.9.17 RULES ON THE DIGEST AND ON THE DEVICE.** The compiler enters the digest's declared frame and demotion splits **`SUBJECT-CHANGED`** (the proof is invalid; re-prove) from **`TOOLCHAIN-CHANGED`** (the proof may hold; the evidence that it applies *here* does not; repairable by the cheap ULP invariant). **Both demote, neither becomes claimable — but a run faulting 97 entries on a `glslc` upgrade must not look like 97 kernels changed.** Do not narrow to a source hash: the breadth is protective, since a compiler that miscompiles correct source is a real correctness event with no other instrument here. **Found while ruling, and larger than the ruling: `parse_ledger` does the opposite of its own comment.** It says *"a stale entry demotes ITSELF and nothing else"*; the branch pushes to `faults` as well as `demoted`, and `lookup_key` faults the **whole ledger** on any non-empty `faults`. Measured, not inferred — `census-counters-dev0-ledger_digest_drift.json`: `ledger_faults=1, ledger_gate=FAULTED, ledger_hits=0`. Two doc comments in one file contradict each other and the code implements the unintended one. **On the device: it goes in the PREDICATE, never in the key** — a key carrying the device makes the ledger a set of per-machine fingerprints and returns `KeyAbsent` for well-proved forms. **And `device0` must not be what the predicate reads:** it is a selector ordinal (`gen_proof_ledger.py`: `device = args.device_name or f"device{args.device}"`), it names different vendors on different boxes, and **a predicate comparing `device0` to `device0` would return `PROVEN` across two vendors while looking exactly like a working guard** — the same failure wearing a check's clothes, and harder to see. The field is `vendor_id:device_id:driver_version` plus the device name, read from physical-device properties, never from the selector. **(3) THE M0 TALLY RE-DERIVED ROW BY ROW FROM ARTIFACTS, WITH R9's THIRD GENERALISATION AS A GATE ON EVERY `MET` ROW** — name the run that would fail it, say whether that run is reachable. **Five met (3, 4, 5, 6, 7), seven not (1, 2, 8, 9, 10, 11, 12).** **Row 1 demoted:** "Linux via CI" was a promise; `--all-targets` clippy is red on 11 bindgen typing errors and seven Linux steps behind it are `GATED_NEVER_RUN`. **Row 8 demoted as OUT OF FRAME:** its parity counters are `abi_version: 2` and the mirror has since been shown to swap two counters silently — a parity result read through a corrected mirror is not a parity result, and the two files named `counters-full` record 4 and 3978 dispatches. **Row 3 promoted, with an amendment to a condition I wrote myself, made in the open:** *"from a run whose verdict is an attributed `MATCH`"* is unsatisfiable while row 10 is `DIVERGENT`, and it existed only to exclude a CPU-fallback run with nothing to validate; a dispatch count plus an **in-frame liveness arm** discharges that purpose more strongly. I would have refused the substitution had it been weaker on that axis. **Row 9 needs restating, not more evidence:** a continuously-assessed consistency criterion has a failing run that is always reachable and never absent, so it is unmeetable by construction — restate as *consistent at a named commit*. **Row 11: (c) is DELIVERED without qualification** — four arms in two pairs differing in one key component, three mutations all `CAUGHT`, and the identical-file control arm that makes the others detections rather than a check that fails on everything — **and the row still does not close, on the clause `no build silently claiming unproven forms`, because §8.9.17's device finding falsifies it with a specimen in this tree:** `wiring_census-dev1.json` reads `ALL-PROVEN, ledger_hits=6` on the Iris Xe against 97 entries all recording `device0`, which dev0's own census shows is the RTX 4060. That is a falsifier firing, not a condition added. **Row 12 additionally loses its pair:** the two device censuses read `ledger_entries` of 97 and 95 — two censuses of two different binaries, R12's fourth generalisation arriving inside the criterion built to catch it. **(4) MY OWN ULP PREDICTION IS SCORED AND REFUTED.** On record before measuring: flat at 1–3 across all 32 layers. Measured: median **1** over outputs, with three exceedances — output 0 (logits head) at **12**, outputs 63/64 (last layer key/value) at **4**. **Refuted in the useful direction: a step, not a curve, and located at the head rather than distributed across depth.** **(5) AND ONE THAT IS MINE TO CARRY:** the artifact behind my withdrawn 2026-08-02T02:02 closure cited both attribution witnesses present and agreeing; **the file at that same path today** reads `witnesses_present: ["ort_profile"]`, `witness_agreement: "UNOBSERVABLE"`. A stable filename now holds a different frame. A verdict artifact must name its frame in its filename or refuse to overwrite one.) · *prior revision 2026-08-02T21:24:34-07:00* (**THE PROOF LEDGER FAILS OPEN ACROSS DEVICES AND FAILS STRICT ACROSS COMPILERS. A PROOF IS A PROPERTY OF A FORM ON A DEVICE — AND THE DICHOTOMY THAT FOLLOWS IS FALSE.** **(1) VERIFIED IN SOURCE:** `LedgerEntry` carries `device`, `ort_build` and `tolerance` — **the entry records its frame in full** — and **no predicate reads `device`.** Not `get`, not `lookup_key`, not `ledger_contains`; there is not one use of an entry's `.device` in `registry.rs`. The field is on 74 of 75 entries and is inert. **Recording a frame is not carrying it. A field no predicate reads is not a guard, it is a comment with a schema.** **(2) RULED R12, NO NEW RULE — this is R12's sharpest specimen**, and the third costume of the session's recurring defect beside RAI-011's early return and `'<absent>'`: not an instrument that goes red and changes nothing, but **one that cannot go red at all while looking exactly like one that could.** Link's asymmetry is the reason it is urgent: **a digest disagreeing fails safe; an entry matching on a device that proved nothing fails open.** **(3) THE DICHOTOMY IS FALSE.** Per-device proofs make a new GPU unusable until ninety-five forms are proved on it — fatal for a cross-platform EP. Device-independent proofs assert proven-anywhere-is-correct-everywhere, which `timestampPeriod` already falsified on Intel. **Both horns are correctly costed and the choice is not required, because one entry is answering for two different jobs with one bit** — that the *form* is implemented correctly, and that the form is correct *here*. **A proof is a property of a form on a device, and the remedy is R12's remedy: carry the frame and name the extrapolation.** Three states replace a two-state predicate: **`PROVEN`** (entry's device matches), **`PROVEN-ELSEWHERE`** (sound entry, another device — **claimable on purpose, but counted, disclosed and named**), **`UNPROVEN`**. **This is not a softening: that extrapolation already happens on every non-`device0` run, silently, indistinguishable in every artifact from a proof obtained here.** The test that it is a strengthening — **after the change a divergence on a new device arrives with a named suspect list; today it arrives with 74 entries all claiming to be proofs.** **(4) AND THE PROMOTION PATH IS CHEAP, WHICH IS WHAT DISSOLVES THE COST ARGUMENT.** Device-dependence, when real, is subgroup width, fp16 rounding and driver behaviour — which move a residual **by ULPs**, and R9's dual has just established the ULP series as the instrument that sees them. **`PROVEN-ELSEWHERE` is promoted by the cheap per-device invariant, not by re-running ninety-five differentials: the expensive proof establishes the form, the cheap invariant establishes the port.** **(5) THE TOOLCHAIN COUPLING IS SEPARATELY WRONG AND IS R13, NOT R12.** `shader_digest_for` hashes SPIR-V **bytes**, so a different `glslc` faults all 74 entries with no kernel change; the digest's declared frame names formula, index space, workgroup, binding, deletion and rename, **and the compiler is on none of them.** **But do not narrow it** — a compiler that miscompiles correct source is a real correctness event and a source digest would be blind to it. The digest is **over-broad, not fabricated**, and the breadth is protective. **Declare, do not narrow:** the compiler enters the declared frame and the demotion splits into **`SUBJECT-CHANGED`** and **`TOOLCHAIN-CHANGED`** — both demote, neither becomes claimable, **but a run that faults seventy-four entries on a `glslc` upgrade must not look like seventy-four kernels changed.** **(6) A DRAFTING COMPANION, ON LINK'S SHARPEST OBSERVATION:** fixing an unrelated clippy failure would have turned op-correctness green having asserted nothing — *"the narrowing you forbade, reached without anyone narrowing anything."* **A prohibition on an act is blind to the state that act would have produced, when that state is already the default.** State prohibitions as **invariants over states with a count** — not *do not narrow the lane*, but **the lane asserts N and publishes N, and a run reporting more skips than assertions is not green.** **(7) `GATED_NEVER_RUN` ENDORSED AND JOINS THE THIRD-STATE FAMILY as R7:** a red step skips the seven behind it, so `device.op_correctness` was never *"never observed to fail"* — **it has never run**; deleting its `observed` date was the better half, because **a date is a claim that something happened.** **(8) A SECOND `misnamed` SPECIMEN NOTED:** a portability failure shipping under the name `Clippy (all warnings as errors)` — *"that one was wrong by 50× in a number; this one by a whole platform in a priority."*) · *prior revision 2026-08-02T15:15:12-07:00* (**THE ACCUMULATION QUESTION HAS A FALSE PREMISE — EVERY f16 KERNEL ALREADY ACCUMULATES IN fp32. THE RESIDUAL IS fp16 ULPs AND THE CURVE IS MAGNITUDE, NOT ERROR. R9's DUAL, RECORDED AND NOT NUMBERED.** **(1) `should f16 kernels accumulate in f32?` — THEY ALREADY DO, EVERYWHERE, AND HAVE.** Verified by symbol: `q_gemv.comp` — *"Accumulation is fp32 regardless of storage, which is also what ORT's `SQNBIT_CompFp32` path does"*; `simplified_layer_norm_f16.comp` — *"Sum-of-squares, tree reduction, rsqrt and the gamma multiply are all fp32"*; `skip_simplified_layer_norm_f16.comp` — *"All arithmetic … is fp32"*; `gqa_f16.comp` declares `float acc[128]` and runs its dot products and online softmax in `float`. **fp16 is a storage format in this EP and has never been an accumulation format.** No cost decision, no occupancy trade, **and no invalidation of the 74 re-proved ledger entries — ruling on the economics as framed would have bought a property the tree already has.** **(2) THE RESIDUAL IS NOT ACCUMULATION ERROR.** Of the 65 per-output residuals, **64 are exact negative powers of two and the 65th is `3 × 2⁻⁹`** — small integer multiples of the **fp16 ULP** at each tensor's magnitude. KV magnitude grows with depth, the ULP grows with it, **and the absolute residual therefore rises with depth for a correct implementation.** The curve is a plot of magnitude. **(3) THE TOLERANCE ARGUMENT IS WRONG FOR A SECOND REASON, NOT THE ONE OFFERED.** That the pass/fail line falls mid-curve is true and well found; the deeper fault is that **`atol` is an absolute bound applied to tensors of growing scale** — §10.0.4's *prefer the ratio* arriving as a defect. **The unit is wrong, not the number, and fixing the unit may make the gate TIGHTER, which is why this is not a relaxation.** Replacement stated with its prediction so it can fail: record the residual **in ULPs**; **predicted flat at order 1–3 across all 32 layers**; flat ⇒ no defect and no curve, a **step** at layer L ⇒ a real defect, localised. **(4) RECORDED AT R9 AS THE DUAL OF THE THIRD GENERALISATION AND NOT NUMBERED** *(fifth decline; self-check run in the open since this is the second consecutive unnumbered finding)*: everything ruled this session was a check whose reading does not move when its subject is wrong; this is a reading that moves when its subject is fine. **An observable that is true whatever happens cannot convict; an observable that degrades whatever happens cannot acquit. The reading must be a function of the claim and of nothing else.** **(5) INSTRUMENT CAUTION:** the depth series must be quoted absolute or in ULPs, **never `max_rel_diff`** — layer 2's key reads `0.4559`, above every layer from 3 to 30, on an unremarkable absolute residual, because that ratio is attained at near-zero elements. **(6) `argmax = 30751`, `top10 = 10/10`: comfort declined on arithmetic, not scepticism — it is one token.** The rank invariant is the right invariant; **N = 1 is not a stated N.** **(7) CRITERION 10's REOPENING GROUND IS MEASURED ABSENT** (`oracle_outputs_degenerate = 0`, 65/65 compared, planted control refuses) **and the row stays open only on the unit.** `verdict = DIVERGENT` is honest and must not be flipped by moving `atol`. **GQA's 1.37× margin is untouched by any of this and stays open.**) · *prior revision 2026-08-02T04:30:29-07:00* (**CRITERION 10 REOPENED THREE HOURS AFTER I CLOSED IT: `model_output_equivalence` COMPARES ONE OUTPUT OUT OF SIXTY-FIVE. COVERAGE DOES NOT COMPOSE — RECORDED UNDER R9 AND DELIBERATELY NOT NUMBERED.** **(1) CRITERION 10 IS REOPENED and the closure was mine and wrong.** Verified in source after Fact Checker raised it in Devil's Advocate mode: `_compare_run_to_cpu` compares `vk_out[0]` against `cpu_out[0]` and derives every oracle fact from the logits alone; `test_phi35.py`'s oracle is the same shape behind a **structural** length assertion; **no KV output is compared against CPU anywhere in the tree.** The all-65 gate is `outputs_bit_equal` — **cross-run identity, which proves determinism and cannot prove correctness, because a deterministically wrong KV write passes it by being consistently wrong.** **(2) THIS IS NOT INCOMPLETE COVERAGE BUT THE ABSENCE OF A FALSIFIER FOR THE DEFECT THE ROW WAS REOPENED FOR.** The 2026-07-31 ground was 50 KV outputs never written, *"giving cross-run divergence on a dirty arena"* — divergence is the symptom of a **dirty** arena; on a clean or zero-initialised one the same unwritten output is **stable** and every gate goes green. The codebase already documents the mechanism — `test_phi35.py` Guard 1, an output outside the descriptor set *"is never written… reads back as all-zero"* — and points that guard at **output 0, the one tensor that already has an oracle.** **(3) THE ESCAPE IS REFUSED.** That the criterion *"was always about logits"* would require renaming the measurement to `logits_equivalence` **after** seeing the broad reading fail, which is narrowing a criterion because it has just failed — the mirror of the move refused three hours ago in this same cell. **A criterion may not be hardened because it is about to pass, nor narrowed because it has just failed; the rule runs both ways or it is not a rule.** **(4) RECORDED UNDER R9 AS A FOURTH SPECIMEN OF THE RED-INSTRUMENT TEST, WITH NO NUMBER**, running the self-check this register carries: the remedy is unchanged — a different instrument — so no amendment and no generalisation. The content that is new: **two gates whose extents differ compose to the weaker extent and the stronger name**, and **a record with two gates owes two extents**. Why nobody saw it is R11 rather than R9: the artifact carries `outputs_compared: 65` among oracle facts when 65 is a **cross-run** count — obligation 4, name–content, and obligation 1, extent undeclared per gate. **(5) THE READER IT CAUGHT WAS ME.** I quoted `max_abs_diff = 0.0625` into a criteria row without stating over what, three hours after diagnosing that same obligation in criterion 12 against someone else. **(6) DISCHARGE CONDITIONS STATED IN FULL** so none can be added later: all-65 oracle with per-output tolerance justified and two named extent keys; a planted control that is wrong **and stable** (all-zero), since an unstable plant is caught by cross-run identity; a non-triviality guard on both sides, because 64 pairs of zeros satisfy an all-65 oracle perfectly and *an oracle that passes on the absence of data* is `0.0 == 0.0` in a fourth costume; and the existing attribution evidence re-emitted, not re-argued. **Fact Checker's session-aggregate attribution argument is recorded as OPEN and explicitly NOT folded in as a condition** — it is analysis without an artifact, and R13's second clause says a result pointing the way I am already going deserves more scrutiny. **(7) ON METHOD:** I verified every field of the artifact and closed wrongly anyway; the coordinator, having supplied the evidence, put it to an adversary *because* he had supplied it. **Content verification by the party ruling is weaker than adversarial review by a party with no stake.**) · *prior revision 2026-08-02T02:02:23-07:00* (**CRITERION 10 CLOSES ON THE CONDITION WRITTEN IN ADVANCE, WITH NO NEW CONDITIONS; CRITERION 12 CONFIRMED NOT MET AND ITS FOUR CONJUNCTS ENUMERATED; A WITNESS IS NOT A DISCHARGE, SECOND INSTANCE, STILL NO NEW RULE.** **(1) CRITERION 10 IS MET.** Verified by me from `bench/results/criterion10-dev{0,1}.json` rather than from the report: both devices `MATCH`/`AGREE`, three consecutive runs of one session, `per_run_comparison` all `AGREE`, `executed_by` showing 3 `VulkanExecutionProvider` island executions against 24 CPU from **ORT's own profiler**, both attribution witnesses present and agreeing, dispatches 1066/1186, argmax 30751 on every run, and **`cross_run_identical_to_run1 = true` on all three** — which is the cross-run divergence that reopened the row, resolved on its own terms. **The condition was stated in advance in exactly these words to bind me, and it binds me now that the news is good:** the coordinator supplied the artifacts and *declined to close the row*, so artifact-supply and tally already sit in different hands — the separation criterion 11 lacked — and **requiring an independent re-run after seeing the result would be hardening a criterion because it is about to pass, which is the mirror of the rescue argument rejected on the 40.201 ms figure and no better for pointing the other way.** The re-run is recorded as a **standing falsifier, not a condition**. Switch and Trinity, the owners whose incompleteness reopened the row, are recorded as having delivered. **Closure does not reach Defect 2's unwitnessed KV write path or the arena-lifetime item; criterion 10 was never the instrument for either and they keep their own owners.** **(2) CRITERION 12 REMAINS NOT MET**, against a report that it was closed. The census was run and returned `unwired: []` on both devices — that is a **witness**, and this row asks for four conjuncts: the census, declared extent, the decomposition identity against an independently-measured whole, and the name–content check. Three are open. They are now **enumerated in the cell**, because a conjunctive criterion whose parts are recoverable only from prose invites closure on whichever part the reader happens to hold. **(3) DIAGNOSED AS R11's FIRST OBLIGATION TURNED ON THE READER** — *declare the extent of what you are reporting* — one conjunct verified, the conjunction reported: **a decomposition presented as closed**, R11's own sentence arriving in a status report rather than in a measurement. The aggravating form is the reporter's own and is kept: ***the thing I verified myself was the thing I over-weighted***; personal verification raises confidence in a part and does nothing for the whole. **No rule minted, for the second time tonight**, on the standing ground that the remedy is already written and a register that grows an entry each time an existing rule is walked past has begun counting its own traffic.) · *prior revision 2026-08-02T01:42:02-07:00* (**THE WITHHELD SENTENCE RESTORED; THE CLASSIFIER'S FAILURE MODE DECLINED AS A RULE; R12's FOURTH GENERALISATION.** **(1) RESTORED to criterion 11's row after a merge where neither side was a superset and the coordinator correctly declined to splice my prose: *WHAT IS WITHHELD IS THE TALLY, NOT THE WORK.*** The row is open because discharge needs an observable that moves, and for no other reason. **Mouse's evidence is not rejected, not doubted, and not diluted by being uncounted** — the ledger, `e4436e93c19c8744` → `331003e0ff88df3f`, the two-armed `mul_f16_unproven` control, the `MatMulNBits` ± `zero_points` pair, and `355 → 0` **paid unsoftened and predicted in writing beforehand**. *I do not want this register becoming a way of declining people's findings, and a lead who can only ever withhold is running a different instrument from the one he thinks he is.* **Three constructions meet the standard I set and are named in the row rather than in a decision file nobody re-reads:** (a) **provenance an enumeration cannot forge** — a dispatch count only exists after a session executed, and the half I did not think to require is the better one, ***absent is treated exactly like zero, and a QUOTED count exactly like absent, because a writer that stringified its counters did not read a counter***; (b)(iii) **the identical-file arm, which is what makes the other two arms detections rather than a check that fails on everything** — R9's falsifier-polarity discipline arriving unprompted inside someone else's control; (d) **`NeverAttempted` derived and never counted, since recording it would be a lookup, which is what it asserts did not happen** — the cleanest statement of R13's instrument/subject boundary written on this project, mine included. **Row 11 closes on (c), Trinity's, and on nothing else.** **(2) THE CLASSIFIER'S OWN FAILURE MODE — DECLINED as a new rule; R13's second clause already is it.** Specimen, brought by the coordinator against himself: having named "union defects" as a pattern that afternoon, he reported clippy's `manual_contains` as *the fourth union defect today* in a table of five; Mouse checked rather than accepted, `-D warnings` gave **five** errors not one, and `git show origin/main:<file>` on each showed **four of the five present on `origin/main` verbatim** — clippy was already red on main, independent of any merge, and only `registry.rs:2261` was a union defect. **R13's second clause reads *a result that confirms a prediction deserves more scrutiny than one that contradicts it — quote the failure text, never the failure count*, and the specimen is that clause with nothing added**: a pattern named, a confirmation **counted** rather than **quoted**. Mouse's remedy was R13's performed literally — retrieve each failure's text from its own source. **What is new is scope, not obligation, and gets one sentence and no machinery:** every prior R13 specimen was a *mechanism* mis-reporting; this one is a **person** — the classifier, not the check. > ***A newly named pattern begins attracting cases that do not belong to it, and the cost is borne by the real instances, which are diluted by the false ones.*** *A class assembled from a tally has the same evidential weight as an instrument that cannot go red.* **And this is where the register stops growing today** — three rules declined, two amendments and three generalisations in one session; the request to be declined rather than grow a rule per incident is correct **and is itself the shape of the error above**: a register that grows by one entry per named pattern is a register attracting cases to its own new categories. **(3) R12's FOURTH GENERALISATION — for a test result, the frame is the BINARY that ran it.** Two specimens, both Mouse's, both caught by their author: a build that linked a sibling's in-flight `registry.rs` from a shared worktree and produced a **false `ALL-DECLINED` he nearly wrote up as a finding**; and **`Copy-Item` preserving `LastWriteTime`, so cargo's fingerprint does not notice a restore-from-backup and re-runs the MUTATED binary** — a persistent false failure he came close to "fixing" **by weakening a correct assertion**, the most expensive outcome available and the one this document exists to prevent. R12 already reads *a reported quantity carries the identity of its frame* — counter/**device**, verdict/**executor**, rationale/**date**; add **test result/binary**, and *a mutation harness that restores sources by timestamp-preserving copy has no claim about which binary that was.* Remedy is R12's unchanged: **the harness touches or hashes the restored file and asserts the rebuild happened before reading any result as a control.** **This is a cross-platform note, not a Mouse-specific one** — the trap exists wherever a build system fingerprints on mtime alone. **A false red that gets "fixed" by softening the assertion it fired on converts a working control into a decoration in one commit**, and the failure arrived disguised as the thing we most want: a check that goes red. **Merged state: clippy green, Rust suite 459/0 across ten targets, both device censuses `unwired=[]` on a rebuilt cdylib.** **Also now a test rather than a paragraph: the bound's narrow half** — `the_substituted_extent_under_counts_on_a_long_prefill_and_the_bound_evaporates`, 128 over-counting at decode extent 1 and under-counting at 4096, the inequality reversing, mutation-tested in both polarities. **Anyone touching `slot_bytes` now breaks a test rather than a bound**, which is what §10.0.4 asks for and what a standing falsifier in prose never achieves.) · *prior revision 2026-08-01T23:36:43-07:00* (**MERGE RATIFIED, ROWS 11 AND 12 RECONCILED, AND TWO GENERALISATIONS THE DAY EARNED.** **(1) Criterion 11's merge resolution is ratified, not reversed.** My cell survived against one reading *"MET 2026-08-01T21:15:16-07:00"*, authored by the agent who supplied the ledger **in the change that supplied it** — *a row closed by the person who supplied its artifact, in the same act, is an identity whose two sides come from one source*, and the coordinator declined it on that ground. **Mouse's evidence is not rejected and not lost** (`0f589ef`, the ledger, digest `e4436e93c19c8744`, the `MatMulNBits` ± `zero_points` pair, `355 → 0`, all on main and all cited): **what is withheld is the tally, not the work.** **(2) Rows 11 and 12 are not in contradiction and row 12 now says so.** Keeping Mouse's row 12 was right — the coordinator ran the census himself on both devices, `unwired: []`, `ledger_lookup ALL-PROVEN proven_key_lookups=6 ledger_hits=6 ledger_entries=9`. **Row 12 is a claim about a MECHANISM (it ran, it was consulted, it reported a value it computed); row 11 is a claim about a CRITERION (is it false-able).** *The census answers whether a mechanism ran; a criterion answers whether a claim is false-able. A census line can never discharge a criterion, and a criterion's failure is never evidence that a mechanism is unwired.* **A wired mechanism beside an undischarged criterion is the normal state of a row being taken seriously**; the abnormal state is the one where supplying the artifact and closing the row are the same act. **(3) R9's THIRD GENERALISATION — the red-instrument test applied to a CRITERION rather than to a claim. Explicitly NOT a new amendment**, recorded as a generalisation because the remedy is R9's unchanged (a different instrument, or a reachable disagreeing state) and only the scope moves: *a milestone criterion is a claim like any other, evaluated by a reading.* > ***A criterion is discharged by an observable that changes when the claim is false, never by one that is true whatever happens.*** **Three specimens in one day, three costumes:** RAI-011's *always evaluated, no branch in front of it* (an unconditional early return **inside** the gate satisfies every word; `bypasses` `0` forever); **Link's screen reading `ONNXRUNTIME_EP_VULKAN_TRACE_FILE`**, a variable nothing defines, which reported the same value on every run it ever made **and would have done so had the tracer been deleted**; and **Switch's assertion comparing two values both exactly `0.0`**, with no reachable failing state. **The first and third are green checks and the second is a negative one — the class is indifferent to polarity**, *an always-false screen and an always-true screen are equally blind*, now a general property rather than an incident. **Operational form, cheap and the thing to actually do: before recording a criterion met, name the run that would have failed it and say whether that run is reachable. If it is not reachable the criterion is not met — it is unfalsifiable, and the tally should say so rather than say `MET`.** **(4) THE DANGLING REFERENCE — R13 amendment 1's class, named, because it generalises past probes and past this document.** My own `partition.rs:475` went stale within the hour and every line reference in a conflicted region moved during one merge. > **A reference that resolves to nothing, and reports that as a value, is R13 whatever the reference is made of** — a key with no emitter (`alloc_device_spans` → `'<absent>'`), an environment variable nothing defines (`…TRACE_FILE` → `OPTIONAL-UNWIRED`), **a line number** (`partition.rs:475` → a different statement, silently, **with no error at all**). **The reference does not fail; it succeeds against the wrong thing, and the reader receives a well-formed answer.** **The line number is the worst of the three and the one nobody instruments**, because there is no lookup to fail — the reader does it by hand and gets a plausible statement — so the remedy is **not making the reference**: cite the **symbol**, line numbers only as a convenience beside it. **A symbol that stops existing produces a failed search, which is a reader-visible `ERROR(instrument)`; a line number that stops being right produces confident nonsense.** **Where this stops:** not every stale reference is R13 — a broken URL fails loudly and is merely broken; **the class is references that resolve anyway**, which is the test because it is the remedy's test. **(5) §10.0's `attribution_witnesses` example CORRECTED (Trinity's finding, and she was right not to edit this document):** it showed **two** witness keys where the record emits `attribution_witnesses_present`, `attribution_witness_agreement`, `counters_witness_reason`, the profile digest/mtime/path pair and `counters_dispatches_executed` typed **int-or-`"UNOBSERVABLE"`** rather than nullable, so a witness that could not report fails arithmetic loudly instead of reading as zero. **A schema example is a claim about the record's extent**, and R11's first obligation binds a document's example exactly as it binds a producer's output — an example that is a strict subset tells a reader the record is complete when it is not, and **two live keys were missing from it for a day.** Remedy is the one required of everyone else: **the example is regenerated from an artifact, not written from memory** — `bench/results/criterion10-dev0.json`. **Still owed: `MEASURED_PHI35_DEV0` → `ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED` and the `TRANSFER_DOMINATED` test rename, Mouse's, sequencing held by the shared worktree.**) · *prior revision 2026-08-01T22:25:29-07:00* (**§5.4.1(a) — THE ESTIMATOR'S FIRST HALF IS FIXED; THE CONCURRENCE IS WORTH LESS THAN IT LOOKS AND A BOUND IS WORTH MORE; PLUS CRITERION 11's DISCHARGE LANGUAGE.** Verified on `squad/mouse` before ruling: internal island edges were charged to the boundary; `ep.rs` now consults a whole-graph per-value consumer map and charges an output only when a node **outside** the island reads it or nothing does — **89,199,100,032 B → 13,936,509,056 B**, a 6.4× overcharge gone, and with the exemption off the gate now **claims on its own economics** (`net_benefit_sole_island_overrides` 1 → 0). **DOES THIS CHANGE FINDING #2 (the exemption's silence set includes "the byte estimator is broken")? Materially, in one direction only, and not by the argument offered.** **I decline the concurrence argument in the form it arrived**: `slot_bytes` still substitutes **128 for every unknown dim**, every Phi-3.5 boundary tensor is runtime-extent, residual **~16,268×** — and, decisively, **agreement between two things fed the same fabricated input is not a second opinion.** *An identity whose two sides come from the same source is a falsifier that cannot fire.* **A verdict flipping from `TRANSFER_DOMINATED` to `Claim` because its input moved 6.4× while remaining 16,268× wrong moved for a reason unrelated to the truth of the proposition.** **What the same fact DOES support is stronger, and should be used instead:** `transfer_ns` is **monotone increasing in bytes**; the gate claims at 13,936,509,056 B; the instrumented boundary for that run is 856,720 B, which is **smaller**; therefore it claims *a fortiori* on the true bytes — **the claim verdict survives a 16,268× adversarial inflation of the term that opposes it.** Not an estimate, not an agreement: **a bound**, taken from a number we do not trust in the only direction where not trusting it is safe. **§10.0.4's invariance preference in a third form — after *prefer the count* and *prefer the ratio*, PREFER THE BOUND YOU CAN SIGN.** **Licence and limits, stated because this is the kind of argument that has failed here when it favoured us:** a modelled quantity known wrong is quotable as a bound only when **(a)** the model is **monotone** in the perturbed input, **(b)** the perturbation's **sign is established for that window by an independent measurement**, and **(c)** it is used only in the direction the sign licenses — **absent (b) it is not a bound, it is a guess with a confident tone.** This is §10.0 obligation 8's companion rule applied to a *modelled* quantity: *a figure is quotable with the record that fixes which world it came from, and not otherwise.* **And (b) is exactly what is not general here** — the 128 substitution's sign is **not known a priori** (Mouse says so and is right: a larger substituted dim pushes towards rejection, a smaller towards claiming); Phi-3.5's measured window happens to over-count, and **a long-prefill `sequence_length` above 128 under-counts, flips the sign, and the bound EVAPORATES rather than weakens.** **Named falsifier beside §5.4.1's first: a configuration whose real extents exceed 128 on a boundary tensor.** `symbolic_boundary_slots` / `boundary_is_fabricated()` are the right instrument and are correctly built to **report rather than judge**. **So the silence set shrinks and does not empty:** *the exemption is masking a verdict that would differ on this island's true bytes* has left it — we can sign that now; *the estimator fabricates its input, the fabrication's sign is unestablished, and no production partition is sensitive to any of it* remains. **`MEASURED_PHI35_DEV0` — RULING: RENAME, disclosure is not sufficient.** A constant named *MEASURED* holding an **estimate wrong by 6.4×**, beside `MEASURED_PHI35_DEV0_REAL_BYTES` which holds the measurement, is R11's name–content obligation with nothing to interpret. Mouse's doc comment is **exemplary** — it discloses the split, the residual, and that parking the total in `output_bytes` charges one `fixed_ns` instead of two and so **biases every test towards claiming, the direction that makes his own conclusions harder to reach**; disclosing a bias that works against you, unprompted, is the standard. **Still not enough, and the register adopts the coordinator's sentence: NAMES OUTLIVE DOC COMMENTS** — a doc comment is a different artifact from a symbol at every call site, and *a caveat in a different artifact from its number is not attached to it.* → **`ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED`**, doc comment verbatim, matching his own new `ESTIMATED_…`; **and the `TRANSFER_DOMINATED` test renames to say which estimate it is about** — it took three steps to establish the green test and the shipping behaviour are not in conflict, and **a reader who stopped at the test name would conclude the opposite of what ships.** Keeping the old constant beside the new rather than overwriting it is **correct** and is why the history is legible; only the name is wrong. Owner Mouse, same change as the `Verdict::Claim` reason field. **LINE-NUMBER CITATIONS — convention, from today's other ruling:** my `partition.rs:475` no longer resolves (the early return is at **536**), and **a line number is a reference that decays without failing** — it does not error, it silently points at something else, which is `'<absent>'`'s defect in a different costume (R13 amendment 1). **This document cites a symbol — `partition.rs::evaluate`'s anchor-exemption early return — with a line number only as a convenience beside it, never alone;** existing citations are not swept, stale ones are converted rather than repaired. **CRITERION 11 — DISCHARGE LANGUAGE WRITTEN WHILE THE ROW IS STILL OPEN AND DELIBERATELY BEFORE THE TALLY MOVES**, which costs nothing and retracts nothing. The words are *no form claimed without a ledger entry under its proof key*, and **the cheapest satisfaction is a ledger generated from the claim table** — derive it from the same enumeration that produces the claims and the criterion is true **by construction**, `ledger_hits == proven_key_lookups` forever, `6/6` identical under both readings. **Four discharge conditions, none of them "the ledger exists":** entries written by a **proof run** with per-entry provenance, never by the claim table's enumeration; **three planted controls in the lane and not `#[ignore]`d** (removed entry → `Unproven`; a key differing only in `opt_inputs`/`shape` must miss — the `MatMulNBits` ± `zero_points` pair is already there and is the all-zero-logits defect, so the control is a regression test too; a baked digest disagreeing with disk must **refuse to claim**, not warn); **`ledger_hits` shown to move with its input** (identical across two different inputs is `UNWIRED` however green — the census's own `gpu_tracer` screen defect is the standing specimen); and a **three-token** miss path. **And the sentence the coordinator asked to survive, generalised:** RAI-011 reads *always evaluated, no branch in front of it*, and the cheapest satisfaction is an unconditional early return **inside** the gate — every word true, `bypasses` `0` forever, **the check's reading does not move when its subject is wrong.** ***A criterion is discharged by an observable that changes when the claim is false, never by one that is true whatever happens.*** Owners: Trinity for the tally and the controls' lane membership, Mouse for provenance and the digest refusal. **Unchanged throughout: 355 of 363 in one fused island, `MATCH` both devices.**) · *prior revision 2026-08-01T22:02:39-07:00* (**§5.4.1 — THE ANCHOR EXEMPTION IS THE DECIDING TERM, AND THE ATTRIBUTION IS WITHDRAWN WHILE THE RESULT STANDS.** Mouse held every input fixed but one: **anchor exemption off → `viable_islands_retained` 0, `net_benefit_sole_island_overrides` 1, reason `TRANSFER_DOMINATED`, at every `fixed_ns` across §7.12's swept range.** Shipping reads retained 1 / overrides 0 — **the override did not fire; the exemption decided.** **Two corrections against the source, both making it more precise.** (i) `partition.rs:475` is an **early return** — `if island.anchors > 0 && policy.anchor_exemption { return Verdict::Claim }` — placed *above* `transfer_ns` and `compute_ns`, so on an anchor-bearing island the economics arithmetic is **not evaluated at all**; stage 3 is a **constant function** and **no property of the island can change its answer**. It is not that the economics arm decided wrongly and lost. Mouse's correction that the predicate does return false on real input (the census lane's one-node chain) is a fact about **3a `TOO_SMALL`** — it establishes stage 3 is not a decoration; it does **not** establish that **3c** has ever decided anything outside a probe, and no artifact shows it has. (ii) **The diagnosis inverts on reading `is_anchor`** (`MatMul`, `Gemm`, `Conv`, `ConvTranspose`, `Attention`, `MatMulNBits`, `GroupQueryAttention`, `MultiHeadAttention`, `QMoE`, `LinearAttention`): **every non-trivial island of any transformer contains one**, so "the economics model does not decide our partition" is **not a Phi-3.5 accident and not a defect — it is the design working.** 3c exists to kill anchor-free elementwise scatter, and our fused island is not that. **What is actually wrong is narrower, and stated without inflation:** **(1)** the exemption is load-bearing for a question it was not designed to answer — its warrant (*an anchor is by definition heavy enough to justify a boundary*) is **asserted, not measured**, and is now the sole term deciding every production partition on both devices; **falsifier is a future exposure, not a present one — an anchor-bearing island that genuinely should be declined** (one small `MatMul` inside large boundary traffic; a small-model or edge-shape graph), which 3b claims unconditionally and 3c never sees. **This is the cross-model generality risk and it is where I expect this to bite first.** **(2)** **the exemption's silence set includes "the byte estimator is broken"** — §7.12.1's **104,116×** discrepancy (89,199,100,032 B estimated vs 856,720 B measured) is *why* 3c declines the graph we ship when allowed to answer, so the exemption does two jobs, the intended one and concealing that from every production run; **R9's silence-set rule applies to a policy term and not only to an instrument**, recorded here in that general form. **(3)** **`Verdict::Claim` is three findings wearing one name — R11 at the value level**: `Claim` from 3b, from 3c passing, and from `SoleIslandOverride` are different facts, and the counters record the verdict rather than the arm, which is why "the exemption decided this" is an **inference from two runs rather than a field in an artifact**. Remedy is R11's: **`Verdict::Claim` carries its reason**, owner **Mouse in `partition.rs`**; his refusal to re-derive the arm at the `ep.rs` call site is **endorsed** — that is a second copy of the predicate and RAI-011 reappearing inside the fix for its own sibling. **§7.12 owes less than feared: it already says this in these words**, so **the defect is placement and propagation, not omission** — the sentence sits under a subsection about calibrating a parameter shown not to matter, and did not reach the two places a reader forms a belief: **§5.4's stage list, which never named the exemption (FIXED — stage 3 now shows 3a/3b/3c and a fifth binding property), and the M1 optimisation ordering's rank 2 (RE-QUALIFIED — two mis-attributions: the 321 → 33 collapse was the clustering wiring, not `retain_viable`, as §10.0.1's R10 table already said; and `retain_viable` has produced zero declines on a production graph. Ranked position withdrawn; the row stays as the record of the withdrawal).** The misled reader is misled by **this** document, not by `OP_COVERAGE.md`. **What must not be done:** remove the exemption "so the model decides" (**deferring to a model measured to be wrong by five orders of magnitude is not rigour**, and it loses M0); fix the estimator in the commit that makes the partitioner observable (**Mouse declined and is right — you lose attribution; it fails safe towards the CPU**); or soften the warrant into "it works, so it is justified" (**it works because every island we have partitioned is anchor-bearing and large — a fact about one model, and generality is checked continuously**). **The drafting rule gets its second live example:** RAI-011's criterion is *the gate is always evaluated, with no branch in front of it*, and the cheapest thing satisfying those words is **an unconditional early return inside the gate** — the branch moves from the call site to line one of the body, `net_benefit_gate` reads `EVALUATED`, `bypasses` stays `0` forever, and every word is true. **Not an accusation** — 3b predates RAI-011 and is legitimate policy in the right module — but **RAI-011's observables cannot tell the two apart**, which is what makes (3) required rather than nice to have. Sits alongside *the cheapest way to pass a steadiness test is to run at a stable wrong clock.* **What survives untouched: 355 of 363 nodes in one fused island, `MATCH` on both devices — a count and a verdict, both observed**, and §10.0.4's invariance preference applies unchanged: the result does not depend on *why* the island was retained. **Withdrawn is the attribution, not the result.**) · *prior revision 2026-08-01T20:39:12-07:00* (**§10.0.1 R13 AMENDMENT 1 — THE DEFAULTING LOOKUP, AND NO NEW RULE IS OWED.** Specimen, verified independently: `bench/results/probe_sec65.py` requests a counter key **`alloc_device_spans`** whose exact string occurs **once in the repository — at the line that requests it.** No emitter, in Rust or anywhere, ever; the read is `data.get(k, '<absent>')`, so the probe has printed `alloc_device_spans = '<absent>'` on **every run since it was written** and no exception has ever been raised. **RULING: not a new rule, and not R11 either.** R11 governs a **reported quantity** — the relation between a name and the content under it — and this is a **request**, on the reader's side of an artifact R11 constrains on the writer's side. Its obligations cannot even be evaluated here: extent of what; no parts; no table; and name–content agreement **requires content**. *A mismatch needs two relata and this specimen has one* — **a name that means nothing is not the extreme case of a name that means the wrong thing, it is a different failure on the other side of the artifact.** It is **R13**, and the coordinator's own diagnosis is R13's sentence verbatim: `'<absent>'` reads as *"the counter reported nothing"* when the fact is *"there is no such counter"*, **opposite diagnoses with opposite fixes** (go wire it / go fix the name), rendered indistinguishable — one token doing the work of two, one of which is an **instrument error counted as a detection**. **R13's costume, R10's face.** Remedy applies unchanged: **`ERROR(instrument): no emitter for this key`** vs **`UNEMITTED_THIS_RUN`** vs the value. Had those been three tokens this would have been caught on run one — **everything dangerous about it (the longest latency in this register, the hole *filled* rather than left open, the appearance of evidence of absence, a reader dispatched to hunt a mechanism that does not exist) follows from the token and not from the name.** What is new is the **surface**: every prior R13 specimen failed **loudly** (Guard D's `NameError`, the census's `TimeoutExpired`) and was mis-rendered downstream; this one has no exception anywhere, manufactured by a construct whose entire purpose is not to fail. **Amendment: a defaulting read (`dict.get`, `unwrap_or`, `?? fallback`, `getattr`) converts a reader-side failure into a subject-side value, silently; where the key set is knowable, the default is not a value and absence is not a reading.** A sentinel is admissible only where absence is a genuine finding *about the subject*, and is then named for that finding. **Why not R14, stated because I declined a new rule yesterday too and a habit of declining is its own defect:** remedy-identity is applied in both directions — refusing to grow the register to preserve its shape is the same error as growing it to reward a finding — and this remedy *is* three tokens. **The day something arrives whose remedy is not any remedy here it gets a number, and this is not that day.** **THE KEY CENSUS — standing obligation, the reader-side counterpart to criterion 12's wiring census:** *every key a probe or report requests resolves, by **exact string match**, to a literal emitter in the source that produces that artifact; an unresolvable key is `ERROR(instrument)`, loudly, and never a value* — **two tiers, both required**: runtime (the shared read helper refuses at the point of request; the tier that cannot be skipped) and static (a census over every probe; the tier that sees keys on unexercised paths). **Owner Tank with Niobe**, importing `audit_instruments.py`'s vocabulary rather than minting a second one. Four cheapest satisfactions named: **delete the key instead of resolving it** (a census that turns a phantom key into a silent deletion destroys what the phantom key destroyed — so classify it *wanted-and-non-existent* vs *typo* before deleting); **fuzzy or substring resolution** (`alloc_device_spans` is one word from `alloc_device_backed_spans`, so a lax matcher **certifies the specimen** — exact equality and nothing else); **wildcarding the emitter side** (literal strings today; a computed key is declared, and an emitter side matching `alloc_*` resolves everything and proves nothing); and **static-only in a lane nobody runs**. The census carries a **planted-phantom positive control** per R9 rule 3, or it is a check of unknown polarity. **M0 criterion 12 is NOT reopened or amended** — a probe is not a mechanism the M0 table relies on, no §6.5 or M0 claim rests on `probe_sec65.py` (verified: `probe_sec65` appears nowhere in `docs/`; §6.5 runs through `probe_indexspace.py`), and **bolting a probe obligation onto a milestone criterion because a bad probe was found today would be hardening a criterion to punish a bad week.** **§6.5.3 DISCHARGED**, and the audit it triggered found two more instances — a bare `backed_spans = 9` numerator now travels with its `alloc_allocations` denominator (*nine of nine and nine of nine hundred are different findings*), and the phantom key above. **Niobe's `span_accounting()` reports without judging — UPHELD**; an accounting note able to withhold `ONE_INDEX_SPACE` would be *a different instrument wearing this one's name*, and after `gpu_steady_tail` that needs no restating. **But "feeds no check" is not "has no teeth"** — R9's clause is unforgiving, an instrument that goes red and changes nothing is decoration — so the teeth are **attachment, not verdict-moving**: *the classification travels in the same artifact as every span count it describes*, the `executed_by` lesson, since **a caveat in a different artifact from its number is not attached to it**; named trigger, if a criterion is ever read against a span count the classification becomes its precondition and the no-judging call is re-made with the stakes it then has. **One defect in it, and it is this ruling's own subject one level up:** `NOT_A_NUMBER` fires on `not isinstance(auth, int)` while the extract is still built with `data.get(k, "<absent>")`, so a **missing or phantom key** classifies as `NOT_A_NUMBER` reporting *"a string state and not a count; the type is the answer"* — **false and affirmatively reassuring**, because `'<absent>'` is the probe's own sentinel and the type discipline has answered nothing. `"UNOBSERVABLE"` and `'<absent>'` are an EP-side finding and a reader-side failure wearing one token. **Not Niobe's error to carry** — the defaulting read is inherited — and it is the argument for fixing the **lookup** rather than the classifier: fix the read once, or fix N consumers and acquire an N+1th next time. **Three sightings in one day of an instrument-side absence rendered as a subject-side state — the recurrence is the evidence the obligation is warranted; any one alone would have been an anecdote.**) · *prior revision 2026-08-01T18:59:38-07:00* (**§6.5 CLOSED — AND THE CLOSURE IS OF A *CONDITIONAL*, WHICH IS STATED WITH ITS LANE OR NOT STATED.** Raised by the coordinator against his own earlier report, by scoring standing predictions rather than by new work. The two-armed artifact holds and is not withdrawn: `indexspace.json`, allocator factory index **`1` on selector 0 (NVIDIA) and `0` on selector 1 (Intel)**, matching each arm's offered index, both `SHARED`, both `alloc_device_buffer_binds = 6`, verdict `ONE_INDEX_SPACE` — the R10 artifact whose content varies with its input, which the earlier one-armed check was not (the pre-fix state read `SHARED` on selector 0 by **coincidence of two index spaces**). **But the probe *arms* the device-memory provider** with `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1`, and the ordinary inference lane reports `alloc_device_frame = OFF`, `binds = 0`. These do not contradict — §6.5 is the conditional *when both sides exist they are on one index space*, and the probe makes its antecedent true — **but a closure statement that omits the lane is not a shorter version of the true sentence, it is a different and false one**, because a reader who runs an ordinary inference finds the mechanism switched off and nothing told them to expect it. Canonical form now fixed in §6.5.1. **`OFF` is a third state and its existence is why this was catchable at all**: the standing prediction was `SHARED` xor `SPLIT-DEVICE`, and the instrument **declined to pick one of the two offered options rather than forcing itself into the nearer one** — the *every way of not knowing gets a name a machine can print* family paying out in the way that is hardest to arrange deliberately, since **a binary prediction met by a third token is a refutation the predictor cannot talk himself out of.** Had `OFF` been folded into `SPLIT-DEVICE` (both are "not SHARED") the prediction would have scored a clean pass and the scope gap would still be invisible. The scoring discipline is ratified as **R12 applied to predictions: a prediction is scored only against an artifact from the lane it described** — wrong lane is `UNSCORABLE`, no artifact is `UNSCORED`, both count as non-passes, and **the denominator never shrinks to flatter the numerator.** **§6.5.2 RULING — `alloc_device_frame = OFF` on the default path is INTENDED, it is NOT the `offer_shared_device` gap, and its recorded justification HAS EXPIRED.** Not the gap: `offer_shared_device` has a production caller (`vk/session.rs`), so §6.5 proves a property of a **wired seam on the only lane where the seam has two sides**, not of a path users never take. Intended: the allocator is opt-in behind `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY`, gated in `factory.rs`, already written down. **But the reason written down with it is no longer true** — it says advertising device memory is a package deal requiring an `OrtDataTransferImpl` without which *every session fails at `Run`*, and that the transfer *"cannot be written until the handle→`VkBuffer` seam is filled"*. **That precondition is discharged**: `CreateDataTransfer` is registered unconditionally, `transfer::VulkanDataTransfer` exists, and armed-lane sessions complete — nine spans, 9,437,184 bytes through the provider, and 427 allocations / 2.09 GB of Phi-3.5 at model scale. **The condition the switch was waiting for was met and nobody went back to the switch.** Recorded as **R12 with a date as the frame** — *for a counter the frame is a device; for a correctness verdict the frame is an executor; for a rationale the frame is a date* — and **a default whose stated reason has expired is indistinguishable from one that is still needed**, the `retain_viable` shape arriving in a justification instead of a call graph. **The default is not changed by this ruling and I am not ruling on whether it should be**: there is a *live* reason for `OFF` that the source does not give — the switch currently buys **host memory wearing a device handle**, risk with no measured benefit — and **a default defended by a reason its own documentation does not give is a default nobody has re-decided.** Re-justify on current evidence, dated, or re-decide; owner Tank with Switch, by M2 entry. Generalised: *a gate, flag, default or `Staged(why)` whose stated precondition has since been discharged is re-justified or removed*, with the precondition stated as something that has an artifact — cheapest abuse named, a precondition too vague to be observed discharged. **§6.5.3 RULING — `alloc_device_authoritative_spans = 0` in the closure artifact is a MEASURED ZERO AT A MEASURED CEILING OF ZERO, not R12.** Right question, wrong shape: the counter is emitted as the **string** `"UNOBSERVABLE"` out of frame and `"UNWIRED"` unrun, and as an **integer** only when measured — it printed `0`, a number, so **the three-state type discipline answered the question before it was asked**; its unconditional twin moved (`residency_evaluations = 9` from the same call site, so `0`/`9` is nine measured negatives, not a silent no-op); and `ceiling = backed − staged = 9 − 9 = 0`, so zero is **the only value it could correctly take** — every device-backed span is still host-staged because the engine reads through `host_backing_for`, and `binds = 6` with `authoritative = 0` is the consistent, honest description. **`UNOBSERVABLE` would be a stronger and false claim**: R12 is a question that *cannot be asked* in this frame; **a zero at a zero ceiling is contingent** — asked nine times, answered no nine times, and it flips the day a span is allocated device-only. **What does need fixing is the probe, not the counter:** `probe_indexspace.py` drops `alloc_staged_spans` and `alloc_device_authoritative_ceiling` from its extract — the two keys that make the zero interpretable — so it presents `backed=9, evaluations=9, authoritative=0` as a sufficient set when it is not, and a careful reader of that artifact **correctly** could not tell a measured zero from a pinned one. **R11's shape in a probe's *selection* rather than in a name.** Remedy, Niobe's and Switch's to make: the extract carries the ceiling and the staged count, or it does not carry the authoritative count at all. M1's residency clarification amended in place — **the pin is lifted and the instrument choice is unchanged for a new reason**: `alloc_device_authoritative_spans` was unusable because it was pinned, and is unusable because it is at its ceiling on the armed lane and has no frame at all on the default one; `device_upload_bytes` stands, and the day the ceiling rises is the day that clarification gets re-read.) · *prior revision 2026-08-01T13:19:00-07:00* (**§10.0.1 R9 AMENDMENT 5 — THE ANTI-CORRELATED FALSIFIER, AND THE REGISTER DOES NOT GROW.** `gpu_steady_tail` is a variance test over a suffix and **cannot see a bias**: measured `STEADY` at **126.647 ms, RSD 0.79%** with three foreign GPU processes outliving a truncated run (**10.99× wrong**), and `STEADY` at **246.720 ms, RSD 0.1163%, zero discarded** on a verified sole-tenant board held at its **210 MHz idle clock against a 3105 MHz boost** (**21.4× wrong**). **In both failure modes the wrong number carried the better RSD than the right one** — *a low clock does not raise RSD, it lowers it*, so a run that is uniformly wrong produces the gate's most confident possible verdict. **RULING: this is not R11 and it is not R14.** R11's four obligations, applied faithfully, *certify* the specimen — there is no decomposition, no flat table, no inclusive parent, and name–content agreement passes — and **a rule that would have certified the specimen does not cover it**, which is the same test used to refuse folding R11 into R10. It is **R9**: bias in a series' level is in a dispersion statistic's **silence set**, and the remedy is R9's and no other, *a different instrument*. **The register individuates by remedy, not by flavour**; a second name for one failure class would be two names for one measurement, appearing to close. What is new is a **mechanism inside R9**: R9 as written describes plural instruments **jointly silent**, this is a single instrument whose confidence is **anti-correlated with the error** — **R9 rule 5: ask which way a check moves when its subject is wrong; if it moves with the reader's confidence, it cannot be repaired by tightening its threshold** (a tighter bound admits *more* of the failure), it is demoted from gate to precondition, and the claim is `UNMEASURED` until a **second quantity from outside the series** records the state of the thing measured. **Precision is not accuracy and this register had never had to say so.** **§10.0 EIGHTH DISCLOSURE OBLIGATION — THE DEVICE-STATE COMPANION, and it withdraws a sentence of mine:** *contention inflates host work but cannot touch the GPU clock* is **false twice over** — foreign GPU work inflates device-busy directly, and the board's own governor varies it **14.8×** with nothing foreign running. **A device-clock figure is quotable only when a device-state record covers the same window as the statistic — the suffix, not the run — carrying a tenancy verdict and a clock min/median/max against the board's advertised maximum**; absent it, `STEADY_UNCERTIFIED`, a fourth state that is neither `STEADY` (read as quotable) nor `ERROR` (the statistic computed). Three tightenings on the drafting rule *what is the cheapest thing that satisfies the words without the intent?*: stated as a **record, never a tool** (cross-platform by mandate; `nvidia-smi` is one vendor's implementation); **the absence of the companion is never a waiver**, or the cheapest pass is to measure on a platform with no telemetry — and the Intel iGPU, which shares its power budget with loaded CPU cores, is the platform that loophole would most reward; and **a missing probe is `ERROR(instrument)`, never `SOLE_TENANT`** (R13). Plus **8b — two device-clock figures are comparable only if their device-state records agree**; a before/after whose "before" predates the requirement **is not a pair**, which upholds Switch's own ⛔ on his barrier result (*"probably sound is not the standard"*). **THE 40.201 ms REGIME-SEPARATION RESCUE FAILS; THE FIGURE IS RE-QUALIFIED, NOT WITHDRAWN.** The argument — two regimes 21× apart, so a run's regime is recoverable from its magnitude — dies on **there not being two regimes**: the board ranged **210 → 2490 MHz within a single run**, a governor is continuous, and *"the two clock states I sampled do not overlap"* was generalised into a claim about the device. Three more: the band rests on **two samples of one build**; the margin protecting **40.201 ms is 6.1×, not 21×**, and it sits at the *top edge* of the boosted band, the least protected position in it; and the rescue is about **clock** while foreign-GPU contention inflates **continuously**, with no regime structure to grip and no tenancy verdict on that run. What survives is arithmetic: every catalogued perturbation has a **non-negative** sign on time, so **40.201 ms is quotable as *≤ 40.201 ms, device state unrecorded*** — **RSD 0.033% loses its certifying role and keeps its descriptive one**, and the 40.390 → 11.525 within-series comparisons are **not certified either**, for the reason Switch himself supplied and did not carry across. **§10.0.4 THE INVARIANCE PREFERENCE — prefer the invariant that survives the contended machine.** Switch's correction to his own arithmetic: `min()` over inferences is an **upper** bound, not a lower one (`observed = true + delay`, `delay ≥ 0`), and **two upper bounds do not bound a difference from below** — "≤ 14.414 ms before" and "≤ 2.704 ms after" does not establish an improvement, let alone 5.33×. What rescued that result was **a count, not a clock: 147,618 `VkBufferMemoryBarrier` structs per inference before, 354 after** — *counts do not care whether the box is busy* — exactly as **byte counts (1997.6 MiB → 0.756 MiB)** carried weight residency when no timing was admissible. **Where a claim can be supported by a quantity the environment can perturb or by one it cannot, the unperturbable quantity is the claim of record and the perturbable one is at most an estimate of magnitude**; declare the sign, and a difference needs bounds on **opposite** sides. Its own cheapest abuse, named: *report the invariant as what it is — the reader may not be handed a count and left to supply the clock.* **M1 CRITERIA NEED NO RESTATING: no threshold moves, no criterion is rescoped.** Criteria **1, 2 and 4 are untouched and that is the finding** — bytes and counts, invariant under contention, tenancy and clock, and the only criteria that needed no correction through a week in which every timing figure was withdrawn twice. Criteria **3, 5 and 6 gain obligation 8 as an admissibility interlock**, and **6 is not exempt for being wall clock**: this EP is host-bound, so contention inflates the two arms by different factors and a ratio is not protected by being a ratio. **Neither surface is unqualified now and there is no third to retreat to.** Named attack on criterion 5: **run on a board stuck at idle clock** — device time inflates ~21×, host recording does not, the recording *share* collapses far below 5%, and every gate reports its most confident verdict; **a share-of-a-total criterion is satisfiable by inflating the total, and a device-state record is the only thing that would notice.**) · *prior revision 2026-07-31T07:45:10-07:00* (**§10.0 THIRD METRIC AMENDMENT — `MATCH` is not a verdict about this EP unless it carries what executed the model.** Specimen: ORT printed `EP_FAIL … Falling back`, re-ran the whole graph on CPU without raising, `get_providers()` still listed VulkanEP because the provider list is fixed at session-create time — and `model_output_equivalence` returned **`MATCH` for a run in which this EP executed zero nodes**. Wired, invoked, correctly named, arithmetically correct, **about a different world: R12 arriving at a verdict rather than a counter, where the frame is not a device but an executor.** The verdict becomes a **record** carrying `executed_by`, parsed on this run from **ORT's profiling trace — an instrument we do not own** — with `MATCH` **unrepresentable** at a zero own-provider count (constructor obligation, not an assertion beside the value), both witnesses recorded and disagreement emitting `SPLIT-FRAME`, and a fourth state **`UNATTRIBUTED`** that is emphatically **not** `DIVERGENT`: *the model was not wrong, the subject was.* **§10.0.1 R13 — an instrument's failure is not distinguishable from the condition it detects, and the reader who most needs the distinction is the one who predicted the red.** Trinity's Guard D — the fix for exactly the hole above — raised `NameError` before reading one profiling event; I watched the suite go `8 passed` → `5 failed` and reported the guard as working. **Three terminal tokens, always: `PASS` / `FAIL(condition)` / `ERROR(instrument)`; an instrument error never counts as a detection; a guard must state what it observed even when it fails; and the remedy is a second witness with a different failure mode, not a better first witness** — the lane now fails on the `Falling back` line itself, five sightings and every gate green. Second clause, and the more dangerous half, the inverse of R6 amendment 4: **a result that confirms a prediction deserves more scrutiny than one that contradicts it, because the contradiction gets checked automatically and the confirmation does not — quote the failure text, never the failure count.** First rule in this register about the reader rather than the instrument. **M0: criteria 2 and 10 REOPENED; four met, six partial, two not met, of twelve.** Criterion 10's closing evidence is **void, not narrow** — scope narrows a true statement and cannot repair one whose subject was absent — and the reopening is priced in advance: **three consecutive attributed `MATCH` runs in one session close it, same day, no new conditions.** Genuine and incomplete on the other side: **ORT profiling reports 354 of 364 nodes on the GPU in one fused island, 10 on CPU matching Mouse's declines exactly, `argmax 30751` == CPU** — the first attributed execution this project has recorded — against a multi-run picture that is red (weight cache OOM → silent fallback; 50 KV-cache outputs never written). Criteria 3, 4 and 5 advanced in substance and moved no row, because **I have not seen the artifacts** (R10), applied on a good day to mechanisms I asked for. **The ranked performance order stands — residency, net-benefit declines, fence-wait idle, kernels — because it was derived from counts and ratios, never from wall clock**; rank 1 keeps its place and changes its content from *make the weights resident* to **make residency bounded**, and **a performance mechanism that fails into silent CPU fallback is a correctness defect wearing a performance costume.** Residency landed on bytes: **1997.6 MiB → 0.756 MiB per inference, ratio 0.0004, forty times inside M1's threshold — and the criterion stays open**, which is the clearest evidence yet that the interlocks are the criterion. **Every wall-clock figure this project holds is withdrawn, 3.1× and 3.7× included** — taken during CPU fallback — so §10.0's disclosure obligation currently publishes **`UNATTRIBUTED`**, not a stale number; my own 22:13 clause, *no timing figure is quotable from a run whose verdict is not `MATCH`*, is what voids them, and **a rule that first bites its author was aimed at the right thing.**) · *prior revision 2026-07-30T22:13:37-07:00 (**STANDING DIRECTIVE (Justin): 「要确保我们性能是非常高 一致向高性能推进」** — *ensure performance is very high; push toward high performance continuously.* **RULING: it changes the calendar and not one gate.** It does **not** overturn the M0 performance ruling — a directive to be fast is exactly the condition under which a speed *gate* becomes dangerous, because the cheapest way to pass a ratio criterion is always to do less GPU work; it raises the value of the interlocks, not the case for the gate. It **does** make performance work continuous and parallel with correctness (`一致` is a **rate** obligation, so **the instrument for it is a series, not a value**, falsifiable by a flat line). Added on my own authority: **no timing figure is quotable from a run whose verdict is not `MATCH`, and every benchmark asserts EP presence and a non-zero claimed count before starting a clock — a fast wrong number is the failure mode this directive creates, not partial credit toward it.** **The tail (Windows + Linux + software rasteriser + CI) is unchanged and stays at the front**: it contends with residency for nothing — not a person, not a file, not a machine — and *a standing directive is a reason to re-examine a placement, never on its own a reason to move it.* **M1's weight-residency criterion stands exactly as written** (< 1% of constant-initializer bytes; today **1.0002**), both interlocks intact. **§6.5 RULING — exactly one `VkDevice` per (physical device, EP instance); the second one is a defect, not a design.** Tank's memory provider created its own device, so the session cannot bind its buffers; §2.3 already said `VkDevice` lifetime is EP-scoped, **so the document was right and the code diverged and nothing in between could tell**. No legitimate reason for two survives inspection (queue families, extension unions and external memory all fail to apply) and the split *costs* compatibility rather than buying it. **Seam owner: Switch** (the side that owns the lifetime, never the side that owns the caller); the allocator changes from *creating* to *receiving*. **§10.0.1 R12 — two instruments can each be correct about a different world, and a counter reading zero may be structurally incapable of reading anything else.** Specimen: `vulkan.cmd_upload` 15.2 s against `alloc_device_upload_bytes: 0`, both correct, different `VkDevice`s. **A quantity carries the identity of its frame; a counter whose event cannot occur in its frame reports `UNOBSERVABLE`, never `0`** — the fourth member of the family with `UNMEASURED`, `UNWIRED` and `SPLIT-DEVICE`, because **every way of not knowing gets a name a machine can print, since prose is where knowledge of a caveat goes to die.** **No criterion may name a pinned instrument**, so residency is read against `device_upload_bytes` on the session's device, never `alloc_device_authoritative_spans`. **R12 is not R11**: R11's remedy is available to the writer, R12's is structural. **Third disclosure obligation added — frame provenance — plus a seventh, positive one: independent corroboration is stated, not reconstructed** (Switch's span-derived 98.0% and Tank's counter-derived 95.8–98.4% for one quantity). Corrected picture, selector 0 = RTX 4060: **`cmd_upload` is ~71% of wall**, GPU kernels ~15%, and **pipeline lookup 0.4% / descriptor allocation 0.3% kill both standing hypotheses about per-island recording cost**) · *prior revision 2026-07-30T20:58:11-07:00* (**§10.0.1 R11 — a measurement's name is not its definition, and a decomposition that appears to close is the hardest kind of wrong.** R10's companion, found within hours of R10 by a specimen R10 certifies: `Phase::Record` is wired, invoked, correct, input-varying — **and misnamed by ~50×**. It is an *inclusive* interval containing the host staging copy, which reports into `phase_us[Upload]` and emits no `ph:"X"` span, so a span aggregation structurally could not see it: **upload is 95.8–98.4% of the "recording" phase; real command-buffer recording is 1–3% of wall.** The dominant cost is **the EP re-uploading the entire weight set on every inference** — 1997.6 MiB/inference, ratio **1.0002** against `device_upload_bytes`, exactly linear over 1/2/3 runs, in:out **2481:1**, transfer ceiling **~94.8% of wall discrete / ~44.0% UMA**. **The old table summed to 99.0% and appeared to close — because the missing cost was *inside* a row, so the residual was zero by construction.** R11 obliges: **every phase declares its extent (inclusive or exclusive of children); a flat table is an assertion of disjointness; the parts are summed against a whole measured by a *different* instrument and the residual published; and any row above 50% has its name checked against its content.** The register now reads **R6 our tooling manufactured a number · R7 a negative · R9 sound instruments jointly silent · R10 never called · R11 called, correct, and misnamed** — R11 is the hardest because *every check we have passes*. **The M0 tally does not move: six met, four partial, two not met, of twelve** — criterion 12 is *strengthened* rather than reopened, which is the whole benefit of its having been left open. **§10.0's disclosure obligation stands as written and is strengthened**: the phase decomposition was wrong by 50× while the wall-clock ratio (3.1× / 3.7× slower than CPU) was correct, **because the ratio has no internal structure to misattribute** — *a metric's robustness is inversely proportional to the number of naming decisions between the measurement and the reader; decompose to diagnose, report the coarse invariant.* A decomposition may accompany the ratio, never replace it, and is publishable only with its identity check. **R6 amendment 4 — the device labels were inverted team-wide**: `enumerate_capable_devices()` sorts best-first and `select_device` indexes the sorted list while `epctl --probe-loader` prints unsorted enumeration order — **`DEVICE=0` is the RTX 4060, `DEVICE=1` is the Iris Xe**, the "Intel beats the discrete GPU" finding dissolves, and **a result surprising enough to be a discovery is first a reason to check the instrument**. M1's lead performance criterion is corrected to **weight residency** — `device_upload_bytes`/inference below 1% of constant-initializer bytes, admissible only at or above last-published coverage with `MATCH` — with recording amortisation demoted to secondary) · *prior revision 2026-07-30T19:05:03-07:00* (**§10.0.1 R10 — a mechanism that exists in the source tree and not in the call graph is indistinguishable from one that was never written, and review cannot tell them apart.** R9's blind spot: *a falsifier that is never invoked is indistinguishable from one that never fires*. Five specimens in one day, all with correct code — `ops/partition.rs` (worth **3.7×**: islands 321 → 33, Intel 2954.6 ms → 807.2 ms when wired), the tracer, `model_output_equivalence`, `retain_viable`, and the EP-side validation messenger (loaded layer, no listener). **The falsifier for "X is wired" is an observation of an artifact X produced whose content varies with X's input — never a reading of X's code, never a flag its author set.** Uninvoked reports **`UNWIRED`**, distinct from empty; **the identity case is an explicit red state** (`island_count == claimed_count` was one line and was true for the whole life of the defect); **wiring is a property of an entry point, not of a file**; **review of a mechanism is not complete until the reviewer has seen an artifact it produced.** **§7.0.2 companion — a claim is a scheduling decision, not a capability statement: a correct claim can be a wrong claim**, net benefit is a property of the op *in a graph at a coverage level*, it lives in the partitioner and never in the registry, and it carries its own decline code. **M0 criterion 10 is MET — `model_output_equivalence = MATCH` on both devices** (argmax 30751 == CPU, top-10 10/10, max diff 0.031 / 0.035; **the non-identity is the correct answer** for fp16 accumulation order). Root cause was **binding arity, not dtype** — a 4-entry pipeline layout against a shader writing binding 4, silently dropped by both drivers — which is the strongest possible vindication of §8.9's `populated_optional_input_set` key component. **Criterion 2 closed on the promise made when it was reopened; criteria 4 and 5 stay partial — a correct model does not give an unknown-polarity check a polarity. Criterion 12 added (wiring census). Six met, four partial, two not met, of twelve.** **RULING: no performance criterion belongs in M0** — slowness is loud, wrongness is silent, and the cheapest way to pass a ratio criterion is always to do less GPU work; **M2 keeps the first threshold** (end-to-end ratio `< 1.0` on one discrete GPU with `MATCH`, every device in the matrix reported). **M0 gains a §10.0 disclosure obligation instead of a gate: the end-to-end wall-clock CPU ratio may never be omitted — currently 3.1× slower on Intel, 3.7× on NVIDIA.** **M1 gains a recording-amortisation criterion** checked with a counter before a clock, on the first honest trace: **68.3% command-buffer recording, 14.1% GPU kernels, 0.3% submit — 85.9% of runtime with no GPU work happening**, fixed per `Compute` call rather than per dispatch, which falsifies the fixed-per-submission hypothesis. **Sequencing: the tail resumes at the front; the 68.3% starts in parallel as Tank's M1 work and lands only through criterion 10's gate**, because a cached binding table is exactly the shape of today's defect) · *prior revision 2026-07-30T06:32:18-07:00* (**§8.9 RULING — unproven is a claim-path state; claiming is gated on evidence and `Live` stops being a thing we write down.** §7.0.1 companion: *evidence shortfalls degrade op coverage, not device availability, identically to capability shortfalls* — the frozen §7.2 gate is untouched. The table declares only `Staged(why)` / `Ready`; **claimability is derived per form from a harness-generated proof ledger**, keyed on `(domain, op_type, opset_bucket, every input/output dtype, kernel_variant_key, shape_class, populated_optional_input_set)` — so **§8.7's expression-vs-path distinction becomes mechanical: an expression difference leaves the key equal, a path difference changes it**, and an f32 proof can never be returned for an f16 node. **A `DIVERGENT` model verdict demotes every form that participated, automatically.** Escape hatch is **a list of proof keys and nothing else** — no `1`, no `*`, no wildcard, C1's shape — with WARN at session creation, `unproven_forms_enabled` in the counters artifact, and `epctl --check-counters` failing on a non-empty list. **Honest cost: Phi-3.5's claimed count goes 161 → 0** — and per §10.0's gate that 161 was already void. **M0 criterion 11 added; four met, four partial, three not met.** **Link's lanes: the gate is a precondition for a lane being declared *green*, not for a lane being brought *up*** — with a per-lane **gate artifact** rather than Phi-3.5 on a rasteriser) · *prior revision 2026-07-30T05:48:29-07:00* (**§10.0.1 R9 — a set of individually sound instruments can be jointly silent on the property that matters; *for every claim, name the instrument that would go red if the claim were false*.** Phi-3.5 on both devices: 161 `MatMulNBits` claimed **and accepted by ORT**, `compute_failures: 0`, `dispatches_executed: 161`, suite green — and `vk argmax 0` against `cpu argmax 30751`, top-10 overlap 0/10. **§9.1.3 RULING — `compute_failures` is an execution-status counter and may never be read as a correctness signal**; prose cannot close that reading, a verdict emitted next to the counters must. **Metric of record gated on `model_output_equivalence` ∈ {MATCH, DIVERGENT, UNMEASURED}**, default `UNMEASURED`. **M0 criteria amended: criterion 10 added (model-level correctness); criteria 2, 4 and 5 REOPENED; criterion 8 relabelled parity-only.** **Sequencing: criterion 10 outranks the Windows/Linux/lavapipe/CI tail in order, and does not replace it as a gate**) · *prior revision 2026-07-29T21:14:03-07:00* (**§8.8 RULING — dynamic shapes are a claim-path capability, not a kernel feature**, and move **ahead of** the three kernels, §10.0.3; measured on the first end-to-end real-model run: **258 nodes declined on symbolic shapes vs 100 on missing kernels**, and the decline codes are first-match so 258 is a *floor*; **§1.2's dynamic-shape non-goal reversed**; **M1 gains a second-token exit criterion** — one session, two concrete values of a symbolic dimension; **OQ-15 promoted to blocking**; **§10.0.1 R8 — we planned against the ops a model contains, having never measured why its nodes are declined**) · *prior revision 2026-07-29T19:42:07-07:00* (**M0 criterion 8 MET — both barrier backends executed, bit-exact on two vendors**; **45 op rows `Live`**; **criterion 3 ruled not discharged — a validation lane needs a positive control**; **criterion 9 not met** — `PLATFORMS.md` LVP2 still carries §7.2's false premise; **§10.0.1 R7 — our instruments fabricate negatives**, *derive, do not declare*; **§8.7 template evidence covers a different expression, never a different path**) · *prior revision 2026-07-29T16:00:55-07:00* (**`Add` executes through ORT — M0 criterion 2 MET**; **§7.2's R5 rationale corrected**, re-grounded on §7.0; **§10.0.1 R6**; criterion 8 amended so a skip cannot satisfy it) · *prior revision 2026-07-29T15:02:55-07:00* (**§8.5 third strengthening**; **metric triple `(coverage, island_count, largest_island_flops)`**; **T3 demonstration target is Phi-3.5**; **R5**) · *prior revision 2026-07-29T09:47:45-07:00* (**first shader dispatch**; **M0 assessed criterion by criterion**; **§7.9 capability probing**; §8.5 *producer **at version***; R4) · *prior revision 2026-07-29T08:13:58-07:00* (**§8.5 producer-relative**; **§8.6 crate evaluations**; **§10.0.2 `ai.onnx::Attention` first**; R1 narrowed + R3) · *prior revision 2026-07-28T22:28:08-07:00* (**OQ-4 §7.8**; **OQ-M6 ruling** §8.4; **OQ-3 §6.4** reserved-VA, no BDA; C2 **item 7**; `retain_viable` §5.4; eleven contrib ops; OQ-16; **§9.1.1 oracle validated**; **R1**)
**Author:** Morpheus (Lead / EP Architect)
**Repo:** `onnxruntime-ep-vulkan`
**Reference architecture:** `onnxruntime-mlx` (Justin Chu's MLX plugin EP for Apple Silicon)
**Sibling documents:** [`ENGINE.md`](./ENGINE.md) (Switch — Vulkan runtime & shaders), [`PLATFORMS.md`](./PLATFORMS.md) (Link — platform & hardware matrix), [`OP_COVERAGE.md`](./OP_COVERAGE.md) (Mouse — **authoritative op-coverage plan**, ratified), `THIRD_PARTY.md` (Rai — licence compliance)

---

## 0. What this EP does today — and what it does not

> **Read this section first, and read both halves of it.** It exists because the record grew large
> enough that its absence was itself a defect: there was no document a newcomer could read to learn
> the state of the project, only 5,900 lines of rulings written for people who already knew. Every
> figure below is derived from a named artifact or symbol in this tree, never from a status report
> and never from another section of this document. Where a figure is a *computation* rather than a
> *measurement*, it says so, because **a §0 that only lists wins is a marketing document and will be
> distrusted, correctly, by the first reader who checks one line of it.**
>
> Last re-derived from artifacts: **2026-08-04, at `main` = `3365221`.**

### 0.1 What it does

**It runs two model classes, not one, and the second one arrived on 2026-08-03.** Until `Conv`
(f32) landed, every model this EP had ever claimed a node of was a decoder-only transformer.
`bench/results/op_census_mobilenetv2_r13_after.json` reads `graph.nodes: 105`, `claimed_nodes: 97`
on MobileNetV2-12 — a convolutional image classifier, previously **0 of 105**. The delta table is
`docs/OP_COVERAGE.md` §13.9.6. The 7 declines are one contiguous
`Shape→Gather→Unsqueeze→Concat→Reshape` classifier tail plus `GlobalAveragePool` and `Gemm`, not
seven scattered holes. Grouping is not a second kernel: `rust/shaders/glsl/conv_f32.comp` computes
the general grouped form with `cpg = pc.c / pc.group`, so depthwise is the `group == C` case and
dense is `group == 1`. Read §0.2 before reading that as coverage of `Conv`.

**It loads into a stock ORT and claims a real model.** On Phi-3.5-mini-instruct (ONNX, int4
`MatMulNBits`), `bench/results/op_census_phi35_r13_after.json` reads `graph.nodes: 366`,
`claimed_nodes: 355`, and ORT fuses them into **one island**:
`bench/results/phi35_claim_reading_summary.json` — `claimed_nodes: 355`, `islands_offered: 1`,
`viable_islands_retained: 1`, `ledger_hits: 355`, `unproven_forms_claimed: 0`. On gpt-oss-20b
(`op_census_gptoss20b_r13_after.json`) it is **293 of 374**. The remaining nodes
decline for stated reasons; three of them decline as `[unproven]`, which is the ledger refusing, not
a gap in the kernels. Attribution is ORT's own profiler, not ours: `bench/results/criterion10-dev0.json`
records `executed_by = {VulkanExecutionProvider: 3, CPUExecutionProvider: 24}` over three inferences —
three island executions, one per run.

**It agrees closely with the CPU EP, and it is not bit-identical to it.** Over all 65 model outputs
(logits + 32 layers × key/value), `criterion10-dev0.json` records **62 of 65 within tolerance**, a
**median of 1 fp16 ULP** across outputs, `argmax_cpu == argmax_vk == 30751` and `top10_overlap = 10`.
Three outputs are outside: output 0 (the logits head) at a median of **12 ULP**, and outputs 63 and 64
(the last layer's key and value) at **4 ULP** each, against a ceiling of 3 predicted in writing before
the measurement. The largest absolute logit disagreement is **0.0625** on a max logit of 13.14. The
verdict on record is therefore `DIVERGENT`, on both devices, and it is honest — see §0.2.

**Its op table is 96 rows, of which 78 carry a kernel.** Read from
`epctl --dump-capabilities --json` (`rust/src/bin/epctl.rs`), not from counting `ops!` lines: 46
rows report `live`, 32 report `ready` and **18 report `staged`** — described, claim-tested, and
declined at runtime with a named blocker carried in the dump's own `staged_reason` field. **The 78
is `has_kernel == true` on the dump row, which is *not* `status == "live"` (46).** §8.9.25's rename
has landed: the row used to spell the noun `live` twice with two denotations — a boolean meaning
*this row has a kernel* and a status token that is the deprecated `OpStatus::Live` alias — so a
reader checking the 78 against the field named `live` got 46. The boolean is now **`has_kernel`**
and the `status` token keeps its three values, so the field name answers the question a reader is
asking. **This count has been misstated in this document and in status reports five times; the dump
is the only reading of it that is not a hand tally, and it does not stop being wrong until it stops
being written by hand.** `OpStatus::Live` is a deprecated alias of `OpStatus::Ready`
(`rust/src/registry.rs::OpStatus`); neither grants a claim. **Claimability is a ledger fact, not a
table field.**

**Nothing is claimed without a proof under its own key.** `evidence/proof_ledger.jsonl` holds **121
entries** (`ledger_entries: 121`, `bench/results/_ledger_counters.json`), each recording the device,
the ORT build, the tolerance, the artifact and its sha256, the nodes claimed and dispatches executed
on the proof run, the shader stems dispatched, and — since §8.9.19 — **two** digests over those
stems, one of the SPIR-V and one of the source closure, so that a compiler change is distinguishable
from a kernel change. A form with no entry declines with `[unproven]`; a build whose kernels have
moved declines too. The gate has been shown to move with its input rather than with the enumeration
that feeds it: `bench/results/census/criterion11c-ledger-arms-devunset.json` reads
`ALL-PROVEN`/`ALL-DECLINED` across four arms differing in exactly one proof-key component each, and
`bench/results/criterion11c_mutations-dev0.json` records three mutations of that test, all `CAUGHT`.

**The ledger now has an invariant that asks whether a proof went *missing*, and it is tested against
a real deletion rather than a planted one.** Every check before it compared the entries that are
*present* against the build; none could see an entry that is absent, which is the one failure a
merge produces. `ci/check_ledger_census.py` compares the ledger against the attempt log and fails on
a key that was recorded `MATCH` and is no longer there, with retirement by name and reason as the
only exemption. Its controls are `ci/negative_control_ledger_census.py`, six arms, all passing
(`bench/results/_probe_ledger_loss/result.json`): arm 3 **replays `eb84364`**, a real merge in this
repository whose conflict resolution silently dropped three real `Cast` proofs, and convicts it. The
same probe records why nothing caught it at the time — default history simplification hid the
proving commit from the file's own log, so the census is run with `--full-history` and asserts the
denominator by the numbers (13 revisions simplified, 55 unsimplified).

**The KV cache can decline the host round trip.** `bench/results/kv_chain_readback-{nvidia,intel}.json`
both read `ROUND_TRIP_REMOVED`: steady-state device→host traffic falls from **1792 to 0 bytes per
step** at *identical* dispatch counts, with the resident lane agreeing with the host lane to **0.0**
on all three outputs. Measured on both an RTX 4060 and an Iris Xe. No wall-clock is quoted anywhere
in that artifact, by design — and none of it reaches a user today, for the reason in §0.2.

**The claim path is conservative by construction and the conservatism is tested.** Zero Vulkan
devices are advertised when there is no ICD and when the build carries no shaders
(`bench/results/criterion4_icd_witness-dev{0,1}.json`, `criterion5_shaderless_witness-dev{0,1}.json`),
the layering lint runs in CI over permanently-planted violations
(`rust/tests/layering.rs::detects_planted_ort_abi_violations`), and a real Phi-3.5 inference produces
**0 in-frame VUID messages in a frame demonstrated to carry validation output** — the liveness arm of
`bench/results/criterion3a_phi35-dev0.json` shows 14 messenger lines arriving inside the same window
that reports zero errors.

### 0.2 What is not true yet

**The device-memory path ships OFF, so none of the KV work above reaches a user.**
`factory::device_memory_enabled` is a read of the `ENV_DEVICE_MEMORY` environment variable and
nothing else; unset means false, and `allocator::device_memory_requested` delegates to it. A default
run therefore reports `alloc_device_frame: "OFF"` and, in its own words,
`"no device-memory provider exists in this process, so the allocator side is on no VkDevice at all"`
(`bench/results/_ledger_counters.json`). `ROUND_TRIP_REMOVED`, the resident KV lane, and the 5.51 GB
ctx-8192 arena are all behind that flag. **A capability behind an opt-in flag is a capability the
project has, not one the product ships**, and §0.1 must not be read as the latter. The flip was
scored against four pre-registered gates
(`bench/results/device-memory-flip-gates-prediction.md`) and one blocker survives.

**There is no timing of any kind here, and the design that was going to produce it is refuted.**
Not withheld pending a quiet box — *refuted*. The paired interleaved A/B alternation was built on
the assumption that contention is common-mode across the two arms, and every artifact says it is
not: `bench/results/paired_ratio_dev0.json` returns `PAIRING_FAILS(apparatus_asymmetry)`, with
`cpuload` moving the ratio 0.560× (vk 3.32× against cpu 5.94×) and `gpuload` moving it 0.722×.
`paired_ratio_dev1.json` and `paired_ratio_resident_dev0.json` fail the same way, four control
failures across the set. **Two of them are worse than a failed control.** Under foreign *GPU* load
our own arm gets **faster** (`vk_lift_x: 0.771`), which no contention model predicts; and the
granularity that would make contention symmetric is the same granularity that manufactures the
device-axis asymmetry, so **the failure is not tunable** — there is no window size at which both
arms are disturbed alike. A ratio is not published, a ratio is not withheld: the apparatus that
would produce one does not measure what it claims to. Every duration in this repository is either
absent or carries a device-state record. §0 quotes counts and bytes only.

**Criterion 10 is `DIVERGENT`, and it is now known that the defective statistic was not the
reason.** §8.9.22 ruled that a `max` over a relative measure whose denominator can go degenerate
measures the degeneracy, and replaced criterion 10's observable. Re-scored under the ruled
statistic, `bench/results/criterion10_rescore_8922-dev0.json` reads `comparison_outcome: DISAGREE`,
`outputs_within_tolerance: 62` of 65, `failing_indices: [0, 63, 64]`, `outputs_degenerate: 0` —
**the verdict is unchanged and the ruling bought it nothing**, which is the correct outcome for a
repair and the wrong one for a hoped-for reprieve. The logits head and the last layer's key and
value are genuinely outside tolerance. It must not be closed by moving `atol`.

**And it cannot be closed by calling the tolerance unsatisfiable, which was the next thing tried.**
§8.9.24 rules that criterion 10's predicate — `|a − b| ≤ atol + rtol·|b|` — is satisfiable at
**every representable fp16 value with at least 20.48 element-ULPs of margin** (`rtol · 2¹⁰`, on
normals; ≥ 16,777 ULP on subnormals from the `atol` term alone), so **every failing element failed
by more than twenty representable fp16 steps at its own magnitude.** The contrary reading — *"`atol`
is 0.128 ULP-at-scale, finer than fp16 can express"* — quoted one term of a two-term sum and divided
by the spacing at the *tensor maximum* while the predicate evaluates per element; the full allowance
at each tensor's own scale is **33.628 / 29.796 / 32.404 ULP-at-scale**. No motion on the tolerance,
unit, predicate or verdict structure is admissible until outputs 0, 63 and 64 have a float64 answer
to **which side is wrong** — at the final RMSNorm Vulkan is bit-exact and ORT's CPU EP carries the
1 ULP, and that question has never been asked of these three.

**That answer now exists at model scale, and §8.9.25 rules that it does not close the row and may
not loosen it.** Trinity's layer-at-a-time float64 chain (`bench/results/criterion10_chain-dev{0,1}
.json`, 355 nodes, 32 layers each proven live, both devices, both reference variants, reading
initialisers and `input_ids` only so that neither EP enters its own derivation) reports: **output 0
— `cpu`**, ORT's CPU EP the further side, unanimous on five discriminators in both variants on both
devices, **83 vs 70 element-ULP from true**; **outputs 63 and 64 — `direction: null`**, no fact of
the matter. **The answer is not uniform, and a tolerance motion built on it would admit three
outputs on a direction that exists for one.** The depth series demanded on 2026-08-02 also exists
and returned the convicting branch: `kv_depth_curve` is flat at 0–2 ULP for layers 0–30,
`kv_depth_largest_step` is **1.0**, and the sole exceedance is **layer 31's key and value at 4 ULP**
— which are outputs 63 and 64. And both EPs sit much further from the weight-only reference than
from each other (**70/83 against 12 apart** on the logits; **12/12 against 4** and **6/7 against 4**
at layer 31), so **an AGREE bounds only the difference and never the distance from truth** — while a
DISAGREE is untouched, because a shared error cancels in a difference.

**102 of the 121 ledger entries record no specialisation state.**
`ledger_specialisation_unrecorded_entries: 102` (`bench/results/_ledger_counters.json`). A
specialisation constant is resolved at pipeline creation and changes the code the driver emits, so
an entry that does not record the value it was proven under is a proof whose subject is only partly
named. The `spec_digest` field exists and the newest entries carry it; the majority do not. This is
the §8.9.19/§8.9.21 specialisation debt with a number on it, and it is unowned as a residual.

**Its Linux lane compiles and runs, and its op suite is red.** `8 failed / 633 passed / 50 skipped
/ 3 xfailed` (`docs/PLATFORMS.md` §7.26.3), decomposed there rather than quoted as a total: 3 are a
missing validation layer on the host, 1 is an unexplained `NO_PROGRESS` instrument error, 2 are real
numeric divergence in `Asin`/`Acos` on llvmpipe, 1 is closed, 1 is a declared accepted red. The
previous reading of `19 failed / 622 passed` was mostly one defect — a deliberately-broken
zero-shader build inheriting `CARGO_TARGET_DIR` and overwriting the real artifact mid-run. M0
criterion 2 says *green*, and green is not what either lane reports. **Criterion 1 is not met.**

**And one absent optional dependency can make that suite assert nothing while reporting green.**
`tests/ops/test_shape_inference_delta.py` imports at module level, so an `ImportError` aborts collection
of the whole directory — 292 skipped, nothing asserted, exit status uninformative. A suite's verdict
must be a function of its assertions; a lane needs a **declared expected execution count**, not an exit
code (§8.9.19 part 4).

**Criteria 2 and 10 are reopened and 11 and 12 are not closed.** See the M0 table in §10.

**A `Conv` proof says nothing about `group`, `strides`, `dilations` or `pads`.** The four entries
are `{bias, no bias} × {static, runtime-extent}` and that is the whole key space for this op;
those four attributes are not proof-key components and §8.9.23 rules that they should not become
them, because they are push-constant values read by one uniform code path rather than selectors of
different code. What follows from that ruling is a **disclosure** obligation, not a key one: until
the claim line names the axes its key does not cover, a reader of the session disclosure sees
`Conv ... proven by ...` and has no way to learn that a stride-2 asymmetrically-padded depthwise
convolution is spoken for by `tests/ops/test_conv.py`'s twelve CI-time combinations and by nothing
that ran in their session. The gap was written down plainly by the kernel's own author, unprompted,
in `docs/OP_COVERAGE.md` §13.9.4 — **the documentation of the limit is excellent and the runtime
disclosure of it is silent**, and it is the second one a user reads.

**`Conv`'s key names no shader, and that is false.** The four entries render their variant component
as the literal `metadata`, whose documented meaning in `registry::variant_key` is *"this row has no
shader"* — while the same entries record `"shaders": ["conv_f32"]` and a real `shader_digest`. The
cause is `ops/conv.rs`'s `kernel!(None)` row: the module is chosen inside `translate`, where the key
does not look. **The subject knows the shader; the key denies it exists.** §8.9.23 rules this a
defect in the key and owes the repair to the op author.

**`REACHED_USER` means the write succeeded, not that a human read it.** The §8.9.7 session
disclosure reaches the console on a default run by a direct write to process stderr that bypasses
ORT's severity threshold entirely — measured, both devices, both polarities, by
`rust/tools/probe_disclosure_reachability.py`, with the escalation arm proving the WARN path fires
only when the quiet channel is actually gone. But a stderr redirected to a writable sink — a log
file, a pipe nobody reads — returns success exactly as a terminal does, and no return code
distinguishes the two **from inside the process**. `session_disclosure_info_reach` therefore reports
the last event on our own side of the boundary and cannot report the next one. This is a bound, not
a bug: see §8.9.23's statement of the general form. Do not over-trust the token.

**"Weight read amplification = 1.000000" is an identity, not a measurement.** The figure comes from
`bench/results/island_bytes_phi35.json::weight_reread_amplification`, whose four fields are literals in
`bench/results/probe_island_bytes.py`. Its numerator is the blob count from
`bench/results/gemv_counts.json::bytes.blobs_per_inference` and its denominator is the weight bytes
from the same census; since a blob **is defined as** 16 weight bytes, the ratio is `16/16` for every
model, every shape and every kernel, including a broken one. It is `x/x`. What the SPIR-V walk behind
it genuinely establishes is narrower and still useful — the packed GEMV issues **one 128-bit load per
16-byte weight blob** rather than four 32-bit ones (`gemv_counts.json::arms`) — but that is an
instruction count, not DRAM traffic, and cache re-reads are explicitly excluded from it. **No artifact
in this tree measures how many times the device reads a weight byte from DRAM.**

**The growing-context KV term is a lower bound, not a measurement.** `kv_chain_readback-*.json` runs
six steps at a *fixed* past extent of 4 and says so in its own `why` list. `island_bytes_phi35.json`'s
`kv_cache_MiB` series across past lengths 0 → 8192 is computed from declared shapes, not observed.
`bench/results/kv_bytes_earned.json` earns the **write** side to the byte (393,216 B per past token,
ratio 1.000000, zero spread across two segments) and explicitly declines the **read** side: how the
past KV becomes device-resident is unobserved, and the artifact refuses to report it as zero.

**A proof is not portable, and the ledger only half says so.** Entries record a `device` field, and
for the older ones `device0` is a *selector ordinal*, not a device identity
(`rust/tools/gen_proof_ledger.py`: `device = args.device_name or f"device{args.device}"`). The
predicate now reads the field and reports `DEVICE-UNATTRIBUTED` rather than claiming silently —
`device_unattributed_forms` in `bench/results/_ledger_counters.json` carries the reason in full —
so the state is disclosed. It is not repaired: a proof run on a second GPU still cannot be told from
a proof run on the first for any entry naming an ordinal.

**No performance claim is made here.** The development machine is permanently contended by several
agents.

---

### 0.3 What it is, in one paragraph

`onnxruntime-ep-vulkan` is an **out-of-tree ONNX Runtime plugin Execution Provider** that runs
fused ONNX subgraphs on any Vulkan compute device. It is loaded by a **stock, unmodified** ORT
build through the plugin-EP C ABI. No ORT fork, no ORT rebuild, no link against
`libonnxruntime`.

| Field | Value |
|---|---|
| Repository / vendor string | `onnxruntime-ep-vulkan` |
| Cargo crate | `rust/` — `onnxruntime-ep-vulkan` |
| Library artifact | `libonnxruntime_vulkan_ep.so` / `onnxruntime_vulkan_ep.dll` / `libonnxruntime_vulkan_ep.dylib` |
| Registered EP / device name | **`VulkanExecutionProvider`** |
| Crate type | `cdylib` |
| ORT ABI | plugin-EP C ABI. **Built against ORT 1.28** (`ORT_API_VERSION`); **minimum runtime API 1.24** (`ORT_API_VERSION_MIN = 24`) with version negotiation. ORT 1.28 fixes a null-allocator PrePack crash that would have hit us on 1.27; 1.24 is where the `OrtEpFactory` surface we rely on stabilized. |
| Version scheme | `0.<ORT_API_VERSION>.<patch>` → `0.28.0` |
| Backend | Vulkan compute, GLSL → SPIR-V, `ash` bindings |
| **Device requirement** | **Vulkan 1.1 core + a compute queue + four limits (§7.2).** No required extensions. Everything else is probed and degrades op coverage, never device availability. |

The architecture is **deliberately the same shape as `onnxruntime-mlx`**: a registry-driven,
claim → fuse → compile → run plugin EP with conservative node claiming and clean CPU fallback.
Every module in `onnxruntime-mlx` has a counterpart here. Where we diverge, §12 records the
divergence and the reason.

**The single biggest divergence from the MLX reference:** MLX runs on Apple unified memory, so
the MLX EP advertises *no device allocator* and copies out with one `memcpy` at the subgraph
boundary. Vulkan has **explicit, non-coherent, non-unified device memory**. That reshapes the
tensor/memory contract (§6), the factory surface (§2.5), the compile step (weight prepacking,
§5), and the milestone plan (§10). It is the reason this document exists rather than a
find-and-replace of the MLX design.

---

## 1. Goals and non-goals

### 1.1 Goals for v1

1. **Cross-platform GPU inference from one codebase.** Windows, Linux, Android, and macOS
   (MoltenVK) on NVIDIA / AMD / Intel / Adreno / Mali, plus the **lavapipe** software rasterizer so
   CI is possible without a GPU runner. No vendor-specific code path is
   permitted to be load-bearing for correctness.
2. **Zero ORT fork.** Ship a single shared library that a stock ORT loads via
   `RegisterExecutionProviderLibrary`.
3. **Correctness before performance.** The ORT CPU EP is the oracle. A claimed op must match it
   within a stated tolerance on every supported platform before we quote a single speedup number.
4. **Conservative claiming with clean CPU fallback.** Claim only node forms whose exact
   dtype / attribute / shape / layout contract the Vulkan translator implements. Everything else
   runs on ORT CPU. Falling back is a feature, not a gap.
5. **Compile-once, replay-many execution.** Weight upload and prepacking happen at `Compile`
   time; per-inference work is command-buffer submission, not graph construction.
6. **A layering that survives contact with contributors.** The ORT C ABI never reaches op code;
   raw Vulkan handles never reach op code. Enforced by module privacy, not by review vigilance.

### 1.2 Non-goals for v1 — explicit and ruthless

These are **out of scope**. Each is a decision, not an oversight. Re-opening any of them requires
a decision record.

> **SUPERSEDED IN PART BY USER RULING, 2026-07-28T20:54:42-07:00.** Justin ruled directly:
> **「contrib op 要做」 — the `com.microsoft` contrib operator domain is in scope.** That decision is
> settled and is not mine to re-open; the `ai.onnx`-only position originally stated in this section
> is void. Record: `.squad/decisions/inbox/copilot-directive-contrib-ops.md`. What remains mine, and
> what §1.4 now contains, is **how we admit it safely** — claim-predicate discipline, the schema-drift
> protocol, and clean CPU fallback. Admission settled; constraints binding.

> **Amended 2026-07-28T19:16:08-07:00 by the ratification of [`OP_COVERAGE.md`](./OP_COVERAGE.md)
> (OQ-11).** Four rows below were reversed. The reversal is not a loosening of ambition-control; it
> is a correction of a factual error on my part. I wrote these rows believing that "run a Qwen
> graph" and "support the `com.microsoft` domain" were separable. Mouse verified from the ONNX
> Runtime GenAI model builder source that they are not: the builder **emits** contrib ops directly,
> so an EP that declines `com.microsoft` cannot run a Qwen graph at all — not slowly, not partially,
> at all. A non-goal that makes the project's named target unreachable is not ruthless, it is wrong.
> The reversed rows are marked **REVERSED** below and the constraints that now attach to them are in
> §1.4.

| Non-goal | Why |
|---|---|
| **Training / gradients** | ORT training EPs are a different ABI surface and a different correctness problem. Inference only. |
| **ONNX opset completeness** | Still a non-goal *as stated* — we do not chase the spec index. But see §8: coverage is now driven to ~174 inventoried ops by model family, and the reason the original row gave ("Vulkan supplies nothing — every op is a shader we write") is answered by kernel-template leverage, not by refusing breadth. What remains a non-goal is claiming ops **no target graph contains**. |
| **Dynamic shapes in the fast path (M0–M2)** | **REVERSED — 2026-07-29T21:14:03-07:00, on measurement.** This row has now been wrong twice, each time in the same direction. It was first narrowed (A5: LLM-path kernels take dimensions in push constants from tier 3). It is now **removed as a non-goal for the claim path entirely**: on the first real model we ran end to end, **symbolic dimensions declined 258 of 363 nodes while missing kernels declined 100** — and the 258 are nodes that had already passed registration, opset, schema and status, so shape is the *sole remaining blocker* for every one of them. See §8.8 for the ruling and §10.0.3 for the sequencing. What remains a non-goal: **data-dependent** shapes (next row), which is a different problem with a different answer. |
| **Data-dependent output shapes** | `NonZero`, `Unique`, value-dependent `Reshape` targets, `NonMaxSuppression`. These need a mid-graph host readback that a recorded command buffer cannot express. Permanent CPU fallback. Mouse inventoried and permanently declined these. |
| **fp64** | Most consumer GPUs have no usable double precision and Vulkan makes `shaderFloat64` optional. Permanent CPU fallback. |
| **Quantized ops (int4/int8 matmul, `MatMulNBits`, `GatherBlockQuantized`)** | **REVERSED — in scope, tier 4 (M3).** An int4 Qwen graph is the variant people actually run; without `MatMulNBits` it shatters into hundreds of islands and the EP is worse than useless on it. Constraints in §1.4. |
| **Attention fusion (GQA / MHA / SDPA / flash attention)** | **REVERSED for `GroupQueryAttention` — in scope, tier 3 (M2/M3).** My original row conflated two things. Performing a fusion is out of scope. `GroupQueryAttention` is not a fusion we perform: it **arrives as a single node from the exporter**, and decomposing it would materialize a `[B,H,S,S]` score matrix in VRAM — gigabytes at S=4096. Implementing it is the *conservative* choice. Fusion patterns we would have to *detect* remain out of scope. |
| **Graph-level op fusion** | Unchanged non-goal for patterns we must *detect* (llama.cpp's `MUL_MAT+ADD`, `RMS_NORM+MUL`). Note this is distinct from a single ONNX node whose semantics are internally fused (`Softmax`, `LayerNormalization`, `SkipSimplifiedLayerNormalization`, GQA) — implementing those as one kernel is not graph fusion, it is implementing the op. |
| **Mobile-first tuning** | Android must *work* (M3) and must not be architecturally excluded (§7). Tile sizes, memory budgets, and Adreno/Mali-specific tuning are not v1. |
| **Images / texture-backed tensors** | Buffers only in v1 (see `ENGINE.md` §3.6). Named trigger for re-evaluation: `Conv` at tier 5c/6. |
| **Multi-GPU / multi-queue overlap** | One `VkDevice`, one compute queue, one submission per subgraph execution. |
| **Cooperative matrix / tensor-core paths** | Optional extension on every baseline. Capability-probed later, never required. |
| **Custom / contrib domain ops (`com.microsoft`)** | **REVERSED — NO LONGER A NON-GOAL. `com.microsoft` is in scope by user ruling, 2026-07-28T20:54:42-07:00.** The nine ops needed for the Qwen3.5 target are enumerated in §1.4 and are the *admitted set for v1*; admitting a tenth is now a scoping decision within an in-scope domain (a decision record, not a re-opened non-goal). The engineering discipline of §1.4 — per-op claim predicates, no domain-wide opt-in in code, census-in-CI — is unchanged and binding, because it was never about whether the domain was permitted; it is about not claiming a schema we have not read. |
| **Shipping wheels on PyPI in v1** | The Python package exists for testing from M0. Publishing is a release decision, not an architecture one. |

### 1.3 Why "conservative claiming" is a hard requirement, not a preference

Because the fallback is not free but it *is* always correct. The failure mode we are designing
against is not "we didn't claim enough ops" — it is "we claimed a node form our shader gets
subtly wrong on one driver, and a user gets silently wrong logits." Every claim predicate is a
promise. The rule from the MLX reference stands verbatim: **when in doubt, do not claim.**

Nothing in the ratified coverage plan relaxes this. A faster schedule changes *how many* ops land
and *in what order*; it does not change what claiming means. If anything, the higher the op
throughput, the more load-bearing this rule becomes.

**Amendment 2026-07-30T06:32:18-07:00 — this paragraph has been in this document since day one and
we shipped its exact failure anyway.** On 2026-07-30 the EP claimed 161 `MatMulNBits` nodes on
Phi-3.5 and produced `argmax 0` against the CPU EP's `30751` — *"a user gets silently wrong logits"*,
verbatim, from a rule written to prevent it. **A prose commitment without a mechanism is not a
commitment** (§9.1.3, R7). The mechanism is §8.9: claiming is gated on a harness-generated proof
ledger keyed on the dispatchable form, and "when in doubt, do not claim" becomes "when there is no
ledger entry, you are in doubt — by construction."

### 1.4 `com.microsoft` contrib ops — in scope; the constraints that attach

**Scope status: settled by user ruling, 2026-07-28T20:54:42-07:00.** The `com.microsoft` domain is
in scope. The deciding fact, verified by Mouse from the ORT GenAI model builder source, is that the
builder *emits* contrib ops directly, so an EP that declines the domain cannot run a Qwen graph at
all — and Qwen3.5 is a named target. I am not re-litigating that here.

**The admitted set for v1 is these eleven.** Nine were admitted on 2026-07-28T19:16:08-07:00 as the
ops a Qwen3.5-class GenAI-built graph actually contains:

`GroupQueryAttention`, `RotaryEmbedding`, `SimplifiedLayerNormalization`,
`SkipSimplifiedLayerNormalization`, `MatMulNBits`, `LinearAttention`, `CausalConvWithState`,
`QMoE`, `GatherBlockQuantized`.

**Two more are ratified here, 2026-07-28T22:28:08-07:00**, both staged by Mouse in `5ae991a` and
both accepted — this is the "a tenth requires a decision record" clause working as intended, and the
record is this paragraph:

- **`MultiHeadAttention`** — the non-GQA fused attention form that ViT and BERT-style encoders
  export. Admitted for the same reason as `GroupQueryAttention` and not by analogy to it: it
  *arrives* as a single node, and decomposing it materializes a `[B,H,S,S]` score matrix. It is
  tier 5c's vision-tower dependency (§10 M3+), and it should share `ops/attention.rs`'s kernel with
  GQA rather than becoming a twelfth thing to write.
- **`MoE`** — the float-expert sibling of `QMoE`. Admitted because a routing implementation that
  only exists in its quantized form cannot be differentially tested against a float oracle, which
  makes `QMoE` much harder to verify, not easier. Cheaper to carry both than to debug one.

Adding a twelfth is the same clause again: a decision record and a tier assignment, inside an
in-scope domain. What follows is what I still own, and it matters *more* now
that the domain is admitted, not less. Contrib ops are a genuinely more dangerous surface than
`ai.onnx` and the discipline below is the whole reason admitting them is safe.

#### C1 — No domain-wide opt-in exists anywhere in the code

`com.microsoft` being in scope as a *product* decision must not become "we accept
`node.domain == "com.microsoft"`" as a *code* decision. There is no such predicate and there must
never be one. The registry is keyed by `(domain, op_type)` and an unregistered contrib op declines
through exactly the same path as an unregistered `ai.onnx` op. **The test that enforces this**: a
graph containing a fabricated `com.microsoft::NotARealOp` node must be declined with the ordinary
`NoHandler` reason and must run correctly on CPU — Trinity, as an M-tier regression test from the
first contrib op onward.

#### C2 — Contrib ops have no opset, so version-gating is by ORT release plus a drift alarm

This is the substantive difference from `ai.onnx` and the constraint everything else hangs off.
`ai.onnx` gives us a monotonic opset number we can range-check in the claim predicate; a schema
change is *visible* as a number we did not accept. Contrib schemas carry no such number — they are
versioned by ORT *release*, and several of the eleven (`LinearAttention`, `CausalConvWithState`,
`QMoE`, `MoE`) are new and still moving. A contrib schema change therefore does not bump anything we
can test against: it silently changes what our claim predicate *should* accept, while our predicate
goes on accepting what it always did.

The protocol, and it is a hard precondition on the first contrib op landing — not on tier 3
generally, on **op #1 of the eleven**:

1. **The ORT version is pinned per release** and recorded in `Cargo.toml`, `docs/`, and the CI
   matrix. Trinity has pinned 1.28. A contrib claim predicate is only ever validated against a
   pinned version.
2. **Each of the eleven records the ORT version its predicate was written against**, in the registry
   entry, next to the predicate. Not in a comment in a design document — in the table, where it can
   be printed by `--dump-capabilities` and diffed.
3. **`tools/graph_census.py` runs in CI against pinned `.onnx` artifacts** and reports per-op claim
   rate. This is the drift alarm: when a schema shifts under us, the census claim rate for that op
   moves, and CI fails on the delta rather than on a numerical comparison months later.
4. **On an ORT version bump, the census delta is a review gate.** A bump that changes any contrib
   claim rate does not merge until the owning op's predicate has been re-read against the new
   schema. The failure we are buying insurance against is the quiet one: a new optional attribute
   appears, our predicate does not know to reject it, and we claim a node whose semantics changed.
5. **When a schema changes shape under us, the correct response is to narrow the predicate and
   decline, not to guess.** Declining is a performance regression that CI reports as a claim-rate
   drop; guessing is wrong logits that nothing reports at all. This ordering is not negotiable and
   it is the contrib-specific restatement of §1.3.
6. **A contrib row whose baseline is not a released ORT version may not be flipped from `Staged` to
   `Live`.** *Added 2026-07-28T22:28:08-07:00 — see below.*
7. **The fingerprints themselves are re-verified against the pinned schema, on a schedule — they are
   not trusted inputs.** *Added 2026-07-28T22:28:08-07:00.* Items 3 and 4 check *observed nodes
   against the fingerprints*. Nothing was checking *the fingerprints against the schema*, which
   means the baseline of the drift alarm was itself unaudited. Mouse's own arity self-audit proved
   this is not hypothetical: he found two errors in the `GroupQueryAttention` fingerprint —
   `min_inputs` recorded as 3 when the true minimum is 7 (optional contrib inputs are *positional*,
   so `seqlens_k` and `total_sequence_length` occupy slots 5 and 6), and a recorded note asserting a
   1.28-vs-main difference that did not exist. **Both errors were in the permissive direction, and
   that is the part that matters.** A drift detector whose baseline is too permissive does not fire
   late — it fails *silently*, in the one direction where silence is expensive: it will accept a
   node it should have rejected and report nothing, because from its point of view nothing drifted.
   A too-strict fingerprint, by contrast, shows up immediately as a claim-rate drop that someone
   investigates. The asymmetry means fingerprint errors cannot be left to be discovered by their
   consequences.

   Concretely: a CI job re-derives arity, attribute names and type constraints for all eleven rows
   from the pinned ORT release's schema — from the ORT Python API's registered schemas where it can,
   from the pinned `defs.cc` otherwise — and fails on any disagreement with the recorded
   fingerprint. It runs on the ORT version bump (item 4) *and* on a schedule, because a fingerprint
   can be wrong on the day it is written, which is exactly what happened here. Where the derivation
   cannot be automated for a row, that row carries an explicit `hand_verified` marker with a date,
   so "not automatically checked" is visible rather than indistinguishable from "checked and
   agreed". Owner: Mouse, with Fact Checker as second reader on any row a bump changes. This is a
   T3 precondition, not an M0 one — but it lands before the first contrib row goes `Live`.

**C2's implemented shape is confirmed** (Mouse, `registry.rs`; Tank, `sys.rs`; both in `5ae991a`).
`SchemaBaseline` is nested *inside* `ContribSchema` and surfaced through `OpSpec::schema_baseline()`,
with build-failing tests requiring a baseline on every `com.microsoft` row and forbidding one on
every `ai.onnx` row. This is better than the flat side table I would have accepted, and the reason is
worth stating as a general rule: **the nested placement makes it impossible to record a schema shape
without recording where the shape came from.** A parallel table can be half-filled; a nested field
cannot. Tank arrived at the same requirement from `sys.rs` within the hour, they reconciled as *sys
owns the type, registry owns the data*, and he deleted his side table — the right call, because two
places recording the same fact is a hazard best fixed by deleting one rather than by testing that
they agree. Ratified as the C2 mechanism of record. The asymmetry (default-domain rows print
`n/a (opset-versioned)`) is also right: their compatibility contract *is* the opset window, and a
baseline there would dilute the signal on the rows where it matters.

**On Tank's 18-line cross-owner edit to `registry.rs`: ratify now, Mouse reviews after.** The shape
is ratified on its merits and does not depend on the wording of one replaced test. Blocking a
ratification on a review of a test that replaced a now-uncompilable test would stall four other
people to protect a low-risk change that is already green and already covered from Tank's side in
`layering.rs` (every contrib row has a baseline, no default-domain row does, no baseline claims a
release newer than the one we compile against). Mouse should still review it — it is his file, he
should own the final wording, and Tank asked for exactly that — but as a follow-up, not a gate.
Tank yielding placement to Mouse's design and *saying why it was better* rather than just yielding
is the behaviour I want between owners; that is how a boundary gets stronger instead of just being
defended.

**The main-branch-only schemas — ruling.** Four of the eleven (`LinearAttention`,
`CausalConvWithState`, `QMoE`, `MoE`) carry `MAIN_BASELINE` — `main (post-1.28.0)` — because they do
not appear in the pinned release at all. Recording that honestly is exactly what C2 is for, and
Mouse recording it rather than writing `1.28.0` across the board is the constraint doing its job on
its first real test. Three rulings follow:

1. **The situation is contained today and must stay contained by mechanism, not by intention.** All
   four rows are `OpStatus::Staged`, so `claim_decision` declines them with a machine-readable
   `[staged]` reason and the graph runs on CPU. Nothing is claimed against an unreleased schema, so
   there is no correctness exposure right now. But "we will remember not to flip these" is not a
   control, so: **C2 item 6 — a contrib row whose baseline is not a released ORT version may not be
   flipped from `Staged` to `Live`.** Enforce it the same way C1 is enforced, as a test rather than
   a convention: a row that is `Live` with a non-release baseline fails the build. This is the same
   pattern as A2 — convert the warning into a gate, because a warning is what gets dropped under
   schedule pressure.
2. **The registry test asserting agreement on the verification *date* while not comparing release
   strings is correct**, and I want the reasoning recorded rather than left as a compromise.
   Comparing release strings would force `MAIN_BASELINE` to lie in order to pass. The date is the
   fact that actually matters — *when did a human last read this schema* — and it is the only field
   both baselines can honestly share.
3. **The milestone consequence, which is real: `LinearAttention` and `CausalConvWithState` gate
   T5a, the named Qwen3.5 target, and they are targeting a schema that has never shipped.** That is
   a genuine schedule risk and it is not Mouse's to absorb quietly, so it is now **OQ-16** in §11:
   the release these ops first appear in, and whether their schema changed between our fingerprint
   and that release, is a tracked question with an owner. Practically, this means the T5a kernels
   may be written twice, and the fingerprints will need re-verification against the release the
   moment it exists. The correct report upward if that slips is *not* that Qwen3.5 slipped for
   Vulkan reasons — it is that we are gated on an upstream schema stabilizing, which is a different
   risk with different mitigations.

#### C3 — Contrib declines use the ordinary machine-readable decline path, never a special case

Every decline — contrib or not — emits Mouse's machine-readable reason through the same mechanism
(`ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1`, and the profiling-JSON claim assertion Trinity landed). No
contrib-specific logging path, no contrib-specific decline enum, no `if domain == "com.microsoft"`
anywhere in the diagnostics. Two reasons: a bespoke path is one more thing that can be wrong in the
place we look when something is already wrong; and the whole value of a uniform decline vocabulary
is that a user debugging a Qwen graph and a user debugging a ResNet read the same output. Any new
decline reason a contrib op needs (`UnsupportedAttributeValue`, `UnverifiedSchemaVersion`) is added
to the shared enum for everyone.

**The claim log is live, and it is append-and-flush per decision, not written at teardown**
(Mouse, 2026-07-28). Ratified, and the reasoning is worth keeping because it generalizes: the
plugin-EP lifecycle gives us **no point at which we are reliably told the session ended**, so any
report-at-exit design is a flaky test waiting to happen — and a diagnostic that is unreliable
exactly when something went badly wrong is worse than none, because it will be trusted anyway. JSON
Lines appended per decision survives an abort, a crash, and a process that simply never tears down.

This also sharpens C1's runtime half. Trinity's assertion can now check `code == "not-registered"`
directly rather than inferring from a zero-node count — and a zero-node assertion cannot distinguish
"declined for the right reason" from "declined for the wrong reason" from "crashed before it ever
reached claiming". All three produce zero claimed nodes and only one of them is the property we
meant to test. **General rule: assert the reason, not the absence.** An absence is consistent with
too many different worlds.

#### C4 — Every one of the eleven gets a hand-written claim predicate

Not the shared `caps`-column helper. Their attribute surface — `num_heads`, `kv_num_heads`,
`do_rotary`, `rotary_interleaved`, `scale`, `softcap`, `bits`, `block_size`, `accuracy_level`,
`g_idx`, the `LinearAttention` recurrence rule — is exactly where a silently-accepted-but-
unimplemented variant produces wrong output rather than a decline. The generated table still owns
their dtype/capability gating; the *semantic* gate is written by hand and reviewed by hand.

#### C5 — Claim only the configuration Trinity has a real artifact for

No claiming a `LinearAttention` recurrence rule we have never seen emitted. Mouse's UNVERIFIED-row
discipline (`OP_COVERAGE.md` §2.1) is ratified and extended: **an UNVERIFIED contrib-op row may not
be claimed at all**, not merely kept off exit criteria. An op moves from UNVERIFIED to VERIFIED by
`graph_census.py` finding it in a pinned artifact, at which point its real attribute values are
known and the predicate can be written against them instead of against the schema docs.

#### C6 — CPU fallback is the safety net and it must stay intact

ORT's own CPU implementation of every one of the eleven is the reference and the fallback. **A contrib
op we claim but get subtly wrong is strictly worse than one we decline** — the decline costs a
transfer boundary and shows up in `largest_island_flops`; the wrong claim costs correctness and
shows up nowhere. Concretely: the eleven are claimed incrementally, one at a time, each landing with
its differential test before the next begins; and **numerical verification for contrib ops is
per-layer, not final-logits**. Comparing final tokens on a 1.7B model against ORT CPU will pass with
a broken kernel far more often than anyone expects — sampling hides a great deal. Trinity owns the
per-layer mechanism; this is a binding constraint on the coverage plan, not a suggestion.

#### C7 — The XL kernels are budgeted as XL work, not as coverage

Three of them (`GroupQueryAttention`, `MatMulNBits`, `LinearAttention`) have no kernel-template
leverage and each gates a separate tier. They must never be counted in an op-coverage number as if
they were eleven of the 174. §10.0 and §1.5 state the schedule consequence honestly.

### 1.5 Two different claims, deliberately kept apart

Ratified from `OP_COVERAGE.md` §11.1 because it is the single most important thing in that document
for expectation-setting:

- **"High op coverage" — the ~121-op elementwise/shape/indexing/reduction/GEMM surface — is a
  weeks-scale goal**, contingent on the kernel-template infrastructure being built *before* op #1.
- **"Qwen3.5 runs end-to-end on Vulkan" is a months-scale goal**, gated on three XL kernels and on
  M2's device allocator.

The op count will look excellent long before any LLM runs. That gap is precisely where a coverage
project deceives itself, which is why the metric of record is `largest_island_flops`, not op count
(§9.2).

---

## 2. How it plugs into ONNX Runtime

### 2.1 The plugin-EP model

ORT exposes a public C ABI for registering an out-of-tree EP as a shared library. The host
resolves two symbols by name:

```c
OrtStatus* CreateEpFactories(const char* registered_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories,
                             size_t max_factories,
                             size_t* num_factories);

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);
```

`rust/src/lib.rs` exports both. ORT is reached **only** through the `OrtApi` function-pointer
table handed to `CreateEpFactories`; we never link `libonnxruntime`. Ownership crosses the C
boundary with `Box::into_raw` / `Box::from_raw`. Every `extern "C"` entry point that runs real
logic is wrapped in a panic guard that converts a Rust panic into an `ORT_EP_FAIL` status —
unwinding into ORT's C++ is undefined behaviour and a plugin must never take down its host.

Usage from the application side:

```python
import onnxruntime as ort
import onnxruntime_ep_vulkan

onnxruntime_ep_vulkan.register_execution_provider_library()
sess = ort.InferenceSession(model, providers=["VulkanExecutionProvider", "CPUExecutionProvider"])
```

### 2.2 Object lifecycle

```
dlopen / LoadLibrary
   └─ CreateEpFactories ────────────► VulkanEpFactory      (process-lived, one per registration)
        ├─ GetName / GetVendor / GetVendorId / GetVersion
        ├─ GetSupportedDevices(OrtHardwareDevice[]) ──────► OrtEpDevice[]   (device enumeration)
        ├─ CreateAllocator(OrtMemoryInfo) ────────────────► OrtAllocator    (device memory)
        ├─ CreateDataTransfer() ──────────────────────────► OrtDataTransferImpl
        └─ CreateEp(devices, session_options, logger) ────► VulkanEp        (one per session)
                ├─ GetName
                ├─ GetDefaultMemoryDevice ────────────────► OrtMemoryDevice
                ├─ GetCapability(OrtGraph, OrtEpGraphSupportInfo)      ← node claiming
                ├─ Compile(OrtGraph[], OrtNode[], OrtNodeComputeInfo[]) ← plan build + prepack
                │     └─ OrtNodeComputeInfo { CreateState, Compute, ReleaseState }  ← inference
                └─ ReleaseNodeComputeInfos
   └─ ReleaseEpFactory
```

The `VulkanEpFactory` struct embeds `OrtEpFactory` as its **first field** under `#[repr(C)]`, so
the pointer ORT holds is pointer-identical to our Rust struct at offset 0. Same for `VulkanEp` and
`OrtEp`, and for the per-subgraph compute-info object and `OrtNodeComputeInfo`. This is the exact
pattern proven in `onnxruntime-mlx/rust/src/factory.rs`.

### 2.3 Device enumeration — where Vulkan differs from MLX

The MLX EP's `GetSupportedDevices` picks *the first* `OrtHardwareDeviceType_GPU` ORT presents and
advertises exactly one `OrtEpDevice`. Apple Silicon has one GPU; that is sufficient there.

Vulkan does not have that luxury. The factory must:

1. Create a `VkInstance` (once per plugin load) and enumerate physical devices.
2. For each physical device, evaluate the **capability gate** (§7): does it meet the required
   feature set? If not, it is not advertised — an unusable device must never be offered to ORT.
3. Correlate each usable `VkPhysicalDevice` with the `OrtHardwareDevice` entries ORT presents.
   The correlation key is vendor ID + device ID, both of which appear in
   `VkPhysicalDeviceProperties` and in ORT's hardware-device metadata. Where correlation fails
   (software rasterizers, virtualized GPUs, MoltenVK), we fall back to type matching (GPU, then
   CPU) and record which strategy was used in the EP device metadata.
4. Create one `OrtEpDevice` per usable device via `EpApi::CreateEpDevice`, attaching EP metadata
   (Vulkan API version, device name, driver version, vendor) and EP options.

Consequences that follow, and which the MLX design never had to answer:

- **Device selection is a user-visible session option.** Multi-GPU machines are normal on Windows
  and Linux. `ep.device_index` selects among advertised devices; the default is the
  highest-scoring device (discrete > integrated > virtual > CPU), matching `ENGINE.md` §2.2.
- **`VkInstance` lifetime is factory-scoped, `VkDevice` lifetime is EP-scoped.** Two sessions on
  the same physical device share the instance but get independent logical devices, queues,
  allocators, and pipeline caches. Sharing a `VkDevice` across sessions is a post-v1
  optimization with real thread-safety cost; we do not take it now.
- **Enumeration must never abort the host.** A machine with no Vulkan loader, no ICD, or a broken
  driver must produce zero advertised devices and a warning — not a crash and not an error status
  that fails session creation. This is a tested requirement (Trinity, M0).

### 2.4 Session options

Prefixed `ep.` and read in `CreateEp` from the `OrtSessionOptions`. The v1 set is small on
purpose:

| Option | Type | Default | Meaning |
|---|---|---|---|
| `ep.device_index` | int | auto | Which advertised Vulkan device to bind. |
| `ep.enable_validation` | bool | `false` (release), `true` (debug) | Enable `VK_LAYER_KHRONOS_validation`. |
| `ep.pipeline_cache_path` | string | platform cache dir | On-disk `VkPipelineCache` blob location. |
| `ep.max_claim_ops` | string list | unset | Restrict claiming to a named op set. Debugging and bisecting only. |
| `ep.disable_device_memory` | bool | `false` | Force the M0 host-memory I/O path (see §6.3). Escape hatch for driver bugs. |
| `ep.force_legacy_barriers` | bool | `false` | Force the legacy `vkCmdPipelineBarrier` backend on a device that supports `synchronization2` (§7.5). Exists so CI exercises both barrier backends on the same hardware; also an escape hatch for a broken sync2 driver. |

Environment variables mirror the MLX EP's convention for observability, and are *not* a
configuration surface: `ONNXRUNTIME_EP_VULKAN_VERBOSE`, `ONNXRUNTIME_EP_VULKAN_TRACE=<path>`,
`ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG`, `RUST_LOG=onnxruntime_ep_vulkan=<level>`.

### 2.5 Node claiming (`GetCapability`) and subgraph compilation

**Claim.** For each node in the `OrtGraph`, `ep.rs` builds a `NodeView` (a read-only FFI wrapper
over `OrtNode` exposing op type, domain, since-version, input/output slot info, and attributes)
and asks the registry a single question: `claimable(&NodeView)`. There is **no per-op logic in
`ep.rs`**. This is the invariant that makes "claimed" and "translatable" impossible to
desynchronize, and it is inherited directly from the MLX reference.

**Fuse.** Claimed nodes are grouped into **maximal convex connected clusters** — the union-find +
reachability-bitset algorithm from the MLX EP. Convexity is not optional: a non-convex fusion
creates a cycle in the partitioned graph and ORT rejects it. Each cluster is handed to
`EpGraphSupportInfo_AddNodesToFuse`.

**Compile.** ORT calls `Compile` with one fused `OrtGraph` per cluster. For each we:
1. Extract a `NodeDesc` per node — op type, domain, since-version, generically-copied attributes
   (ints / floats / int arrays / float arrays / strings / tensors), and input/output tensor refs.
2. Build a `Plan`: the topologically ordered `NodeDesc` list plus the I/O binding table.
3. **Prepack**: read every constant initializer, convert it to the layout the shader wants, and
   upload it into a device-local buffer owned by the plan. This happens **once**. (§5.4)
4. Create or fetch the `VkPipeline` for every node in the plan, so the first inference does not
   pay shader compilation.
5. Hand ORT an `OrtNodeComputeInfo` that owns the plan.

**Run.** `Compute` binds ORT's input tensors, records (or replays) a command buffer, submits it,
waits on a fence, and makes the outputs visible to ORT.

### 2.6 CPU fallback

Unclaimed nodes are assigned to ORT's CPU EP by the ORT partitioner. ORT inserts the required
memcpy nodes at partition boundaries, using the `OrtDataTransferImpl` our factory supplies
(§6.2). Nothing about fallback is our code path — which is precisely why it is trustworthy.

The cost model, which the op-coverage strategy in §8 is built around: **one unclaimed node in the
middle of a graph splits it into two islands with a device round-trip between them.** Claim rate
is a bad metric; fused-region compute volume is the good one. The MLX EP learned this the
expensive way and we inherit the lesson rather than repeating it.

---

## 3. Repository and crate layout

Mapped one-to-one against `onnxruntime-mlx`. New paths carry a ✨.

```text
onnxruntime-ep-vulkan/
├── README.md                          # what it is, how to build, how to run
├── LICENSE
├── docs/
│   ├── DESIGN.md                      # ← this file: architecture of record
│   ├── ENGINE.md                      # ✨ Switch: Vulkan runtime, memory, shaders, sync
│   ├── PLATFORMS.md                   # ✨ Link: platform/driver matrix, toolchains, CI lanes
│   ├── OP_ARCHITECTURE.md             # Mouse: op registry + authoritative coverage table
│   └── BENCHMARKS.md                  # ✨ Niobe: methodology + published baselines
├── rust/
│   ├── Cargo.toml                     # cdylib crate, lib name onnxruntime_vulkan_ep
│   ├── build.rs                       # bindgen(ORT C ABI) + GLSL→SPIR-V compile+embed
│   ├── README.md                      # crate-level notes for contributors
│   ├── shaders/                       # ✨ GLSL compute sources (Switch)
│   │   ├── include/                   #    shared GLSL headers (indexing, broadcast, dtype)
│   │   ├── elementwise_binary.comp
│   │   ├── elementwise_unary.comp
│   │   └── ...
│   └── src/
│       ├── lib.rs                     # CreateEpFactories / ReleaseEpFactory, panic guards
│       ├── factory.rs                 # OrtEpFactory vtable: devices, allocator, data transfer
│       ├── ep.rs                      # OrtEp vtable: GetCapability, clustering, Compile, Compute
│       ├── engine.rs                  # Plan, NodeDesc, DispatchContext — the op-facing API
│       ├── recorded.rs                # ✨ command-buffer recording cache (shape-keyed replay)
│       ├── registry.rs                # op registry + NodeView / GraphView + claim helpers
│       ├── allocator.rs               # ✨ OrtAllocator over device memory
│       ├── transfer.rs                # ✨ OrtDataTransferImpl: host↔device staging copies
│       ├── vk/                        # ✨ the Vulkan layer — Switch owns, nothing else enters
│       │   ├── mod.rs                 #    re-exports the safe surface only
│       │   ├── instance.rs            #    VkInstance, layers, debug messenger
│       │   ├── device.rs              #    physical-device scoring, VkDevice, queues
│       │   ├── caps.rs                #    Capabilities struct: the single capability oracle
│       │   ├── memory.rs              #    gpu-allocator integration, DeviceBuffer, StagingPool
│       │   ├── pipeline.rs            #    VkPipeline creation, VkPipelineCache, spec constants
│       │   ├── descriptor.rs          #    descriptor set layouts and pools
│       │   ├── command.rs             #    command pool/buffer recording, barriers, submission
│       │   └── shaders.rs             #    embedded SPIR-V module table + variant selection
│       ├── ops/                       # per-family ONNX handlers + claim predicates (Mouse)
│       │   ├── mod.rs
│       │   ├── elementwise.rs
│       │   ├── math.rs
│       │   ├── reduction.rs
│       │   ├── shape.rs
│       │   ├── matmul.rs
│       │   └── norm.rs
│       ├── sys.rs                     # raw bindgen output for the ORT plugin-EP C ABI
│       ├── logging.rs                 # in-crate `log` subscriber, env-gated, silent by default
│       └── trace.rs                   # env-gated Chrome/Perfetto tracer + GPU timestamp queries
├── rust/modelrunner/                  # ✨ Tank: Rust-native real-model validation (§9.3)
│   ├── Cargo.toml                     #    workspace member, NOT a default member; host-only
│   └── src/
│       ├── main.rs                    #    the `--check-model-agreement` CLI
│       ├── run.rs                     #    the six guards and the evidence document
│       ├── ortapi.rs                  #    RAII wrappers over the ORT C API (the unsafe surface)
│       ├── ortlib.rs                  #    dlopen/LoadLibrary discovery + API-version gate
│       ├── provenance.rs              #    SHA-256 + size pin against model_provenance.json
│       ├── foundry.rs                 #    Foundry Local cache resolution (exactly-one rule)
│       ├── feeds.rs                   #    deterministic input generation, free-dim pinning
│       ├── compare.rs                 #    the per-model tolerance policy and the comparator
│       ├── evidence.rs                #    guards, counters snapshot, artifact writing
│       ├── sha256.rs                  #    in-tree SHA-256 (no crate: PyPI-free means dep-free)
│       └── json.rs                    #    in-tree JSON reader/writer
├── tests/
│   ├── README.md
│   ├── ops/                           # pytest op-correctness: Vulkan EP vs ORT CPU EP
│   │   ├── conftest.py                #    registers the plugin from ONNXRUNTIME_VULKAN_EP_LIB
│   │   ├── _models.py                 #    ONNX IR model builders, shared with bench/
│   │   └── test_*.py
│   ├── backend/                       # ONNX backend node tests through the EP
│   └── conformance/                   # opt-in broader conformance (onnx-tests harness)
│       ├── README.md
│       ├── RESULTS.md
│       ├── claimed_ops.txt
│       └── run_conformance.sh
├── bench/
│   ├── README.md
│   ├── bench.py                       # per-op-family timings, Vulkan vs CPU
│   ├── cases.py
│   └── compare.py                     # base-vs-PR regression table for CI comment
├── python/
│   ├── README.md
│   ├── pyproject.toml
│   ├── hatch_build.py                 # builds the cargo cdylib into the wheel
│   └── src/onnxruntime_ep_vulkan/
│       ├── __init__.py                # register_execution_provider_library(), EP_NAME, paths
│       └── py.typed
└── .github/workflows/
    ├── ci.yml                         # fmt, clippy, build matrix, op tests on lavapipe (Linux + Windows)
    ├── conformance.yml                # opt-in workflow_dispatch
    ├── bench.yml                      # informational perf comment on PRs
    └── publish.yml                    # wheel build + release
```

### 3.1 Naming decisions

- **EP name `VulkanExecutionProvider`.** Matches ORT's naming convention for every other EP and
  is what a user will guess. Frozen — changing it later breaks every user's provider list.
- **Library base name `onnxruntime_vulkan_ep`,** pinned via `[lib] name` so it is stable
  regardless of the crate name, exactly as the MLX EP does. Python, tests, CI, and any downstream
  runtime load it by that exact filename.
- **Version `0.<ORT_API_VERSION>.<patch>`.** A plugin EP is bound to one plugin-EP C-ABI version,
  so the version must state which ORT it works with. `0.27.0` pairs with ORT 1.27.x. When ORT
  ships API version 28, we move to `0.28.0`. The EP reports this to ORT from
  `env!("CARGO_PKG_VERSION")` so it can never drift from the manifest.
- **Vendor ID.** Unlike the MLX EP (which reports Apple's `0x106B`), there is no single hardware
  vendor here. The factory reports the **Vulkan `vendorID` of the bound physical device** from
  `VkPhysicalDeviceProperties`, so a user querying EP devices sees NVIDIA/AMD/Intel/Qualcomm/ARM
  correctly. Open question OQ-6 covers the no-device case.

---

## 4. Module responsibilities and boundaries

### 4.1 Layer map

```
┌──────────────────────────────────────────────────────────────────────────┐
│ L0  ORT C ABI boundary        lib.rs · factory.rs · ep.rs · allocator.rs │
│                               transfer.rs · sys.rs                       │
│     Owns: OrtEpFactory/OrtEp/OrtNodeComputeInfo/OrtAllocator vtables,     │
│           Box::into_raw ownership, panic guards, OrtStatus construction.  │
│     Owner: Tank                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│ L1  Plan & dispatch           engine.rs · recorded.rs · registry.rs      │
│     Owns: NodeDesc, Plan, DispatchContext, the op registry, NodeView,     │
│           claim dispatch, command-buffer recording cache.                 │
│     Owner: Morpheus (contract) · Tank (plumbing) · Mouse (registry)      │
├──────────────────────────────────────────────────────────────────────────┤
│ L2  ONNX op semantics         ops/*.rs                                   │
│     Owns: per-op claim predicates and translate handlers. Reads           │
│           attributes, validates dtypes/shapes, requests dispatches.       │
│     Owner: Mouse                                                         │
├──────────────────────────────────────────────────────────────────────────┤
│ L3  Vulkan engine             vk/*.rs · shaders/*.comp                   │
│     Owns: VkInstance/VkDevice/VkQueue, allocator, staging, descriptors,   │
│           pipelines, barriers, submission, SPIR-V modules.                │
│     Owner: Switch                                                        │
├──────────────────────────────────────────────────────────────────────────┤
│ L4  Raw bindings              ash · gpu-allocator · bindgen(ORT)         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 The two hard rules

**Rule 1 — The ORT C ABI never leaks into op code.**
No type from `sys::ort` may appear in a signature, field, or local in `rust/src/ops/`. Op handlers
see `NodeDesc`, `NodeView`, `TensorRef`, and `DispatchContext`. They never see `OrtNode`,
`OrtValue`, `OrtKernelContext`, `OrtStatus`, or an `OrtApi` function pointer. The exception that
proves the rule: `NodeView` and `NodeDesc` are *where* the ABI is translated into safe Rust, and
they live in `registry.rs` / `engine.rs`, not in `ops/`.

*Enforcement:* `sys` is `pub(crate)` but a CI lint (`ci.yml`) greps `rust/src/ops/` for `sys::`,
`Ort`, and `unsafe` and fails the build on a hit. A rule that is not mechanically checked is a
suggestion.

**Rule 2 — Op code never touches raw Vulkan handles.**
No `ash::vk::*` type may appear in `rust/src/ops/`. Op handlers cannot hold a `vk::CommandBuffer`,
call `vkCmdDispatch`, allocate memory, or create a pipeline. They express intent through
`DispatchContext`:

```rust
// The entire vocabulary an op handler has. Illustrative signature — the real one lands with M0.
pub trait DispatchContext {
    fn resolve(&mut self, r: &TensorRef) -> Result<BufferView, EpError>;
    fn bind_output(&mut self, o: &OutRef, desc: TensorDesc) -> Result<BufferView, EpError>;
    fn alloc_temp(&mut self, desc: TensorDesc) -> Result<BufferView, EpError>;
    fn dispatch(&mut self, k: KernelRequest) -> Result<(), EpError>;
    fn read_const_i64(&self, r: &TensorRef) -> Option<Vec<i64>>;
}
```

`BufferView` is an opaque handle. `KernelRequest` names a shader variant, its specialization
constants, its push-constant payload, its bindings, and a workgroup count. The engine decides
descriptor sets, barriers, pipeline selection, and submission. `ENGINE.md` §1 states the same
boundary from the engine side and enforces it with module privacy: the wrapper types in `vk/` are
not `pub` outside the engine.

*Enforcement:* the same CI lint greps `rust/src/ops/` for `ash`, `vk::`, and `unsafe`. It also
enforces the barrier seam of §7.5: `cmd_pipeline_barrier`, `cmd_pipeline_barrier2`,
`BufferMemoryBarrier`, `DependencyInfo`, `PipelineStageFlags*` and `AccessFlags*` may appear
**only** in `rust/src/vk/barrier.rs`, and `Capabilities::synchronization2` may be read **only** in
`rust/src/vk/barrier.rs` and `rust/src/vk/caps.rs`.

**Why this matters enough to reject a working PR over.** The MLX EP got a mature backend that
handled memory, scheduling, and dtypes. We do not. Every op we add is a shader, a descriptor
layout, a barrier, and a workgroup calculation. If those details are allowed to bleed into 60 op
modules, the first driver quirk Link finds becomes a 60-file change instead of a 1-file change.
The boundary is the only thing that keeps op coverage a linear cost.

### 4.3 What each module may depend on

| Module | May use | May **not** use |
|---|---|---|
| `lib.rs`, `factory.rs`, `ep.rs`, `allocator.rs`, `transfer.rs` | `sys::ort`, `engine`, `registry`, `vk` (safe surface) | `ash` directly, `ops::*` internals |
| `engine.rs`, `recorded.rs` | `vk` safe surface, `registry` | `sys::ort` FFI calls outside `NodeDesc` construction |
| `registry.rs` | `sys::ort` (for `NodeView` only), `engine` types | `ash`, `vk` |
| `ops/*.rs` | `engine::{DispatchContext, NodeDesc, TensorRef}`, `registry` helpers | `sys`, `ash`, `vk`, `unsafe` |
| `vk/*.rs` | `ash`, `gpu-allocator` | `sys::ort`, `ops` |

---

## 5. Execution flow, end to end

### 5.1 Library load

`RegisterExecutionProviderLibrary` → `dlopen`/`LoadLibrary` → `CreateEpFactories`.
Initialise logging, negotiate `ORT_API_VERSION 27` (fail with a clear status if the host is
older), construct one `VulkanEpFactory`. **No Vulkan work happens yet** — a plugin must be cheap
to load even on a machine that will never use it.

### 5.2 Device enumeration

ORT calls `GetSupportedDevices`. *Now* we create the `VkInstance`, enumerate physical devices,
apply the capability gate (§7), score and sort, correlate with ORT's `OrtHardwareDevice` list,
and create one `OrtEpDevice` per usable device. Zero usable devices → advertise none, log a
warning, return success. The instance is kept alive on the factory.

### 5.3 Session creation

ORT calls `CreateEp`. We read session options, select the physical device, create the `VkDevice`,
compute queue, `gpu-allocator` arena, command pool, descriptor pools, staging pool, and load the
on-disk `VkPipelineCache`. The `VulkanEp` owns all of it and drops it in `ReleaseEp`. RAII, not
manual teardown — this is where the MLX rewrite found a real per-session leak that three lines of
`impl Drop` fixed, and we take the same posture from day one.

`GetDefaultMemoryDevice` returns our device's `OrtMemoryDevice` (M2+) or null (M0/M1, §6.3).

### 5.4 `GetCapability` — claiming

Per node: build `NodeView` → `registry::claim_decision(&view)`. Rejections carry a *reason string*
and are aggregated per op type; with `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` or tracing on, the EP
prints exactly which ops were declined, how many, and why. This is the single most valuable
diagnostic the MLX EP has, and it is cheap. It ships in M0.

Claimed nodes → convex clustering → **minimum-viable-subgraph filter** →
`EpGraphSupportInfo_AddNodesToFuse` per surviving cluster.

**Where `ops::partition::retain_viable` is invoked — the answer to Mouse's question.** Exactly one
call site, in `ep.rs`'s `GetCapability`, positioned as the **third of four stages** and never
anywhere else:

```
GetCapability(OrtGraph) →
  1. per-node claim      registry::claim_decision(&NodeView)      → claimed / declined + reason
  2. cluster             maximal convex connected components of the claimed set → Vec<Island>
  3. FILTER              ops::partition::retain_viable(&islands, &model, &policy)
                                                                   → (kept, dropped + RejectReason)
                         stage 3 is TWO gates and an exemption, in this order:
                           3a. size       nodes < min_nodes AND anchors == 0  → TooSmall
                           3b. EXEMPTION  anchors > 0 (shipping default on)   → Claim, and RETURN
                           3c. economics  compute_ns < margin × transfer_ns   → TransferDominated
  4. report              AddNodesToFuse(kept) ; declines for step 1 and step 3 both go to the
                         one decline vocabulary and the one CLAIM_DEBUG dump
```

Four properties of that placement are binding, not incidental:

1. **After clustering, never before.** The rule is about an *island*, not a node — whether the work
   inside pays for the transfers at its edges. There is no meaningful per-node form of it, and a
   per-node approximation would re-introduce exactly the shredding it exists to prevent.
2. **Before `AddNodesToFuse`, so a rejected island is never handed to ORT at all.** Filtering after
   the fact would mean un-claiming, which the ORT surface does not offer.
3. **A step-3 rejection is a decline like any other**, carrying `DeclineCode::Partition` with the
   modelled numbers (`~Xns compute against ~Yns transfer, below the Zx margin`) into the same
   `CLAIM_DEBUG` output as a step-1 decline. This is §8.4 A3's requirement, and it is why the
   dropped set is returned rather than discarded: a silently-declined region is otherwise
   indistinguishable at the console from a missing op, and we would spend real time mis-diagnosing
   one as the other.
4. **`retain_viable` stays pure — data in, verdict out.** It does not touch the graph API or the
   device. `ep.rs` builds `Island` values from the cluster and the `TransferModel` from the device
   calibration, calls the function, and translates the verdicts back into ORT calls. This is the
   §4.2 boundary rule applied to partitioning: policy in `ops/`, ABI in `ep.rs`.
5. **Stage 3b returns before 3c reads any island property.** *Added 2026-08-01, see §5.4.1.* On an
   anchor-bearing island in the shipping configuration, stage 3 is a **constant function**: 3a is
   skipped by its own `anchors == 0` conjunct, 3b returns `Claim`, and no byte count, FLOP count,
   margin or `fixed_ns` is ever read. This is by design — 3c exists to kill *anchor-free* elementwise
   scatter — but it means **stage 3's verdict on our production graph is not a decision.** Anything
   that reads `viable_islands_retained == 1` as evidence that the economics model endorsed the
   island is reading a field that is silent on that question.

#### 5.4.1 The anchor exemption is the deciding term, and it is not an exemption from a working model — RULING (2026-08-01)

**The finding, as reported and then corrected against the source.** The coordinator predicted that
Phi-3.5's 355-node island is retained by `net_benefit_sole_island_overrides`. It is not: the
shipping configuration reads `viable_islands_retained: 1`, `net_benefit_sole_island_overrides: 0`.
Mouse moved one input and held the rest fixed — **anchor exemption off → retained 0, overrides 1,
reason `TRANSFER_DOMINATED`, at every `fixed_ns` across §7.12's swept range.** The exemption is the
deciding term.

**Two corrections to the framing, both in the direction that makes it more precise.**

**(i) In the shipping configuration the economics arithmetic on our graph is not merely
outvoted — it is not evaluated.** `evaluate` at `rust/src/ops/partition.rs:475` is an early return:
`if island.anchors > 0 && policy.anchor_exemption { return Verdict::Claim; }`, placed *above*
`transfer_ns` and `compute_ns`. So it is not that the economics arm decided wrongly and lost; it is
that on any anchor-bearing island **no property of the island can change stage 3's answer.** The
coordinator's "the economics arithmetic is genuinely reached" is true of the *counterfactual* run
and of anchor-free islands; it is not true of the graph we ship. And Mouse's correction — that the
predicate does return false on real input, the census lane's one-node chain rejecting at shipping
defaults — is a fact about **3a**, `TOO_SMALL`. It establishes that *stage 3* is not a decoration.
It does not establish that **3c** has ever decided anything outside a probe, and I can find no
artifact in which it has.

**(ii) The diagnosis inverts once you read `is_anchor`.** The anchor set is `MatMul`, `Gemm`,
`Conv`, `ConvTranspose`, `Attention`, `MatMulNBits`, `GroupQueryAttention`, `MultiHeadAttention`,
`QMoE`, `LinearAttention` (`partition.rs:308`). **Every non-trivial island of any transformer
contains at least one.** So "the economics model does not decide our partition" is not a Phi-3.5
accident and is not a discovered defect — **it is the design, working.** 3c was written to kill
cheap elementwise scatter whose tensors outweigh its arithmetic, and Phi-3.5's fused island is not
that. The doc comment at `partition.rs:458` says so in as many words.

**So what is actually wrong is narrower, and I want it stated without inflation.** Three things:

1. **The exemption is load-bearing for a question it was not designed to answer.** Its stated
   warrant is *an anchor is by definition heavy enough to justify a boundary on its own.* That is a
   claim about anchors on LLM-sized weights. It is asserted, not measured, and it is now the sole
   term deciding every production partition we ship on both devices. **Falsifier, and it is a
   future exposure rather than a present one: an anchor-bearing island that genuinely should be
   declined** — one small `MatMul` surrounded by large boundary traffic, which is a small-model or
   edge-shape graph, not ours. Under 3b we claim it unconditionally and 3c never sees it. That is a
   cross-model generality risk and it is where I expect this to bite first.
2. **The exemption's silence set includes "the byte estimator is broken."** §7.12.1's 104,116×
   estimator-versus-measured discrepancy (89,199,100,032 B estimated against 856,720 B measured) is
   the reason 3c declines the graph we ship when it is allowed to answer. The exemption is doing two
   jobs: the intended one, and concealing that from every production run. **R9's silence-set rule
   applies to a policy term and not only to an instrument** — the exemption is a mechanism whose
   correct operation makes a known-broken model unobservable. That is worth writing down as the
   general form.
3. **`Verdict::Claim` is three findings wearing one name, and that is R11 at the value level.**
   `Claim` returned from 3b, from 3c passing, and from `GateOutcome::SoleIslandOverride` are
   different facts about a run, and the counters record the verdict rather than the arm — which is
   why "the anchor exemption decided this" is an inference from two runs rather than a field in an
   artifact. R11's remedy is name–content agreement, and Mouse has it right: **`Verdict::Claim`
   carries its reason.** I endorse his refusal to re-derive the arm at the `ep.rs` call site — that
   is a second copy of the predicate and it is RAI-011 reappearing inside the fix for its own
   sibling. **Owner: Mouse, in `partition.rs`, once the worktree is clear.**

**What §7.12 owes, and it is less than the coordinator feared.** §7.12.1 already states this, in
these words: *"the only reason the model is claimed at all is the anchor exemption. Remove the
exemption and the EP declines the whole graph, at every `fixed_ns` I swept."* The section is not
silent and does not need to be rewritten. **The defect is placement and propagation, not omission**
— the sentence sits at the bottom of a subsection about calibrating a parameter that has been shown
not to matter, and it did not propagate to the two places a reader forms a belief about the
partition: §5.4's stage list, which described stage 3 as a filter and never named the exemption
(**fixed above**), and the M1 optimisation ordering's rank 2 (**re-qualified below**). The reader
the coordinator is worried about is misled by **this** document, not by `OP_COVERAGE.md`. §7.12
gets a pointer to §5.4.1 and nothing else; `OP_COVERAGE.md` is Mouse's file and his account of his
own finding is accurate.

**What must not be done, stated because two of the three are tempting.**

- **Do not remove the exemption "so the model decides."** The model is known wrong by five orders of
  magnitude on the only island we have measured. Adopting its verdict would decline the entire
  graph and lose M0. *Deferring to a model you have measured to be wrong is not rigour.*
- **Do not fix the estimator in the commit that makes the partitioner observable.** Mouse declined
  this and he is right; the reason is exactly the one he gave — you lose the ability to attribute the
  behaviour change. It fails safe, towards the CPU.
- **Do not soften the exemption's warrant into "it works, so it is justified."** It works because
  every island we have ever partitioned is anchor-bearing and large. That is a fact about our one
  model, and the constraint on this project is that generality is checked continuously.

**And the drafting rule gets a second live example.** *For every criterion, ask what the cheapest
thing is that satisfies the words without satisfying the intent.* RAI-011's criterion is **the gate
is always evaluated, with no branch in front of it**. The cheapest thing that satisfies those words
is **an unconditional early return inside the gate**: the branch moves from the call site to line
one of the body, `net_benefit_gate` reads `EVALUATED`, `net_benefit_gate_bypasses` stays `0`
forever, and every word of the criterion is true. I am **not** saying that is what happened — 3b is
older than RAI-011 and is legitimate policy in the right module. I am saying **RAI-011's observables
cannot tell the two apart**, which is why item 3 above is required rather than nice to have. This
sits alongside the variance gate — *the cheapest way to pass a steadiness test is to run at a stable
wrong clock* — as the second standing example of the rule catching something real.

**THE DRAFTING RULE'S COMPANION, AND IT IS ABOUT PROHIBITIONS RATHER THAN CRITERIA — added
2026-08-02T21:24:34-07:00, on Link's finding, which is the sharpest governance observation anyone has
made this week.** I had instructed the team **not to make a red lane green by narrowing it**. Link then
found that fixing an unrelated `clippy` failure would have turned the op-correctness lane green *having
asserted nothing* — the lane reports `2 passed, 36 skipped` because `conftest` reports the EP's own
decision as *"No Vulkan device available"* on a box whose gate had just passed. His sentence:

> **the narrowing you forbade, reached without anyone narrowing anything.**

> **A prohibition on an act is blind to the state that act would have produced, when that state is
> already the default. "Do not narrow the lane" is satisfied perfectly by a lane that was already
> narrow.** State prohibitions as **invariants over states, with a count**, never as bans on actions:
> not *do not narrow the lane*, but **the lane asserts N things and publishes N, and a run reporting
> more skips than assertions is not green.**

This is the same defect as everything else ruled this session wearing its governance costume — **a
reading that does not move when its subject is wrong** — and it is the reason it matters that I write
criteria rather than instructions. An instruction constrains the next edit. **An invariant constrains
the state, including the state nobody edited into place.**

**What survives untouched.** 355 of 363 nodes in one fused island, `MATCH` on both devices, is a
**count and a verdict**, both observed. §10.0.4's invariance preference applies unchanged: the
result does not depend on why the island was retained. **What is withdrawn is the attribution, not
the result** — no claim that the partitioner's economics model justifies our island has ever been
earned, and where this document implied one it now says so.

##### 5.4.1(a) The estimator's first half is fixed, and the concurrence is worth less than it looks — but a bound is worth more (2026-08-01T22:25:29-07:00)

**Verified on `squad/mouse` before ruling**, at `rust/src/ops/partition.rs` and `ep.rs`: the
estimator counted **every** claimed node's outputs as boundary, including edges wholly inside the
island. `ep.rs` now consults a whole-graph per-value consumer map and charges an output only when a
node **outside** the island reads it, or nothing reads it. Re-estimated on the same model and
device: **89,199,100,032 B → 13,936,509,056 B**, a 6.4× overcharge removed — and with the exemption
off the gate now **claims** Phi-3.5's island on its own economics,
`net_benefit_sole_island_overrides` 1 → 0. My own line citation has moved with the commit: the
early return is at 536 now, and see the citation ruling at the end of this subsection.

**Does this change §5.4.1's finding #2 — that the exemption's silence set includes "the byte
estimator is broken"? It changes it materially, in one direction only, and not by the argument
offered.**

**The concurrence argument, and why I decline it in the form it arrived.** *The economics arm now
agrees with the exemption on a real model, so the exemption is concurring rather than masking.*
Two things are wrong with taking that as corroboration. First, the remaining defect is
untouched — `slot_bytes` still substitutes **128 for every unknown dim**, every boundary tensor in
Phi-3.5 is runtime-extent, and the residual is **~16,268×**. Second and more important, **agreement
between two things fed the same fabricated input is not a second opinion.** That is the shape this
register was built around: *an identity whose two sides come from the same source is a falsifier
that cannot fire.* The exemption and the economics arm are not independent witnesses to "this island
is worth claiming"; one of them is reading an invented number. **A verdict flipping from
`TRANSFER_DOMINATED` to `Claim` because its input moved 6.4× while remaining 16,268× wrong is a
verdict that moved for a reason unrelated to the truth of the proposition.**

**What the same fact does support, and this is stronger than the concurrence and I want it used
instead.** `transfer_ns` is **monotone increasing in bytes**. The gate claims when fed
13,936,509,056 B. The instrumented boundary for that same run is 856,720 B, which is **smaller**.
Therefore the gate claims *a fortiori* on the true bytes: **the claim verdict survives a 16,268×
adversarial inflation of the term that opposes it.** That is not an estimate and not an agreement —
it is a **bound**, obtained by monotonicity from a number we do not trust, used in the only
direction where not trusting it is safe. It is §10.0.4's invariance preference arriving in a third
form: after *prefer the count* and *prefer the ratio*, **prefer the bound you can sign.**

> **The licence, and its limits, because this is exactly the kind of argument that has failed here
> when it favoured us.** A modelled quantity known to be wrong may be quoted as a **bound** only
> when (a) the model is **monotone** in the perturbed input, (b) the perturbation's **sign is
> established for that window by an independent measurement**, and (c) the bound is used in the
> direction the sign licenses and no other. **Absent (b) it is not a bound, it is a guess with a
> confident tone.** This is the device-clock companion obligation (§10.0, obligation 8) applied to a
> modelled quantity rather than a measured one, and the same sentence disposes of both: *a figure is
> quotable with the record that fixes which world it came from, and not otherwise.*

**And (b) is precisely what is not general here.** The 128 substitution's sign is **not known a
priori** — Mouse says so himself, and he is right: a larger substituted dimension inflates the
estimate and pushes towards rejection, a smaller one towards claiming. Phi-3.5's measured
configuration happens to be one where 128 over-counts. **A long-prefill `sequence_length` above 128
under-counts, the sign flips, and the bound above evaporates — it does not merely weaken.**
`Island::symbolic_boundary_slots` and `boundary_is_fabricated()` are the right instrument for this
and they are correctly built to *report rather than judge*. **Named falsifier, and it belongs beside
§5.4.1's first one: a configuration whose real extents exceed 128 on a boundary tensor.** Until the
sign is established per run, the bound argument is licensed for the measured window and nowhere else.

**So the silence set shrinks and does not empty.** What has left it: *the exemption is masking a
verdict that would be different on this island's true bytes.* It is not — we can now sign that. What
remains in it: *the estimator fabricates its input, the fabrication's sign is unestablished, and no
production run's partition is sensitive to any of it.* Finding #2 stands, narrowed to that.

**On `MEASURED_PHI35_DEV0` — RULING: rename, and disclosure is not sufficient.** The constant is
named *MEASURED*, holds an **estimate** now known wrong by 6.4×, and sits beside
`MEASURED_PHI35_DEV0_REAL_BYTES`, which holds the actual measurement. That is R11's name–content
obligation with no interpretation required, and the disagreement is at the top of the scale, not near
50%. Mouse's doc comment is exemplary — it discloses the estimate/measurement split, the residual,
and that parking the whole total in `output_bytes` charges one `fixed_ns` instead of two and
therefore **biases every test towards claiming, the direction that makes his conclusions harder to
reach**. Disclosing a bias that works against you, unprompted, is the standard. **It is still not
enough, for the reason the coordinator gives and I am adopting as the register's sentence: names
outlive doc comments.** The register already holds the general form — *a caveat that lives in a
different artifact from the number it qualifies is not attached to it* — and a doc comment is a
different artifact from a symbol at every call site. His own new constant already follows the right
convention (`ESTIMATED_…`), so the fix is to make the old one match it, not to invent one.

- **`MEASURED_PHI35_DEV0` → `ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED`**, keeping the doc comment
  verbatim.
- **The test asserting `TRANSFER_DOMINATED` renames to say which estimate it is about.** It took the
  coordinator three steps to establish that the green test and the shipping behaviour are not in
  conflict, and *a reader who stopped at the test name would conclude the opposite of what ships*.
  A test name is a claim; this one currently makes a false one.
- **Not a criticism, and not a new obligation.** Keeping the old constant beside the new one rather
  than overwriting it is the correct move and is why the history is legible at all. Only the name is
  wrong. **Owner: Mouse, with the `Verdict::Claim` reason field, same change.**

**On line-number citations — a convention for this document, arrived at from today's other
ruling.** My `partition.rs:475` no longer resolves; the early return is at 536. **A line number is a
reference that decays without failing** — it does not error, it silently points at something else,
which is `'<absent>'`'s defect in a different costume (§10.0.1 R13 amendment 1). **This document
cites a symbol — `partition.rs::evaluate`'s anchor-exemption early return — and a line number only
as a convenience beside it, never alone.** Existing citations are not swept; new ones follow this,
and any citation found stale is converted rather than repaired.

**Unchanged by all of the above: 355 of 363 nodes in one fused island, `MATCH` on both devices.**
A count and a verdict, both observed, and neither depends on which term retained the island.

The `TransferModel` handed to it is the one calibrated at device init (`OP_COVERAGE.md` §7.2;
§8.4 A3), not a constant — with `TransferModel::UMA` / `DISCRETE` as the pre-calibration defaults so
M0/M1 have a defined behaviour before Niobe's measurements exist. `CoverageReport`, computed from
the same `(kept, dropped)` pair, is what produces `largest_island_flops` for §10.0's milestone
reporting — the rule and its metric live in one module deliberately, because their drifting apart is
exactly how a coverage number becomes a lie.

### 5.5 `Compile` — plan build and prepacking

For each fused subgraph, in order:

1. **Extract.** `NodeDesc` per node, attributes copied generically into typed maps, inputs and
   outputs resolved to `TensorRef`/`OutRef` with dtype and static shape where known.
2. **Validate.** Every node must resolve to a registry entry with a claim predicate that still
   accepts it. A mismatch here is an internal invariant violation, not a user error — fail the
   compile loudly.
3. **Plan shapes.** Compute static shapes for every intermediate. Determine the workgroup counts
   and the temporary-buffer set. Assign temporaries to a shared arena with a greedy
   liveness-based packing (ExecuTorch's `SharedObject` idea) so a 40-node subgraph does not
   allocate 40 buffers.
4. **Prepack weights.** For each constant initializer: read host bytes through the ORT graph API,
   convert to the shader's expected layout/dtype, and upload into a **device-local buffer owned by
   the plan**, via a staging buffer. Done once, at compile time. This is the ExecuTorch
   `prepack()` model and the direct analog of the MLX plan's repacked-weight cache, and it is the
   reason inference does not re-upload weights.
5. **Warm pipelines.** Create every `VkPipeline` the plan needs, populating the `VkPipelineCache`.
   First-inference latency is a real user-visible cost on Vulkan; paying it at compile time is the
   right trade.
6. **Wrap.** Return an `OrtNodeComputeInfo` owning the plan.

### 5.6 `Compute` — inference

```
ORT calls Compute(state, OrtKernelContext)
 ├─ 1. Resolve inputs.
 │      M0/M1 (host I/O):   ORT hands host pointers → upload into device buffers via staging.
 │      M2+  (device I/O):  ORT hands device "pointers" our allocator produced → resolve to
 │                          (VkBuffer, offset) with no copy at all.
 ├─ 2. Shape key.  Hash the concrete input shapes. Look up recorded.rs.
 │      hit  → reuse the recorded VkCommandBuffer.
 │      miss → record: for each NodeDesc in topo order, registry::translate() runs the handler,
 │             the handler calls DispatchContext::dispatch(), the engine binds descriptors,
 │             emits the memory barrier the dependency edge requires, and records vkCmdDispatch.
 │             Cache the recording under the shape key.
 ├─ 3. Bind outputs. KernelContext_GetOutput for each subgraph output.
 ├─ 4. Submit once to the compute queue. One submission per subgraph execution.
 ├─ 5. Wait on the fence.
 └─ 6. Outputs.
        M0/M1: download device → staging → ORT's host output tensor.
        M2+:   nothing — the output already lives in the device buffer ORT allocated.
```

**Where CPU fallback happens.** Three distinct places, and it is worth keeping them straight:

1. **Claim time (the main one).** A node the registry declines is never in a plan. ORT assigns it
   to the CPU EP and inserts partition-boundary copies through our `OrtDataTransferImpl`. This is
   the designed path and it is always correct.
2. **Compile time.** If a plan cannot be built (a shape that cannot be resolved statically, an
   initializer that cannot be read, a pipeline that fails to create), `Compile` returns an error
   status and ORT falls the whole subgraph back to CPU. Loud, logged, and rare by construction —
   §5.5 step 2 makes it an invariant violation.
3. **Runtime.** A device-lost, out-of-memory, or panic condition returns `ORT_EP_FAIL` from
   `Compute`. ORT surfaces the failure. We do **not** attempt a silent per-node CPU rescue inside
   `Compute`: a partially-executed command buffer with half-written outputs is not a state we can
   reason about, and silently producing CPU results after a GPU fault hides real bugs.

---

## 6. Tensor and memory model

This is the section where Vulkan and MLX genuinely part ways, so it states the **contract**;
`ENGINE.md` §3 owns the implementation.

### 6.1 The problem

MLX: unified memory. An ORT CPU tensor and an MLX array can point at the same bytes. The MLX EP
therefore advertises no device allocator, returns null from `GetDefaultMemoryDevice`, and copies
out once per subgraph with a `memcpy`.

Vulkan: explicit device memory. Device-local memory is generally not host-visible; host-visible
memory is generally not device-local; and on discrete GPUs the two are across a PCIe bus. There is
no pointer we can hand ORT that a shader can also read. Everything below follows from that.

### 6.2 The contract

| Concern | Contract |
|---|---|
| **Tensor identity** | A device tensor is `(VkBuffer, offset, size, dtype, shape)`. The `VkBuffer` may be a suballocation of a larger arena. Op code sees this only as an opaque `BufferView`. |
| **Layout** | Row-major dense, ONNX semantics, no implicit padding, no implicit transposition. Any op needing a different internal layout materializes it explicitly as a temporary. Prepacked *constants* may use a shader-specific layout; **activations may not**. |
| **Alignment** | Every allocation satisfies `minStorageBufferOffsetAlignment` for the bound device. Enforced by the allocator, never by op code. |
| **Who allocates weights** | The plan, at `Compile` time, device-local, uploaded once, freed when the plan drops. |
| **Who allocates activations at partition boundaries** | ORT, through our `OrtAllocator` (M2+) or ORT's CPU allocator (M0/M1). |
| **Who allocates intermediates** | The engine, from a plan-owned arena sized at compile time, reused across inferences. |
| **Who transfers** | Only `transfer.rs` (`OrtDataTransferImpl`) and the engine's staging path. Op code never initiates a transfer. |
| **When transfers happen** | At partition boundaries (ORT-inserted) and, in M0/M1, at subgraph entry/exit. Never per node. |
| **Coherence** | Non-coherent host-visible memory is flushed/invalidated by the engine around every host access. This is not optional and it is not the op author's problem. |
| **Synchronization** | The engine emits the barriers the plan's dataflow edges imply. A read-after-write between two dispatches always gets a barrier. Correctness first; barrier-batching optimization is Niobe's ticket, not a design assumption. |

### 6.3 Avoiding a copy on every inference — the phased plan

This is the crux, so it is explicit.

**M0/M1 — host I/O (the MLX shape).** `GetDefaultMemoryDevice` returns null; the factory's
`CreateAllocator`/`CreateDataTransfer` return null (valid — ORT tolerates it, as the MLX EP
proves). Subgraph I/O lives in CPU memory; `Compute` uploads inputs and downloads outputs each
call. Weights are still uploaded once at compile time, so the per-inference traffic is
activations only.

*Why start here:* it removes the entire allocator/data-transfer/ORT-memory-placement surface from
M0, which is the highest-uncertainty part of the ABI. It gets a correct, cross-platform,
CPU-oracle-verified elementwise op running on Windows and Linux in the shortest path. It is
honestly slow for anything small, and we will say so rather than benchmark it.

**M2 — device I/O (the real model).** The factory implements `CreateAllocator` (returning an
`OrtAllocator` backed by the device arena) and `CreateDataTransfer` (host↔device staging copies).
`GetDefaultMemoryDevice` returns the device's `OrtMemoryDevice`. ORT then:
- places tensors that only ever cross Vulkan partitions in device memory, so **two adjacent Vulkan
  subgraphs separated by a CPU node still avoid one of the two round-trips**;
- uses our data transfer for the boundary copies it does need;
- lets a user with `IoBinding` keep inputs and outputs resident on the device across inferences,
  which is what makes a per-inference copy disappear entirely.

The one hard problem M2 must solve: **ORT's allocator API is pointer-based, and a `VkBuffer` is
not a pointer.** This is OQ-3, and it is now **decided** — see §6.4.

**M3+ — persistence.** Keep prepacked weights and, where a graph allows, activation buffers
resident across `Compute` calls; shapeless recording so a growing dimension does not retrace.

### 6.4 OQ-3 RESOLVED — reserved virtual address space, resolved through an opaque-handle registry

> **Decided 2026-07-28T22:28:08-07:00** on Tank's proposal (`.squad/decisions/inbox/tank-oq3-allocator-proposal.md`,
> D-T8). **Accepted in full, including the part that goes further than my own framing.** Binding on
> `allocator.rs`, `transfer.rs`, `ENGINE.md` and every op that receives an ORT data pointer.

**The decision.** `Alloc(size)` sub-allocates real Vulkan memory through `gpu-allocator`, carves a
matching span out of a large region of **reserved-but-uncommitted virtual address space**
(`VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` on Windows; `mmap(PROT_NONE, MAP_NORESERVE)` on
Linux/Android/macOS), records `span_base -> (VkBuffer, offset, size, generation)`, and returns
`span_base` as the `void*`. Resolution to `(VkBuffer, offset)` happens **once per binding when a
descriptor set is built** — not per element, not per dispatch — and is cached in the compiled plan.
`Free` quarantines the span and bumps a generation rather than recycling it immediately.

**`VK_KHR_buffer_device_address` is not carried at all.** I had written "registry primary, BDA an
optimization on top". Tank's argument that BDA is **not an optimization of the registry but a second
shader architecture** is correct and it changes the answer, so I am adopting his position rather
than the one I brought. A `VkDeviceAddress` is unusable by a descriptor-bound shader: consuming one
requires `GL_EXT_buffer_reference` / `PhysicalStorageBuffer` addressing, which is a second shader
family, a second variant axis in the manifest, and a second set of conformance runs — a permanent
cost on Mouse's and Trinity's surface, not a branch in Tank's. And it *does not even remove the side
table*, because building a descriptor set still needs a `VkBuffer`, so address → buffer resolution
survives regardless; BDA only pays off under a fully-bindless design, which is a much larger bet
nobody has proposed. The platform evidence points the same way: `PLATFORMS.md` MVK3 (BDA needs
Metal 3 / Apple Silicon — Intel Macs excluded), MVK4 (MoltenVK explicitly advises the explicit
binding model), and my own §7.2 freezing BDA as probed-not-required. **Two paths where one is
unreachable on Apple and on older Android drivers is not a dual design; it is one design plus an
under-tested liability**, which is the same failure shape I rejected the `synchronization2` layer
shim for. Nothing forecloses BDA later: if a bindless GEMM path is ever adopted and measurement
shows descriptor construction to be a real cost, BDA slots in *alongside* the registry for shaders
that opted into buffer references. Revisit then, **with numbers**.

**Why reserved VA rather than a synthetic token — the part that decided it.** My stated fear was
that ORT performs pointer arithmetic on values it believes are addresses, and that the resulting bug
would be invisible until some ORT-internal path `memcpy`s from an allocator pointer. Reserved VA
answers that **by construction rather than by convention**, which is a different quality of answer:

- `base + offset` from the memory-pattern planner stays inside the same allocation because the span
  *is* contiguous reserved VA of exactly that size; an interior pointer resolves to
  `(VkBuffer, offset + delta)` correctly. A token scheme cannot provide this at all.
- `align_up(ptr, 256)` works exactly; span bases are 2 MiB-aligned, which dominates
  `minStorageBufferOffsetAlignment` on all target hardware.
- Uniqueness against real heap pointers is **OS-guaranteed** — the kernel will not hand that range
  to anyone else. A hand-rolled "reserved" numeric range can collide; a real reservation cannot.
- A stray dereference is an **MMU fault at an address we recognise**, with a stack trace naming the
  culprit, instead of a read or write to whatever happens to live at `0x1000`. This converts my
  worst case from silent corruption into a crash at the scene.
- Use-after-free resolves to a quarantined span and becomes a loud `OrtStatus` rather than a silent
  alias onto a different live tensor.

I want the general principle recorded, because it will recur: **when a design can make a hazard
impossible by construction, prefer it to a design that makes the hazard merely unlikely, even if
the second is simpler.** The cost here is a page-table reservation that consumes no physical memory.

Two supporting points I specifically endorse. Tank notes the NV reference calls
`DisableMemPattern()`; **we must not rely on that** — an EP cannot force a caller's session options,
and a design that only works with mem-pattern off breaks the first time somebody leaves it on. And
the resolution cost (a subtract, a shift, one flat-array load, under an `RwLock` taken for write
only on `Alloc`/`Free`) is stated plainly now precisely so it cannot later be cited as a reason to
reopen BDA without measurement.

**The Android sub-question: a tuning parameter, not a blocking dependency.** Android's narrower
virtual address space (39-bit on many devices) means the reservation size must be **derived from the
platform at runtime, not hard-coded**. That is a requirement on the implementation, and it is
Tank's to satisfy now — not a dependency on Link's matrix. Rationale: the design does not change
with the answer, only a constant does, and the correct implementation is to *probe and back off*
rather than to look a number up in a table. Binding form:

1. The region size is chosen at allocator construction by attempting a reservation and **halving on
   failure** down to a documented floor, rather than by consulting a per-platform constant. A table
   of platform address-space widths is a table that will be wrong about some device we have never
   seen; a reservation that either succeeds or does not is correct on every device by construction.
2. The chosen size, the granularity, and the number of back-off steps taken are recorded in the
   capability dump. On a 39-bit device this is the line that will explain a later allocation
   failure.
3. Falling below the floor is a **clean allocator-construction failure with a specific message**,
   which degrades the EP to M0/M1 host-I/O behaviour or to no device at all — never a silent
   success with a region too small to serve the model.
4. Link's platform matrix should still record observed address-space widths, because it tells us
   whether the back-off is ever actually taken. That is *information*, and it does not gate M2.

**What this means for OQ-13.** An imported external buffer becomes an ordinary registry entry
(Tank's D-T9 step 6), so nothing downstream — descriptor construction, translate handlers, data
transfer — needs to know a buffer came from outside. That is the registry serving the importer, and
it is a further argument for a single resolution mechanism.

### 6.5 RULING — exactly one `VkDevice` per (physical device, EP instance). The second one is a defect, not a design.

> **Ruled 2026-07-30T22:13:37-07:00**, on Switch's finding that Tank's device-backed memory provider
> creates its own `VkDevice`, so the session cannot bind those buffers. Binding on `allocator.rs`,
> `transfer.rs`, `device.rs` and `ep.rs`.

**The decision, in one sentence: a Vulkan object graph has exactly one owner per EP instance, that
owner is the EP's device context, and any component needing a queue, an allocator or a buffer takes
a handle to it and never constructs one.** §2.3 already said `VkDevice` lifetime is **EP-scoped** and
§1.2 already listed *"one `VkDevice`, one compute queue, one submission per subgraph"* as the
non-goal boundary. **So this is not a new architectural decision and I am not making one. It is an
existing invariant that was violated without anyone noticing, which is a different and more
uncomfortable finding** — the document was right, the code diverged, and nothing in between could
tell. It is R10's lesson in a new place: *the architecture is a claim about the object graph, not
about the prose.*

**Is there a legitimate reason for two? I looked, and no.** The three real ones do not apply here:
separate queue families (one `VkDevice` exposes all of them); differing enabled-extension sets
(solved by enabling the union at device creation, which is what capability probing is *for*, §7.9);
and cross-process/external-memory sharing (that is `VK_KHR_external_memory`, a different mechanism
with a different lifetime, and we are explicitly not doing it in v1). **Compatibility outranks
elegance, and this is not an elegance argument** — the split does not buy compatibility, it costs
it: two devices double the device-lost surface, double the memory budget accounting, and on drivers
that meter per-device allocations they halve the headroom we can address.

**Four consequences, all binding:**

1. **The memory provider takes a borrowed device handle; it does not create one.** The seam moves to
   device-context construction, which happens once, in `CreateEp`, before either the allocator or the
   engine exists. Neither of the two current owners is downstream of the other today, which is
   exactly why they each built one.
2. **Owner of the seam: Switch, with Tank as the consumer of the fixed interface.** Not because the
   defect is his — it is not, it is nobody's and that is the point — but because the device context,
   queue and command pool live in his files, and the rule from §7.0 forward is that **the seam is
   owned by the side that owns the lifetime, never by the side that owns the caller.** Tank's
   allocator changes from *creating* to *receiving*, which is the smaller and safer edit, and it is
   the edit that unpins `alloc_device_authoritative_spans`.
3. **Until it is fixed, every allocator-side byte counter carries its device identity in its name or
   in the artifact, and a run with two devices reports `SPLIT-DEVICE` on the transfer accounting.**
   Not `0`. A run where `vulkan.cmd_upload` is 15.2 s and `alloc_device_upload_bytes` is 0 has two
   correct instruments describing two different worlds, and **the artifact must say which world**,
   because the reader cannot.
4. **`alloc_device_authoritative_spans` may not be used as the instrument for any criterion while
   the split exists**, per R12 below. It is structurally pinned at zero; a pinned instrument is not
   evidence, in either direction.

**The trap if we leave it, stated plainly because it is the real cost.** Two correct owners will
keep building correct mechanisms that cannot observe each other, every cross-cutting number will
need a caveat naming which device produced it, and the caveats will be dropped in transit — which is
precisely what happened to the phase table four hours ago. **A seam that requires a caveat on every
number crossing it is not a seam, it is a fork.**

#### 6.5.1 CLOSED — and the closure is of a **conditional**, in a lane that must be named with it

*Ruled 2026-08-01T18:59:38-07:00, on the coordinator's report that `alloc_device_frame` reads `OFF`
on the ordinary inference path. He raised this against himself, scoring old predictions rather than
doing new work, after telling the team §6.5 was closed. The closure holds. The scope qualifier he
asked for is owed, and this is it.*

**What is closed, and it is not withdrawn.** `bench/results/indexspace.json`: two arms, allocator
factory index **`1` on selector 0 (NVIDIA) and `0` on selector 1 (Intel)**, matching each arm's
offered index, both `SHARED`, both `alloc_device_buffer_binds = 6`, verdict `ONE_INDEX_SPACE`. This
is the R10 artifact the earlier one-armed check was not: the pre-fix state read `SHARED` on selector
0 by **coincidence of two index spaces**, and only a probe whose content varies with its input can
tell those apart. It does. The seam `vk::host_device_memory::offer_shared_device` now has a
production caller (`vk/session.rs`), which is the R10 falsifier the census asked for.

**What the closure is a claim about.** §6.5 is a **conditional**: *when both a session and a
device-memory provider exist in one process, they are on one `VkDevice` and one index space.* The
probe establishes the conditional by making its antecedent true. It does that by setting
`ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1`, which is the **only** thing that brings a device-memory
provider into existence. In the default lane the antecedent is false — no provider exists — and the
counter correctly reads `alloc_device_frame = OFF`.

> **The closure of §6.5 is stated with its lane or it is not stated.** Canonical form: *"§6.5 is
> closed as a conditional, established two-armed in the `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1`
> lane (`bench/results/indexspace.json`); the default inference lane instantiates no device-memory
> provider and reports `alloc_device_frame = OFF`, which is neither a violation nor a confirmation
> of it."* A closure statement that omits the lane is **not a shorter version of this sentence, it
> is a different and false one**, because a reader who runs an ordinary inference will find the
> mechanism switched off and nothing in the closure told them to expect that.

**`OFF` is a third state and its existence is the reason this was catchable.** The prediction on the
record was `SHARED` xor `SPLIT-DEVICE`. The instrument returned neither. **It declined to pick one
of the two offered options instead of forcing itself into the nearer one** — the family discipline
of *every way of not knowing gets a name a machine can print* (§10.0 obligation 6; `UNMEASURED`,
`UNWIRED`, `UNOBSERVABLE`, `SPLIT-DEVICE`, `UNATTRIBUTED`), paying out in the one way that is hard
to arrange deliberately: **a binary prediction met by a third token is a refutation the predictor
cannot talk himself out of.** Had `OFF` been folded into `SPLIT-DEVICE` — both are "not SHARED" —
the prediction would have scored as a clean pass and this scope gap would still be invisible.

**And the scoring discipline that surfaced it is ratified as a rule.** Of six standing predictions,
one was graded `UNSCORABLE` because the artifact lane dispatched 8 and the prediction described the
N=3 Phi-3.5 lane, and one `UNSCORED` for having no artifact at all. Both were counted as **not
passes** rather than quietly dropped. That is R12 applied to predictions:

> **A prediction is scored only against an artifact from the lane it described.** A prediction
> scored against a different lane is `UNSCORABLE` and counts as a non-pass, never as a pass and
> never as absent. A prediction with no artifact is `UNSCORED` and also counts as a non-pass. The
> denominator never shrinks to flatter the numerator.

#### 6.5.2 `alloc_device_frame = OFF` on the default path — RULING: intended, **and its recorded justification has expired**

*The coordinator offered two readings — intended-and-undocumented, or the `offer_shared_device` gap
— and said they have very different consequences. He is right that they do, and the answer is
neither of them.*

**It is not the gap.** The gap was `offer_shared_device` having no production caller. It has one
(`vk/session.rs`). The census's own proxy for that seam — `alloc_device_frame` reading `SHARED`
rather than `SPLIT-DEVICE` once the function acquires a caller — is exactly what the two-armed probe
observes. **§6.5 is not proving a property of a path users never take; it is proving a property of a
seam that is wired, on the only path where the seam has two sides.**

**It is intended.** The device allocator is opt-in behind `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY`,
documented as such, and `factory.rs` gates advertisement on it. That is a decision and it is already
written down. **But the reason written down with it is no longer true**, and that is the finding:

> The recorded justification is that advertising `OrtDeviceMemoryType_DEFAULT` is a package deal —
> ORT then requires a registered `OrtDataTransferImpl`, without which **every session fails at
> `Run`** — and that *"the data transfer cannot be written until the handle→`VkBuffer` seam is
> filled, which is Switch's side."*

**That precondition is discharged.** `CreateDataTransfer` is registered unconditionally on the
factory, `transfer::VulkanDataTransfer` exists, and sessions in the armed lane do not fail at `Run`
— they complete, three sequential sessions per arm, nine spans, 9,437,184 bytes uploaded through the
provider, and at model scale 427 allocations and 2.09 GB of Phi-3.5 through the registry. **The
condition the switch was waiting for has been met and nobody went back to the switch.**

**This is R12 with a date as the frame, and I am recording it as an instance rather than a new
rule.** R12 has already been generalised twice — *for a counter the frame is a device; for a
correctness verdict the frame is an executor* — and this is the third: **for a rationale the frame
is a date.** The doc-comment is correct about the world of 2026-07-29 and is being read in the world
of 2026-08-01, and nothing in it says which. A default whose stated reason has expired is
indistinguishable from one that is still needed — the `retain_viable` shape, arriving in a
justification instead of in a call graph.

**Three obligations, and note what I am deliberately not doing.**

1. **The flag's default is not changed by this ruling and I am not ruling on whether it should be.**
   There is a live reason for `OFF` that is *different* from the recorded one and that this ruling
   does not evaluate: with `alloc_device_authoritative_spans` at a measured ceiling of zero (§6.5.3),
   turning the switch on today buys **host memory wearing a device handle** — risk with no measured
   benefit. That may well be sufficient. It is not what the source says, and **a default defended by
   a reason its own documentation does not give is a default nobody has actually re-decided.**
2. **The justification is re-stated on current evidence, dated, or the default is re-decided —
   owner: Tank (allocator) with Switch (transfer seam), by M2 entry.** Whichever they choose, the
   artifact that would falsify the new reason is named with it. `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY`
   is `factory.rs`, so the edit is theirs and not mine; **this document records that the reason is
   owed, not what it must say.**
3. **Generalised, because this one switch is not the only expired reason we are likely to hold.**
   *A gate, flag, default or `Staged(why)` whose stated precondition has since been discharged is
   re-justified or removed; the check is that the sentence naming the precondition is re-read
   whenever the thing it names lands.* Cheapest satisfaction, named per the drafting rule: writing a
   precondition so vague it can never be observed to be discharged. So the precondition is stated as
   something with an artifact — as this one was, which is why it was catchable.

#### 6.5.3 `alloc_device_authoritative_spans = 0` in the closure artifact — RULING: a **measured zero at a measured ceiling of zero**, not R12

*The coordinator flagged nine spans backed and evaluated against zero authoritative, and asked
whether the honest answer is `UNOBSERVABLE`. It is the right question to ask of that shape, and the
answer here is no. Asking it was correct; the artifact already contains the discriminator.*

The full counters artifact, which the probe's own extract does not reproduce:

```
alloc_device_backed_spans           = 9
alloc_staged_spans                  = 9
alloc_device_residency_evaluations  = 9
alloc_device_authoritative_ceiling  = 0        <- backed - staged
alloc_device_authoritative_spans    = 0        <- an int, not a string
```

**Three independent reasons it is a real zero.**

1. **Its type says so.** The counter is emitted as the JSON string `"UNOBSERVABLE"` when the frame
   is not `SHARED`, as `"UNWIRED"` when the frame allows it but the increment point has never run,
   and as an **integer** only when it is a measurement. Here the frame is `SHARED` and evaluations
   are 9, so the artifact prints `0`, a number. **The type discipline this project adopted for
   exactly this question answered it before the question was asked** — that is what the three-state
   design is for, and it worked.
2. **Its unconditional twin moved.** `on_residency_evaluated` increments the unconditional
   `residency_evaluations` counter and the conditional authoritative one **from the same call site**,
   so a zero authoritative count with a nine evaluations count is a measurement of nine negatives.
   A silent no-op would show `0`/`0`.
3. **The zero is at its own measured ceiling, and the ceiling is the explanation.** All nine
   device-backed spans are also host-staged, so `ceiling = backed − staged = 0` and zero is the only
   value the counter could correctly take. That is not the counter being blind; it is the counter
   reporting the true state of the world — **the engine still reads through `host_backing_for`, so
   staging remains authoritative and the device buffer is a mirror.** `binds = 6` and
   `authoritative = 0` are consistent, and the pair is the honest description of where the transfer
   path currently stands.

**Why it is not R12, stated as the distinction rather than as the answer.** R12's `UNOBSERVABLE` is
for a question that **cannot be asked** in this frame — the event is structurally impossible, and no
run in this frame could ever produce a different number. Here the question **was** asked, nine
times, in the right frame, and the answer was no nine times. **A zero at a zero ceiling is
contingent: it is a fact about this run's allocation behaviour, and it flips the day a span is
allocated device-only.** `UNOBSERVABLE` would be a *stronger and false* claim — it would assert the
counter could not move, when what is true is that nothing moved it.

**One thing does need fixing, and it is in the probe rather than the counter.**
`bench/results/probe_indexspace.py` extracts nine keys and **drops `alloc_staged_spans` and
`alloc_device_authoritative_ceiling`** — the two that make the zero interpretable. Its extract
therefore presents `backed=9, evaluations=9, authoritative=0` as a sufficient set when it is not,
and a careful reader of that artifact **correctly** could not tell a measured zero from a pinned
one. That is the R11 shape in a probe's *selection* rather than in a name: a set of numbers
published as though it closed. **Remedy: the extract carries the ceiling and the staged count, or it
does not carry the authoritative count at all.** `bench/` is Niobe's and the probe is Switch's work;
the change is two strings in `KEYS` and it is theirs to make, not mine.

**DISCHARGED 2026-08-01T20:39:12-07:00, and the audit it triggered found two more — one of them a
different rule.** Niobe restored both companion keys plus `alloc_allocations`, so a bare numerator
(`backed_spans = 9`) now travels with its denominator — *nine of nine and nine of nine hundred are
different findings* — and added `span_accounting()`, which names on the face of the output which
kind of zero a zero is. The second instance from that audit is the phantom key
`alloc_device_spans`, ruled at §10.0.1 R13 amendment 1 (it is R13, not R11, and no new rule is
owed). Two rulings on her instrument follow.

**`span_accounting()` reports without judging — UPHELD, and it is the right instinct.** It feeds no
check; an accounting note able to withhold `ONE_INDEX_SPACE` would be, in her words, *a different
instrument wearing this one's name*, and after `gpu_steady_tail` the case for keeping describers out
of verdicts needs no restating. **But "feeds no check" is not the same as "has no teeth", and R9's
teeth clause is unforgiving: an instrument that goes red and changes nothing is decoration.** The
teeth this one needs are not verdict-moving, they are **attachment**:

> **A `span_accounting()` classification travels in the same artifact as every span count it
> describes.** It must be impossible to quote `alloc_device_authoritative_spans` without its
> classification arriving alongside — the `executed_by` lesson, where the observation became a
> constructor argument because *a caveat that lives in a different artifact from the number it
> qualifies is not attached to it.* Named trigger for revisiting the no-judging call: **if any
> criterion is ever read against a span count, this classification becomes that criterion's
> precondition at that moment**, and the choice gets re-made with the stakes it then has.

**One defect in it, and it is this ruling's own subject committed one level up.** `NOT_A_NUMBER`
fires on `not isinstance(auth, int)` — while the extract is still built with
`data.get(k, "<absent>")`. So a **missing or phantom key** classifies as `NOT_A_NUMBER` and reports
*"a string state and not a count. The type is the answer; no arithmetic applies"* — which is
**false and affirmatively reassuring**: `'<absent>'` is not a state the counter emitted, it is the
probe's own sentinel, and the type discipline has answered nothing. `"UNOBSERVABLE"` and
`'<absent>'` are an EP-side finding and a reader-side failure wearing one token. **This is not
Niobe's error to carry** — she inherited the defaulting read from the original probe — and it is the
argument for the remedy being at the **lookup** and not in the classifier: fix the read once, or fix
N consumers and acquire an N+1th the next time someone adds one. Until the key census lands, the
minimum is that `NOT_A_NUMBER` splits, with an unresolvable key reported as `ERROR(instrument)`.

**Three sightings in one day of an instrument-side absence rendered as a subject-side state** — a
dropped companion key, a phantom key, and a sentinel swallowed by a type test. **The recurrence is
the evidence the obligation is warranted; any one of them alone would have been an anecdote.**

---

## 7. Vulkan API baseline — decision

> **Status: FROZEN as of 2026-07-28T19:16:08-07:00.** OQ-1 is **resolved** with measured data
> (Link, [`PLATFORMS.md`](./PLATFORMS.md) §8, vulkan.gpuinfo.org pulled 2026-07-28) and the answer
> **reversed the provisional §7.2 requirement set**. This section is now the binding contract for
> Switch's [`ENGINE.md`](./ENGINE.md) and Link's CI matrix. Changing it requires a new decision
> record, not an edit.
>
> **Governing directive (Justin, 2026-07-28):** 「如果 1.3 兼容性不好 那 1.2 更好。可以保证兼容性
> 是最好。」 — *broad device compatibility is the top-priority property of this decision.* Where
> device coverage and engine-code simplicity conflict, **coverage wins**. Every ruling below is
> made under that constraint, and where it costs Switch complexity, it costs Switch complexity.
>
> Justin separately ratified the *framing* — a capability set rather than a version number
> (「拿能力集很聪明，听你的」). What changed is where the bar sits, not how the bar is expressed.

### 7.0 The frozen principle

**The device gate is minimal. Capability shortfalls degrade op coverage, not device availability.**

This one sentence replaces the previous "require the two features Switch wants" posture. A device
that lacks an optional capability must still load, still be advertised, and still run every op
that does not need that capability. It declines the ops it cannot do correctly, and ORT's
partitioner sends those to CPU (§2.6). We never refuse a device for a reason that only affects
*some* ops.

Consequences, stated so they are not re-litigated per op:

- A hard device requirement must be justified by *"no op we will ever ship can work without it."*
- A per-op requirement is expressed as a claim predicate in `ops/` (§8), never as a device gate.
- Anything we make optional, we must be able to run **both ways in CI** (§7.5 item 5, §9.1).

**Companion, added 2026-07-30T06:32:18-07:00 — §7.0.1. This does not modify the frozen gate and
adds no device requirement; it is recorded here because readers come to §7.0 for the principle and
would otherwise find a category it does not mention.** §7.0 contemplates ops we **cannot** run.
Coverage work created a third category: ops we **can** dispatch and have **not proven correct**.

> **§7.0.1 — Evidence shortfalls degrade op coverage, not device availability, and they degrade it
> identically to capability shortfalls. An op we have not proven correct on a form is, for claiming
> purposes, an op we cannot run on that form.**

The full ruling, the proof-key granularity, the escape hatch and the cost are §8.9. Nothing in it
touches the five §7.2 requirements or the device gate, which remain frozen.

**Second companion, added 2026-07-30T19:05:03-07:00 — §7.0.2. Same disclaimer: no device
requirement is added and the §7.2 gate stays frozen.** §7.0 contemplates ops we **cannot** run.
§7.0.1 added ops we **can** run and have not proven correct. Measurement has now produced a fourth
category that neither sentence reaches: **ops we can run correctly, and which make the model worse
by being claimed.**

> **§7.0.2 — A claim is a scheduling decision, not a capability statement. Correctness is necessary
> and not sufficient: an op we can run correctly on a form may still be wrong to claim in a given
> graph, and declining it there is not a coverage shortfall. Whether a claim is net-positive is a
> property of the op *in a graph at a coverage level*, never a property of the op.**

The concrete case is Mouse's, measured rather than argued: promoting
`SkipSimplifiedLayerNormalization` from `Staged` to `Ready` moved Phi-3.5 from 257 claimed / 257
islands to 321 claimed / 321 islands — **+64 claimed nodes and +64 islands, nothing merged** — and
made the model slower. The same op, after the partitioner was wired and islands collapsed 321 → 33,
helps. Nothing about the kernel changed between those two readings. This is §10.0's `Cast` result
(coverage 28% → 54%, islands 52 → 125) arriving a second time on a different graph, and two
observations of the same shape stop being anecdotes.

**Four consequences, and the third is the one that will be argued with.**

1. **The decision lives in the partitioner, not in the registry.** `Staged`/`Ready`/the §8.9 proof
   ledger answer *"may we?"* — capability and evidence, both properties of the op and its form, both
   stable across graphs. Net benefit answers *"should we, here?"* and is evaluated at
   `GetCapability` against the graph in front of us (`ops/partition.rs::evaluate`, §5.4). **These
   two must never be conflated**, and in particular a net-negative result must never be recorded by
   demoting a row to `Staged` — that would encode a graph-dependent fact in a graph-independent
   place, and it would make an op that works look like an op we have not written.
2. **A net-negative decline is its own decline code and is never folded into `staged` or `dtype`.**
   `DeclineCode::Partition` already exists and is machine-readable. Folding it in would make our own
   decline histogram (R8) report a kernel gap where there is none, and R8's whole point is that the
   histogram drives what we build next. A decline that lies about *why* misdirects the roadmap.
3. **This is the only decline in the system that is discretionary, so it is the only one that needs
   a guard against itself.** The other three categories decline because we must; this one declines
   because we judge. A discretionary decline can hide a slow kernel behind a partitioning argument,
   and it can be tuned until the numbers look good — which is why **a net-negative decline must be
   measured per artifact at producer-at-version and re-measured when the neighbourhood changes**,
   never asserted from the shape of the graph. The SkipNorm case is precisely one where the correct
   answer *flipped* without the op changing.
4. **The mechanism must be observed to run** (R10). `partition::evaluate` is wired into
   `GetCapability` as of 2026-07-30 and executes per cluster; `retain_viable`, the batch wrapper the
   doctrine was written around, is still called only from `partition.rs`'s own `#[cfg(test)]`
   module. **"partition.rs is wired" is true of one entry point and false of another**, which is
   R10's sub-rule in miniature: wiring is a property of an entry point, not of a file.

**One hazard recorded rather than fixed, because I do not own the code.** `GetCapability` currently
bypasses `evaluate` entirely when there is exactly one surviving cluster, on the stated grounds that
*"there is no competing partition to choose between, so the economics check is moot."* That premise
is wrong in the direction §7.0.2 is about: **the competing partition is always CPU fallback.** A
single island that costs more than it earns is exactly the case where claiming nothing is the right
answer, and it is the case the bypass excludes. It did not bite on Phi-3.5 (33 clusters), and it is
the shape a fully-claimed graph converges to, so it will bite at T4. Owner: Mouse, with the
measurement, not with an argument.

### 7.1 The evidence

| Source | Finding |
|---|---|
| Justin's proposal | Vulkan 1.3, citing llama.cpp. |
| llama.cpp `ggml-vulkan.cpp` | Hard runtime floor is **Vulkan 1.2** — `if (api_version < VK_API_VERSION_1_2) throw`. `VkApplicationInfo::apiVersion` is set to *whatever the instance reports*, not to a hardcoded 1.3. |
| llama.cpp `vulkan-shaders-gen.cpp` (Fact Checker, claim 1, SHA `3e6b395`) | **Base shaders are compiled with `--target-env=vulkan1.2`.** Only the cooperative-matrix-2 (`_cm2`) variants — an NVIDIA Ampere+/Ada optimization path — target `vulkan1.3`. The `--target-env=vulkan1.3` in CMakeLists is an extension-availability probe, not the default. **Verdict: the "llama.cpp requires 1.3" claim is contradicted.** |
| ExecuTorch `vk_api/Runtime.cpp` (Fact Checker, claim 2, SHA `8001512`) | Hardcodes `VK_API_VERSION_1_1` in `VkApplicationInfo`; `Device.cpp` branches feature queries at `>= VK_API_VERSION_1_1`; VMA is initialized with `VK_API_VERSION_1_0`. **Verdict: contradicted** — ExecuTorch targets 1.1. |
| MoltenVK (Fact Checker, claim 3) | **Verified:** MoltenVK 1.3.0 (2025) advertises Vulkan 1.3 on macOS/iOS. Older MoltenVK does not. |
| Android (Fact Checker, claim 4 — *unverified*, plausible) | Vulkan 1.3 ≈ **26%** of active Android devices; Vulkan 1.1 ≈ **62%**. The Android CDD does not mandate 1.3 at any API level as of Android 15. Link (`PLATFORMS.md` §4) reports ~89% for 1.1 measured against *devices that expose Vulkan at all* — a different denominator, same conclusion. |
| lavapipe / SwiftShader (Fact Checker, claim 5) | **Verified:** both support Vulkan 1.3 and both pass 1.3 conformance. Adequate for GPU-less CI. *Update 2026-07-28T21:01:56-07:00: Trinity evaluated SwiftShader and **rejected** it — no usable prebuilts and a ~20-minute build from source. **Both CI lanes use lavapipe**: Linux via `apt`, Windows via mesa-dist-win 26.1.3.* |
| Link (`PLATFORMS.md` §4) | Recommends 1.2 core + mandatory device features. Explicitly does **not** recommend a hard 1.3 baseline if Android coverage is a goal. |
| Switch (`ENGINE.md` §8) | Exactly **two** features materially simplify the engine: `synchronization2` and `subgroup_size_control`. Both are core in 1.3 — **and both are available as standalone extensions on 1.1/1.2 drivers.** `shaderFloat16`, `bufferDeviceAddress`, and cooperative matrix must be capability-probed at runtime *regardless of baseline*. |
| **Link, OQ-1 (`PLATFORMS.md` §8), vulkan.gpuinfo.org, pulled 2026-07-28** | **`VK_KHR_synchronization2`: Android 68.57%, Windows 87.78%, Linux 99.05%, macOS 97.5%, iOS 100%.** A **31.43-point Android gap** and a **12.22-point Windows gap.** The Android shortfall is concentrated in Adreno 5xx (Snapdragon 625–660, frozen pre-2021 OEM blobs), Adreno 6xx on unupdated Android 10/11, and Mali Bifrost (G52/G57/G72/G76) especially on MediaTek — populations with no update cadence, so this does not decay with time on any schedule we control. Link's verdict: the hard requirement is **not safe**. |
| **Link, OQ-1** | **`VK_EXT_subgroup_size_control`: Android 85.88%, Windows 93.33%, Linux 98.81%, macOS/iOS 100%.** A **14.12-point Android gap.** |
| **Link, OQ-1 — the MoltenVK artifact** | The macOS/iOS 100% figure is **extension-string presence only**. MoltenVK reports Vulkan 1.3, which promotes `subgroup_size_control` to core, so the string is always there — but the `subgroupSizeControl` **feature flag is `VK_FALSE`**, because Metal cannot control SIMD-group width per pipeline. **Requiring the feature flag to be `VK_TRUE` would silently exclude all of macOS and iOS** — and probably lavapipe too, which has a single fixed CPU SIMD width. |
| **Link, OQ-1 — limits that *are* safe** | `maxComputeWorkGroupInvocations ≥ 256`: ~1% of 8,206 Android reports show 128. `maxComputeSharedMemorySize ≥ 16 KiB`: the Vulkan spec minimum, universal. Subgroup `BASIC`: spec-guaranteed in the compute stage on 1.1+. Subgroup `ARITHMETIC`: >95%, but *query, never assume*. |
| **Morpheus, layer-shim feasibility research (2026-07-28, primary sources below)** | The Khronos `VK_LAYER_KHRONOS_synchronization2` shim **cannot be shipped by us on Android.** The AOSP Vulkan loader does not read `VK_LAYER_PATH`, does not use JSON manifests, and searches only the **host application's** `nativeLibraryDir` (derived from the installed APK via `GraphicsEnv::getAppNamespace()`) plus `/data/local/debug/vulkan` (debuggable/userdebug only). Khronos' own `docs/synchronization2_layer.md` states the `.so` "needs to be packaged **inside the APK**". A plugin `.so` `dlopen`ed into someone else's process has no mechanism to add a layer search path. Sources: `developer.android.com/ndk/guides/graphics/validation-layer`; `KhronosGroup/Vulkan-Loader` `docs/LoaderLayerInterface.md` ("The Android loader does not use manifest files"; "There is No Support For Implicit Layers on Android"); `KhronosGroup/Vulkan-ExtensionLayer` `docs/synchronization2_layer.md`. |
| **Morpheus, prior-art check on barrier strategy** | **wgpu, Dawn, and Godot all use legacy `vkCmdPipelineBarrier` exclusively and none of them ships the sync2 layer.** `gfx-rs/wgpu` `wgpu-hal/src/vulkan/command.rs` calls `cmd_pipeline_barrier` in `transition_buffers`/`transition_textures` with no sync2 variant and no sync2 entry in its `Workarounds` bitflags; `google/dawn` `src/dawn/native/vulkan/CommandBufferVk.cpp` calls `fn.CmdPipelineBarrier` and mentions sync2 only in a spec comment; `godotengine/godot` `drivers/vulkan/rendering_device_driver_vulkan.cpp` calls `vkCmdPipelineBarrier`. The cited precedent for Option B does not survive contact with the source. |

The premise that motivated 1.3 — "llama.cpp requires it" — is contradicted by llama.cpp's own
source at both the runtime check and the shader target, and independently verified as contradicted
by Fact Checker. That does not make 1.3 wrong; it makes the *reason* wrong, and I would rather we
decide this on the two features Switch identified than on a misattribution.


### 7.2 Decision — the frozen capability set

**We require a capability set, not a version number.** The set is deliberately small.

A physical device is advertised to ORT **if and only if** it satisfies all of:

| # | Hard requirement | Why it is a *device* gate and not a per-op gate |
|---|---|---|
| R1 | Vulkan **≥ 1.1** core, instance and device | `VkPhysicalDeviceFeatures2` / `VkPhysicalDeviceProperties2` chains and `VkPhysicalDeviceSubgroupProperties` are core at 1.1. Below 1.1 we cannot even *ask* what a device can do, so no op can be claimed safely. This is also the Android floor. |
| R2 | A queue family with `VK_QUEUE_COMPUTE_BIT` | Without it there is nothing to dispatch to. |
| R3 | `maxComputeWorkGroupInvocations ≥ 256` | Every shader skeleton we will write assumes a 256-invocation workgroup. ~1% of Android reports fall below. |
| R4 | `maxComputeSharedMemorySize ≥ 16384` | The Vulkan spec minimum; universal. Listed so the assumption is written down, not because it filters anything. |
| ~~R5~~ | ~~Subgroup `BASIC` in the `COMPUTE` stage~~ | **REMOVED from the gate 2026-07-29 — see below.** Now a probed capability, `Capabilities::subgroup_basic_in_compute`. |
| R6 | At least one `DEVICE_LOCAL` memory type and at least one `HOST_VISIBLE` memory type | The staging path (§6) has no meaning otherwise. |

**That is the entire gate.** It is satisfied by essentially every device that exposes Vulkan 1.1
at all, on every platform, including MoltenVK and lavapipe.

**R5's removal — the decision stands, and the reason recorded for it was false.** *Corrected
2026-07-29T16:00:55-07:00. The previous text of this passage said R5 was removed because lavapipe
reports `supportedStages = 0`. That was wrong and the correction matters more than the fix.*

- **What actually happened.** Subgroup BASIC was demoted from the device gate to a probed
  capability. That demotion is **correct and it is kept**, because §7.0 says a capability shortfall
  must degrade *op coverage*, not *device availability* — and subgroup arithmetic is exactly such a
  shortfall: it selects between the subgroup-reduction and shared-memory tree-reduction shader
  variants (§7.2's capability table), which is a variant choice, not a reason to refuse a device.
  A device that cannot do subgroup reductions can still run every elementwise kernel we have.
- **What the recorded reason claimed, and why it was false.** The premise was that lavapipe reports
  `VkPhysicalDeviceSubgroupProperties::supportedStages = 0`, i.e. no subgroup support in any stage,
  and that R5 therefore excluded both of our CI lanes. Switch re-checked this on request: **Mesa
  26.1 lavapipe does support subgroup `BASIC` in compute**, and the `supportedStages = 0` reading
  was almost certainly the §7.9 Bug 1 `push_next` probe failure — *our own bug*, producing a number
  we then treated as a device fact and used to reopen a frozen architectural decision.
- **The two things are independent, which is the only reason this is a correction and not a
  reversal.** §7.0 argues R5 out of the gate from first principles and never mentions lavapipe.
  Had the lavapipe number been the *load-bearing* reason rather than the *presented* one, this
  correction would have had to restore R5. It does not. **A right answer reached through false
  evidence is still an unaudited answer**, and the audit is what this paragraph is.
- **Downstream consequences.** `PLATFORMS.md` §6.3 quirk LVP2 and its "reason §7.2 removed subgroup
  BASIC" attribution are wrong on the *reason* — Link should re-observe lavapipe's `supportedStages`
  with the fixed probe and restate LVP2 as observed-or-retracted. `ENGINE.md` §-on-the-gate and the
  `caps.rs` / `instance.rs` comments carry the same false premise and should cite §7.0 instead.
  **No code behaviour changes**: R5 stays out of the gate, the scalar fallback path stays, and the
  `instance.rs` test pinning R5's absence stays — it is pinning the right policy for the right
  reason once its comment is corrected.
- **Recorded as §10.0.1 R6**, because the episode is a cleaner specimen of our characteristic
  failure than anything else in the register: every step was reasonable, the conclusion was right,
  and the evidence was manufactured by our own tooling.

**Everything else is capability-probed** into a single `vk::caps::Capabilities` struct, read once at
device init, and used in exactly two ways: (a) to select an implementation strategy inside the
engine, or (b) to gate an op's claim predicate. Nothing on this list may ever become a device gate
without a new decision record:

| Capability | Probed how | What it changes |
|---|---|---|
| `synchronization2` | 1.3 core **or** `VK_KHR_synchronization2` device extension | Selects the barrier backend (§7.3). **Not required.** |
| `subgroup_size_control` **properties** | 1.3 core **or** `VK_EXT_subgroup_size_control` — *properties queryable only* (§7.4) | Narrows the known subgroup-size range; enables the subgroup-cooperative shader variants. |
| Subgroup `BASIC` (formerly gate R5) / `ARITHMETIC` / `BALLOT` / `SHUFFLE` | `VkPhysicalDeviceSubgroupProperties::supportedOperations` + `supportedStages` | Gates the subgroup-reduction shader variants. Absent → shared-memory tree-reduction variant. **Never a device gate** (§7.2, R5's removal). |
| `shaderFloat16`, `storageBuffer16BitAccess` | `VkPhysicalDeviceVulkan12Features` / `VK_KHR_shader_float16_int8` + `VK_KHR_16bit_storage` | Gates fp16 op variants; absent → those ops are not claimed for fp16. |
| `shaderInt8`, integer dot product | extension probe | Gates future quantized ops. |
| Timeline semaphores | 1.2 core or `VK_KHR_timeline_semaphore` | Post-v0 multi-stream pipelining. Unused in v0. |
| `bufferDeviceAddress` | 1.2 core or `VK_KHR_buffer_device_address` | **Not used.** OQ-3 is resolved against it (§6.4); probed for diagnostics only, and a future bindless GEMM path would have to re-argue it with numbers. |
| Cooperative matrix | `VK_KHR_cooperative_matrix` / `VK_NV_cooperative_matrix2` | Post-v0 GEMM variants, llama.cpp's `_cm2` split. |

`VkApplicationInfo::apiVersion` is set to `min(vkEnumerateInstanceVersion(), VK_API_VERSION_1_3)`
— llama.cpp's pattern. We ask for the highest the loader will give us and then *use* only what the
device actually reports.

### 7.3 `synchronization2` — dropped from the hard requirement; Switch carries a legacy path

**Ruling: Option A.** `synchronization2` is **not required**. Switch implements a legacy
`vkCmdPipelineBarrier` backend alongside the `vkCmdPipelineBarrier2` backend, selected once at
device init (§7.5 defines the seam).

This reverses the provisional §7.2 of 2026-07-28T17:59:54-07:00, which required it.

**Why.** Under the compatibility-first directive, a 31.43-point Android exclusion and a
12.22-point Windows exclusion cannot be traded for one internal code path. The Windows number
matters as much as the Android one and is easy to overlook: nearly one desktop Windows device in
eight in Link's sample would be silently declined. The missing Android population is
*structurally* missing — Adreno 5xx blobs frozen before the 2021 extension, Mali Bifrost on
MediaTek with no update cadence — so it does not shrink on any timeline we control.

The cost is bounded and one-time: two implementations of a five-function internal API, written
once, tested in CI on every run (§7.5). The cost of the alternative is unbounded and permanent:
every device we decline is a device we can never win back with engineering.

#### Ruling on the layer-shim proposal (Link's Option B) — **rejected as a shippable mechanism**

The coordinator asked me to examine rather than adopt this. I did, and the concern is correct and
decisive.

| Platform | Can *our plugin* enable `VK_LAYER_KHRONOS_synchronization2`? | Basis |
|---|---|---|
| **Retail Android (non-rooted, non-debuggable)** | **No.** | The AOSP loader does not read `VK_LAYER_PATH`, does not use JSON manifests, and enumerates layers only from the **host application's** `nativeLibraryDir` (set by the framework at process launch from the installed APK via `GraphicsEnv::getAppNamespace()`) and from `/data/local/debug/vulkan`, which requires a debuggable app or a userdebug build. Khronos' own layer documentation says the `.so` must be "packaged inside the APK". We do not own the APK. |
| Windows | Conditionally yes | `SetEnvironmentVariable("VK_ADD_LAYER_PATH", …)` before *our* `vkCreateInstance` works, because the desktop loader re-scans layer paths at `vkCreateInstance`, not at load time. **Fails silently** if the host process runs at High Integrity Level (`loader_secure_getenv` returns NULL). Mutating the environment of a host process we do not own is also a `setenv`/`getenv` data race in any multi-threaded host. |
| Linux / macOS | Conditionally yes | Same mechanism; fails under setuid/setgid (`secure_getenv`). Manifest must carry an absolute `.so` path. Same race. |

Two independent reasons to reject it even where it technically works:

1. **It does not solve the platform it was proposed for.** Android is 100% of the reason we were
   considering it, and Android is the one platform where it cannot work from a plugin.
2. **The cited precedent does not exist.** wgpu, Dawn, and Godot were offered as evidence that
   shipping this layer is normal practice. Reading their source, all three use legacy
   `vkCmdPipelineBarrier` exclusively and none of them ships the sync2 layer. The precedent
   actually supports Option A.

Add to that: silently mutating a host process's environment variables from inside a `dlopen`ed
plugin is behaviour I would reject in code review on its own merits, independent of Vulkan.

**What survives.** Nothing that we ship. If an *Android integrator* independently packages
`libVkLayer_khronos_synchronization2.so` in their own APK and enables it, our sync2 backend will
light up automatically — because we probe the extension, and the layer's documented behaviour is
to advertise the extension and disable itself when the driver already provides it. That is a
**documented, optional, integrator-side deployment note** in `PLATFORMS.md`, not a mechanism we
depend on and not a substitute for the legacy path. Labelled as materially weaker, exactly as the
coordinator required.

**Option C (scope Android to a 2021+ population) is rejected outright** — it is the directive read
backwards.

### 7.4 `subgroup_size_control` — required as a *query*, never as a *feature*

**Ruling.** `subgroup_size_control` is **not** a device gate at all, and where we do consult it we
require only that the **properties struct is queryable**. We **never** require
`VkPhysicalDeviceSubgroupSizeControlFeatures::subgroupSizeControl == VK_TRUE`, and we never call
`vkCmdSetRequiredSubgroupSize`-style per-pipeline sizing as a correctness dependency.

Precisely what the engine does:

1. **Always** read `VkPhysicalDeviceSubgroupProperties::subgroupSize` and `supportedOperations`
   (Vulkan 1.1 core, universally available). This is the baseline knowledge.
2. **If** `VK_EXT_subgroup_size_control` is present *or* the device reports 1.3 core, chain
   `VkPhysicalDeviceSubgroupSizeControlProperties` into `vkGetPhysicalDeviceProperties2` and record
   `minSubgroupSize` / `maxSubgroupSize` / `requiredSubgroupSizeStages`. Treat this as *better
   information about the range*, nothing more.
3. **Only if** the `subgroupSizeControl` feature flag is additionally `VK_TRUE` may a pipeline be
   created with `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`. This is an *optimization
   path*, gated at pipeline-creation time.
4. **A shader whose correctness depends on a specific subgroup width may only be selected when the
   width is known exactly** — i.e. `minSubgroupSize == maxSubgroupSize`, or the required-size
   pipeline path from (3) is available and was used. Otherwise the engine selects the portable
   variant, which uses shared memory and workgroup barriers and makes no subgroup-width assumption.

Rule 4 is the substantive part and it is a correctness rule, not a performance rule. Assuming a
subgroup width silently produces wrong numbers in cooperative GEMM and reduction shaders; that was
the original reason for wanting this extension, and this formulation preserves the guarantee
without excluding anyone.

**Why this matters beyond macOS.** Requiring the feature flag would have excluded all of
macOS/iOS (MoltenVK reports `VK_FALSE`; Metal has no per-pipeline SIMD-group width control) and
very likely lavapipe — meaning both of our own CI lanes. A requirement that excludes the
machines you test on is a requirement you have not tested. Requiring the extension *string* would
still have cost 14.12 points of Android for information we can approximate from 1.1 core.

**Link's third open item — the 12.22% Windows `synchronization2` gap — is moot** under this
ruling. We accept nothing, because we require nothing; those devices run the legacy backend.

### 7.5 The barrier abstraction contract — binding on `ENGINE.md`

Switch wrote `ENGINE.md` §6.2 around a single `vkCmdPipelineBarrier2` path (§6.3 already noted a
fallback, but only as a sentence). This section is the contract that replaces it.

**Rule: one internal barrier API, two backends, selected exactly once at device init. Not
`if caps.sync2 { … } else { … }` at call sites.** A dual path scattered across the recorder is how
this decision turns into a bug farm; a single seam is how it stays a one-time cost.

The seam lives in **`rust/src/vk/barrier.rs`** and is the *only* file in the crate permitted to
name `vkCmdPipelineBarrier`, `vkCmdPipelineBarrier2`, `VkBufferMemoryBarrier`,
`VkBufferMemoryBarrier2`, `VkDependencyInfo`, or the `VK_PIPELINE_STAGE*` / `VK_ACCESS*` flag
families. The layering lint (§4.2) is extended to enforce this: those tokens outside
`vk/barrier.rs` fail CI.

Shape of the seam (illustrative — Switch owns the final signatures):

```rust
// rust/src/vk/barrier.rs — the ONLY module that names Vulkan barrier types.

/// Our own closed set. Deliberately contains no `None`/`NONE` variant: `VK_PIPELINE_STAGE_2_NONE`
/// has no legacy equivalent, so the abstraction must not be able to express it.
pub(crate) enum Access { ShaderRead, ShaderWrite, TransferRead, TransferWrite, HostRead, HostWrite }

pub(crate) struct BufferDep {
    pub buffer: vk::Buffer, pub offset: u64, pub size: u64,
    pub src: Access, pub dst: Access,
}

pub(crate) enum Barriers { Sync2(Sync2Backend), Legacy(LegacyBackend) }

impl Barriers {
    /// Chosen ONCE, in `Device::new`, from `Capabilities`. Never re-evaluated.
    pub(crate) fn select(caps: &Capabilities, dev: &ash::Device) -> Self;

    pub(crate) fn buffer_deps(&self, cb: vk::CommandBuffer, deps: &[BufferDep]);
    pub(crate) fn execution_only(&self, cb: vk::CommandBuffer, src: Access, dst: Access);
}
```

Binding requirements on the implementation:

1. **`Barriers::select` is called once, in `Device::new`, and the result is stored on the device
   handle.** `recorded.rs` and every op path call `dev.barriers().buffer_deps(...)`. No call site
   anywhere else may branch on `caps.synchronization2`.
2. **`Access` and `Stage` are our own closed enums, not Vulkan flag re-exports.** This is what makes
   the legacy backend total rather than best-effort: every value we can express has an exact 32-bit
   legacy equivalent by construction. `VK_PIPELINE_STAGE_2_NONE`, `SHADER_STORAGE_*`-only bits, and
   any other sync2-only concept are simply not representable.
3. **The mapping is one table, in one place.** `ShaderRead → (COMPUTE_SHADER, SHADER_READ)`,
   `ShaderWrite → (COMPUTE_SHADER, SHADER_WRITE)`, `TransferRead → (TRANSFER, TRANSFER_READ)`,
   `TransferWrite → (TRANSFER, TRANSFER_WRITE)`, `HostRead → (HOST, HOST_READ)`,
   `HostWrite → (HOST, HOST_WRITE)`. The sync2 backend widens the same table to the `_2_` flag
   names. If the two tables ever disagree, that is a bug in one file.
4. **Batching semantics are identical in both backends.** `buffer_deps` takes a slice and emits
   **one** barrier command covering all of them — `VkDependencyInfo` with N
   `VkBufferMemoryBarrier2` for sync2, one `vkCmdPipelineBarrier` with N `VkBufferMemoryBarrier`
   and OR-ed stage masks for legacy. The barrier-placement algorithm in `ENGINE.md` §6.2 does not
   change at all; only the emission does.
5. **Both backends are exercised on every CI run.** A new session option
   **`ep.force_legacy_barriers`** (bool, default `false`) forces `Barriers::Legacy` on a device
   that supports sync2. Trinity runs the differential suite twice per lane — once default, once
   forced — so the legacy path is never the untested path. Without this, the ~99%-Linux-coverage
   sync2 path is the only one our CI ever sees, and the 31% of Android we just bought would be
   running code no test has executed.
6. **`ENGINE.md` §6.2's worked example must be rewritten in terms of `buffer_deps`**, and §6.3's
   `vkCmdPipelineBarrier2` row must point at this section. §6.2's *reasoning* — per-edge barriers
   rather than one global barrier, one barrier per consumer edge — is correct and unchanged.

### 7.6 Why this and not the alternatives

**Why not a hard 1.3 baseline (Justin's original proposal).** It buys the two features Switch
wanted, which we have now stopped requiring anyway. It costs roughly **36 points of Android
installed-base coverage** (Fact Checker: 26% at 1.3 vs 62% at 1.1) and any MoltenVK older than
1.3.0, for zero engine simplification. The premise that motivated it — "llama.cpp requires 1.3" —
is contradicted by llama.cpp's own source at both the runtime check (floor 1.2) and the shader
target (base shaders `--target-env=vulkan1.2`), independently verified by Fact Checker (claims
1–2). It is also flatly incompatible with the compatibility-first directive.

**Why not a hard 1.2 baseline.** On Android the 1.2 tier barely exists — devices jumped 1.1 → 1.3
— so a 1.2 floor pays nearly the full Android cost of a 1.3 floor while getting less than 1.3
gives on desktop. The only 1.2 core feature we care about is timeline semaphores, which v0 does not
use and which are available as `VK_KHR_timeline_semaphore` on 1.1 when we do.

**Why not keep the two-extension requirement (the provisional 2026-07-28T17:59:54 position).**
Because Link measured it and it costs 31.43 points of Android and 12.22 points of Windows. It was
a defensible position under "we don't know the number"; it is indefensible now that we do.

**Why not require `synchronization2` on desktop only, and legacy on Android.** A per-platform
requirement is the worst of both: we still write both backends, and we additionally get a matrix
where a Windows-only contributor cannot reproduce an Android-only code path. If we are writing the
legacy backend at all, it must be the one that runs everywhere it is needed and gets tested
everywhere.

**What we lose by being this permissive.** Two things, both accepted: (1) `Barriers` has two
implementations forever, and the CI matrix doubles for barrier-sensitive tests (§7.5 item 5);
(2) a device can now be advertised, claim an op, and produce a slow result where a stricter gate
would have declined the device and let CPU handle it. Mitigation for (2) is Niobe's job, not the
gate's: if a device class is measurably worse than CPU, that is a *scoring* and *claim* decision
(§8), recorded per device class in `PLATFORMS.md` — not a reason to refuse to load.

### 7.7 Shader targets

SPIR-V is compiled with `--target-env=vulkan1.1` by default (SPIR-V 1.3), which every device
meeting §7.2 can consume. This is one notch below llama.cpp's `vulkan1.2` default and is the
conservative choice for Android breadth; if a base shader ever needs a 1.2-only SPIR-V capability
we raise the default and record it. Variants needing higher SPIR-V (fp16 arithmetic, integer dot
product, cooperative matrix) or a known subgroup width (§7.4 rule 4) are compiled as **separate
variants** and selected at runtime from `Capabilities` — the same split llama.cpp uses for its
`_cm2` shaders. Never a single fat module with runtime-dead capabilities; some drivers validate
the whole module.

Note that shader targets are **independent of the barrier decision**: `synchronization2` is a
host-side API, not a SPIR-V capability. Nothing in `shaders/` changes because of §7.3.

### 7.8 OQ-4 RESOLVED — the Vulkan SDK is a hard build dependency; there is no checked-in SPIR-V

> **Decided 2026-07-28T22:28:08-07:00.** This **changes the provisional decision** recorded in
> OQ-4 ("build-time `glslc` with a checked-in SPIR-V fallback so a plain `cargo build` works") to
> match what `build.rs` actually does. The doc was wrong, not the code. Binding on Switch, Tank,
> Link and Trinity.

**The decision.** `glslc` from the Vulkan SDK (or on `PATH`) is a **required build prerequisite**.
There is no checked-in `.spv` fallback and none will be added. A build without `glslc` fails with an
actionable error naming the SDK, `PATH`, and the escape hatch.

**Why I changed the decision rather than the code.** The fallback was meant to prevent exactly the
outcome the coordinator hit — a new contributor cloning the repo and getting a build failure. That
is a real cost and I am not dismissing it. But a checked-in fallback buys convenience by creating a
**silent-divergence hazard**, and it is a bad trade at our shader count:

1. **A checked-in `.spv` that no longer matches its `.comp` is undetectable at a glance and changes
   what runs.** CI (which has the SDK) would compile from source while a contributor without it ran
   a stale binary — and the two would differ in numerical behaviour with no signal. That is the
   "silently wrong" failure class this project's whole claim discipline exists to avoid, relocated
   into the build system. Freshness could be enforced by hashing, but then a stale artifact fails
   the build *anyway* and the fallback has bought nothing.
2. **"Reviewable diffs" — the original argument for checked-in SPIR-V — does not survive contact
   with the artifact.** SPIR-V binary diffs are not reviewable, and the shader-variant table already
   generates **168** modules from 69 rows. Every shader edit would rewrite a large set of binaries,
   making PR diffs unreadable and producing binary merge conflicts.
3. **The prerequisite is normal for this ecosystem and it is honest.** Several Vulkan projects
   require the SDK. A dependency stated in the README costs a contributor one install; a stale
   binary costs somebody a day of debugging a numerical difference that is not in the source.

This is the same principle as §6.4: **prefer the design where the hazard cannot occur to the one
where it is merely unlikely**, and pay a visible cost rather than accept an invisible one.

**Five binding conditions**, because a hard dependency is only defensible if it fails well:

1. **The failure message must be actionable without documentation.** It names the missing tool, how
   many shaders exist, the SDK, `PATH`, and the escape hatch. It already does; this pins it.
2. **The escape hatch (`ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1`) is for lint-only and
   docs-only lanes, and it must stay loud.** It emits a `cargo:warning` stating that the artifact
   can create no pipeline and must not be shipped. It already does; this pins it.
3. **A shader-less artifact must be inert at runtime, not subtly broken.** If `SHADER_MODULES` is
   empty, the EP must **advertise zero devices and claim nothing**, with a specific reason
   (`built without shaders`) in the log and in `CLAIM_DEBUG` — never load, claim nodes, and fail at
   pipeline creation. An escape-hatch build that looks like a working EP is worse than one that
   does not build. *Switch owns this guard; it is the one thing the current code does not yet
   enforce.*
4. **No release artifact may be produced from an escape-hatch build.** The release workflow asserts
   `SHADER_MODULES` is non-empty. *Trinity/Link.*
5. **The prerequisite is documented where a contributor meets it first** — root `README.md` (done),
   `rust/README.md` (*Tank*), and the CI setup step that installs it (*Link*, already true on both
   lanes). Plus a from-clean-clone lane that proves the documented prerequisites are sufficient,
   because a prerequisite list nobody tests is a prerequisite list that is wrong.

**What would reverse this.** Evidence that the SDK requirement is actually blocking contribution —
a platform where `glslc` is genuinely hard to obtain, or repeated contributor friction. The remedy
then is **not** checked-in SPIR-V; it is vendoring a compiler (`shaderc` as a Cargo dependency, or
`naga`), which removes the prerequisite without introducing a second source of truth. That is the
right escape route and it stays open.

### 7.9 Capability probing must distinguish "not supported" from "not asked correctly"

*Added 2026-07-29T09:47:45-07:00, from two bugs found by running on two vendors for the first time.*

§7.2 makes the capability probe load-bearing: it decides which devices we accept, which shader
variants are legal, and which ops may be claimed. That gives it a failure mode nothing else in the
system has — **a probe that fails returns the same answer as a device with no capabilities**, and
that answer is silently conservative, so nothing alarms.

**Bug 1 — the chain that was never sent.** `let _ = props2.push_next(..)` discarded the entire
`VkPhysicalDeviceProperties2` `pNext` chain (`ash`'s builders are `#[must_use]` and return the
modified value rather than mutating in place). Every chained capability therefore read as zero, and
subgroup size appeared to be 0. Nothing was wrong with the device, the driver, or our understanding
of the spec: we never asked the question. **This exact ambiguity had already misled us once, and as
of 2026-07-29T16:00:55-07:00 we know it did.** lavapipe's `supportedStages = 0` was read as a device
fact; Switch re-checked with the fixed probe and **Mesa 26.1 lavapipe does support subgroup `BASIC`
in compute**, so that reading was almost certainly this same bug. It had already been used as the
recorded reason for removing R5 from the frozen device gate (§7.2). The probe bug did not merely
mislead a diagnostic — **it manufactured the evidence for an architectural decision.**

**Bug 2 — the plausible-but-wrong UMA predicate.** `detect_uma` returned `true` for the *discrete*
RTX 4060, because ReBAR maps the VRAM heap `HOST_VISIBLE`. The correct predicate is that **every**
heap is `DEVICE_LOCAL`, not that some heap is host-visible. This one is worse than a mistake — it
is the *natural* mistake: Niobe hit it independently in the benchmark harness, two people reaching
the same wrong answer from the same reasonable intuition. It would have skipped the staging copy on
discrete hardware, which §6 depends on, and it would have been fast and wrong rather than slow and
wrong.

**The rules, binding on `vk/caps.rs` (Switch) and on anything that reads a device property:**

1. **A capability probe reports three states, not two: supported, not supported, and *not
   determined*.** "Not determined" is a distinct value in `Capabilities`, it is never silently
   coerced to "not supported", and reaching it is an error condition worth logging loudly even
   though it degrades safely.
2. **Every chained query is validated after the call, not assumed.** A `pNext` chain that comes back
   entirely zeroed on a device that reports Vulkan ≥1.1 is treated as *probe failure* until proven
   otherwise, because a modern conformant device returning zeros for everything is far less likely
   than our having mis-built the chain.
3. **`--dump-capabilities` prints the raw values it read, not only the derived booleans.** A derived
   boolean cannot be audited; the number it came from can. This is the mechanism that would have
   caught both bugs in minutes.
4. **Predicates over heaps and memory types are stated positively and universally where the safe
   answer is universal.** UMA is "every heap is `DEVICE_LOCAL`", not "a heap is `HOST_VISIBLE`".
   Where a predicate's two plausible readings differ in *which direction they fail*, the one that
   fails toward the extra copy wins — §6's staging path is the safe default and skipping it is the
   optimisation, so the burden of proof sits on skipping.
5. **Capability-derived behaviour is tested on at least one integrated and one discrete device
   before it is trusted.** Both bugs were invisible on a single device and on lavapipe. The local
   Intel + NVIDIA pair is now the minimum bar for any change to `caps.rs`, and the Intel part is the
   more valuable half — it is the stricter implementation, which makes it a conformance oracle
   rather than a second sample.

**Why this is in the design document and not only in `ENGINE.md`.** The capability probe is where a
*silent* wrong answer propagates furthest: into device acceptance, into variant selection, into
claim predicates, and finally into numbers. It is the same failure class as C2 item 7's permissive
fingerprint and as §9.1's shared-misreading hazard — an error that cannot announce itself and must
therefore be designed against rather than tested for after the fact.

**Implementation status — the rules are enforced, not aspirational.** *2026-07-29T16:00:55-07:00.*
Switch has landed rule 1 as `Capabilities::subgroup_probe_valid` (a genuine third state, not a
coerced boolean), rule 2 as a loud warning from `probe()` on an all-zero chain, and rule 3 as
`epctl --probe-loader`, which prints raw capability values per device — the precise output that
would have caught the original bug in minutes rather than letting it manufacture §7.2's R5
rationale. **The rule that a derived boolean cannot be audited but the number behind it can is now
a command anyone can run.** Rule 5's two-device bar (one integrated, one discrete) is a standing
review requirement on `caps.rs`, not a test, and is enforced at review.

**One consequence for prior observations, and it is not optional.** Any capability reading recorded
in `PLATFORMS.md` or `ENGINE.md` **before** the probe fix is provisional and must be re-observed
with `--probe-loader` before it is cited again — starting with lavapipe's LVP2 quirk. A number taken
with a broken instrument is not evidence merely because it was written down.

---

## 8. Op coverage strategy and the v0 op set

### 8.1 Strategy

> **RATIFIED 2026-07-28T19:16:08-07:00 — OQ-11 closed. [`OP_COVERAGE.md`](./OP_COVERAGE.md) (Mouse)
> is the authoritative op-coverage plan and supersedes §8.2 and §8.3 of this document.** Verdict:
> **ratify with five amendments** (§8.4). §8.1's seven principles below are unchanged and Mouse
> endorses all seven; §8.2 is retained only as the M0/M1 floor; §8.3's qualitative fragmentation
> rule is **replaced** by Mouse's quantitative Minimum Viable Subgraph rule (`OP_COVERAGE.md` §7.2),
> which is a strict improvement — it converts a principle I could only state into a rule the
> partitioner can enforce, with a transfer cost *measured at device-init time* rather than a
> hardcoded constant that would be wrong on half our platform matrix.
>
> **Why I ratify.** Three things earn it. (1) The op list is derived from **emitted graphs** — the
> GenAI model builder source, contrib schemas, the ORT WebGPU EP registries — not from the ONNX spec
> index, which is the difference between a coverage plan and a wish list. (2) Every row carries a
> VERIFIED/UNVERIFIED mark and no UNVERIFIED row may be load-bearing for a tier exit criterion; that
> is the discipline I would have had to impose, arriving pre-imposed. (3) The central thesis —
> **87 tier-1 ops served by ~5 kernel templates, and the template infrastructure must exist before
> op #1** — is correct and is the only honest answer to "MLX did this in days". MLX did it in days
> because MLX already owned the kernels. Our leverage has to be manufactured, and §5 of that
> document is a credible manufacturing plan.
>
> **The part I most want on the record** is `OP_COVERAGE.md` §11.1's refusal to collapse two
> different claims into one: ~121 ops is weeks-scale; Qwen3.5 end-to-end is months-scale. A lead who
> lets those merge gets a project that reports 90% op coverage while running nothing. See §1.5.
>
> Constraints from the prior brief are all honoured: conservative claiming (§7.1 of that doc,
> restated verbatim), clean CPU fallback (unchanged), minimum viable subgraph size (§7.2, now
> quantified), and test-plus-platform-row on the same PR (§10). Ratification does not license
> shortcuts on any of them.

Mouse owns `docs/OP_COVERAGE.md`, `docs/OP_ARCHITECTURE.md` and the registry. The architectural
constraints on that work:

1. **One op = one handler + one claim predicate + one registration line, in one
   `ops/<family>.rs`.** Zero edits to `ep.rs`, `engine.rs`, or the registry core. If adding an op
   requires touching the boundary layer, the boundary layer is wrong — that is a bug report
   against L1, not a reason to edit it.
2. **Claim and translate share one table.** Claimed can never outrun translatable.
3. **Claim predicates validate everything:** domain, op type, opset range, input/output count and
   presence, dtypes, required attributes, static-shape availability, and broadcast form. When in
   doubt, do not claim.
4. **Every claimed op ships with a differential test against ORT CPU on the same PR.** No
   exceptions, no "tests in a follow-up."
5. **Every claimed op ships with a `PLATFORMS.md` row or an explicit "untested on X" note.** An op
   verified only on lavapipe is not verified.
6. **Coverage is grown in families, not alphabetically.** A family shares a shader skeleton, a
   descriptor layout, and a test file, so families amortize; scattered ops do not.
7. **fp32 first.** fp16 is a variant per family, gated on `shaderFloat16` +
   `storageBuffer16BitAccess`, added family by family once fp32 is green. No fp64, ever.
   *Amended:* this holds through tier 2. From tier 3 the LLM path is **fp16-native** — an fp32 KV
   cache for a real model is a memory-footprint failure, not a slow path. See §8.4 amendment A4.

### 8.2 The M0/M1 op floor

> Superseded as a *plan* by [`OP_COVERAGE.md`](./OP_COVERAGE.md), ratified 2026-07-28T19:16:08-07:00
> (§8.1). Retained as the minimum that must exist for the pipeline to be provable; nothing here is
> a ceiling. `OP_COVERAGE.md` T0/T1 subsume this and Mouse explicitly left T0 unchanged.

**M0 — one op, end to end.** `Add`, fp32, identical shapes, 2 inputs, 1 output, static shape.
That is the whole M0 claim set. It exists to prove the ABI, the device, the memory path, the
dispatch, and the test harness — not to be useful.

**M1 — the elementwise and shape families.**

| Family | Ops | Constraints |
|---|---|---|
| Binary elementwise | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Min`, `Max` | fp32; equal shapes or suffix/scalar broadcast |
| Unary elementwise | `Neg`, `Abs`, `Sqrt`, `Exp`, `Log`, `Reciprocal`, `Floor`, `Ceil`, `Round`, `Sign`, `Erf` | fp32 |
| Activations | `Relu`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Elu`, `HardSigmoid`, `Softplus`, `Clip`, `Gelu` | fp32; `Clip` with constant or absent min/max |
| Comparison / logic | `Equal`, `Greater`, `Less`, `GreaterOrEqual`, `LessOrEqual`, `And`, `Or`, `Not`, `Where` | fp32/bool; same broadcast rule |
| Cast | `Cast` | fp32 ↔ int32 ↔ bool only |
| Shape (metadata-only) | `Reshape`, `Squeeze`, `Unsqueeze`, `Flatten`, `Identity` | constant target shape; **no data movement** |
| Shape (copying) | `Transpose`, `Concat`, `Slice`, `Gather` | fp32/int32; constant axes/indices where the op allows |

Explicitly **not** claimed in M1, and each with a stated reason in the coverage table: anything
fp16/fp64, anything with a data-dependent shape, `Resize`, `Pad` with non-constant pads, and every
`com.microsoft` op.

**M2 — the compute families that make the EP worth using.**
`MatMul`, `Gemm`, `ReduceSum`/`ReduceMean`/`ReduceMax`/`ReduceMin`/`ReduceProd`, `Softmax`
(last-axis), `LogSoftmax`, `LayerNormalization`, `RMSNormalization`, `ArgMax`/`ArgMin`.
This is the first milestone where a real model (a small MLP, then a small CNN once `Conv` lands)
has a fused region big enough for a speedup to be meaningful rather than dispatch-bound.

### 8.3 The fragmentation rule (superseded — see `OP_COVERAGE.md` §7.2)

> Retained for its reasoning. The enforceable form is Mouse's Minimum Viable Subgraph rule.

A new op is worth claiming when it **connects** existing claimed regions or **extends** one at the
edge. An op claimed in isolation, in the middle of a graph of unclaimed ops, makes the graph
*slower* — two extra device round-trips for one dispatch. Mouse prioritizes by "does this merge
two islands", not by "is this op easy." Niobe's benchmark must report island count and largest
fused region, not just wall time, so this is measurable rather than folklore.

### 8.4 OQ-11 ratification — the five amendments

Ratified **with amendments**. Each amendment is binding on `OP_COVERAGE.md` at its next revision;
none of them requires a respin of the document.

**A1 — Contrib ops: in scope, with per-op discipline.** *Revised 2026-07-28T20:54:42-07:00 by user
ruling.* The domain question is settled above my level: `com.microsoft` is in scope. My amendment
survives the ruling intact because it was never about permission — it is about **how**. The three
parts that bind: an **UNVERIFIED contrib-op row may not be claimed at all** (Mouse's rule kept them
off *exit criteria*; I extend it to claiming, because a contrib op has no opset number to
range-check and a wrong claim there produces wrong logits rather than a decline); **there is no
domain-wide opt-in predicate in the code**, only per-op registry entries with hand-written semantic
gates; and **`tools/graph_census.py` running in CI against pinned artifacts is a precondition on
contrib op #1**, not on tier 3 — it is the only drift alarm we have for a surface with no version
number. The full protocol, including what to do when a schema changes shape under us, is §1.4
C1–C7. Contrib declines flow through the ordinary machine-readable decline path (§1.4 C3); no
special-casing.

**A2 — The template infrastructure is a milestone deliverable with its own exit criterion, not a
preamble.** `OP_COVERAGE.md` §11.1 is explicit that the schedule dies if ops #1–#20 are hand-written
before `indexing.glsl`, the `build.rs` variant generation, the `ops!` macro and the shared claim
helpers exist. I am making that structural rather than advisory: **M1 does not begin until the
template infrastructure is merged, and M1's exit criterion includes the ops-per-hand-written-kernel
ratio (≥ 8 in tiers 1–2) as a reported number.** A ratio that collapses is the earliest possible
signal that the thesis is failing, and it is measurable in week one rather than month three.

**A3 — `OP_COVERAGE.md` §7.2's MVS rule is adopted, with one addition.** The measured transfer-cost
calibration at device init is right, and the ~2 ms cost is acceptable. The addition: **the
calibration result and the MVS decision for every rejected candidate subgraph must be dumpable**
(`ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1`). A partitioning heuristic that silently declines a region is
indistinguishable from a missing op at the console, and we will otherwise spend real time
mis-diagnosing one as the other. Also: the SAFETY factor 3.0 and the `node_count >= 4` /
`64 KiB` floors are **provisional constants that must be re-derived from Niobe's measurements at the
M2 retrospective**, not settled values.

**A4 — fp16 is promoted from "a variant per family" to a tier-3 precondition, and its coverage risk
is escalated.** `OP_COVERAGE.md` §6.1 item 2 is correct that an fp32-upcast LLM path is a
memory-footprint failure, not a slow path. Under the frozen §7.2, `shaderFloat16` and
`storageBuffer16BitAccess` are **probed, not required** — so this is a live product question, not a
detail: if a meaningful fraction of Android lacks 16-bit storage, the LLM story is desktop-first no
matter what the op plan says. This is Mouse's OQ-M2 and I am escalating it to **OQ-14** in §11 with
Link as owner, because it decides a product boundary and not just a shader variant.

**A5 — Shape-agnostic push-constant kernel parameters are authorized for the LLM path from tier 3,
and this is a contract on Switch, not a request.** `OP_COVERAGE.md` §6.1 item 3 / OQ-M1 is right
that KV length growing every token makes this structural. It is also much cheaper to decide now
than to retrofit through every LLM kernel. Ruling: **from tier 3, every LLM-path kernel takes its
dimensions in push constants; the recorded command buffer is length-agnostic for the decode loop.**
The §1.2 non-goal on dynamic shapes is amended accordingly and narrowed to "no dynamic-shape *fast
paths* outside the LLM decode path in M0–M2". Switch must reflect this in `ENGINE.md`'s recording
model alongside the barrier seam. Note the interaction the coverage plan did not call out: a
push-constant-parameterized dispatch still needs a **workgroup count** that depends on the sequence
length, so either the recording is re-issued per shape bucket anyway or we need
`vkCmdDispatchIndirect` with a device-computed count. **I want indirect dispatch evaluated, not
assumed** — it is the same mechanism `QMoE`'s data-dependent routing will need (`OP_COVERAGE.md`
§11 risk 8), so one answer serves both. Assigned to Switch as OQ-15.

**Not amended, explicitly.** Mouse's `ops!` macro with the machine-readable `caps` column
(`OP_COVERAGE.md` §5.7) is a genuine improvement on the MLX reference and I endorse it without
qualification — a hand-maintained support matrix across five vendors, two dtypes and
optional-capability shader variants *is* the "claims support but silently gets it wrong" failure
mode my §8.1 item 2 exists to prevent. Generating `OP_SUPPORT.md` and `--dump-capabilities` from the
same table is exactly right. Likewise the compose-before-bespoke rule (§5.6) with its short fusion
allowlist, the three claim-policy traps inherited from the MLX project's scars (§7.1), and the
metric contract with Niobe (§7.3) are adopted as written.

**Licence dependency, now closed.** Mouse names reading llama.cpp's MIT-licensed Vulkan shaders as
the single largest available accelerant for the three XL kernels. **Rai ruled OQ-M6 🟢 Green on
2026-07-28T19:16:08-07:00** (`.squad/decisions/inbox/rai-oq-m6-license-ruling.md`,
`docs/THIRD_PARTY.md`): reading is permitted with no attribution obligation; substantial adaptation
triggers a file header, a `THIRD_PARTY_NOTICES.md` entry, a commit-message note, and distribution of
the notices file with any binary containing the adapted SPIR-V. My timeline view in §1.5 assumes
this accelerant is used — **read the tiling and subgroup strategies, do not port the code**, and
treat any shader that ends up substantially adapted as triggering Rai's conditions. Without it I
would widen the tier-3/4/5a estimates rather than hold them.

**Clarification, 2026-07-28T22:28:08-07:00 — the accelerant is still available and the timeline
does not widen.** `OP_COVERAGE.md` §13.2 records every XL kernel as an *independent implementation
with no obligation attaching*, which reads at first glance as though the accelerant I priced in had
evaporated. It has not, and the distinction is the same one I stated when I priced it. Mouse's rows
say, in their own text, "read llama.cpp's flash-attention Vulkan shaders for tiling and
subgroup-reduction strategy" and "read `mul_mat_vec_q*` for the memory-access strategy" — that *is*
algorithm study, which is what I assumed and what Rai's 🟢 permits without obligation. What he
declines is **source adaptation**, which I also declined ("do not port the code"), and his reason for
declining it on `MatMulNBits` is stronger than mine: llama.cpp's block formats are not ONNX's, so
adaptation would not even work there. His §13.2 closes by saying the study "removes the largest
single unknown from the XL-kernel estimates". Those are the same position, and no obligation
attaching is the *expected* outcome of using the accelerant correctly, not evidence of its absence.

So: **the estimates hold and I am not widening T3/T4/T5a.** What I would need from Switch's
independent read to keep them — stated now, before his answer arrives, so it is a test rather than a
rationalization:

1. **For `GroupQueryAttention`**: that the flash-attention tiling schedule and the subgroup-reduction
   shape are transferable *independently of data layout* — i.e. that the useful content is the loop
   structure, the online-softmax rescaling order, and what lives in shared memory versus registers.
   If Switch reports the schedules are entangled with ggml's KV layout such that a reader gains
   little, T3 widens.
2. **For `MatMulNBits`**: that dequant-in-register patterns and the memory-access strategy transfer
   even though the block format differs. This is the one I would most expect to be weaker, and it is
   the one Mouse already flagged. If the answer is that the access strategy is a consequence of the
   block format and does not survive changing it, T4 widens.
3. **For `LinearAttention`**: nothing — there is no reference to read, Mouse says so, and my estimate
   never assumed one. This tier's risk is OQ-16 (an unreleased schema), not licensing.

If Switch's read is negative on 1 or 2, I widen the corresponding tier and say so plainly rather
than absorbing it, per §1.5. A timeline that quietly absorbs a lost accelerant is a timeline that
has started lying.

**RULING, 2026-07-28T22:28:08-07:00 — Switch's read has landed and the estimates HOLD. T3, T4 and
T5a are not widened.** I am stating this explicitly rather than letting the estimates stand by
default, because a schedule that survives a challenge by silence has not actually survived it.

Switch (D-S4-10), who read llama.cpp's shader pipeline closely while writing `ENGINE.md`, judges
Mouse's "adaptation would not even work" too strong. His finding, against my pre-committed test
above:

| Test | Switch's answer | Result |
|---|---|---|
| 1. GQA — flash-attention tiling and subgroup-reduction shape transfer independently of layout | **Yes.** GEMV-vs-GEMM tile-size specialisation-constant structure and the per-lane partial dot → `subgroupAdd` reduction shape are layout-independent. | **T3 holds** |
| 2. `MatMulNBits` — dequant-in-register and access strategy survive a different block format | **Yes, partially and usefully.** The ONNX nibble layout genuinely is incompatible with llama.cpp's K-quant structs — no code moves — but dequant-in-register *patterns* and the memory-access strategy transfer as algorithmic reference. | **T4 holds** |
| 3. `LinearAttention` — no reference assumed | Unchanged; there is nothing to read. Its risk is OQ-16. | **T5a holds** (on this axis) |

**The two of them are not actually in conflict, and it is worth being precise about where they
differ**, because the difference is small and the licensing conclusion is identical. Both agree we
write our own code and that **no obligation attaches** — that is settled and Rai's 🟢 covers it.
They differ only on whether *reading* still saves time given the block-format mismatch. Mouse
reasoned from the artifact (the structs are different, so nothing can be copied), which is correct
and is exactly why no obligation attaches. Switch reasoned from the *schedule* (what does a reader
learn that they would otherwise have to derive), which is the question my estimates actually depend
on. **The distinction that resolves it: the block format dictates the innermost unpack, not the
tiling schedule or the reduction shape** — and it is the latter two that cost weeks to get right on
unfamiliar hardware, not the former.

So the accelerant survives in Switch's narrower form, and that narrower form is the one I priced in:
tiling and reduction structure as algorithmic reference, never the quantization layout. Two riders
on holding the estimates:

- **Budget the study time explicitly.** Switch asks that items 1–2 carry real algorithm-study time
  rather than being treated as free. Agreed — "we may read llama.cpp" is not the same as "reading it
  is instantaneous." That reading is inside the T3/T4 estimates, not a discount applied to them.
- **The `MatMulNBits` unpack is ours from first principles.** The one part Mouse is unambiguously
  right about gets no reference and no schedule relief; it is dictated by the ONNX contrib schema.

**General rule from this exchange, since it will recur:** when two owners appear to disagree, check
whether they are answering the same question before adjudicating. Here one answered "can this be
copied?" and the other "does reading it help?" — both correctly, about different things. My error
would have been to widen three tiers on the first answer when my estimates depended on the second.

### 8.5 Op coverage is relative to a **producer**, not to a model architecture

*Added 2026-07-29T08:13:58-07:00. Ratifying and generalizing `OP_COVERAGE.md` §4.18 (Mouse).*

This belongs in the architecture document rather than only in the coverage plan, because it is a
class of error the entire plan was exposed to and it will recur.

**What happened.** Our op inventory was derived from emitted graphs — which is why I ratified it
(§8.1) and still would. But it was derived from emitted graphs *of one exporter*, the ORT GenAI
model builder, and then reasoned about as though it described **"what a Qwen3 graph looks like"**.
When Mouse read Justin's own `justinchuby/onnx-genai-models` (the `mobius` builder) on Justin's
direction, it built the same target models out of a substantially different op set:

| Our table held (contrib) | `mobius` emits |
|---|---|
| `com.microsoft::GroupQueryAttention` | `ai.onnx::Attention` @ opset 23, 6 inputs |
| `com.microsoft::SimplifiedLayerNormalization` | `ai.onnx::RMSNormalization` |
| `com.microsoft::RotaryEmbedding` | `ai.onnx::RotaryEmbedding` |
| fused skip-norm | *nothing* — not emitted |

`MatMulNBits` is the only op both toolchains agree on. Because the registry held only the contrib
column, a Qwen3 built by **Justin's own toolchain** would have declined roughly five nodes per
layer across 28 layers as `[not-registered]` — **for want of a table row, not a kernel.** Every one
of those ops was already implemented or planned. The nodes would have been handed to CPU, the
`largest_island_flops` metric would have been near zero on the artifact that matters most, and the
diagnosis would have looked like an engine problem.

**The generalization, which is now a standing rule.** *A coverage number is meaningless without
naming the producer it was measured against.* "We support Qwen3" is not a well-formed claim; "we
support Qwen3 as emitted by producer P at version V" is. Model architectures are not expressed in
ONNX — *exporters* are, and two exporters targeting identical weights and identical mathematics can
disagree on domain, on fusion boundaries, and on which optional inputs exist. Nothing in this
finding was a misreading: everything was read correctly, nothing was missing from the list, and the
list was answering a narrower question than we thought it was. That is the §10.0.1 failure shape
exactly, and it is why §10.0.1 exists.

**What it obliges us to do.**

1. **The corpus is indexed by producer.** Every artifact in `tools/graph_census.py`'s corpus
   carries the producer and version that built it, and the census reports **per producer**. A
   coverage figure that averages across producers hides precisely the gap this finding exposed.
2. **A target model is only "covered" when it is covered for a named producer**, and the tier exit
   criteria say which. Where two producers disagree, both columns are scope or one is explicitly
   out of scope with a recorded reason — never left ambiguous.
3. **Standard-domain and contrib forms of the same computation get separate claim predicates**,
   even where they share a kernel. Mouse registered `RMSNormalization` sharing
   `simplified_layer_norm`'s handler (asserted by function-pointer identity, which is the right way
   to state "same kernel" so it cannot silently drift) while giving `ai.onnx::Attention` its own
   predicate. **Ratified, and the reasoning is binding:** attribute names, illegal-combination sets
   and optional-input indices all differ, so a single predicate spanning both would be wrong about
   one of them **in the permissive direction** — accepting a node whose semantics it never checked.
   That is the exact asymmetry C2 item 7 was added for. Shared kernels, separate gates.
4. **Prefer the standard domain where a producer offers one.** `ai.onnx::Attention`,
   `RMSNormalization` and `RotaryEmbedding` are opset-versioned, which restores the monotonic
   range-check that §1.4 C2 exists to compensate for the absence of. Every op we can serve from the
   standard domain is an op outside the contrib risk surface. The contrib rows stay — the ORT GenAI
   path is real and external — but standard-domain support is not merely an alternative, it is the
   lower-risk of the two.

**Windowing.** The standard-domain LLM rows are windowed `OPSET_STD_LLM(23)..=OPSET_ANY`. Opset 23
is where these ops enter the standard domain; the open upper bound is consistent with §1.3's
conservative claiming only because the claim predicate validates the node's actual shape rather
than trusting the version — if a future opset revises `Attention`'s attribute surface, the
predicate declines on the attribute, not on the number. That is a load-bearing assumption and it is
worth stating so it can be attacked.

**Producer *and version* — amendment, 2026-07-29T09:47:45-07:00.** Mouse is re-deriving the
producer analysis against `onnxruntime/mobius` (the authoritative repo; Justin corrected the earlier
reference) at its default **opset 24**, and he has been asked to raise with me whether §8.5's rule
needs a version alongside the producer name. **It does, and I am amending it now rather than
waiting**: the rule is *producer **at version***. The evidence is already in hand — the same builder
at a different default opset changes the op set we must serve, which is the identical failure the
rule was written for, one level finer. Concretely: every corpus artifact records producer **and**
producer version **and** the graph's opset imports; the census reports on that triple; and a tier
exit criterion naming a model names all three. A producer name alone would have let an opset-24
`mobius` graph decline against rows windowed for what an earlier `mobius` emitted, and the diagnosis
would again have looked like a missing kernel. Note this also stress-tests the open upper bound
above: opset 24 is exactly the case `OPSET_STD_LLM(23)..=OPSET_ANY` is claiming to handle by
validating shape rather than trusting the number, so it is the first real test of that assumption
and should be treated as such rather than as a formality.

**THE THIRD STRENGTHENING — the rule of record, 2026-07-29T15:02:55-07:00.** Adopting Mouse's
formulation verbatim, because he earned it and because it is stronger than anything above:

> **A claim about what a producer emits is not evidence until it has been read off a graph that
> producer actually produced. Builder source is intent; the model file is the fact.**

**The recurrence is the finding, and I want it on the record as such.** This lesson has now arrived
three times at successively finer grain, and each time we correctly narrowed the claim and still
landed one level short of the evidence:

| Pass | What we corrected | What was still wrong |
|---|---|---|
| §4.18 | We had inventoried against the **wrong producer** (ORT GenAI only) | The new producer was still read from *builder source* |
| §4.19/§4.20 | We were reading the **wrong repository/revision** of the right producer | Still builder source, now at the right revision |
| §4.21 | We pinned the **right producer at the right revision** | **Its output had never been read.** Source is a statement of what a builder means to emit under conditions we were also inferring |

Three iterations of narrowing a claim without once closing the gap between *intent* and *artifact*
is not three mistakes; it is one mistake with a stable shape. Builder source describes a decision
tree; the flags, defaults and model-specific branches that select a path through it are exactly the
part we were guessing at. **The model file is downstream of every one of those guesses.**

Practically, this supersedes rather than extends the two amendments above:

1. **A registry row, a tier criterion, or a coverage figure may cite builder source as
   *motivation*, never as *evidence*.** Where only source has been read, the row is UNVERIFIED
   under Mouse's existing discipline and may not be claimed (C5) or be load-bearing for an exit
   criterion.
2. **The corpus is graphs, not builders.** Every artifact records producer, producer version, opset
   imports, **and the file it was read from**. `graph_census.py` reports over files.
3. **Where no artifact exists for a target, that is stated as the gap it is** — "we have not read a
   graph from this producer" is a specific, closeable status, and it is not the same as "this
   producer emits X".

This is the same epistemics as §9.1's CPU-EP oracle, C2 item 7's fingerprint audit, and §7.9's
capability probe: in each case we had a plausible derived answer and no check against the thing
itself. That is now four instances, which makes it the project's characteristic failure and not a
run of bad luck. **The general form: when a claim can be checked against an artifact, checking it
against a description of the artifact is not checking it.**

**FOURTH STRENGTHENING — 2026-07-30T05:48:29-07:00. Producer-at-version now indexes a *correctness
verdict*, not only a coverage figure.** §8.5 has been about making coverage numbers well-formed:
*"we support Qwen3"* is not a claim, *"we support Qwen3 as emitted by producer P at version V"* is.
The R9 event (§10.0.1) shows the well-formed version is still not a *useful* claim, because on
Phi-3.5 at a named producer-at-version our coverage went **0 → 161 nodes** and the model went from
**correct** (via CPU fallback) to **wrong** (via GPU). Every §8.5 discipline was honoured. The
artifact was read, the producer was named, the version was pinned, the census was over files. And
the resulting number described a regression as an advance.

So the triple carried per producer-at-version is now **gated on `model_output_equivalence`** (§10.0),
and the census reports the verdict in the same row as the triple. Concretely, for §8.5's own rules:
every corpus artifact already records producer, producer version, opset imports and the file it was
read from; it now additionally records, for each run that claimed a non-zero node count, whether
that artifact's outputs matched a CPU-only run of the same session — `MATCH`, `DIVERGENT`, or
`UNMEASURED`, with `UNMEASURED` as the default and never as a silence. **"Covered for producer P at
version V" means covered *and correct* for that artifact; without the verdict it means claimed, and
claimed is a statement about our partitioner, not about the model.** This is §8.5's own lesson
arriving a fifth time in its most expensive form: we checked our claim against the artifact's
*structure* and never against the artifact's *values*.

### 8.6 External crate evaluations — deferred, with named triggers

*Added 2026-07-29T08:13:58-07:00.* Justin directed the team to evaluate his own crates
(*"这都是我们的项目"*). Mouse's evaluations came back mostly negative, and I am recording them here
because **the reasons do not expire and a "no" without a trigger becomes a question that gets asked
every quarter.**

| Crate | Verdict | Reason | Revisit trigger |
|---|---|---|---|
| `onnx-ir-rust` | **Deferred, no expected revisit** | 20% complete by its own status file; use-def tracking commented out in source; no protobuf deserialisation. | None set. Re-evaluate only if its status file changes materially. |
| `onnx-shape-inference` | **Adopted — as an oracle, not a dependency** | Pure Python, so the dependency question does not arise. Run as a preprocessing step in Trinity's harness it resolves symbolic dims, converting `[dynamic-shape]` declines into claims **with zero Rust changes**. | Adopted now. |
| `onnx-genai` / `onnx-runtime-ir` | **Deferred on structural grounds** | Genuinely good, and that is not the issue: **ORT hands us `OrtGraph`/`OrtNode` across a C ABI and we never see a protobuf**, so any external IR means copying the entire graph into a second representation inside someone else's process. | **Adopt the day we need a graph representation that outlives a single `GetCapability` call.** |

*Amendment, 2026-07-29T09:47:45-07:00:* Justin has withdrawn the trust objection to
`onnx-runtime-ir` and Mouse is re-evaluating. **The structural objection is unaffected and stands on
its own merits** — it was never a judgement about the crate's quality or provenance, which is
exactly why it was recorded as a structural fact with a named trigger rather than as a preference.
If the re-evaluation reverses the deferral it must do so by defeating the structural argument or by
meeting the trigger, not by noting that the original objection has weakened; those are different
arguments and only two of the three are reasons.

Two things worth extracting. First, **`onnx-shape-inference` is the cheapest coverage in the whole
plan** — it converts declines into claims without touching a kernel, a predicate or a line of Rust,
and it should be sequenced accordingly rather than treated as harness polish. Second, the
`onnx-runtime-ir` deferral is the right *kind* of "no": it names a structural fact about our
position in the process (we are a guest inside ORT's address space, working from ORT's own graph
view) rather than a maturity judgement that would be obsolete in six months, and it names the
trigger that would reverse it. Deferrals in this project should look like that one.

### 8.7 What template evidence covers — a different *expression*, never a different *path*

*Added 2026-07-29T19:42:07-07:00, adopting Mouse's sharpening (`OP_COVERAGE.md` §7.1.3) as a design
rule because it constrains what §8.4 A2's template infrastructure is allowed to buy.*

§8.4 A2 makes the kernel-template infrastructure a milestone deliverable on the argument that ~87
tier-1 ops are served by ~5 templates. That argument has an obvious and dangerous corollary — *if
one instance of a template is verified, its siblings are verified* — and the corollary is only
sometimes true. The boundary:

> **Template evidence covers a different expression inside an already-exercised code path. It never
> covers a different code path.**

The case that produced it: the attribute-carrying activations (`Elu`, `LeakyRelu`, `Selu` and
friends) satisfy every stated condition for promotion by template evidence, and Mouse declined to
promote them, because they introduce a **push-constant tail** — a new code path, not a new
expression inside an exercised one. **A wrong offset for `params[0]` is invisible to every currently
`Live` op**, since they all push zeros there and read none of them. The verified ops cannot fail in
the way the new ops would fail, so their greenness carries no information about it.

**The operational test, applied before any promotion by template evidence:** *what could be wrong
with the new op that the exercised ops are structurally incapable of detecting?* If the answer is
"nothing", the template evidence is real. If it names anything — a new push-constant, a new binding,
a new descriptor layout, a new dispatch geometry, a new dtype width — that thing needs its own first
execution, and the op is `Staged` until it has one.

**Why this belongs in the design record and not only in `OP_COVERAGE.md`.** Template leverage is the
whole basis of the M1 estimate; it is therefore the place where the project is most tempted to
convert *structural similarity* into *evidence*, and §1.5's split (what we claim vs what we have
verified) is exactly what would be eroded. It is also the same shape as R7's "derive, do not
declare" one level up: **similarity is not a measurement**, and a status derived from a family
resemblance is a declaration wearing a derivation's clothes.

### 8.8 Dynamic shapes are a **claim-path capability**, not a kernel feature — RULING

*Decided 2026-07-29T21:14:03-07:00, on the first end-to-end run of a real model through the EP.*

**The measurement.** Phi-3.5 (2.2 GB, external data, fp16) loaded through the EP on the RTX 4060,
ran, produced 65 outputs, declined every node with a machine-readable reason, fell back to CPU
cleanly, and was bit-identical across two sessions. The decline histogram: **`dynamic-shape` 258**,
`staged` (no kernel yet) 100, `not-registered` 5.

**Read the histogram correctly before ruling on it — the codes are first-match, not a partition.**
`claim_decision_uninstrumented` checks key → opset → contrib schema → **status** → predicate, and
attributes a node to the *first* thing wrong with it. Two consequences, pointing in opposite
directions:

- The `staged: 100` nodes were declined at the **status** check and **never reached the shape
  check**. Their shape viability is *unknown*. Landing all three planned kernels therefore unlocks
  **at most** 100 nodes, and plausibly far fewer, because an unknown number of them are also
  symbolic.
- The `dynamic-shape: 258` nodes had already passed registration, opset, schema **and** status.
  Every one is a node we have a registered, released, non-staged row for. **Shape is the sole
  remaining blocker on 258 of 363 nodes.**

So the asymmetry is not 2.5× — it is **larger than 2.5×**, and the 258 is the only figure in the
histogram that is not an upper bound. Worth stating precisely, because the naive reading understates
the case against the plan we had.

**RULING.** Dynamic-shape handling is **not** a tier-3 kernel feature and **not** an optimisation.
It is a capability of the **claim path and the dispatch path**, it is a precondition for claiming
anything at all on a decoder, and it moves **ahead of** the three planned kernels in sequencing.
Concretely:

1. **The claim predicate's shape contract changes.** A symbolic dimension is no longer *per se* a
   decline. The predicate must distinguish (a) **rank known, extents symbolic** — claimable if the
   kernel takes its extents as runtime parameters; (b) **rank unknown** — decline; (c)
   **data-dependent** extents (`NonZero`, `Unique`, value-dependent `Reshape`) — permanently
   declined, unchanged, and §1.2's next row still governs them. The current `claim::` helpers reject
   (a) together with (b) and (c). **That was correct for a static-shape EP and is wrong for an LLM
   EP** — and it is being amended because it was measured, not because it was argued.
2. **§8.4 A5 is generalised.** "LLM-path kernels take their dimensions in push constants from tier
   3" becomes **"shape extents are runtime parameters for every kernel whose claim depends on them,
   from M1"**. Switch's push-constant path already does this for the elementwise family, which is
   why option (c) may be much closer than it looks for a large share of the 258.
3. **Workgroup count is the open mechanism and it is now on the critical path.** A
   push-constant-parameterised dispatch still needs a group count derived from the extents: either
   the command buffer is re-recorded per shape bucket, or `vkCmdDispatchIndirect` computes it on
   device. That is **OQ-15** (Switch), hereby promoted from "evaluate before tier 3" to **"answered
   before M1's shape criterion can be met"**.
4. **`onnx-shape-inference` does not solve this and must not be allowed to look like it does.**
   §8.6 adopted it as an oracle and it remains the cheapest coverage in the harness — but it
   resolves symbolic dims **statically**, which converts declines into claims *for a fixed-shape
   test artifact*. **In real inference the sequence length genuinely varies per call**, so an EP
   that claims only statically-resolved shapes claims nothing on the second token. A preprocessing
   step that improves our test numbers without making inference work is the purest form of the
   §9.1.2 hazard; §8.6's row is qualified accordingly.

**What this ruling is not.** It is not a licence to widen predicates to reach a number. Report the
**cost** of each option — extents known at `Compile`, known at `Compute`, or fully dynamic in-kernel
— and how many of the 258 each reaches. A node claimed on a shape the kernel cannot actually handle
is R5's permissive direction with a graph-sized blast radius.

### 8.9 Unproven is a claim-path state — RULING: claiming is gated on evidence, and `Live` stops being a thing we write down

*Decided 2026-07-30T06:32:18-07:00, on the R9 event (§10.0.1) and on the state of `main` at
`557bf24`.*

**The situation being ruled on is live, not hypothetical.** As of `557bf24`, anyone who pulls `main`
and points it at Phi-3.5 gets silently wrong answers at full speed: 161 nodes claimed, 161
dispatched, zero failures reported, and `argmax 0`. The tests know. The EP does not. Before this
week that could not happen — the EP declined everything, so it was useless but never wrong.

#### 8.9.1 The third category, and why §7.0 does not cover it

§7.0 is frozen and stays frozen: *"the device gate is minimal; capability shortfalls degrade op
coverage, not device availability."* It contemplates ops we **cannot** run — a missing `shaderFloat16`,
an absent subgroup op — and routes them to a decline. It has nothing to say about ops we **can**
dispatch and have **not proven correct**. That is a third category and coverage work created it:

| Category | Can we dispatch it? | Do we know it is right? | §7.0 verdict | Correct behaviour |
|---|---|---|---|---|
| Incapable | No | n/a | decline | decline |
| Proven | Yes | Yes, on this form | claim | claim |
| **Unproven** | **Yes** | **No** | **silent — the gap** | **decline** |

**RULING, and it is a companion to §7.0 rather than a modification of it — the frozen device gate is
untouched:**

> **§7.0.1 — Evidence shortfalls degrade op coverage, not device availability, and they degrade it
> identically to capability shortfalls. An op we have not proven correct on a form is, for claiming
> purposes, an op we cannot run on that form.**

I am ruling **yes: claiming is gated on proven correctness**, and concretely **`MatMulNBits`
declines until it produces a `MATCH` verdict at its producer-at-version** — see §8.9.5 for what that
costs.

**The argument I am rejecting, stated at its strongest.** An EP that declines is useless; gating
claims on proof risks a design in which nothing is ever claimed for the first time; and every day a
kernel spends unproven is a day of coverage we do not have. That is real and §8.9.4 exists to answer
it. It does not survive contact with the asymmetry:

- **A decline is loud.** It shows up as a claim-rate drop, an island-count rise, a CPU-fallback line
  in the claim log, and — since yesterday — a voided metric triple. Someone notices within a day. We
  have measured this: the 363-node Phi-3.5 decline run produced a decline histogram inside hours.
- **A wrong claim is silent by construction** (R5). It produces numbers. It produced 161 of them.
- **A fast wrong answer is more dangerous than a slow right one precisely because it does not
  announce itself.** A crash is a report. Zeros at full speed are a report of success.

Justin's standing ruling is that **compatibility outranks API elegance**. I read silently-wrong
numerical output as **the most severe compatibility failure available to this project** — an EP that
changes a model's answers has broken compatibility with the only contract a user actually has, which
is *"ORT computes this graph"*. Declining a node keeps that contract perfectly (§2.6, C6). So the
compatibility ruling does not merely permit this gate; it requires it. And §1.3 already said so on
day one: *"the failure mode we are designing against is not 'we didn't claim enough ops' — it is 'we
claimed a node form our shader gets subtly wrong and a user gets silently wrong logits.'"* **That
sentence has been in this document since 2026-07-28 and we shipped its exact failure anyway**, which
is the strongest possible argument that a prose commitment without a mechanism is not a commitment
(§9.1.3, R7).

#### 8.9.2 The mechanism — `Live` stops being written down, and the unit is the dispatchable form

The registry today declares `OpStatus::Live` or `OpStatus::Staged(why)` **in the table, by hand**.
`Live` is therefore a hand-written duplicate of a machine-known fact — *the differential harness
already knows which forms it proved* — and R7's rule of record is **derive, do not declare**: a
hand-written duplicate of a machine-known fact is a fork, and it drifts in the permissive direction.
`Add-i32` carrying `live=True` against an f32-only predicate was the first specimen. `MatMulNBits`
`Live` on a `FLOAT` mask with an f32-only proof is the second, and the second one shipped.

**Ruling on the mechanism:**

1. **The table declares only facts about the source.** `Staged(why)` means *no kernel exists*;
   `Ready` means *a kernel exists*. Both are statements a human can truthfully write, because both
   are about code in the repository.
2. **Claimability is derived, per form, from a proof ledger.** `claim_decision` claims a node only
   if the row is `Ready` **and** the ledger holds an entry under the node's **proof key**. No entry
   means decline, with a machine-readable `[unproven]` decline code that names the missing key. This
   is §7.9 rule 1 and R7 a fifth time: **no entry is a third state, not a permission.**
3. **The ledger is generated by the differential harness, never hand-edited.** It is emitted by the
   run that obtained the evidence, carries the device(s), the ORT build, the tolerance policy applied
   and the artifact or builder the case came from, and is baked into the cdylib at build time. A
   hand-edited ledger fails a regeneration check in CI. A ledger you can write by hand is a
   declaration wearing a mechanism's clothes.
4. **Promotion is automatic and demotion is automatic.** A form becomes claimable the moment the
   harness proves it; **a `DIVERGENT` model-level verdict demotes every form that participated in
   that run back to unproven**, without a judgement call and without a meeting. This is what gives
   R9's red instrument teeth: an instrument that goes red and changes nothing is decoration.
5. **`epctl --dump-capabilities --json` is extended additively.** `status` keeps its current meaning
   and its current strings (*does a kernel exist*); a new per-form `claimable` boolean and `proof`
   object are **added**. Trinity's harness reads `claimable`. Compatibility outranks elegance — we do
   not silently change the meaning of a field five consumers already parse. **The one field that has
   been renamed since is the kernel boolean, `live` → `has_kernel` (§8.9.25 ruling 6, landed
   2026-08-05):** that was not a change of meaning but the removal of one — the row spelled `live`
   twice, as a boolean and as a `status` token, and a schema in which one noun denotes two things
   has no compatibility to preserve.

**The proof key — the granularity, chosen so that yesterday's mistake is unrepresentable rather than
discouraged.** The key is the tuple that selects the dispatched code and the layout of what it reads:

```
(domain, op_type, opset_bucket,
 element dtype of every input and output,
 kernel_variant_key — including any spec-constant value that changes the emitted code,
 shape_class ∈ {static, runtime-extent},          # §8.8
 populated_optional_input_set)                    # R5, packed QKV
```

Per producer at version **only where the form is not fully determined by the key** — the key is a
property of the node, and §8.5's producer indexing attaches to the *model-level* verdict (§10.0), not
to the op-level ledger. Two producers emitting the same key emit the same node, and that is the whole
point of keying on the node rather than on the file.

**Why this key and not "per op" or "per op-form".** Per-op is what we had; it is the granularity that
let a `FLOAT` mask carry an f32-only proof into an fp16 model. The key makes §8.7's distinction
mechanical instead of editorial:

> §8.7 says template evidence covers a **different expression** in an exercised path, never a
> **different path**. Under this ruling: **an expression difference is one that leaves the key equal;
> a path difference is one that changes the key.** The predicate looks evidence up *by key*, so it is
> not possible for evidence about one path to be returned for another. There is no judgement call
> left to get wrong.

Applied to the defect: f32 and f16 differ in `element dtype`, therefore differ in key, therefore the
f32 GEMV proof is **not in the table under the f16 key**, therefore `MatMulNBits`-f16 declines. Had
the ledger existed on 2026-07-29, Phi-3.5 would have claimed zero `MatMulNBits` nodes and run
correctly on CPU, and `test_matmulnbits.py` mentioning `f16` exactly twice would have been visible as
a coverage gap instead of invisible as a correctness one.

**Cost of the key, stated because it is real.** The key space is larger than the row space, so the
proven fraction will be smaller than today's `Live` count and the coverage number will drop the day
this lands. For the template families that is cheap and mostly automatic: `build.rs` already
generates variants per dtype and capability, so the harness can enumerate the key space for a
template family and prove it variant by variant in one run. For the XL kernels it is not automatic
and should not be — an XL kernel proven on one dtype is exactly the situation this rule exists to
name.

#### 8.9.3 Two tiers, because op-level proof is necessary and not sufficient

An op-level `MatMulNBits`-f16 GEMV proof would very likely **not** have caught this defect, which
reproduces at N=161 with 161 descriptor sets and 161 readbacks. So:

- **Tier 1 — per-form op proof.** Gates **claiming** (§8.9.2). Owner: Mouse's predicates, Trinity's
  harness.
- **Tier 2 — per-producer-at-version model proof.** Gates **reporting** — the `model_output_equivalence`
  metric gate (§10.0) and M0 criterion 10. Owner: Trinity.

Neither substitutes for the other, and the interaction is the load-bearing part: **Tier 1 says what
may be claimed; Tier 2 can retract Tier 1.** A form may be op-proven and still participate in a
`DIVERGENT` model run — that is precisely today — and when it does, §8.9.2 rule 4 demotes it. An
op-level proof is evidence about a kernel; a model-level verdict is evidence about the kernel *plus*
the descriptor path, the readback path, the island boundaries and the allocator. The second can
falsify the first and must be allowed to.

#### 8.9.4 The escape hatch — an allowlist of keys, never a switch

Development must be able to run unproven kernels or nothing new is ever written. **The bootstrapping
objection is answered by construction: the ledger is produced by the ordinary differential run, so
the path from unproven to proven *is* the normal development loop, not a ceremony.** A form is
unproven exactly as long as nobody has run the differential on it, and no longer.

**Ruling on the hatch, and it takes C1's shape deliberately.** C1 says: *no domain-wide opt-in may
exist anywhere in the code; the registry key **is** the allowlist.* The same shape applies here.

1. **`ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN` takes a list of proof keys and nothing else.** There is
   no boolean form, no `=1`, no `=all`, no `*`, and no domain or op-type wildcard. **A parser that
   can express "everything" must not exist**, and that is enforced the way C1 is enforced — as a
   test, not a convention: a planted `*`, a planted `1`, and a planted bare op-type must each be
   rejected with an error, and the session must then claim nothing. The list of keys **is** the
   allowlist.
2. **The default is the safe setting and it requires no act.** Unset ⇒ unproven forms decline.
   `UNMEASURED` and *declined* are what you get by doing nothing (this is the same design property
   the coordinator required of Trinity's verdict, applied to claiming).
3. **No build can silently be in the unsafe setting.** Three independent disclosures, because R9's
   whole lesson is that one instrument is silence waiting to happen:
   - the EP logs at **WARN at session creation**, naming every enabled key;
   - the counters/verdict artifact records `unproven_forms_enabled: [...]`, absent meaning empty;
   - **`epctl --check-counters` fails** when that list is non-empty unless `--allow-unproven` is
     passed explicitly, so a CI lane cannot be green while claiming unproven forms.
4. **The hatch is available in release builds.** Not because it is nice, but because Trinity's and
   Link's lanes run release builds and a mechanism that only works in debug is a mechanism the
   evidence path cannot use. **Availability is not the risk; silence is** — and item 3 removes the
   silence. This is the compatibility ruling again: a feature-gated hatch would fork the shipping
   artifact from the tested one, which is a worse failure than the one it prevents.
5. **A run that enables a key is `UNMEASURED` for every key it does not measure.** Enabling a form in
   order to prove it is the intended use; enabling a form and then reporting the triple is not.

#### 8.9.5 The cost, stated without smoothing

**On the morning this lands, Phi-3.5's claimed-node count goes 161 → 0.** `MatMulNBits` has no f16
proof, so it declines; nothing else on that graph is claimable; the coverage member of the metric
triple for Phi-3.5 at its producer-at-version reads **zero**. That is a visible regression in a
number Justin has been watching, and I am not going to present it as anything else.

Two things are true about that zero and both should be said:

1. **It is a regression in the reported number, not in the code.** The kernels are exactly as good
   tomorrow as today. What changes is that we stop counting nodes we cannot vouch for.
2. **Per my own metric gate (§10.0), the 161 was already void.** The verdict on that run is
   `DIVERGENT`, and a `DIVERGENT` verdict voids the triple rather than discounting it. **The honest
   number was already zero; the only thing this ruling changes is that the EP's behaviour now agrees
   with the reporting instead of contradicting it.** A 161 that voids to zero on inspection and a 0
   are the same fact; one of them is just harder to misread.

I would rather hand Justin an honest zero than a dishonest 161, and I want it recorded that I said so
on the day the number went down rather than afterwards.

#### 8.9.6 Where I expect to disagree with Rai, and what I will do about it

Rai is being asked in parallel whether silently-wrong numerical output from an inference EP is a
responsible-AI concern in its own right. **My ruling does not depend on his answer**, and I want the
independence on record so that a later reader cannot mistake agreement for corroboration — which is
R9's exact failure mode applied to people instead of counters.

- **If Rai rules it *is* an RAI concern:** the ruling is unchanged and gains a second, independent
  justification. Per R6 rule 1, the load-bearing reason stays the engineering one — the claim/decline
  asymmetry — so that if the RAI framing were ever withdrawn the gate would not move.
- **If Rai rules it is *not* an RAI concern:** **the ruling still stands**, on §1.3 and on the
  compatibility ruling, and the disagreement is recorded here rather than resolved into a
  compromise. An EP that silently changes a model's answers is a defect whether or not it is also a
  responsible-AI finding.

#### 8.9.7 Rai's verdict — convergent, and it names one gap this ruling did not close

*Added 2026-07-30T06:32:18-07:00, after Rai's independent verdict landed in `2c35eac`.*

Rai returned **RAI-008 🔴 Critical** — *the architecture permits silently-wrong inference output at
any layer with no disclosure* — and **RAI-007 🟡 Advisory** for the fp16 kernel itself, correctly
separating the instance from the class. **We converged, and I want it stated that we converged
independently rather than agreed**: his load-bearing reason is the autoregressive amplification (one
zeroed-logit dispatch produces an unbounded stream of fluent wrong tokens, indistinguishable to a
user from "this model is bad"); mine is the claim/decline asymmetry and the compatibility ruling.
Per R6 rule 1 the ruling's load-bearing reason stays the engineering one, so §8.9 does not move if
the RAI framing is ever revised. Two independent arguments reaching the same gate is worth more than
either — but only because they are *different* arguments, which is R9 read the right way round: what
makes a second reading evidence is that it could have come out differently.

**RAI-009 is the part I did not cover, and he is right.** *"There is no runtime WARNING when claimed
forms carry UNMEASURED proof status"* — §8.9.4 item 3 discloses only when the **escape hatch** is
enabled, and §9.1.3's verdict lives in a counters file a user never sees. A user who receives a
session object and calls `Run` observes nothing either way. Under §8.9 the unproven forms are no
longer claimed, so the acute hazard is closed by the gate; the residual is that **the user still has
no positive statement of what was claimed and on what evidence**, which is R9 aimed at the user
instead of at us: silence reads as fine.

**Closing it, as an addition to §8.9.4 rather than a separate mechanism:**

- **At session creation the EP logs, at INFO, one line per claimed form: the proof key and the
  ledger entry backing it** — device, ORT build, artifact. A user can see what we vouched for and on
  what basis, without reading a counters file.
- **At WARN if any claimed form's evidence is `UNMEASURED`** — which under §8.9 can only happen via
  the escape hatch, so the two disclosures are the same mechanism at two severities and there is no
  second thing to maintain.
- **If the EP claims zero nodes, it says so at INFO**, naming the top decline codes. "The EP claimed
  nothing" is a finding a user is entitled to, and it is the state §8.9 will produce on Phi-3.5
  tomorrow morning.

This is disclosure, not a gate, and it must not be mistaken for one: **a log line is an instrument
with no red state** (R9), so it may never substitute for the ledger. It exists because the ledger
protects the user from us, and this tells the user that it did. Owner: Tank at session creation,
Mouse for the per-form data. Tracked as part of M0 criterion 11.

---

#### 8.9.17 RULING — the digest's frame gains the compiler, demotion splits `SUBJECT-CHANGED` from `TOOLCHAIN-CHANGED`, and the device belongs in the **predicate**, not in the key (2026-08-02)

*The §8.9.x series is shared with `OP_COVERAGE.md`, which carries 8.9.8–8.9.16. This is a DESIGN-level
ruling and lives here. Two questions were put to me together and they are not the same question.*

**(1) THE TWO DEMOTIONS ARE NOT THE SAME EVENT AND MUST NOT SHARE A TOKEN.**
`registry.rs::shader_digest_for` hashes the compiled SPIR-V bytes of exactly the stems a proof run
dispatched. Its declared frame — stated in the function's own doc comment — names the formula, the
index expression, the workgroup size, the binding order, deletion and rename. **The compiler is on
none of them, and the compiler is in the digest.** A `glslc` upgrade today moves all 97 digests with
no kernel change and no source change, and `parse_ledger` reports that in the same words it uses for a
kernel that was replaced: `STALE-SHADER`, *"the entry describes a kernel that has been replaced"*.

**Do not narrow it.** Hashing GLSL source instead would remove the false alarm and go blind to a
compiler that miscompiles correct source, which is a real correctness event and one this project has no
other instrument for. **The digest is over-broad, not fabricated, and the breadth is protective.**
Declare, do not narrow. The compiler enters the declared frame, and the single demotion splits in two:

| Token | Condition | What it means | What it licenses |
|---|---|---|---|
| **`SUBJECT-CHANGED`** | the entry's stems are all present, the toolchain identity is unchanged, and the digest moved | **the proof is invalid.** The code that was measured no longer exists | Re-prove. Nothing else discharges it. The entry is gone, not stale |
| **`TOOLCHAIN-CHANGED`** | the digest moved and the recorded toolchain identity differs from this build's | **the proof may well still hold; the evidence that it applies here does not.** A different compiler emitted different bytes from the same source | Re-prove **or** re-establish the port by the cheap per-device/per-toolchain invariant (§8.9.17(4)). The claim it made about the *form* is not withdrawn |

**Both demote. Neither becomes claimable.** The difference is not permissiveness, it is repairability
and it is diagnosis: *a subject change invalidates the proof; a toolchain change invalidates the
evidence that the proof still applies*, which is a weaker and differently-repairable thing. The
operational requirement is blunt: **a run that faults 97 entries on a `glslc` upgrade must not look
like 97 kernels changed.** Today it looks exactly like that, and the first person to see it under time
pressure will reach for the narrowing this ruling forbids.

**Discharge condition, stated in advance so it cannot be renegotiated:** the entry records a
**toolchain identity** — `glslc --version` output, the SPIR-V target env, and the optimisation flags —
as its own field; `parse_ledger` compares it before it compares the digest; and the two tokens appear
in `Ledger::demoted` and in the counters vocabulary as distinct strings. A single token with a longer
message is not a discharge. **A distinction that exists only in prose is not carried by any predicate,
which is the defect this whole section was written about.**

**(2) A DEFECT FOUND WHILE RULING, AND IT IS LARGER THAN THE RULING.**
`parse_ledger`'s own comment on the staleness branch says: *"A stale entry demotes ITSELF and nothing
else. Making it a global fault would let one shader edit disable every claim in the artifact, which is
the blunt shape that gets relaxed the first time someone is in a hurry."* **The code does the opposite
of its comment.** The `STALE-SHADER` branch pushes to `demoted` **and** to `faults`; `Ledger::get`
returns `None` whenever `faults` is non-empty, and `lookup_key` returns `Faulted` on the same
condition. **One stale entry disables all 97 claims.** `LedgerEntry::shader_digest`'s own doc says
*"a disagreement demotes this entry and only this entry"*; `Ledger::demoted`'s doc, thirty lines away,
says *"the matching fault above already makes the whole ledger unusable."* Two doc comments in one
file, in contradiction, with the code implementing the one that was not intended.

**This is measured, not inferred.** `bench/results/census/census-counters-dev0-ledger_digest_drift.json`
reads `ledger_entries: 97, ledger_faults: 1, ledger_gate: "FAULTED", ledger_miss: "LEDGER-FAULTED",
ledger_hits: 0, unproven_declines: 1` — one drifted entry, every claim declined.

**RULING: the code is wrong and the comments were right.** A stale entry demotes itself. Staleness
belongs in `demoted`; only a *structural* problem — an unparseable line, a header digest mismatch, a
baked-vs-disk disagreement — belongs in `faults`, because those are the states in which the ledger
cannot say anything about any key. **The distinction is exactly `LedgerLookup`'s own: `KeyAbsent` is a
finding about the form, `Faulted` is an instrument failure (R13). A stale entry is a finding about
one form and is currently reported as an instrument failure about all of them.** Owner: Mouse. This is
a prerequisite for (1): splitting a token is pointless while both halves fail the whole artifact.

**(3) THE DEVICE BELONGS IN THE PREDICATE. IT DOES NOT BELONG IN THE KEY — AND `device0` DOES NOT
BELONG IN THE LEDGER AT ALL.**
I have already ruled that **a proof is a property of a form on a device**, with `PROVEN` /
`PROVEN-ELSEWHERE` / `UNPROVEN` replacing a two-state predicate. Link's finding sharpens the
question: the field is on 97 of 97 entries and **no predicate reads it** — not `Ledger::get`, not
`lookup_key`, not `ledger_contains`. **A field no predicate reads is not a guard; it is a comment with
a schema.**

The device does **not** go into `ProofKey`. A key is the identity of the *form*, and putting the
device in it makes the ledger a set of per-machine fingerprints: ninety-seven entries per GPU, a new
device unusable until it has been re-proved, and `KeyAbsent` returned for a form that is perfectly
well proved. That destroys the ledger's one portable function. **The device goes into the
predicate**, which reads the entry's frame *after* finding it and returns the third state. Same
lookup, one more comparison, three answers instead of two.

And a second thing must change with it, or the guard is worse than its absence. **All 97 entries
record `"device": "device0"`, which is a selector ordinal, not a device identity** —
`rust/tools/gen_proof_ledger.py` writes `device = args.device_name or f"device{args.device}"`, and
`--device-name` was not passed. `device0` is the Iris Xe on one box and the RTX 4060 on another; this
document's own §10 records a day when every device label written was inverted. **A predicate that
compared `device0` to `device0` would return `PROVEN` across two different vendors and would look
exactly like a working guard.** That is the same failure as today's, wearing a check's clothes, and it
is the more expensive version because it is harder to see.

**RULING:** the frame field is `vendor_id:device_id:driver_version` plus the human-readable device
name, taken from the physical-device properties of the run that produced the proof — never from the
selector. `criterion3a_phi35-dev0.json` already models the discipline exactly and gives the reason in
its own text: the device name is read from `counters.alloc_device_frame_session_devices`, *"not the
`ONNXRUNTIME_EP_VULKAN_DEVICE` selector, which is a request and whose number is not the allocator's
enumeration index."* An entry whose device field cannot be resolved to an identity is
`PROVEN-ELSEWHERE` at best, never `PROVEN`, and the generator should refuse to write it.

**(4) WHAT PROMOTES `PROVEN-ELSEWHERE`, AND WHY THIS IS NOT A COST.**
Real device dependence is subgroup width, fp16 rounding and driver behaviour, and my ruling of
2026-08-02T15:15 established that those move a residual **by ULPs**. So the ULP series is the cheap
per-device invariant: **the expensive differential establishes the form; the cheap invariant
establishes the port.** The same instrument now promotes `TOOLCHAIN-CHANGED`, for the same reason and
with the same cost — a compiler difference that changes numerics changes them by ULPs, and one that
changes nothing shows a flat series. That is the second half of why (1) is a declaration and not a
relaxation: **`TOOLCHAIN-CHANGED` names a repair path that `SUBJECT-CHANGED` does not have, and both
of them refuse the claim until the repair is done.**

**(5) THE ORDER, BECAUSE THE ASYMMETRY SETS IT.** A digest disagreement **fails safe** — loud,
wasteful, and nobody is harmed. A device mismatch **fails open** — silent, unwatched, and the user
gets an answer from a kernel nothing has measured on their hardware. So: the device predicate first,
the demotion split second, and the `faults`/`demoted` correction in (2) before either, since it is
what makes a per-entry demotion mean anything at all.

---

#### 8.9.18 RULING — `PROVEN-ELSEWHERE` loses its promotion licence and keeps its disclosure licence; a stale entry demotes itself; and "decline" is a naming convention, not a measurement

*2026-08-03, against `main` = `7d6f57b`. Three questions put to me: one refutation of my own reasoning
from Fact Checker, one finding about my own bookkeeping, and one routing decision for Mouse. Numbering
note as in §8.9.17: the `8.9.x` sequence is shared between this document and `OP_COVERAGE.md`
(8.9.1–8.9.7 and now 8.9.17–8.9.18 here, 8.9.8–8.9.16 there). I take 8.9.18 and say so.*

---

**PART 1 — THE REFUTATION IS UPHELD. The cheap invariant does not reach a per-form key, and I should
not have said it did.**

The claim under attack is mine, in §10.0.1 R12: *"the expensive proof establishes the form; the cheap
invariant establishes the port."* Fact Checker's counter is that **model-level ULP evidence cannot
promote unexercised per-form keys.** It is correct, and the shortest demonstration is that the rule it
invokes is already written down in the code I was reasoning about. `ProofKey::from_node`'s doc comment:
*"the lookup is by key, so evidence about one path cannot be returned for another."* A model-level ULP
series is not evidence about *another* path. **It is evidence about no path** — its subject is the
composed graph, and its per-element records are indexed by *model output*, not by proof key. Look at
what the artifact actually holds: `criterion10-dev0.json` carries `ulp_curve_median_over_outputs` and
`oracle_failing_indices = [0, 63, 64]`. Those are output ordinals. There is no function from an output
ordinal to a proof key, because output 0 is downstream of every form in the model; that is exactly why
its 12-ULP step could not be attributed to anything when I found it.

**And the "unexercised" half is worse than the "exercised" half, with the arithmetic in-tree.**
`wiring_census-dev1.json` records `proven_key_lookups=6 ledger_hits=6 ledger_entries=95`. One model run
on the second device consults **six keys**. Under the withdrawn paragraph, one clean ULP curve from that
run would have promoted **ninety-five** entries — eighty-nine of which the run never touched, could not
have touched, and about which it emitted not one bit. That is not a weak inference. It is the vacuous
pass in its purest form: **a promotion whose evidence is invariant under the thing it claims to
establish.** §9.1's own trap, one level up, and I wrote it into the risk register while quoting R9 in
the sentence before.

**What survives, and I want the distinction sharp because the state itself is not damaged.**
`PROVEN-ELSEWHERE` was argued for on two legs and only one of them broke:

| leg | claim | status |
|---|---|---|
| the fatal-horn argument | refusing to claim on a second device means declining everything on it, *or else* extrapolating silently — and today the silent extrapolation is what actually happens, indistinguishable in every artifact from a proof obtained here | **stands, untouched.** It is an argument about disclosure and it needs no promotion path at all |
| the cost argument | the port is re-established cheaply, so naming the gap is not a permanent tax | **withdrawn.** There is no cheap per-form instrument in this tree today |

**So the ruling is: `PROVEN-ELSEWHERE` is a disclosure state, not a staging state.** It licenses one
thing — that a run on a device with no matching entry says so, by name, in its own artifact — and it
licenses nothing about becoming `PROVEN` later. An entry reaches `PROVEN` on a device the way the first
one did: by evidence keyed to that form on that device.

**The mechanism that would reach a per-form key, stated so nobody has to invent it under pressure.**
Not a model run. A **per-key replay**: for entry *E*, re-execute *E*'s own recorded case — `artifact`
names it, `tolerance` names the comparison, `shaders`/`shader_digest` name the subject — on the running
device, and compare against the reference the entry was proven against. This is cheaper than the
original proof because the expensive parts are already paid for and stored: constructing the case,
running the CPU oracle, choosing the tolerance, plumbing the witnesses. The port pays only for
re-execution and comparison. **It is per-key by construction, so it can only ever promote keys it
actually ran**, and eighty-nine untouched keys stay `PROVEN-ELSEWHERE` — which is the correct outcome,
not a failure of the mechanism. I am not asking for this to be built now. I am recording it so that the
next person who wants promotion has a shape to build rather than a gap to fill with a model run.

**And the ordering the coordinator has already given Mouse is right and I endorse it rather than
merely permit it.** `PROVEN-ELSEWHERE` means "proven on another device". §8.9.17 established that this
ledger cannot tell devices apart: `LedgerEntry::device` is read by no predicate, and
`gen_proof_ledger.py` writes `args.device_name or f"device{args.device}"`, so the recorded value is a
**selector ordinal**. A state whose definition turns on a distinction the artifact cannot make is not
under-implemented, **it is undefined** — and implementing it first would have produced a predicate
comparing `device0` to `device0` and returning `PROVEN` across two vendors while looking like a guard.
Specify, make `device` load-bearing, then implement. That order is forced by the finding, not chosen
for tidiness.

**One thing I will not do, and it matters that it is on the record.** Fact Checker is advisory and
cannot overrule me, which is precisely why "advisory" must not become "declined by the author whose
reasoning it examined". I upheld this because the argument is right — the key doc comment and the
six-of-ninety-five census are not opinions — and I would have said so with no authority attached to it
at all.

---

**PART 2 — "DECLINE" IS A NAMING CONVENTION. The count is real and it measures the wrong thing.**

The finding: six self-counted declines exist, but the count **excludes other explicitly unnumbered
obligations, so it measures numbering rather than register growth.** I accept it without qualification,
and I will go further than the finding does, because the ⚠️ beside it — *some principles were
re-derived* — is not a separate observation. **It is the proof.** A principle that had to be re-derived
is a principle that was in the record and could not be found. That is what an unnumbered load-bearing
sentence costs, measured, in this project, already.

**The policy, stated so it binds me rather than describing me.**

1. **A rule is anything cited as binding by someone who was not in the conversation that produced it.**
   Not anything I intended as a rule. Intent is unobservable and it is mine; citation is observable and
   it is someone else's. This is the same move as §8.9.17's device finding — the test of a guard is
   whether a predicate reads it, not whether the author meant it as one.
2. **Numbering follows citation.** An unnumbered sentence of mine that is cited twice as binding must be
   given a number or withdrawn, and the choice is not optional or deferrable. "It is in a ruling
   somewhere" is not navigable, and an agent who has never spoken to me cannot be expected to grep six
   thousand lines of prose for a sentence whose wording they do not know.
3. **A decline counts only if the principle stayed out of the record**, in every form. If I decline to
   mint R14 and then write the same obligation into a ruling as prose, that is not restraint. It is the
   register growing off-book, and it is the exact failure the count was supposed to detect. Under this
   definition some of my six declines may not survive re-examination, and I would rather that be found
   than protected.
4. **I am not the one who counts.** This is the structural half of the complaint and it is the half that
   cannot be answered by better behaviour on my part. I am sole author and sole judge of the register,
   and the coordinator who brought me every candidate had an interest in there being fewer rules to
   enforce — so the two people in the loop were biased the same way. **Authorship stays with me; the
   tally does not.** Fact Checker holds the count, derives it from citations rather than from my
   history file, and publishes it where I do not edit it. I asked to be checked on whether the register
   was under-growing; the answer to "your own tally says no" cannot be another tally of mine.

**What this does not concede.** The ✅ finding — *no principle was lost* — is the one I was actually
worried about when I asked, and I take it. The register is under-**numbered**, not under-**populated**.
That is a navigability defect, which is repairable by numbering, and not an integrity defect, which
would not have been.

---

**PART 3 — `parse_ledger`: PER-ENTRY DEMOTION IS CORRECT. The whole-ledger fault is right for a
different class of damage and must be kept for that class only.**

Routed to Mouse. The comment and the code disagree; the comment is right. Both doc comments state their
own position clearly, so this is a real disagreement and not an oversight:

- `LedgerEntry::shader_digest`: *"Recomputed at parse time; a disagreement demotes this entry and only
  this entry."*
- `Ledger::demoted`: *"A demotion **grants nothing** — the matching fault above already makes the whole
  ledger unusable."*

**The principle that resolves it: fault scope is set by the scope of what you cannot locate, not by the
severity of what you found.** A stale `shader_digest` is a *precisely located* fact — this key, this
shader set, this recomputed hash. Escalating a located fault to a global one is not conservatism, it is
**discarding the localisation you already have**, and 96 sound entries are destroyed to punish 1
drifted one. The code already does the correct thing before it does the incorrect one: the `continue`
in the `Some(now)` arm means the stale entry is never pushed into `entries`, so it is unfindable by
`Ledger::get` on its own. The `faults.push` beside it adds nothing to the safety of that key and
removes every other key.

**The decisive argument is the one §8.9.17 created this morning, and it is why this cannot wait.**
`TOOLCHAIN-CHANGED` is **ledger-wide by nature** — a `glslc` upgrade changes every module's bytes at
once. Under the current code, therefore, *every future toolchain bump is a total ledger fault*: 97
entries demoted, `ledger_gate: FAULTED`, `ledger_hits: 0`, every form declining, for a change that
touched no kernel. `parse_ledger`'s own comment predicts what happens next — *"the blunt shape that
gets relaxed the first time someone is in a hurry"* — and the artifact already exists showing the
mechanism firing on a single entry: `bench/results/census/census-counters-dev0-ledger_digest_drift.json`
reads `ledger_faults: 1`, `ledger_gate: FAULTED`, `ledger_hits: 0`. **A fail-safe that is guaranteed to
fire spuriously on a routine maintenance action is a fail-safe with a scheduled date for being turned
off.** That is how this one becomes fail-open, and the date is whenever someone next upgrades a
compiler under deadline.

**The line, drawn where the localisation is:**

| damage | can you say which claims it touches? | scope |
|---|---|---|
| `STALE-SHADER` — recomputed digest ≠ recorded digest | yes: this key | **demote this entry** |
| `NO-SUBJECT-WITNESS` — empty shader set or missing digest | yes: this key | **demote this entry** |
| absent or zero `claimed_nodes`/`dispatches_executed` | yes: this key | **demote this entry** |
| header digest ≠ recomputed body digest | no — the file was hand-edited and *any* line may be affected | **fault the ledger** |
| `declared_count` ≠ parsed entries | no — an entry may have been silently dropped or added | **fault the ledger** |
| duplicate key | no — two entries disagree and neither is authoritative | **fault the ledger** |
| a line that does not parse | no — you cannot tell what it was going to say | **fault the ledger** |

The invariant: **the top group is "this proof is not usable"; the bottom group is "this artifact is not
readable".** Those are different findings and collapsing them costs 96 sound proofs for one located
defect, or — under `TOOLCHAIN-CHANGED` — all 97 for a compiler upgrade.

**Two obligations attach to the correction, or it becomes a weakening.** First, a demoted entry must be
*visible*, not merely absent: `Ledger::demoted` and `Ledger::demotion_for` already carry the vocabulary,
and §8.9.7's session disclosure must print the demotion count and its reasons on every run, so that "96
of 97 proofs are live" is a sentence a reader sees rather than a state they infer. Second, **the count
of demotions is not allowed to be zero-by-construction**: the drift census above is the positive case,
and it must stay a test, because a demotion path never observed in its firing state has no demonstrated
firing state — which is the same rule Niobe is now being held to on the amplification probe.

---

#### 8.9.19 RULING — a Linux run may claim `PROVEN-ELSEWHERE(toolchain)`; the toolchain belongs in the frame, never in the key; and the subject needs two digests because one cannot be both sensitive to the kernel and blind to the compiler

*2026-08-03, against `main` = `152de11`. **Blocking**, not documentary: Link's Linux lane compiles and
runs, four of seven blocked steps pass, and the remaining three are one cause — Ubuntu's shaderc 2023.8
against the Windows SDK's v2026.2 — which faults every ledger entry. He established it is the ledger and
not the platform by perturbing one GLSL template **on Windows** and reproducing the same eleven test
names plus one, a superset. Numbering note as in §8.9.17–18: `8.9.x` is shared with `OP_COVERAGE.md`; I
take 8.9.19.*

---

**PART 1 — THE SCHEMA, because questions 1 and 2 cannot be answered separately and the coordinator is
right that two rulings must become one schema.**

I have now ruled twice on what belongs where and the two rulings need a single structure or they will be
applied as two. Here it is. **A ledger entry has three parts and they answer three different questions:**

| part | question it answers | contents |
|---|---|---|
| **KEY** | *what was proven* — the **form** | `ProofKey::from_node`: `domain::op_type/opset_bucket/dtypes/variant/shape_class/populated_inputs`. **Nothing else, ever.** |
| **SUBJECT** | *what code was proven* | the shader set, plus digests over it (part 2) |
| **FRAME** | *under what conditions the proof was obtained* | device identity, driver, `ort_build`, **toolchain identity**, tolerance |

**The rule that generates every answer below: you look up by KEY; you compare FRAME after you have
looked up; a SUBJECT mismatch means the proof is about something else.** A component of the frame in the
key turns "I have a proof that does not apply here" into "I have no proof", which are different facts
with different repairs, and the second one is unactionable.

**So the answer to question 2 is: no. The ledger must not be keyed per toolchain, and it is not keyed
per device either — those are the same error twice, and it is the same error I ruled on in §8.9.17.**
The device does *not* go in the key; it goes in the frame and is read by the predicate. I said "the
device belongs" and I should have been more careful, so I am being exact now: **the device belongs to the
entry, and to the predicate. It does not belong to the key.** Same for the toolchain. One schema, and
both of my earlier rulings are instances of it rather than exceptions to it.

**And "keyed per toolchain" is not a design choice anyone made — it is a mechanical accident, and I can
name the line.** In `parse_ledger`, the `Some(now)` arm of the `shader_digest_for` comparison pushes a
demotion and then `continue`s, so the entry **never enters `Ledger::entries`**. `Ledger::get` therefore
returns `None`, and `lookup_key` reports the same `LedgerLookup` token it would report for a form nobody
ever proved. **A frame mismatch is currently indistinguishable from a key absence.** That is why Linux
reads as "97 forms were never proven" rather than "97 proofs were obtained under a different compiler".
The repair is the same repair §8.9.18 part 3 already ordered, one level up: **the entry must survive
parsing and carry its mismatch**, and the predicate — not the parser — decides what that mismatch
licenses.

**The status lattice. One state for "proven, out of frame", carrying an enumerated delta — not a growing
enum of combinations.**

| reading | condition | claimable |
|---|---|---|
| `PROVEN` | key present, `MATCH`, witnesses present, **subject identical**, **frame identical** | yes, silently |
| `PROVEN-ELSEWHERE{δ}` | key present, sound, subject identical, **frame differs in exactly the components named in δ** — δ ⊆ {`device`, `driver`, `ort_build`, `toolchain`} | **yes — counted, disclosed, and δ printed in the run record** |
| `UNPROVEN{SUBJECT-CHANGED}` | the subject moved | no |
| `UNPROVEN{NO-SUBJECT-WITNESS}`, `{UNATTRIBUTED}`, `{KEY-ABSENT}` | as today | no |

`PROVEN-ELSEWHERE` **generalises** rather than acquiring siblings. A Linux CI run on lavapipe differs
from the Windows proof in *both* device and toolchain; under a per-combination enum that is a new status,
and the enum grows as a product. Under a delta set it is `PROVEN-ELSEWHERE{device, toolchain}` and the
reader learns more, not less. **This is §10.0's standing obligation — every way of not knowing gets a
name a machine can print — applied to a set rather than a scalar.**

**So: a Linux run may claim, and every claim it makes says out loud that it is claiming out of frame.**
That is the answer to question 1. `TOOLCHAIN-CHANGED` is **not** a demotion to `UNPROVEN`, because
demoting it would mean Linux declines all 97 forms and produces no op-correctness number at all — which
is today's state and is the thing being unblocked. Nor is it silently `PROVEN`, because a different
compiler under `-O` can legitimately produce different arithmetic and the whole point of §8.9.17 was that
the toolchain was on nobody's declared list.

**And the licence has the same limit §8.9.18 part 1 imposed, restated so it is not re-derived later:
`PROVEN-ELSEWHERE{toolchain}` is not promoted by a model run.** It is promoted **per key**, and the
instrument that promotes it is the very thing Link is unblocking: `tests/ops` **is** a per-form
differential against the CPU oracle. Each key the Linux op suite passes earns a Linux entry on its own
evidence. **This ruling is self-discharging** — it grants exactly enough claim for the suite to run, and
the suite is what removes the need for the grant.

---

**PART 2 — THE SUBJECT NEEDS TWO DIGESTS, AND SAYING WHAT EACH IS BLIND TO IS THE POINT OF HAVING TWO.**

The coordinator names the trap exactly: the cheapest way to say "same form, different compiler" is to
compare the GLSL source text, and Link found that **a comment-only edit does not change the SPIR-V
digest** — so source text and digest already disagree about what "the same shader" means.

**The honest statement first: no single hash can be sensitive to the kernel and blind to the compiler,
because the compiler is a function whose output is the only thing that actually runs.** Stop looking for
one. **Take two, and let their disagreement be the instrument.**

| digest | covers | **blind to** | over-sensitive to |
|---|---|---|---|
| `spirv_digest` — today's `shader_digest_for` | the exact bytes dispatched | comment-only GLSL edits (correctly — they do not survive `glslc`); host-side code (already a named residual in its doc) | **the compiler version and its flags** — the defect |
| `source_digest` — new | the `.comp` text, **every file reachable through the `-I` include directory**, the `shader_variants.txt` row (stem, source, specialisation assignments), and the `glslc` argv **excluding the compiler binary and its version** | **compiler behaviour entirely** — a miscompilation, an optimiser difference, a codegen bug | comments and whitespace |

Note the include closure and the argv are not decoration. `rust/build.rs` invokes `glslc` with
`-fshader-stage=compute --target-env=vulkan1.1 -O -I<include_dir>`; a `source_digest` over the `.comp`
file alone would be blind to an edited include and to a changed `--target-env`, which are subject
changes wearing a toolchain costume.

**The decision table, and the fourth row is the one that proves the pair is doing work:**

| `spirv_digest` | `source_digest` | reading |
|---|---|---|
| same | same | `PROVEN` — nothing moved |
| **differs** | **same** | **frame delta `toolchain`** — same source closure, different compiler output. Claimable, disclosed. **This is Linux.** |
| differs | differs | `UNPROVEN{SUBJECT-CHANGED}` — the kernel moved. No claim. |
| same | differs | `SOURCE-COSMETIC` — an edit that produced identical SPIR-V. **`PROVEN`, and *named*.** |

**What the pair is jointly blind to, stated because a ruling that does not name its residual is the thing
I grade other people for.** Three things, and I claim none of them are closed by this:

1. **A compiler bug.** Row 2 licenses a claim whose evidence was produced by a different compiler. If
   shaderc 2023.8 miscompiles a kernel that v2026.2 gets right, the disclosure names the risk and the
   op suite catches the numerics; nothing in the digest pair catches it. **That is the whole reason row 2
   is disclosed rather than silent.**
2. **Host-side change** — already named in `shader_digest_for`'s own doc and not improved here.
3. **Specialisation values chosen at runtime** rather than at build time. The variant row is hashed;
   a runtime-chosen constant is not. Switch's `IsInf`/`Clip` work makes selectors specialisation
   constants, so this residual is growing and somebody should own it.

---

**PART 3 — WHAT MOUSE IMPLEMENTS, IN ORDER, WITH THE MINIMUM THAT UNBLOCKS LINUX MARKED.**

1. **Entry survival.** A subject-or-frame mismatch must not delete the entry from `Ledger::entries`.
   Record the mismatch on the entry; leave the entry findable. Without this every status below is
   unreachable, and this is also §8.9.18 part 3's ordered fix. **Blocking.**
2. **`source_digest`.** Emitted by `rust/build.rs` over the source closure defined in part 2; baked
   alongside the SPIR-V digest; recorded by `rust/tools/gen_proof_ledger.py`. **Blocking** — it is what
   distinguishes row 2 from row 3, and without it Linux cannot tell "different compiler" from "different
   kernel".
3. **Predicate returns the delta set**, and §8.9.7's session disclosure prints it. Not blocking for a
   number, blocking for an *honest* number: without it Linux claims out of frame silently, which is the
   state §8.9.17 called the silent extrapolation.
4. **`toolchain` field** — `glslc --version`, recorded per entry, read by the predicate.
5. **`device` identity** — the real name, not `f"device{args.device}"`. Already ordered.

Items 1 and 2 are the ruling's blocking content. **3 must not lag them by long**, because a claim
granted before its disclosure exists is exactly the trade §8.9.17 refused.

---

**PART 4 — TWO OF LINK'S FINDINGS THAT TOUCH THE RECORD, AND I AM NOT MINTING A NUMBER FOR EITHER.**

**The DLL hash is a one-way instrument and Link is right to retire his own method.** Six builds of an
unchanged tree gave six distinct Windows hashes; the Linux `.so` was byte-identical across four. The
general form: **a fingerprint of an output witnesses its input only if the production is a function**,
and MSVC linking is not one. So an identical hash means *nothing relinked*; a differing hash means
**nothing at all**. It was being read in the direction it does not support. Retiring a method you
authored, on evidence, is the same act as withdrawing a paragraph, and it should be as unremarkable.

**A collection-time `ImportError` zeroes a suite while it reports green, and this one does generalise.**
`tests/ops/test_shape_inference_delta.py` imports at module level; one absent optional dependency aborts
collection of the directory, 292 tests are skipped, and CI — which installs `tests/requirements.txt` and
never sees the gap — reports a step that asserted nothing. **The general form is not "pin the
dependency". It is: a suite's verdict must be a function of its assertions, and an exit status is not.**
The concrete remedy is a **declared expected count**: the lane states how many tests it expects to
*execute*, and a run that executes materially fewer fails on the shortfall regardless of exit code.
`ci.yml`'s own comment already says the right thing about the older version of this defect — *"the pytest
process exited zero. That is all it meant"* — so the lane has met this shape before and closed it for
verdicts without closing it for collection.

**And I am deliberately not giving either of these an `R` number.** §8.9.18 part 2 ruled that
**numbering follows citation, not authorial intent**, and that the register's count is no longer mine to
keep. Minting two numbers in the same document that hands the counting away would be the old habit
wearing the new rule. Both obligations are stated here in full; if they come back cited by someone who
was not in this conversation, they get numbered then, and Fact Checker is the one who will know.

---

#### 8.9.21 RULING — an optional device capability is a **frozen frame constant**, not a claim-time query; the claim path may read frame if and only if the component is resolved-before-first-claim, session-immutable, and passed in; and `shaderInt64` is `synchronization2` with a different name (2026-08-03)

*Against `main` = `af408bf`. Tank declined this inside a merge window and was right to decline it. Three
questions were put to me: whether device-dependent loadability is admissible on the claim path at all;
what the general treatment of an optional device capability is in a system whose proofs are properties
of a form; and whether Mouse's "claim when unsure" and Tank's "answer `true` when unsure" are the same
rule. Numbering note as in §8.9.17–19: the `8.9.x` sequence is shared with `OP_COVERAGE.md`, which now
carries 8.9.8–8.9.16 and 8.9.20 (§7.23). I take 8.9.21.*

---

**PART 0 — WHAT IS ACTUALLY TRUE TODAY, VERIFIED BEFORE RULING, BECAUSE THE RED HAS TWO CODES AND THE
LOUD ONE IS NOT THE ONE THAT BINDS.**

`variants::CAP_INT64` is SPIR-V capability 11, declared by every `_i64` module.
`variants::ENGINE_ENABLED_CAPABILITIES` is `&[CAP_SHADER]` — one element. `vk::device::Device::new`
builds `vk::DeviceCreateInfo` with `queue_create_infos` and `enabled_extension_names` and a
`DeviceFeatureChain` carrying `synchronization2`, and **calls no `.enabled_features(…)` at all**, so
`pEnabledFeatures` is null and `shaderInt64` is off on every device this project has ever run on.
`variants::variant_is_loadable` therefore returns `false` for every `_i64` stem;
`elementwise::only_loadable_variants` turns that into a `[dtype]` decline; and
`registry::unproven_decline_detail` separately emits `[unproven]`. **Two codes, and `[dtype]` is the one
that binds** — R8 again, in its own words: *a decline code names the first failing check, not the only
one.* Tank's measurement stands: `gen_proof_ledger.py --append` answers `no unlockable keys` on both
shape classes for `ai.onnx::Cast/6+/i64>i32/ew_cast_i64_to_i32`, and four of Phi-3.5's five unproven
declines — `Cast` ×2, `Sub` i64, `Greater` i64 — are i64 modules. **This backlog is one device feature,
not five proof runs, and no amount of evidence discharges it.**

What Tank shipped is right and I am not reopening it. `registry::form_is_provable` is pure over the
checked-in SPIR-V and a `const` list — no device handle, no global — so it answers identically before
and after device creation, and an unprovable form no longer receives advice that cannot be followed.
**He did not move the count**, which is the part I want on the record: `unproven_declines: 5` is still
true, and `unprovable_decline_forms` says something that could not previously be said. A ruling that
changed a number while changing what the number means would have been unreadable in both directions.

---

**PART 1 — IS DEVICE-DEPENDENT LOADABILITY ADMISSIBLE ON THE CLAIM PATH? YES, UNDER A TEST — AND THE
TEST IS THE ANSWER TO PART 2, SO I GIVE IT FIRST.**

Tank's objection is that the third of the three edits — *decline the variant on devices that lack the
feature*, as `ENGINE_ENABLED_CAPABILITIES`' own doc comment orders — makes loadability device-dependent
on a path with no device in scope, and that a global written at device creation and read at claim time
is exactly what Mouse refused to build this morning. **The objection is correct about the global and
wrong about the device**, and separating those two is the whole ruling.

**The device is not unknowable at claim time.** A `VkDevice` exists before any node is offered: ORT
selects an `OrtEpDevice`, the EP creates its logical device, and only then is the graph partitioned.
Mouse's `SpecWitness::Unobserved` case is genuinely different in kind — his own doc comment says why,
in `registry::audit_dispatch_specialisation`: *"a claim is decided before any pipeline exists, so a
witness consulted only there would report `SPEC-UNOBSERVED` on every single-session run and the delta
counter's only observable value would be zero."* **That is a fact about ordering, not about scope.** The
pipeline does not exist yet; the device does. Treating the two as one case would import a real
impossibility into a case that only has a plumbing problem.

So the general test, which settles Mouse's refusal and Tank's decline in **opposite directions** and
explains why they differ:

> **THE CLAIM-TIME FRAME TEST. The claim path may read a component of the FRAME if and only if all three
> hold: (a) the component is *resolved before the first claim*; (b) it is *immutable for the life of the
> session*; and (c) it is *passed in as a value*, not fetched from a global. A component that fails (a)
> or (b) is not claim-time-readable at all and must be handled at the boundary where it does become
> readable. A component that fails only (c) has a plumbing defect, not a schema defect — and a plumbing
> defect is not a licence to keep the global.**

Run it on everything currently in the frame:

| frame component | (a) resolved before first claim | (b) session-immutable | (c) passable | claim-time readable |
|---|---|---|---|---|
| enabled capability set (`shaderInt64`, `shaderFloat16`, subgroup features) | yes — at `Device::new` | yes — `Device::caps` is documented "frozen" | yes | **YES** |
| device identity (`vendor_id:device_id:driver_version`) | yes | yes | yes | **YES** — §8.9.17(3) |
| toolchain identity | yes — baked at build | yes | yes | **YES** — §8.9.19 |
| `ort_build` | yes | yes | yes | **YES** |
| bound specialisation (`spec_digest`) | **no** — `vkCreateComputePipelines` has not run | no, by construction | — | **NO** — §8.9.20, and Mouse was right |

**The three tests are not decoration; each one has a specimen on this project.** (a) is Mouse's, this
morning. (b) is what a lazily-initialised `OnceLock` capability set would violate, and this codebase has
shipped a time-dependent global three times by `form_is_provable`'s own count. (c) is the one Tank
flagged, and it is the cheapest to satisfy and the easiest to skip.

**RULING on question 1: device-dependent loadability is admissible on the claim path, because the
enabled capability set passes all three tests. It is inadmissible *as a global*, and that is the only
part of Tank's objection that survives.** The correct structure is stated in Part 2.

**And I reject the two alternatives that were offered, with their costs stated, because "the i64 ops are
worth having" is a constraint and not an argument.**

- **Declare the capabilities statically and fail session creation if the device lacks them.** Rejected.
  This converts *four Phi-3.5 nodes run on the CPU EP* into *no Vulkan EP on this device at all*.
  `shaderInt64` is an optional `VkPhysicalDeviceFeatures` bit and is not universally present; a hard
  requirement makes a whole-EP outage out of a per-form decline, on hardware we have not enumerated. It
  also violates §7.0's frozen principle in spirit: the capability set is the *floor for the EP to
  function*, and i64 elementwise is not that floor. **A capability that gates four nodes may not gate the
  session.** This option is admissible for `CAP_SHADER` and for nothing else currently in view.
- **Claim optimistically and decline at pipeline creation.** Rejected, and it is already rejected in
  writing. `OP_COVERAGE.md` §8.9.16's split table says it in one line: *claiming a node whose module
  cannot be instantiated is an `EP_FAIL` at translate time, not a decline, and no ledger entry could
  make it safe.* A decline sends the node to the CPU EP, which is always right; a translate-time failure
  takes the session down. **Trading a correct answer for a crash is not an optimism, it is a defect with
  a cheerful name.**

---

**PART 2 — THE GENERAL TREATMENT, BECAUSE `shaderInt64` IS THE FIRST AND `shaderFloat16` AND THE
SUBGROUP FEATURES ARE ALREADY BEHIND IT.**

The general answer is short, and it is short because **this project already ruled it once and did not
notice the ruling was general**:

> **An optional device capability is resolved exactly once, at device creation, and frozen into the
> engine instance's capability set. Downstream code — the claim path included — reads the *resolved
> set*, never the feature and never the physical device. The capability set is FRAME; the resolved set
> is a constant of the session; and a predicate over (checked-in SPIR-V, resolved set) is pure.**

**`synchronization2` is the precedent and it is exact.** §7.3 dropped it from the hard requirement and
Switch carries a legacy path; §7.5's barrier abstraction contract is the enforcement, and its rule is
the one I am generalising: *the decision happens once in `Device::new`; no call site branches on
`caps.synchronization2`; call sites use `self.barriers.…`.* There is a layering lint for it and
`device.rs` is not on its permitted-files list. **`shaderInt64` is `synchronization2` with a different
name and one extra consumer.** The only new thing is that the consumer is the claim predicate rather
than a command-buffer recording site, and the claim predicate is further from the device — which is a
plumbing distance, not a difference in kind.

**THE THREE EDITS, RESTATED AS FOUR, BECAUSE THE FOURTH IS THE ONE THAT MAKES THE OTHER THREE SAFE.**
`ENGINE_ENABLED_CAPABILITIES`' doc comment names three: enable the feature in the chain, probe it in
`vk::caps`, decline the variant on devices that lack it. That list is right and incomplete.

1. **`vk::caps` probes `VkPhysicalDeviceFeatures::shaderInt64` and every other optional capability the
   engine knows how to use, and records the *supported* set.** A probe that cannot distinguish "not
   supported" from "not asked correctly" is §7.9's defect and is already ruled on; this probe inherits
   that obligation unchanged.
2. **`Device::new` enables the intersection of what the engine wants with what the device supports, and
   freezes the result as the *enabled* set on `Capabilities` — alongside `synchronization2`, in the
   same struct, resolved at the same moment.** Supported and enabled are two sets and conflating them
   is the mistake `variants.rs`' own capability-accounting comment already names: *Vulkan does not grant
   features by being new enough.*
3. **`ENGINE_ENABLED_CAPABILITIES` stops being a `const`.** It becomes a value carried on the frozen
   capability set, and `variants::variant_is_loadable` takes it as a parameter:
   `variant_is_loadable(stem, enabled)` — still pure, still no device handle, still no global, still
   answering identically for identical inputs. The `const` survives as the **generation-time** list
   (`GENERATED_CAPABILITIES` is already the wider sibling and already has the right doc comment:
   *building a variant no device can load costs a few kilobytes and nothing else… Building one is not
   the bug. Claiming on one is*).
4. **The resolved enabled set is threaded to the claim path as a parameter, and the absence of a
   thread is a cost to be paid, not a licence for a global.** `claim_decision(view: &NodeView)` and
   `claim_audit` take the resolved set the same way they take the node. If that plumbing is expensive,
   the expense is the honest price of (c) in the frame test above, and it is paid once. **The moment
   this becomes `static ENABLED: OnceLock<…>` read from inside `only_loadable_variants`, the ruling has
   been implemented as its own counterexample** — a claim that consults a device-created global is
   reading frame at key time, which is precisely what Tank refused to build and precisely what Mouse
   refused this morning one axis over.

**WHAT THIS COSTS, STATED, BECAUSE IT IS A REAL COST AND IT IS NOT THE ONE PEOPLE EXPECT.** Once an
optional capability is enabled where present, **two devices running the same binary have different claim
sets.** A `[dtype]` histogram from a device with `shaderInt64` is not comparable to one from a device
without it, and nothing in the histogram says so. That is R8's incomparability — *two decline counts are
not comparable without knowing the check order* — arriving through a second door, and it takes R8's
remedy: **every run record publishes the resolved enabled-capability set, and any comparison of claim
or decline counts across runs compares that set first.** A claim census that does not carry its
capability set is an undeclared extent (R11 obligation 1) and is not quotable.

**WHAT IT DOES *NOT* COST, AND THIS ONE DISCHARGES ITSELF.** The ledger needs no new field. A form whose
module cannot be created produces no dispatch, no proof run and therefore **no entry**; so *the
existence of an entry is itself the witness that the capability was enabled when the proof was taken*.
The enabled set does not enter the KEY (it is frame), does not enter the SUBJECT (the SPIR-V bytes are
unchanged by which features are on), and does not need to be stamped on the entry to be recoverable —
`variants::declared_capabilities` reads the requirement straight off the module. **Stating what that is
blind to, as I require of other people's rulings:** it cannot distinguish *enabled and required* from
*enabled and irrelevant*, and it therefore cannot detect a device that enabled `shaderInt64` for a
module that does not declare `Int64`. That is a harmless over-enablement and I am accepting it
explicitly rather than leaving it unnamed.

**ORDERING, FOR THE SAME REASON §8.9.17(5) HAS ONE.** The capability check runs **before** the ledger
lookup, and it already does. A capability failure **fails safe** (the node goes to the CPU EP, which is
always right); a ledger frame mismatch is the thing §8.9.19 spent a ruling making claimable. If the
order inverted, a device without `shaderInt64` would consult the ledger for a form it cannot load and
would report `PROVEN-ELSEWHERE{device}` on a kernel it can never create — a claim, on a module that
cannot be instantiated, wearing a lattice state's clothes. **The `[dtype]` decline binding before the
`[unproven]` one is not an accident of check order; it is the correct order and is now ruled.**

**SCOPE — THE THREE THAT ARE NEXT, NAMED SO THIS IS NOT RE-LITIGATED.**

- **`shaderFloat16` / `VK_KHR_shader_float16_int8`.** Same shape exactly, and the `variants.rs`
  capability comment already records that conflating generated with enabled *"is exactly the mistake
  that made every f16 module unloadable for as long as they existed."* Covered by this ruling with no
  amendment.
- **Subgroup features.** §7.4 already rules that `subgroup_size_control` is required **as a query, never
  as a feature**, and §7.2 freezes the capability set. A subgroup *arithmetic* feature is an optional
  capability and is covered here; a subgroup *size* is a property and stays under §7.4. The two must not
  merge — `Capabilities::has_subgroup_arithmetic` and `Capabilities::subgroup_size_is_exact` are
  different questions and already have different accessors.
- **Switch's spec-constant selectors are NOT covered and this ruling does not reach them**, and I am
  saying so rather than letting the coverage be assumed. A specialisation value is chosen at
  `vkCreateComputePipelines`, fails test (a), and is §8.9.20's territory — the dispatch-time witness,
  not the claim path. **My §8.9.19 carry-forward recorded that runtime-chosen specialisation values are
  outside both digests and that nobody owns the residual — `OP_COVERAGE.md` §7.22 now carries it under
  that name. That debt is not paid by this ruling.** What *is* new is that the frame test now says why the two cases must be handled at different boundaries,
  which is the thing that was missing when I called it a debt.

---

**PART 3 — MOUSE'S DEFAULT AND TANK'S DEFAULT ARE THE SAME RULE, THE RULE IS ALREADY IN THE REGISTER,
AND WHAT LOOKS LIKE A DISAGREEMENT IS THE ARTIFACT CHOOSING THE DIRECTION.**

The asymmetry as put to me: Mouse made a missing `spec_digest` **claim** (`SpecWitness::Unrecorded` —
his comment: *a missing specialisation is a fact about a run that has ended, and no build can recover
it*), while a missing `source_digest` **declines** (repairable from the tree, and the repair is
`--backfill-frame`). Tank's `form_is_provable` **answers `true` whenever unsure** and documents its
output as a **lower bound**. Two agents chose opposite-looking defaults for "we do not know" within
hours.

**They are the same rule and it is R13.** Apply the register's own individuation test — *the register
individuates by remedy* — and run it on the failure each default exists to prevent:

- Mouse's failure mode, had he refused: a field that was never written would be reported as a
  comparison that was performed and disagreed. **An instrument-side absence emitted as a subject-side
  finding.**
- Tank's failure mode, and it *fired*: `variant_is_loadable("metadata")` returned `false` for a stem
  that names no module, and a composite `Gather` form was reported as **unprovable** on the strength of
  a failed lookup. **An instrument-side absence emitted as a subject-side finding**, verbatim, and one
  axis further along — R13 **amendment 1** exactly, the defaulting read: *where the key set is knowable
  from the source, the default is not a value and absence is not a reading.* `variant_is_loadable`'s
  early `else { return false }` is `dict.get(k, sentinel)` written in Rust.

Same failure, same remedy, one rule, no number owed. **What differs is not the rule but which direction
is the silent one, and that is decided by the artifact, not by temperament:**

> **THE LOUD-DEFAULT TEST — a generalisation of R13, deliberately unnumbered. When a mechanism does not
> know, it takes the answer that *leaves a trace*, not the answer that is nominally conservative.
> Refusing is not automatically the safe side: refusal is usually an *aggregate* ("all 103 forms
> declined", "5/5 unprovable") and the permissive answer is usually *itemised* (`SPEC-UNRECORDED` with a
> counter, an ordinary `[unproven]` decline that already names a repair). The aggregate is where a form
> goes to stop being looked at. Choose the answer a reader can still find tomorrow.**

Check it against both specimens and it selects correctly in both, in opposite directions: Mouse's
permissive answer emits a token and moves a counter, so permissive is loud; Tank's permissive answer
leaves the form in the `[unproven]` bucket where it was already counted and already has advice, so
permissive is loud. Neither chose permissiveness; both chose the loud side, and the loud side happened
to be permissive twice.

**And I must name the limit, because a test that only ever agrees with what people already did is not a
test.** The loud-default test **inverts** where the permissive answer is silent. A frame component that
would let a node be claimed with no token, no counter and no disclosure has a silent permissive side,
and there the default is refusal. This is not hypothetical and it is not new: it is exactly why
`PROVEN-ELSEWHERE{δ}` is *"counted, disclosed, and δ printed in the run record"* (§8.9.19) rather than
silently `PROVEN`, and it is why §8.9.17(5) ordered the device predicate first — *a digest disagreement
fails safe; a device mismatch fails open.* **The loud-default test is the same sentence stated as a
choice rule instead of an ordering.** So: no new rule, one generalisation, and the register's shape is
unchanged.

---

**PART 4 — "TOO CLEAN" IS DOING REAL EPISTEMIC WORK AND IT DOES GENERALISE, BUT THE REMEDY IS ALREADY
HERE THREE TIMES AND WHAT IS NEW IS THE TRIP-WIRE.**

Tank's first classifier reported **5/5 unprovable** on Phi-3.5 and he distrusted it because it was *too
clean*. He was right, the reading was an artefact of `variant_is_loadable("metadata")`, and the raw
output rather than the verdict caught it. The general statement:

> **A total is the one reading under which a mechanism's discriminating behaviour is unexercised.** When
> a classifier returns the same verdict for its entire input set — all, none, `0`, `100%` — there are
> two live explanations, *the subject is uniform* and *the mechanism is not discriminating*, and the
> reading itself contains no evidence for either. **A result more complete than the mechanism could
> plausibly produce is evidence about the mechanism, not about the subject.**

**The remedy is not new and I will not pretend otherwise.** It is *demonstrate both polarities*, and
this register already carries it three times under three names: R9 rule 3's planted positive control;
R12's `refused > 0` assertion in `elementwise::no_live_claim_rests_on_an_unloadable_variant`, whose own
comment says *"a zero would let this test pass for the wrong reason the day the `_i64` variants stop
being generated"*; and Niobe's `UNWITNESSED` verdict when `probe_weight_reread.py`'s three positive
controls do not all fire. Remedy-identity therefore says **no number**, and my own §8.9.18 part 2 says
numbering follows citation and the count is Fact Checker's anyway. I record it here in the form I judge
right and mint nothing.

**What *is* new is the trigger, and it is worth stating mechanically because "be suspicious of a clean
result" is advice, and R13's own note says advice does not survive transit.**

> **A mechanism whose verdict is uniform across its whole input set says so, in the artifact, and names
> the arm that went unexercised.** Concretely: emit `UNIFORM(n, verdict)` alongside the verdict, and a
> lane treats `UNIFORM` as **not quotable** until a named positive control has produced the other arm.
> This is `refused > 0` promoted from one test to a discipline, and it costs a counter and a
> comparison.

**Cheapest satisfactions, named as the drafting rule requires, because two of them are how this would
quietly fail.** (i) *Emit `UNIFORM` and quote the verdict anyway* — the token is the trip-wire, and a
trip-wire nothing is required to obey is a comment. (ii) *Satisfy the control with a synthetic input the
real path never takes* — the control must run through the same predicate as the subject, or it
demonstrates a different mechanism's polarity. (iii) *Suppress `UNIFORM` when n is small* — a uniform
verdict over n=2 is weaker evidence, not less in need of a control, and a threshold is where this
obligation goes to be forgotten.

**On the count, which is not mine.** I verified **one** instance directly — Tank's `metadata` stem, from
his own write-up and from `registry::form_is_provable`'s doc comment, which records the specimen in the
code that fixes it. I was told it is the fourth such instance in this session by instruments of his. I
am not asserting four, because §8.9.18 part 2 handed the counting to Fact Checker precisely so that a
tally would stop being kept by the person whose argument it strengthens, and *"fourth time today"* is a
tally. **The pattern is recorded; the count is Fact Checker's; the citation is the specimen above.** If
the other three reconstruct, this belongs in the register with a number and Fact Checker will be the one
who knows.

---

**PART 5 — THE MERGE DRIVER, RULED BRIEFLY BECAUSE SCRIBE WAS TOLD I MIGHT BE NEEDED AND THE TRADE HE
FEARS IS NOT THE TRADE HE HAS.**

`.squad/tools/history_merge_driver.py` decides which side of a merge is a condensation by
`len(ours) < len(base)`. **That is a proxy, and it resolves anyway when it is wrong** — which is my own
dangling-reference class: *the reference does not fail, it succeeds against the wrong thing, and the
reader receives a well-formed answer.* A side that condenses **and** appends can be longer than the base
while still having deleted lines; the driver then takes the `else` branch, treats the merge as two plain
appends, and resurrects exactly what it was written to protect. That is the original defect surviving
inside its own fix, and it will present as a condensation that "did not take" with no error anywhere.

**RULING: a condensation is *declared*, not *inferred*. The driver keys on a marker the condensing
commit writes — `<!-- CONDENSED-AT: <merge-base-sha> --> ` on its own line — and never on a length
comparison.** Cost to Scribe: one line per condensation. Cost to the concurrency guarantee the driver
exists for: **none** — a concurrent agent append never carries the marker, so it never takes the
skeleton path, and the `else` branch stays byte-for-byte the union-of-new-lines behaviour agents rely on
today. **There is no trade here.** Length is a proxy for an intention; the marker is the intention.
Keep the length comparison if you like, but only as an assertion that fires when the two disagree —
that is a second witness with a different failure mode (R13 obligation 3), and it is free.

**And the same class applies to `.squad/decisions.md`, which `.gitattributes` now routes through the
same driver:** Tier-1/Tier-2 archival is a deletion, archival commits are large, and an archival that
also merges a fat inbox round can easily net longer than its base. The marker covers both files with one
mechanism. Owner: Scribe.

---

#### 8.9.22 RULING — a `max` over a relative measure whose denominator can go degenerate is a measurement of the degeneracy; criterion 10's logits observable is replaced, and **the replacement admits nothing** (2026-08-03)

*Against `main` = `3bac325`. Switch measured the int8 KV error budget before writing a kernel, filed
three ruling shapes, **rejected shape 1 (widen the band) in advance** — correctly, it is the move the
standing rule forbids — and named shape 2 without ruling it. He was right to stop where he stopped.
Anchor phrases, per the derived register's §4.1 and because I am about to rule that citations should
have them: **THE DEGENERATE-DENOMINATOR RULE** and **THE WINDOW OF A FIT**.*

**THE FACT THAT MAKES THIS MINE RATHER THAN HIS.** A max-ULP criterion ranks the fp16 GPU path —
**337,178 ULP on the logits** — as *worse than every int8 CPU lane* (7,886 / 45,638 / 38,278).
**Criterion 10 is measured with that observable.** So this is not a request to make room for a
quantised kernel; **the current observable already produces an ordering nobody believes, and it does so
whether or not int8 ever ships.**

**AND SWITCH FOUND THE MECHANISM HIMSELF, IN THE FINDING HE FILED AGAINST HIS OWN INSTRUMENT.** His
cancellation counter read `0` exact zeros while `max_ulp` read 6.3e6; the spacing floor is reached by
**any reference below the smallest fp16 normal**, not only by an exact zero — **18,765 subnormal
references, 0.45% of the worst tensor.** A ULP is `|a−b| / spacing(reference)`, and `spacing()`
collapses by ~3 orders of magnitude in the subnormal range. **The max is located, by construction, at
the reference values carrying the least information in the tensor.**

> **THE DEGENERATE-DENOMINATOR RULE. A relative measure is undefined where its denominator is
> degenerate, and a `max` taken over a set containing degenerate denominators is a measurement of the
> degeneracy rather than of the subject. The unit is not at fault; the statistic is.**

**RULING (1): criterion 10's logits observable is not a max-ULP over the whole tensor.** The
replacement must report **two things and never one number**: the residual over references **at or above
the smallest normal**, and the **count and fraction of references below it**, as a separate named
quantity. Nothing is excluded silently — the subnormal population is *published*, not dropped, which is
the difference between declaring a domain and narrowing one.

**RULING (2), and it is the load-bearing half: this is NOT a narrowing-because-it-failed, and the test
is that the change buys int8 nothing.** I have refused a narrowing after a failure once already this
project, in this same criterion, and I am not going to launder one now. So the check, run before the
ruling: under a domain-split observable, int8 `per_block32` still sits at **18–22 ULP** on the KV
against the fp16 control's **3** — **6–7×** — and Switch's verdict
**`NO_ULP_BAND_ADMITS_INT8_AND_STILL_CATCHES_FP16` survives the fix his own data motivated.** The new
observable is *strictly better* at policing fp16, because the spacing floor was destroying the
discrimination a threshold needs; and it leaves the int8 admission question **exactly where he left
it**. **A change to a criterion that makes nothing pass which did not pass before is a repair. One that
admits the thing whose measurement prompted it is a narrowing.** This is the first.

**And it composes with Trinity's float64 result rather than competing with it.** At the final RMSNorm
Vulkan is bit-exact against float64 and ORT's CPU EP carries the 1 ULP. **So the unit is sound and the
oracle is the imprecise side** — which is the strongest possible reason not to respond to a bad number
by changing the unit or the tolerance. **Keep the unit, fix the statistic.**

**WHAT I AM NOT RULING, SAID EXPLICITLY SO THE SILENCE IS NOT READ AS A GRANT.** Whether int8 KV is
admitted at all is **open** and is not decided here. It needs Trinity's observable work, and it needs a
byte figure that is `MEASUREMENT` rather than `MODEL` — Switch's own provenance discipline, applied to
his own lever, and he already refuses to quote the modelled 1.40×/1.76× as measured. **The ledger's
2.21× / 3.17× / 4.06× do not reproduce from any artifact in this tree, and that disagreement was
written down before the first int8 run** (`bench/results/kv-int8-budget-prediction.md` §3), which is
the only reason it is a finding rather than an explanation. At 1.40× on the footprint int8 is the same
order as the arena's already-banked saving and costs a correctness argument the arena did not. **That
changes the ranking and it is a scope call, which is mine, and it is not made today.**

**Owner of the observable:** Trinity (comparison and verdict constructors), with Switch's
`ulp_residual` used unmodified as it was here. **Owner of the int8 scope call:** me, when the
measurement exists.

---

**AND TWO SPECIMENS THAT ARE ONE RULE, WHICH IS WHY NEITHER GETS A NUMBER.**

The ULP max above does not declare the domain its statistic is valid over. And Switch's second
reversal is the same failure with a different face: he measured an **8-step slope of 1.60 ULP/step**,
filed it as compounding, then ran to `past_len 259` and found the residual **compounds and stops** —
saturated at **29 ULP by past_len ~28**, exponent 0.113, flat along the token axis, old tokens no
worse than new.

> **THE WINDOW OF A FIT. A fit is a claim about a domain, and it is quotable only inside the window it
> was taken over. Carried linearly to ctx 8192, that 1.60 ULP/step predicts ~13,000 ULP. The measured
> value is 29 — wrong by ~450×, in the direction that would have killed the lever.**

**Both are R11 obligation 1 — declare the extent of what you are reporting — with the remedy unchanged,
so no number is owed.** A statistic that does not declare its domain and a fit that does not declare
its window are one obligation wearing two costumes, and stating them together is worth more than
numbering either. **What is new is the magnitude:** this project's refusal to extrapolate a slope now
has **450×** attached to it, out of its own tree, and it cost one CPU-EP lockstep run. *That is worth
more than the principle was, because a principle survives being disagreed with and a number does not.*

**Recording, not ruling, because it is Mouse's and Switch already filed it:** an int8 KV kernel makes
the specialisation constant **load-bearing for correctness** — the wrong group size dequantises with
the wrong stride and returns plausible wrong numbers. That is the first such case here, it lands
squarely on §8.9.20's dispatch-time witness, and it is the specialisation residual §8.9.21 declined to
pay coming back with teeth. **`shaders_dispatched_spec_digest` must move, and a `SPEC-DELTA` on that
kernel is a correctness event rather than a bookkeeping one.**

---

#### 8.9.23 RULING — a key names the **path**; the claim must name the **domain**. `Conv`'s attributes stay out of the key, `Conv`'s key is false today for an unrelated reason, and two classes are settled — the self-witness bound (new) and the stale citation (not new) (2026-08-04)

*Against `main` = `3365221`. Mouse asked for this rather than deciding it himself, which was the
right call and is the second time an op author has escalated a key question instead of widening a
key. Anchor phrases: **A KEY NAMES THE PATH; THE CLAIM MUST NAME THE DOMAIN**, **THE SELF-WITNESS
BOUND**, and **A CITATION IS A PROOF KEY WITH NO SUBJECT DIGEST**.*

##### (1) The question as asked — schema change (mine) or key component (his) — has a third answer

`group`, `strides`, `dilations` and `pads` are **push-constant values consumed by one uniform code
path**. `rust/shaders/glsl/conv_f32.comp` computes `cpg = pc.c / pc.group` and `mpg = pc.m /
pc.group`, indexes with `pc.stride_h` / `pc.pad_h` / `pc.dil_h`, and **branches on none of them**;
its own header says grouping is the general form and nothing special-cases depthwise.
`ops/conv.rs::translate` emits one `KernelRequest` with `spec_constants: vec![CONV_LOCAL_SIZE]` for
every one of these cases. Under §8.7's expression-vs-path distinction they are **expressions**.

Two `Conv` nodes differing only in them are dispatched by the same module with the same bindings —
which is exactly and only what a shared key asserts. **The key is true.** Adding a component would
make it false in the other direction: it would assert that a stride-2 `Conv` runs different code
from a stride-1 `Conv`, and it would demand a separate proof for most of MobileNetV2's 52 nodes.
**A key that over-separates loses the one property a key exists for — *proof of one is proof of the
other*.**

> **THE ATTRIBUTE TEST, which is §8.9.21's frame test one level down and is not a new obligation.**
> An attribute belongs in the proof key **iff it selects the emitted code** — a specialisation
> constant, a module stem, a binding arity. An attribute that only parameterises arithmetic inside
> one code path belongs to the **case**, not to the **form**.

`IsInf(detect_negative)` passes that test and is in a key (`registry::variant_key`'s `@sel` suffix).
`Conv(group)` fails it and must not be. Neither a ledger schema change nor a key component is owed.

##### (2) What *is* owed is that the claim publishes what the key is blind to

`disclosure::disclose_claimed_forms`'s `Proven` arm prints the raw `ProofKey` and the artifact and
nothing about the key's **domain**. A user who does not already know the schema reads "`Conv`
proven" as "`Conv` is correct".

> **A proof key whose blindness a reader cannot see will eventually be read as a claim it never
> made. The remedy is not to widen the key; it is to ship the key's blind list beside it.**

Mechanism, chosen as the cheapest one that cannot drift away from the row it describes: a
`blind_axes: &'static [&'static str]` on the row in `registry::OpSpec` — **a registry field, not a
ledger schema change, therefore Mouse's** — rendered into the disclosure line for any row that
declares one, together with the sentence that those axes are spoken for by a **CI-time** suite
(`tests/ops/test_conv.py`, twelve attribute combinations) and by **nothing that ran in the reader's
session**. That second clause is not decoration: without it the disclosure trades one false
impression for another.

This is also the answer to Rai's related 🟡 (owner Mouse/Tank), and it closes with the same
mechanism. Her severity call is right and worth restating as a rule: **the gap was self-disclosed in
writing by the author of the gap, unprompted, before review** (`docs/OP_COVERAGE.md` §13.9.4,
`tests/ops/test_conv.py`'s docstring). The defect is not that nobody knew; it is that **the
documentation of the limit is excellent and the runtime disclosure of it is silent**, and the
runtime disclosure is the one a user reads.

##### (3) Found while ruling, larger than the question asked, and blocking: `Conv`'s key says it has no shader

All four entries in `evidence/proof_ledger.jsonl` render the variant component as the literal
`metadata` —

```
ai.onnx::Conv/1+/f32,f32,f32>f32/metadata/static/n3
ai.onnx::Conv/1+/f32,f32>f32/metadata/runtime-extent/n2      (and two more)
```

— whose documented meaning in `registry::variant_key` is *"this row has no shader"*, while **the
same entries** record `"shaders": ["conv_f32"]` with a real `shader_digest` and `source_digest`.
Cause: `ops/conv.rs` registers `kernel!(None)`, so `spec.kernel.stem(F32)` is `None` and the
sentinel is taken; the module is chosen inside `translate`, where the key does not look.

**The subject knows the shader; the key denies it exists.** Three consequences, in the order they
bite:

1. **It is false now.** A key component states something about the form that is not so.
2. **The variant component is constant across every present and future `Conv` form**, so the one
   mechanism that would separate a specialised grouped or depthwise kernel from the general one is
   inert on precisely the op that will acquire one first. This is the scenario Mouse was worried
   about — *a key that will eventually claim something false* — arriving through a different door
   than the attributes he named.
3. `registry::form_is_provable` short-circuits on `variant_is_generated("metadata")`, so **`Conv`'s
   provability is answered without ever consulting `conv_f32`.** In a shaderless build its key still
   reads provable.

Consequence 3 is **the loud-default test (§8.9.21 part 3) failing in the permissive direction**, on
a row that is not composite — the case that ruling did not anticipate, because it reasoned about
rows that genuinely have no module and this row has one.

**Repair, Mouse, and it is owed before a second `Conv` kernel variant exists:** the variant
component must be named by **the code that dispatches**, not by a kernel table the row does not
populate. `translate` names `conv_f32`; the key must name what `translate` will name. **Do not
repair by widening the `metadata` sentinel's meaning** — that is the cheapest satisfaction of the
definition and it hides the same blindness one level further in.

##### (4) What the key is still blind to after all of that

Stated because I have twice found that the cheapest satisfaction of a definition is the one that
hides its own blindness.

| Blind to | Why, and whether it is disclosed |
|---|---|
| `group`, `strides`, `dilations`, `pads` | By this ruling. Disclosed by (2). |
| `auto_pad`, spatial rank ≠ 2 | **Not at all** — both are declined at claim time (`ops/conv.rs`), and a declined node reaches no key. The key's silence here is correct and **unreadable as correct**. |
| Accumulation order across `cpg` | Absorbed by the derived `FP32_CONV` tolerance, which is **one vendor's** (`docs/OP_COVERAGE.md` §13.9.5, RTX 4060, re-derive before quoting elsewhere). |
| Any axis nobody listed | The blind list of (2) is **written by hand**. |

The last row is the one that will bite, and it is the same defect the shader-variant list already
carries in `ops/elementwise.rs` (*"a list nobody can falsify is the next thing to go wrong"*). It is
accepted here **explicitly**, in advance, rather than discovered later.

##### (5) THE SELF-WITNESS BOUND

Rai discharged RAI-013 by measurement rather than inference: the default-arm disclosure reaches the
console by a **direct stderr write that bypasses ORT's severity threshold entirely**, on both
devices, in both polarities, with a working negative arm (stderr forced to fail; the WARN escalation
appears on ORT's own sink and the reach token changes). She distinguished the credit from her
RAI-008(a) refusal explicitly, and correctly: that falsifier required a whole criterion; this one is
a single clause and the clause is now affirmatively measured true.

Her residual is a **bound**, and it is not about stderr. The general form:

> **THE SELF-WITNESS BOUND. Where the property of interest lies past a boundary the process cannot
> cross, an instrument can report only its own last action before the boundary. The positive reading
> is a fact about the *attempt*, never about the *arrival* — and no elaboration of the instrument
> moves the boundary, because the elaboration runs on this side of it too.**

It has two arms of opposite sign and both are live in this repository today:

* **Arm (a), the guaranteed antecedent.** A check whose triggering condition is *made true by the
  context in which it runs* carries no information about the thing it warns of. This session's
  canary token is the specimen: a marker that can only be read by the reader whose absence it exists
  to detect.
* **Arm (b), the unavailable observation.** `session_disclosure_info_reach: REACHED_USER` is
  produced by a `write()` that returns success identically for a terminal, a log file, and a pipe
  with no reader.

Arm (a) is a check that **cannot fail**; arm (b) is a check that **cannot distinguish**. Both are
instruments whose output is a function of the apparatus and not of the subject, which is R9's test
arriving through the *observer* rather than through the observable.

**Disposition is disclosure, never repair — and disclosure means the token names our own side of the
boundary.** The §0.2 line is landed. `REACHED_USER` should read `WRITE_SUCCEEDED`; that is a
one-token change owed to whoever owns `disclosure.rs`, and it is **not blocking**, because a bound
correctly named is a discharge and not a gap.

##### (6) A true claim on a stale citation is R13, and R13's remedy does not reach it

Rai verified RAI-012 fixed by rebuilding the crate fresh under WSL and running the real op suite —
42 declines, every one carrying the corrected message, zero instances of the old sentence — while
the three artifact files cited as evidence remain **pre-fix**, committed roughly three hours before
the fix landed and never regenerated. **The claim is true; the citation is not.**

**Same class.** A reference that resolves to a plausible value which is not its referent is exactly
R13's defaulting read. What differs is only who defaults: here it is the **reader** — the file
opens, so the citation is taken as checked.

**One amendment, because the difference is load-bearing.** In R13 the referent is *absent*; here it
is *present at the wrong version*, and a version carries an order that an absence does not.
**R13's loud-default remedy therefore cannot reach it — nothing is silent.** What reaches it is a
mechanism this project already built one floor down and has never applied to prose:

> **A proof entry names its subject by digest and demotes itself when the subject moves. A citation
> names its artifact by path and cannot. A citation is a proof key with no subject digest** — so
> cite a **state**, not a path: the commit an artifact was generated at, checked against the commit
> the claim is about.

Regeneration of the three files is routed to Link. The general form is recorded here **without a
number**, per §8.9.18 part 2 — numbering follows citation and the count is Fact Checker's. **The
classes stay distinct by their remedies, not by their names, and this one shares R13's diagnosis
while needing a different fix.**

---

#### 8.9.24 RULING — criterion 10's tolerance is **satisfiable everywhere in fp16 with a factor of 20 to spare**; `atol` does not move, the verdict does not split by mechanism, and the oracle question is now **blocking** on any future tolerance motion (2026-08-04)

*Against `main` = `9b16f4f`. Trinity re-scored criterion 10 under §8.9.22's statistic, predicted the
headline result before running anything, filed the unsatisfiability finding **and deliberately did
not act on it**, and declined to move `atol` for a third consecutive round — including once at the
moment moving it would have turned two of three failing outputs green. Every part of that is the
right order. The finding is nonetheless refuted, and it is refuted by arithmetic that needs no run.
Anchor phrases: **A TOLERANCE IS UNSATISFIABLE ONLY WHERE ITS ALLOWANCE FALLS BELOW THE SPACING AT
THE POINT THE PREDICATE IS EVALUATED**, **A VERDICT IS A FACT ABOUT A PREDICATE, A MECHANISM IS A
FACT ABOUT A CAUSE**, **THE MOVER IS NOT THE MEASURER**, and **A PREDICTION IS NOT A READING**.*

##### (1) The premise is false, and it fails twice in the same sentence

The finding is *"`atol=1e-3` is 0.128 ULP-at-scale on the logits — the tolerance demands finer than
fp16 can express."* The figure is correct and the inference from it is not, because the quantity
`0.128` is not the tolerance and `ULP-at-scale` is not the unit the predicate is evaluated in.

**The predicate is a sum, and the quoted figure is one of its two terms.** `np.allclose` tests
`|a − b| ≤ atol + rtol·|b|`. Dropping `rtol·|b|` and calling the remainder "the tolerance" is only
sound where `|b|` is small — and where `|b|` is small the spacing is small too, which is the
opposite of the case being argued. At the logits' own scale the full allowance is **33.628
ULP-at-scale**, not 0.128. Same arithmetic on the other two: **29.796** on `present.31.key`,
**32.404** on `present.31.value`. *The reported figure is off by a factor of 263, 116 and 506
respectively, and in each case the missing factor is the term that dominates everywhere it matters.*

**The predicate is evaluated per element; the denominator was taken per tensor.** `ULP-at-scale`
divides by the spacing at the **tensor maximum**. The predicate never evaluates at the tensor
maximum unless the element is the maximum. A residual at a reference of 0.011 judged against the
step size at 5.77 is a numerator and a denominator from two different points, which is the
§8.9.22 defect with the sign reversed — *there* the denominator collapsed and made a sound residual
look catastrophic; *here* it inflates and makes a real residual look like nothing. **The same
instrument now has both directions of the same error on the record, four days apart.** That is the
strongest available argument that the rule is about the construction and not about the tensor.

**The satisfiability bound, which settles it in one line and for every fp16 tensor this project will
ever compare.** For a normal `b`, `ulp(b) = 2^(e−10)` with `2^e ≤ |b|`, so `ulp(b) ≤ |b|·2⁻¹⁰`.
Therefore

> `allowance / ulp(b)  ≥  rtol·|b| / (|b|·2⁻¹⁰)  =  0.02 · 1024  =  20.48`

— independent of `b`, of the tensor, and of the scale. Swept over the whole fp16 normal range the
minimum is **20.48000, attained at `|b| = 32768`**, exactly as the algebra says. On subnormals
`ulp = 2⁻²⁴` and the `atol` term alone gives **≥ 16,777 ULP**. So:

> **RULING (1). Criterion 10's tolerance is satisfiable at every representable fp16 value, with a
> margin of at least 20.48 element-ULPs on normals and 16,777 on subnormals. It is not an
> unsatisfiable criterion. `atol` does not move, `rtol` does not move, and criterion 10 stays
> `DIVERGENT`.**

And the corollary is the part worth carrying, because it inverts the reading of the failures:
**a failing element is one whose residual exceeded an allowance that was already ≥ 20.48 element-
ULPs wide.** Layer 31's key and value do not fail by a sub-step amount. They fail by more than
twenty representable fp16 steps *at their own magnitudes*, and the only thing that made them look
sub-step was measuring them against a step size borrowed from a value roughly 500× larger.

> **A TOLERANCE IS UNSATISFIABLE ONLY WHERE ITS ALLOWANCE FALLS BELOW THE SPACING AT THE POINT THE
> PREDICATE IS EVALUATED. A statistic offered as evidence about a predicate is evaluated in the
> predicate's own units, at the predicate's own granularity, and against the predicate's whole
> expression — not one term of it. A relative tolerance whose ratio exceeds the format's relative
> spacing is satisfiable by construction, everywhere, and no measurement is required to know it.**

I want the general form on the record because the mirror-image reading in the brief is correct and
would have been decisive if the premise had held: **an unsatisfiable criterion is the exact dual of
an unfalsifiable one, and it is demoted for the same reason** — it cannot be met however correct the
implementation is, so it reports nothing about the implementation. That remains true. It is simply
not this criterion. The rule earns its number by being the test that tells the two apart, and the
test is cheap: compute the allowance in the format's own units at the point the predicate looks.

##### (2) The verdict does not split by mechanism — and this is where the narrowing was hiding

The brief's strongest point is that the three failures are two mechanisms: 6,056 elements at up to
8.0 ULP-at-scale on the logits is a bulk residual; 16 elements on `present.31.key` and 2 on
`present.31.value` are a tail. That is **true, load-bearing, and a fact about causes.** It is not a
reason to give them different verdicts.

> **RULING (2). A verdict is a fact about a predicate; a mechanism is a fact about a cause. Both are
> reported and only one is a verdict. Criterion 10 reports one verdict per output under one
> predicate, and the two mechanisms are reported as diagnosis alongside it.**

The reason is procedural and I would rather state it plainly than dress it up. **Splitting a verdict
by mechanism, in the round in which the mechanism was discovered, on precisely the outputs that
would go green, is a narrowing with a taxonomy bolted to the front of it.** I refused a
narrowing-after-failure in this criterion in §8.9.22 and checked my own repair against the test *a
change that makes nothing pass which did not pass before is a repair*. Applying that test here:
splitting the verdict would move outputs 63 and 64 out of `OUTSIDE_TOLERANCE` without any element of
either output moving. **It admits two of the three failures. It is a narrowing.** Nothing about it
survives the test I set for myself, and it does not survive it any better for having arrived through
a true observation about mechanisms.

**Applied to my own ruling, per the same test.** §8.9.24 changes no predicate, no threshold, no
observable and no verdict. Nothing passes under it that did not pass before, and nothing fails that
did not fail before. **It admits nothing because it moves nothing** — which is a weaker claim than
§8.9.22's and I am stating it in the weak form deliberately, since a ruling that only adds
obligations has bought its cleanliness cheaply and should not be credited as if it had been at risk.

##### (3) What the ULP-at-scale figures are still good for, and the reporting obligation that keeps them honest

The statistic is not withdrawn. It is a sound answer to *"is this residual large relative to the
tensor?"* and that is a real question — it is how a reader decides whether a divergence could change
a token. What it may not do is stand next to a pass/fail predicate it does not participate in, in a
unit that predicate does not use, where a reader will take the comparison as a verdict.

> **RULING (3). Any per-output census that reports a residual in `ULP-at-scale` also reports, on the
> same row, (a) the allowance `atol + rtol·|b|` expressed in the *same* unit, and (b) the failing
> set's residual on the element basis. `failing_residual_within_one_ulp_at_scale` may not appear
> without `atol_in_ulps_at_scale`'s companion `allowance_in_ulps_at_scale`.** Owner: Trinity, in the
> comparator that already carries `verdict_predicate`.

`verdict_predicate` is why this round produced a refutation instead of a relaxation, and it deserves
saying. Trinity turned "the predicate does not read a ULP" from an observation into an **assertion
scored against a hand-written `allclose`**, and put it on every per-output entry. That is the loud
form: a reader who tries to argue from a ULP figure to a verdict now has the predicate printed in
the row they are reading from. **The mechanism that caught my error was already in the artifact
before I opened it.**

##### (4) The oracle question is blocking, and the ordering is the ruling

Trinity's float64 result stands unanswered at model scale: at the final RMSNorm **Vulkan is
bit-exact against float64 and ORT's CPU EP carries the 1 ULP.** Nobody has asked which side is wrong
on outputs 0, 63 and 64. §8.9.22 already leaned on that result to refuse a change of unit.

> **RULING (4). No motion to change criterion 10's tolerance, unit, predicate or verdict structure
> is entertained until outputs 0, 63 and 64 have a float64 answer to "which side is wrong". Owner:
> Trinity. Until then the only admissible motions are ones that change no verdict.**

The reason is not caution. **If the reference is the wrong side, the correct remedy is a different
oracle and every tolerance argument made first was an argument about the wrong question** — and it
would have been made in the direction of loosening, using the reference's own error as the budget.
This costs a legitimate relaxation nothing except its place in the queue; it costs an illegitimate
one its only available route.

##### (5) **THE MOVER IS NOT THE MEASURER** — ratified as a rule, because it is already being obeyed

Trinity has now declined to move `atol` three consecutive rounds, and wrote the reason herself before
anyone ruled it: *"the moment to fix a tolerance is not the moment when fixing it turns two of three
failing outputs green, and not by the person whose measurement made them red."* That is a rule, it
has been load-bearing three times, and by §8.9.18 part 2 an unnumbered sentence that is being obeyed
as binding is either numbered or withdrawn. It is numbered here, inside this ruling:

> **A motion to change a criterion is not authored by the party whose run produced the failure it
> would relieve, and not in the round in which that failure was produced. Both clauses are required:
> the second alone permits laundering by delay, the first alone permits laundering by proxy.**

I am recording that she declined, and I am **not** scoring it. §8.9.18 part 4 retired my own decline
tally because "did I mint a number?" and "did the project acquire a binding obligation?" are
different questions and only the first had a counter on it. A tally of someone else's declines has
exactly the same defect and I am not going to build one four days after retiring mine.

---

##### (6) The second item — **A PREDICTION IS NOT A READING**, and the remedy is a token this project already ships

Trinity's first explanation of the Intel arena refusal cited `alloc_device_frame = SPLIT-DEVICE`. The
EP's §6.5 obligation 3 *predicts* exactly that token — *"a run with two devices reports `SPLIT-DEVICE`
on the transfer accounting"* — and the prediction was written down as a reading. One arm per process
with a fresh counters file shows `alloc_device_frame = SHARED` in **both** polarities
(`bench/results/arena_refusal-dev1-noenv-arena.json`, `...-pinned-arena.json`). The real discriminant
is one field over: `alloc_device_frame_allocator_index` reads **`1`** on the refusing arm and **`0`**
on the passing arm, while the session's own device list puts the device it ran on at index `0` in
both. **A right headline with a wrong cause bolted to it** — the refusal is a harness allocator-index
mismatch, which is what the corrected classification says, and it was never a device property.

**Why this is not one of the classes we have.** R13 is an instrument that ran and failed wearing its
finding's costume; R13 amendment 1 is a lookup that resolved to a sentinel; §8.9.23(6) is a citation
that resolved to the wrong *version* of a real artifact. **All three are failures of resolution.**
This one resolves perfectly. The document is present, current, correctly cited, and **true** — it is
a sound conditional prediction that has not been falsified. What changed in transit is not the value
and not the referent but the **modality**: *this is what the mechanism will report* became *this is
what the mechanism reported*. No lookup failed, nothing defaulted, nothing was stale, and there is no
silence for a loud default to fill. **A document that describes what an instrument will say is
indistinguishable, once quoted, from the instrument saying it** — and the more accurate and
better-maintained the document, the more convincing the substitution.

**No new number is owed, and the remedy is already in the tree.** The register individuates by
remedy, and this one's remedy is a mechanism Switch built and this project has been using for weeks:
**a quoted figure carries its provenance class.** §8.9.22 already turned on it — I refused to treat a
modelled 1.40× as a byte measurement and required `MEASUREMENT` rather than `MODEL`. The extension is
one token and one obligation:

> **Provenance classes are `MEASUREMENT`, `MODEL` and `PREDICTION`, and a value sourced from a
> document that states what a mechanism *will* report is `PREDICTION` — including when the document
> is this one. A `PREDICTION` may not be quoted in a position where a `MEASUREMENT` is expected, and
> a claim whose evidence is a `PREDICTION` names the artifact that would have carried the
> `MEASUREMENT` and reports it as absent.**

Per §8.9.18 part 2, numbering follows citation and the count is Fact Checker's: **if this is cited
twice as binding, it is owed a number.** I am recording it unnumbered, exactly as §8.9.23(6) was.

**And a consequence that lands on me rather than on Trinity, which is why this section is here at
all.** The prediction she read back was in **§6.5, in this document, written by me** — a normative
obligation phrased in the indicative present, *"a run with two devices reports `SPLIT-DEVICE`"*.
That phrasing is an instruction to the implementer and a description of the world at the same time,
and a reader who arrives at it from the outside cannot tell which. **The obligation to distinguish a
prediction from a reading cannot rest entirely on the reader when the author wrote them in the same
tense.** So the drafting rule, and it binds this document first: **a normative clause about what an
instrument must report is written in the imperative or with an explicit modal — `must report`, `is
required to report` — never in the bare present indicative.** §6.5 obligation 3 is the specimen and
it is mine; the fix is prose and it is owed at the next §6.5 edit, not urgently, because the class is
now named and the artifact-side remedy does not depend on it.

**This is the fifth time this week that a rule already in the record answered a question I was
treating as new.** §7.5 on Monday, §5.4.1(a) on Monday afternoon, the specialisation debt twice, and
now Switch's provenance classes. The pattern has been stable long enough to stop being an
observation about my memory: **the register is under-indexed by question.** I have said so twice and
recorded no mechanism, which is precisely the decision-versus-mechanism gap I have been grading other
people on. It is Fact Checker's to build if anyone's — an index from *question asked* to *ruling that
answers it* — and I am naming the owner rather than the intention this time.

---

#### 8.9.25 RULING — criterion 10 stays **open** and the oracle answer is the reason it may not be loosened; **an agreement bounds only the difference**; the depth series returned the convicting branch of my own close condition; and a close condition that does not declare its modality defaults permissive (2026-08-04)

*Against `main` = `996a9d8`. Trinity delivered the oracle §8.9.24(4) made blocking, at model scale,
and gated the record with `assert_record_proposes_no_motion` so that the artifact cannot be read as
proposing what it enables. `atol` and `rtol` are untouched for a fifth consecutive round. Anchor
phrases: **AN AGREEMENT BOUNDS ONLY THE DIFFERENCE, NEVER THE DISTANCE FROM TRUTH**, **A CLOSE
CONDITION DECLARES ITS OWN MODALITY — SUFFICIENT OR UNBLOCKING — OR IT IS READ AS SUFFICIENT**, and
**A DIRECTION THAT EXISTS FOR ONE OUTPUT IS NOT A BUDGET FOR THREE**.*

##### (1) The blocking condition is discharged, and discharging a block grants nothing

§8.9.24(4) said no motion on criterion 10's tolerance, unit, predicate or verdict structure until
outputs 0, 63 and 64 have a float64 answer to *which side is wrong*. The answer is in
`bench/results/criterion10_chain-dev{0,1}.json`, and the seam is closed by data flow rather than by
assertion: the chain reads **initialisers and `input_ids` only**, layer L's input is layer L−1's
**reference** output, and `assert_chain_never_reseeded` digests every boundary and raises. Neither
EP appears in its own derivation. The liveness bar *is* re-seeded per layer per side — and its
result never reaches the chain, which is the distinction that made arm F dishonest and that makes a
liveness bar mean anything at all.

| output | direction | evidence |
|---|---|---|
| 0 `logits` | **`cpu`** | unanimous on all five discriminators, both reference variants, both devices; **83 vs 70** element-ULP from true (f64) |
| 63 `present.31.key` | **`null`** | discriminators conflict inside a variant and the variants disagree across it |
| 64 `present.31.value` | **`null`** | same |

**The answer is not uniform, and that is the whole of the ruling on it.** A tolerance motion resting
on *"the reference is the further side"* would admit **three** outputs on a direction measured for
**one**; on the other two it would have to *default*, and defaulting in the permissive direction on
precisely the outputs that would go green is §8.9.21's loud-default test failing in the direction it
exists to catch. A verdict structure keyed on direction is §8.9.24(2) with a new taxonomy bolted to
the front — it moves 63 and 64 out of `OUTSIDE_TOLERANCE` **with no element moving**, which is the
test I set for myself and it fails it identically whether the taxonomy is *mechanism* or *direction*.

> **RULING (1). Criterion 10 stays OPEN and its verdict stays `DIVERGENT`. `atol`, `rtol`, the
> predicate, the unit and the verdict structure are unchanged. §8.9.24(4)'s block is discharged:
> motions may now be *made*, and this is not one being granted.**

I predicted this specimen in advance — *"it would have been made in the direction of loosening,
using the reference's own error as the budget"* — and my own screening rule says the question is not
*is this true?* but *what does it admit?* The observation is **true**: on the logits ORT's CPU EP is
the further side from the real answer, by 13 element-ULP of median. It admits nothing, because the
criterion does not ask whether we are better than the reference. It asks whether we agree with it.

##### (2) The depth series exists — and it returned the branch that convicts

On 2026-08-02 I wrote that criterion 10's gate had a unit defect and that *"it closes when the ULP
series exists and is either flat or has a located step"*, with the reading attached: **flat ⇒ no
accumulation defect; a step at layer L ⇒ a real defect, localised.** Both artifacts now carry it:

- `kv_depth_curve` — 32 rows, median ULP 0–2 for layers 0 through 30, on both key and value;
- `kv_depth_largest_step` — **1.0** (key 2→3, value 0→1);
- `kv_depth_exceedances` — **layer 31, key and value, median 4.0 ULP**, and nothing else.

**Layer 31's key and value are outputs 63 and 64.** The instrument I demanded, in the unit I
demanded, has located a real defect at the one layer whose two outputs are two of the three that
fail. That is the second branch of my own disjunction, and it is the one that convicts.

**And my close condition was written in a modality it does not declare.** *"It closes when the ULP
series exists and is either flat or has a located step"* reads as **sufficient**. It was meant as
the discharge of a blocking objection about the gate's unit. Read as sufficient it closes criterion
10 today — on an artifact whose own `verdict` is `DIVERGENT` and whose `per_run_comparison` is
`['DISAGREE','DISAGREE','DISAGREE']`, which would close a criterion whose text requires an
attributed `MATCH` on the strength of a series that is not the criterion.

> **RULING (2). That condition was UNBLOCKING, not SUFFICIENT. Criterion 10 closes on its own
> text — an attributed `model_output_equivalence = MATCH` over N ≥ 3 — and on nothing else.**
>
> **A close condition on a criterion row declares `SUFFICIENT` or `UNBLOCKING`. An undeclared one is
> read as sufficient, and therefore permissive.**

This is §8.9.24(6) one day later, same author, through the other door: there the substitution was of
**tense** (a prediction read as a reading), here it is of **modality in the deontic direction** (a
condition that removes an objection read as a condition that grants a pass). The remedy is the
identical explicit modal, so by remedy-identity **it takes no number** and lives here as an
extension of that drafting rule. I would rather record that the permissive reading of my own
sentence was available and that I found it myself than have it found for me — while noting that
finding it is not the same as being immune to it, since I wrote it in the first place, in a cell
whose whole subject is not letting a criterion be closed on the wrong evidence.

##### (3) **AN AGREEMENT BOUNDS ONLY THE DIFFERENCE** — and the ratio is per output, not per model

Trinity recorded, and explicitly refused to treat as a budget, that both EPs sit much further from
the weight-only reference than they sit from each other. **I re-derived the ratio per output rather
than quoting the model-scale figure, and the model-scale figure does not survive:**

| output | Vulkan from true (f64) | CPU EP from true (f64) | the two apart | ratio |
|---|---|---|---|---|
| 0 `logits` | 70 (dev1: 71) | 83 | 12 | **5.8–6.9×** |
| 63 `present.31.key` | 12 | 12 | 4 | **3.0×** |
| 64 `present.31.value` | 6 | 7 | 4 | **1.5–1.75×** |

*(median element-ULP, `how_far_both_sides_are_from_true_vs_from_each_other`, both devices.)*

**"Roughly 6× further from true than from each other" is a fact about the logits and is not a fact
about the model.** It falls to 3× at layer 31's key and to about 1.6× at its value. The
generalisation was available, it was wrong, and it was wrong in the direction that makes the finding
more dramatic — which is the direction to check first.

> **RULING (3). `comparison = AGREE` in `compare_all_outputs_to_cpu` means CONSISTENT-WITH-CPU and
> never CORRECT. A correctness claim resting on an AGREE carries, for the output it is made about,
> the measured distance of both sides from a weight-only reference — quoted per output — or it is
> unquotable.** Mechanism: a `means` string on the comparison itself, exactly as
> `output_coverage.means` already does one field over. Owner: Trinity. It changes no verdict.

**The asymmetry is the load-bearing half, and it closes the door this ruling would otherwise open.**
The next move available to a reader of (3) is *"then a DISAGREE is a weak signal too, so criterion
10's red is weak, so relax it."* It does not follow, and the reason is arithmetic rather than
policy: **a shared error cancels in a difference.** `|a − b|` is untouched by any component both
sides carry, so it remains an exact lower bound on the *differential* error whatever the common
error is. **The oracle weakens an AGREE and leaves a DISAGREE exactly where it stood.** Criterion
10's red rests on the half of the picture the common-error finding cannot reach.

**And the common error is not a defect in either EP**, which needs saying before somebody reports it
as one. One graph, one set of weights, one fp16 storage format, one set of op semantics: two
implementations of that are *expected* to share most of their distance from the reals. What the
finding bounds is not this EP's quality — it is **how much correctness evidence any
two-implementation agreement can ever carry**, which is a fact about the method. `compare_all_outputs_to_cpu`
sits underneath most of this project's correctness claims and it is a **consistency** instrument.
It always was. The measurement is what tells us by how much.

##### (4) A quotable direction standing beside the flag that denies it

The chain's roll-up is right: `direction: null` on 63 and 64, with `direction_note` stating that a
variant whose discriminators conflict has no direction and cannot be spoken for by the other. One
level down, `by_reference_variant.f64.which_is_further_from_true` still reads `"vulkan"` on output
63 while `discriminators_conflict` is `true` in the same object, with the caveat carried in a
neighbouring prose `reading` field.

**A record can default loudly at the top and quietly one level down, and the level a reader quotes
from is the deepest one that answers their question.** This is §8.9.21's loud-default rule applied
to nesting rather than to composition.

> **RULING (4). `which_is_further_from_true` carries the absence rather than the value:
> `null` whenever that variant's discriminators conflict. The conflicting verdicts stay in
> `verdict_by_discriminator`, where a reader who wants them must ask for them by name.** Owner:
> Trinity; additive, changes no verdict, and the roll-up already behaves correctly.

##### (5) `Gemm`'s transposes — the axis is not blind, the **entry** is

Mouse left the `transB` question for me and it resolves under §8.9.23 verbatim: `gemm_f32.comp`
selects the index with a ternary on a push constant, one module, one pipeline, one set of emitted
instructions — an **expression, not a path**. The transposes stay out of the key, where Mouse has
already put them (`registry::OpSpec::blind_axes` = `["alpha","beta","transA","transB"]`).

**Three things are true that the reading did not have, and I checked each in the tree rather than
reasoning about it.**

**(a) The axis is not untested.** `tests/ops/test_gemm_and_pool.py::_GEMM_CASES` runs `transB`,
`transAB`, `wide_k` and `mobilenet_head` (`M=1, K=1280, N=1000, C=[1000]` — MobileNetV2's own head
form) through `_models.check`, which asserts the claim **before** comparing, so a decline fails
rather than passes. `rust/tools/ledger_case_models.py` mints `gemm_f32_transb`,
`gemm_f32_transb_nobias` and `gemm_f32_transb_dyn`. The proofs at `transB=1` were run.

**(b) What is blind is the record.** Those cases mint the *same key* as their `transB=0` siblings,
by design and correctly. So the ledger entry **cannot say which values of a blind axis contributed
to it**, and a reader cannot distinguish a key proven at both values from a key proven at one.
Mouse's sentence — *the blind axis exercised at the value it was proven at tests nothing* — is right
about the defect and one noun off about its subject.

> **RULING (5). A non-key `witnessed_at` field on the ledger entry records, per declared blind axis,
> the values its contributing cases actually carried.** Not a key component: that would re-split the
> space §8.9.23 deliberately merged and quadruple `Gemm`'s keys for no code distinction. Additive,
> changes no key and no mintability, and it makes `disclosure::blind_axes_clause`'s second clause —
> *"a CI-time suite varies them"* — **checkable for the first time.** Owner: Mouse.

**(c) The `[rank]` decline is not about `transB`.** `ops::matmul::translate` requires `a.rank() == 2`;
the same head form at rank 2 is claimed and compared in-lane. The decline is a fact about the
neighbouring shape, and reading it as *"the axis cannot be exercised"* mislocates a fact about `A`
onto an attribute of `B`.

**(d) And the one I found by running the two tests rather than reading them.**
`test_transb_is_a_transpose_and_not_a_relabelling` is the only test in the tree that can tell a
transpose from a **relabelling** — asymmetric integer `B`, identity `A`, so a kernel that ignored
`transB` returns a tensor with the identical multiset and every statistical summary unchanged. It
takes **no `require_vulkan` fixture**, unlike its fifteen neighbours. In a worktree with no loadable
EP it therefore runs, falls back to CPU, finds `is_vulkan_claimed` false, and reports
`pytest.skip("EP did not claim the transB identity case")`. **The EP was not present. The skip
reason names a refusal that never happened** — two terminal states, one token, and the token names
the one that would be a finding. The R13 lane summary *did* catch it
(`LANE FAILURE (fallback log): 1`), so the lane is not silent; the test's own reason string is what
lies, and the reason string is what a reader quotes. Fix: take the fixture, and the decline branch's
skip reason becomes true. Owner: Mouse. **No number** — this is *A PREDICTION IS NOT A READING* in a
third costume, a diagnostic that resolves perfectly to the wrong one of two branches, and the remedy
is the same one: name the branch you observed.

##### (6) The README provenance filing is refuted, and the refutation is the better finding

The filing: *"Its op table is 94 rows, of which 76 carry a kernel. Read from
`epctl --dump-capabilities --json`"* sources two numbers to a command that yields one, because the
dump carries no kernel field — only `dtypes`, `live`, `name`, `opsets`, `schema_baseline`,
`staged_reason`, `status`. I ran it: **94 rows; `status` ∈ {`live` 46, `ready` 30, `staged` 18}; and
the boolean `live` is `true` on exactly 46 + 30 = 76 rows and `false` on all 18 staged.**

**The dump does carry the kernel fact. It carries it under the name `live`.** So the citation is
sound and the schema is not: **one JSON row spells the noun `live` twice with two denotations** — a
status token that is the deprecated `OpStatus::Live` alias granting nothing, and a boolean meaning
*this row has a kernel*. A reader checking the 76 against the field literally named `live` gets 46.
The filing's own footnote records the first arm of that collision happening to its author — *"my
first attempt reported kernel-carrying: 0 because I guessed field names"* — without recognising it
as the second.

> **RULING (6). The boolean is renamed `has_kernel`; `status` keeps its three tokens. Owner: Tank
> for `epctl`, Mouse for the registry serialiser. Until it lands, `docs/DESIGN.md` §0 and
> `README.md` state the derivation inside the sentence — *76 = rows with `live == true`, which is
> not `status == "live"` (46)*.**
>
> **A true sentence whose check requires knowing that one word means two things is checkable only by
> someone who already knows the answer.**

**LANDED 2026-08-05 (Tank, issue #3).** `epctl --dump-capabilities --json` emits `has_kernel` and no
longer emits `live` as a key; `status` is unchanged and still takes exactly `live`/`ready`/`staged`.
The human summary carried the same collision one line down — it called the kernel-carrying count
"live" while the status column beside it spelled `live` for a strict subset of those rows — and now
reads `96 row(s): 78 with a kernel (46 live + 32 ready), 18 staged`, which is the decomposition
rather than a number a reader has to trust. Two tests in `rust/tests/dump_capabilities.rs` hold it:
one asserts the NAME (`"live":` must not appear as a key — a rename that left the old key beside the
new one would satisfy every value assertion and none of this ruling) and the arithmetic
`has_kernel == live + ready`, the other that the human summary names the predicate it counted. The
rename moves no total: `has_kernel` is `OpSpec::is_live()`, the same predicate the boolean always
serialised.

**And the observation that opened this round lands harder here than where it was made.** *"45 op
rows are `Live`"* now reads 46, so it looks nearly right, and §8.9 retired `Live` as a thing we
write down. **But the retired noun is not merely still in prose — it is a serialised field name,
twice, meaning two different things.** A noun retired in prose and left in a schema is retired in
the one place nobody quotes from.

§0's count is corrected here from **92 / 74 / 46-28-18** to **94 / 76 / 46-30-18**. That is the
fifth wrong reading of this integer on the record and the fourth in this document; it moved because
the `MatMul` and `Gemm` rows landed, which is exactly what the previous correction said the next one
would be. **It does not stop being wrong until it stops being written by hand** — which is why the
rename in (6) is not cosmetic and why the sentence now carries its own derivation.

##### (7) What this ruling admits, in the weak form, again

§8.9.25 changes no predicate, no threshold, no tolerance, no unit and no verdict. Criterion 10 is
`DIVERGENT` before it and after it; outputs 0, 63 and 64 fail before it and after it. The three
items it opens — `means` on the comparison, `witnessed_at` on the ledger entry, `has_kernel` on the
dump — are additive, and none of them can turn a red row green. **It admits nothing because it moves
nothing**, and that is again the cheap clean bill rather than §8.9.22's earned one.

The one place it could have admitted something is (2). **The permissive reading of my own close
condition closes criterion 10 today**, on an artifact that says `DIVERGENT` three times. That is
what a narrowing looks like when it arrives without an advocate — nobody proposed it, the sentence
was simply lying there waiting to be read the convenient way, and it was written by the person whose
whole job in that cell was to stop the criterion being closed on the wrong evidence.

**Carried forward.** Criterion 10's red now has a **fully characterised cause** — a localised
layer-31 residual of 4 median ULP on two outputs with no oracle direction, and a bulk logits
residual where the further-from-true side is the reference — and a criterion staying red with a
characterised cause is a better artifact than a criterion turned green. Trinity has declined to move
`atol` five rounds running, twice at moments when moving it would have turned reds green; per
§8.9.24(5) that is recorded and **not** scored.

---

#### 8.9.26 RULING — a gate whose trigger and whose remedy are keyed on different variables is **unsatisfiable by construction**; archival eligibility is re-keyed from **age** to **supersession**, and the class beneath it — an **inert mechanism**, a check whose verdict reaches no one who can act on it — is named (2026-08-05)

**Reported by:** Scribe, twice, in consecutive rounds, as a declined gate rather than as a finding.
**Adjudicated by:** Morpheus. **Number to be assigned by Fact Checker** — the count is hers
(§8.9.18/Round 12), and this section is cited before it is numbered on purpose.

##### (1) The finding is the second decline, not the first

`decisions.md` stands at **67,623 bytes** against the Tier-2 gate of **51,200** — up ~20 KB in two
rounds — and **zero entries are age-eligible**, because the eligibility tiers are 30-day and 7-day
and the newest active round is ~1.6 days old. Scribe declined to archive, recorded the breach as a
judgement, and did so **again** the next round.

> **Re-measured rather than quoted** (Scribe's own health-report discipline): `67,623` bytes on
> `main` at `0a73d82`, `67,756` in this worktree after the same merge. Both are over `51,200`; the
> 133-byte gap between two checkouts of identical content is a line-ending artifact and is recorded
> here rather than rounded away, because a figure whose frame is unstated is how a size gate
> acquires its own stale-field defect.

**She was right both times, and the fact that it had to happen twice is the finding.** A gate whose
**trigger** is keyed on *size* and whose only **remedy** is keyed on *age* cannot fire on a project
that writes faster than its own retention window. The condition is not merely unmet; it is
**unmeetable**, and the cost falls on the honest agent, who is left breaching a hard gate on the
record every round and correctly refusing the only cut available.

This is the mirror of the criteria §8.9.22 and §8.9.24 demoted. There the defect was **a check that
can never go red**. Here it is **a remedy that can never apply**. Same disconnection, opposite end:
one cannot report, the other cannot repair.

##### (2) Rejected, by name, with what each admits

The screening question is not *is this true?* but ***what does it admit?*** (§8.9.24). Applied to
the four candidates on the table, none of which arrived endorsed:

- **Age tiers proportional to write rate — REJECTED.** At sufficient velocity the eligible window
  collapses onto the round currently being cited. It admits **archiving a round while it is still
  load-bearing**, which is worse than a large file, and it admits it *precisely when the project is
  moving fastest*, i.e. when the record is most in use.
- **Size-triggered archive of the oldest N regardless of age — REJECTED**, same door, different
  key. A remedy keyed on **rank** or on **rate** can always reach a live round, because neither rank
  nor rate is a fact about whether the content is still doing work.
- **The gate is advisory when no remedy exists — REJECTED, and this is the one to reject loudest.**
  It admits *every* breach, since any breach can be re-described as one for which no remedy applied.
  It would take the defect in (5) and **mint it as policy**: a hard gate that produces a verdict
  nothing is obliged to act on is not a gate.
- **The threshold is simply wrong for this velocity — TRUE BUT INSUFFICIENT.** Raising 51,200 buys
  one or two rounds and reproduces the identical standoff at the new number, because the structure —
  trigger on one variable, remedy on another — is untouched. Not adopted; not needed, given (3).

##### (3) The ruling: eligibility is keyed on supersession, and age is demoted to one sufficient clause

**MODALITY, DECLARED (§8.9.25's drafting rule): the three clauses below are `SUFFICIENT`. Any one of
them makes an entry archive-eligible; none is necessary.** An entry in `decisions.md` is eligible
when:

- **(a) SUPERSEDED** — a *later, active, differently-authored* entry names it and overturns, retires
  or replaces it. The naming must be explicit; "covered by the newer round" is not a supersession.
- **(b) DISCHARGED** — every obligation the entry creates is closed, and **the closing entry is
  named and is not this entry**. Self-asserted discharge does not count; an entry may not certify
  its own completion.
- **(c) AGED** — older than the tier's window (30 days at Tier 1, 7 days at Tier 2). **Unchanged, so
  nothing eligible before this ruling becomes ineligible after it.**

**Why supersession and not age.** Age was always a *proxy* for the only property that makes a cut
safe — *is this content still doing work?* — chosen because nothing measured that property. This
project does measure it: supersession and discharge are **positive facts asserted by a later named
entry**. A remedy keyed on a positive fact **cannot reach a live round by construction**, which is
exactly the guarantee the two rejected candidates in (2) cannot give.

##### (4) The safety condition is **resolution**, not silence

The instinct is to require that an archived entry be *uncited*. That is wrong on this project's own
evidence, for two reasons.

First, an absence is the observation this project keeps failing to make (R13, §8.9.23(6), the
field-level reversions). Second, and decisively: **a citation is not harmed by archival, it is
harmed by archival that breaks resolution.** So the obligation moves off the citation and onto the
archivist:

> **An entry may be archived under (a) or (b) only if the `ARCHIVAL POINTER` records, for that
> entry: its exact heading, the archive file path, and every token it mints — `§`/`RAI-`/`R{n}`
> names, struct fields, tool paths, env vars. A citation that still resolves after the cut is
> unharmed; a citation that dangles after the cut is R13, manufactured by the archivist.**

A citation screen is retained, but **as a veto, never as a grant**: an entry cited by any entry
carrying an **OPEN** obligation is ineligible under (a), (b) *and* (c), regardless of anything else.
Screens withhold eligibility; they do not confer it.

##### (5) What the gate measures once it can fire — and what it means when it still cannot

Run against Round 10 as a demonstration rather than as an assertion: the Fact Checker entry
*"'six declines' is a correct self-tally but not a measure of register restraint"* is **SUPERSEDED
by name** — Round 12 records *"the decline tally is retired; 3 of 8 declines survive"*, a later,
differently-authored entry that overturns it explicitly. Eligible under (a). The remaining four
Round 10 entries are not: Niobe's provenance classes are cited by live work in three agents' lanes,
Mouse's `counters_abi` and `pipeline_variants` records are cited 105 and 35 times respectively, and
the Fact Checker canary-observability entry is standing normative text.

**So the remedy applies — and it does not clear 51,200 this round.** I say that rather than round it
up. One entry is a few KB against a ~16 KB overhang.

**That residue is the ruling's real product.** With eligibility keyed on supersession, a file over
the threshold is no longer an unmeetable condition; it is **a true report that the project is
carrying more live, unsuperseded, undischarged obligations than the threshold assumes**. The correct
response to that reading is *to close obligations*, not to cut the record. The gate stops being
unsatisfiable and becomes **diagnostic** — a backlog reading, which is information the project did
not previously have. A breach that means something is worth more than a breach that could be
cleared.

##### (6) The class underneath: an **inert mechanism** — the register worked, the reader did not exist

Three mechanisms failed the same way this week, and Link's phrasing is the best available:
***"the register worked; the reader did not exist."***

- the **archive gate** — fires correctly, is read by Scribe, and hands her no action she may take;
- **`ci/check_open_reds.py`** — computes a correct red and the red is not read;
- **`ci/check_flake_witness.py`** — wired into `ci.yml` at four call sites and never handed a real
  log to parse.

And a fourth, older and mine: **`LedgerEntry.device`** was recorded and read by **zero predicates** —
*"a field no predicate reads is a comment with a schema, not a guard"* (Round 9). **I wrote the
specimen four days ago and never named the class.**

**It is distinct from every class already on the register, and the discriminator is mechanical.**
An *unfalsifiable observable* is a defect **in the predicate**: it evaluates and can only return one
value. A **dangling reference** (R13) is a defect **in resolution**: the referent is absent or
wrong. The **self-witness bound** (§8.9.23) is a defect **in the observer's position**: it reports
its own side of a boundary. An inert mechanism has **none of these**. Its predicate is sound, its
referents resolve, it observes the right side — **and its verdict changes nothing.**

The cleanest proof of distinctness is that the two specimens in CI **each ship a negative control**
(`ci/negative_control_open_reds.py`, `ci/negative_control_flake_witness.py`). A negative control
demonstrates that a check *can* go red. **It is the exact instrument that refutes "unfalsifiable" —
and it is silent on "inert".** A check can be demonstrably falsifiable and completely inert, and
these two are.

Two forms, both admitted under one name:

- **INERT/UNREAD** — a verdict is produced and no one consumes it (`check_open_reds`,
  `check_flake_witness`, `LedgerEntry.device`).
- **INERT/UNACTIONABLE** — a verdict is consumed by a reader who has **no permitted action** (the
  archive gate; and any "advisory when no remedy exists" rule, which is why (2) rejects it).

**MINTING RULE, binding on every new mechanism from here:**

> **Name the reader and name the action it takes on red, before the mechanism ships.** A check that
> cannot answer *who reads this* and *what changes when it is red* is inert, and an inert check is
> worse than no check: it consumes the attention a real one would have had, and it reports coverage
> it does not provide.

This is the sixth time a rule already in the register answered a question I was treating as new,
and the first time the earlier statement of it was **my own unnumbered prose**. The register is
still under-indexed by question; Fact Checker owns the question→ruling index (§8.9.25 carry-forward)
and this section is the sixth entry for it.

##### (7) What this ruling admits, in the weak form

§8.9.26 **lowers no threshold, relaxes no gate, and turns no red green.** The 51,200-byte and
20,480-byte triggers are unchanged; the 30-day and 7-day windows are unchanged and remain sufficient;
both recorded breaches stay on the record as breaches. The only thing it adds is **two more ways for
an entry to become eligible, each requiring a positive assertion by a later, different entry**, plus
an obligation on the archivist that did not exist before (4). It admits **strictly less** than the
status quo in the one direction that matters — the status quo, taken literally, would eventually
force a cut under duress with no rule saying which; this one forbids that cut.

It is not, however, a free clean bill like §8.9.24's. **(3)(b) is the exposed clause:** "every
obligation is closed" is a judgement about a set, and a set is where an omission hides. The
`not-this-entry` and `named-closing-entry` requirements are what keep it honest, and if a
discharge-under-(b) is ever found to have archived a live obligation, **(b) is withdrawn and (a)
stands alone** — recorded here in advance, so the falsifier is not written after the fact.

---

#### 8.9.27 RULING — `Gemm`'s `witnessed_at` is **withdrawn before it ships**, because the remedy it asks for already exists twice and a third copy would be inert; and the two open items are **one item**, because the discharge rests on a suite one of whose members skips for a reason that never happened (2026-08-05)

**By:** Morpheus. Closes the two items §8.9.25 left open with named owners. **Both close on the
same fact**, which is why they are ruled together.

##### (1) `witnessed_at` — withdrawn; **Mouse is not needed, and must not be asked**

§8.9.25 found the record blind rather than the axis: `transB=1` mints the same proof key as its
`transB=0` sibling, so a ledger reader cannot tell a key proven at both values from one proven at
one. That reading is correct. The proposed remedy — a non-key `witnessed_at` field on
`LedgerEntry`, owner Mouse — is **withdrawn**, on two independent grounds, either of which is
sufficient.

**First: the remedy is already ruled, and already built.** §8.9.23 settled this exact question for
`Conv` and the answer was *disclosure plus a CI-time suite*, not new proofs and not new key
components. Both halves exist for `Gemm` today:

- `registry::OpSpec::blind_axes` on the `Gemm` row declares `alpha`, `beta`, `transA`, `transB`
  (`rust/src/ops/matmul.rs`), rendered onto the claim line the user reads by
  `disclosure::blind_axes_clause` and counted by `Disclosure::blind_axes_disclosed`;
- `tests/ops/test_gemm_and_pool.py` **is** the CI-time suite that disclosure points at — its module
  docstring says so in as many words — and `_GEMM_CASES` runs `transA`, `transB`, `transAB`,
  `alpha_only`, `beta_only`, `negative_alpha` and every legal `C` broadcast shape through
  `test_gemm_matches_cpu_across_the_value_space`, which **does** carry `require_vulkan`.

**A third answer to a question already answered twice is not thoroughness.** This is the sixth
occurrence of the pattern named in §8.9.26(6) and it is the reason that section names an index owner.

**Second, and this is the disqualifying ground: `witnessed_at` as specified is an inert mechanism —
§8.9.26(6), form UNREAD.** It is defined as *non-key*, which means no predicate may read it; a field
no predicate reads is `LedgerEntry.device` again, the specimen from which that class was drawn. And
the alternative is worse: the moment a predicate does read it, a non-key field has become a key
component through the back door, and `Gemm`'s key silently acquires the two axes §8.9.23 ruled out
of it. **The field is inert if it stays honest and unsound if it becomes useful.** Withdrawn.

**Status: CLOSED as a schema note. Mouse's queue is unchanged** — he is on ORT's shape inference,
which is what blocks BERT, and this would have interrupted him for a field with no reader.

##### (2) The skip that names a refusal that never happened — and why (1) depends on it

§8.9.25 found `test_transb_is_a_transpose_and_not_a_relabelling` — the only test in the tree that can
tell a **transpose** from a **relabelling**, because `B.T` and `B` are the same multiset and every
statistical summary of them is identical — takes **no `require_vulkan`**, unlike the fifteen
neighbours that do. With no EP loadable, `_models.is_vulkan_claimed` returns `False` on the
conservative path and the test reports:

> `skip("EP did not claim the transB identity case")`

**That is a skip asserting a claim about the EP's behaviour on a run in which no EP existed.** The
EP did not decline; it was never there. The string is a false statement about a counterfactual, and
it is the strongest specimen of the family this project has spent the week cataloguing, because it
is *green*, *honest-looking*, and one line from correct.

**The two items are one item.** (1) discharges `Gemm`'s blind-axis obligation **by pointing at that
suite** — and a suite is a discharge only if it runs, or says truthfully why it did not. One member
of the suite currently says something false when it does not run. **Fixing (2) is what makes (1)
true**, so (1) is closed *conditional on* (2), not before it.

**Scope, measured, not estimated.** Five call sites in `tests/ops/` share the shape — every
non-parametrised structural discriminator:

| file | test |
|---|---|
| `test_gemm_and_pool.py` | `test_transb_is_a_transpose_and_not_a_relabelling` |
| `test_gemm_and_pool.py` | `test_a_broadcast_column_c_is_added_down_the_rows` |
| `test_gemm_and_pool.py` | `test_the_pool_reduces_its_own_channel_and_not_its_neighbour` |
| `test_conv.py` | `test_depthwise_reads_only_its_own_group` |
| `test_conv.py` | `test_zero_padding_is_skipped_accumulation_not_a_clamped_read` |

The correct form already exists in the tree twice over, which is what makes this a defect rather
than an open design question. `tests/ops/probe_conv_tolerance.py` meets the identical condition and
raises `ERROR(instrument)` — *"the EP did not claim case …"* is loud there, not a skip.
`tests/ops/test_no_cpu_fallback.py` carries `require_vulkan` on **every** test that probes the EP.
And the `test_*_declines_*` family is **explicitly not in scope**: `assert_vulkan_does_not_claim`
documents its own vacuity ("with zero devices, the EP can never claim any node"), and a disclosed
vacuous pass is a different thing from an undisclosed false skip.

##### (3) Owner: **Trinity**, and the fix is not the decorator

**Trinity fixes it**, on the harness ground, not the op ground: `require_vulkan` is her fixture in
`tests/ops/conftest.py`, and the defect is a harness-level one that surfaced in two op suites owned
by someone else. Mouse is not the owner here either.

**And the deliverable is not "add the fixture to five tests."** Five decorators fix five instances
and leave the sixth to be written next week by whoever adds the next structural discriminator. Per
§8.9.26(6)'s minting rule — *name the reader and the action* — what is owed is a screen that makes
the shape unrepresentable:

> **A test in `tests/ops/` that calls `_models.is_vulkan_claimed` and does not request the
> `require_vulkan` fixture is a CI failure.** Reader: the `lane_checks_suite`. Action on red: the
> test is fixed or the exemption is written down with its reason.

That is one predicate over the AST of a directory, it fails today on exactly the five rows above,
and it has a positive control by construction: remove the fixture from any of the fifteen compliant
neighbours and it must go red.

##### (4) What this ruling admits

§8.9.27 **removes** a proposed schema field and **adds** a CI screen. It grants no claim, mints no
key component, moves no tolerance, and cannot turn a red row green — the screen it requires can only
turn green rows red, and is expected to, five times, on the day it lands. The one thing it admits is
`Gemm`'s blind-axis obligation being **CLOSED**; that closure is explicitly conditional on (2)
landing, and if (2) does not land, **(1) reopens**, because the suite it cites would not be a suite
that runs.

---

## 9. Testing and benchmarking strategy

### 9.1 Differential testing against the ORT CPU EP — Trinity


The oracle is **ORT's own CPU EP**, running the same ONNX model. Not numpy, not a reference we
wrote. This is the single most important testing decision in the project: it means a test failure
is unambiguous, and it means we cannot accidentally encode our own misreading of an ONNX spec into
both the implementation and the expectation.

| Layer | Location | Purpose | Gate |
|---|---|---|---|
| Op correctness | `tests/ops/` (pytest) | Per-op, per-dtype, per-shape differential vs ORT CPU. Models built with the ONNX IR API in `_models.py`. | **Required on every PR.** |
| Claim assertion | `tests/ops/test_claim_diagnostics.py` | Asserts the node *actually ran on `VulkanExecutionProvider`*, via the claim diagnostics. Prevents vacuous CPU-fallback passes. | **Required.** |
| ONNX backend node tests | `tests/backend/` | The ONNX project's own node tests through the EP. | Required. |
| Conformance fuzzing | `tests/conformance/` | Bounded property-based fuzzing of claimed ops against the ONNX standard, one op per subprocess so a native crash cannot abort the run. | Opt-in `workflow_dispatch`. |
| Validation layers | all suites, debug builds | `VK_LAYER_KHRONOS_validation` clean is part of "done" for any engine change. | Required in the CI debug lane. |
| Leak / teardown | stress scripts across many sessions | RAII teardown leaves no `VkDeviceMemory`, no pipelines, no descriptor pools. | Required. |
| **Barrier-backend parity** | every lane, run twice | The full suite with the default backend and again with `ep.force_legacy_barriers=1`, asserting **identical** numerical results (§7.5 item 5). Without this, the legacy `vkCmdPipelineBarrier` path we carry for 31% of Android and 12% of Windows would never be executed by any test we own. | **Required on every PR.** |

**The vacuous-pass trap, stated plainly.** Because CPU fallback is always correct, a test that
merely compares outputs will pass whether or not the EP ran anything. Every op test **must** assert
the claim. This is non-negotiable and it is the first thing I will look for in a review.

**Tolerances.** Stated per family in `tests/ops/`, defaulting to `rtol=1e-5, atol=1e-5` for fp32
elementwise. Reductions and GEMM get looser tolerances tied to accumulation order — but a
tolerance is *derived and documented*, never widened to make a red test green. Widening a
tolerance requires Trinity's sign-off and a note in the test.

**Cross-platform.** The same suite runs on every CI lane. As landed by Trinity
(2026-07-28): a **Linux lavapipe lane** (via `apt`) and a **real Windows Vulkan lane** — she found
lavapipe Windows prebuilts via **mesa-dist-win 26.1.3**, driven by `VK_ICD_FILENAMES`, so our
primary development platform has genuine correctness coverage rather than build-only coverage, which
materially raises the value of a green CI run. There is also an always-on no-ICD fallback assertion,
which is exit criterion 4 of M0 tested continuously rather than once. **SwiftShader was evaluated
and rejected** — no usable prebuilts and a ~20-minute build from source — so lavapipe is the only
rasterizer we run and the only one this document should name. ORT is pinned to 1.28 across lanes
(§1.4 C2 depends on that pin). Claims are asserted from the profiling JSON, so "it ran on Vulkan" is
proven rather than assumed. A pass on a software rasterizer alone remains a smoke test, not a
correctness claim — lavapipe does not reproduce driver-specific subgroup, denorm, or precision
behaviour, and **there is currently no CI coverage of any physical GPU, on any platform**; §11.1's
on-device work is what begins to close that gap. Link owns which lanes exist; Trinity owns what runs
on them.

#### 9.1.1 The oracle is validated, not assumed — and what validating it cost

§9.1's premise had one genuinely open risk: the CPU EP is an excellent oracle for `ai.onnx` float
ops, but the quantized contrib path (§1.4) is a different animal, and if the CPU EP could not serve
as an oracle for a GenAI-built int4 graph then the differential strategy for the *entire* quantized
half of the project — the half containing all three XL kernels — would have needed rethinking
before M2. Trinity ran the experiment rather than reasoning about it. **Result (2026-07-28): the
ORT CPU EP is usable as a differential oracle for the quantized path. §9.1 stands unchanged and no
rework is required.**

Two findings emerged that only running it could have produced, and both are now binding:

1. **The oracle is pinned to `accuracy_level=1`.** `MatMulNBits`' `accuracy_level` 4 (int8/VNNI)
   diverges from levels 0–3 by ~3.6e-3 at K=1024, N=512. Unpinned, ORT would select a level based
   on the CPU it happens to be running on, so our reference values would have drifted **silently
   across CI runner hardware** — and the resulting flake would have looked like a bug in our
   kernel. Generalized as a rule: **any oracle knob whose value the runtime chooses from the host
   machine must be pinned explicitly and recorded in the test, not left to default.** An oracle
   that changes with the machine is not an oracle. This applies to future additions too — thread
   count, arena settings, graph-optimization level.
2. **fp16 activations produce NaN/Inf on ORT 1.27**, independently reproducing the null-allocator
   `PrePack` bug Fact Checker found and that drove Tank's version pin. The fp16 oracle test is
   gated on ORT ≥ 1.28 and runs in CI. Three independent lines of evidence — Fact Checker's source
   read, Tank's build experience, Trinity's numerical failure — now converge on 1.28, so the
   `ORT_API_VERSION_MIN = 24` compile floor with a **pinned runtime** of 1.28 is settled on
   evidence rather than on caution.

**The one documented exception to "the CPU EP is the oracle".** Trinity implemented Mouse's
three-regime tolerance policy but checks **dequantize bit-exact against NumPy, not against the CPU
EP**. This is correct and I am ratifying it as the general rule for *layout* semantics: where a
kernel's job is to interpret a bit layout defined by a schema, comparing against another
implementation of that same schema does not test the reading — a shared misreading passes on both
sides, which is precisely the failure mode §9.1 exists to prevent. So: **behaviour is checked
against the CPU EP; bit-layout interpretation is checked against an independently written
specification of the layout.** The two are complementary and neither substitutes for the other.
This is C6 (§1.4 — per-layer verification, never final-logits) applied one level down, and the two
now reinforce each other: with per-layer capture implemented, a wrong kernel is *locatable* without
comparing logits across a 150k vocabulary, and with bit-exact dequant checks a wrong *unpack* is
locatable without running a kernel at all.

**No autouse EP fixture.** `conftest` no longer registers the EP as an autouse fixture, so tests
that do not require Vulkan actually execute instead of skipping wholesale. This changes what a
green run means, which is why it belongs in this document: 206 collected and 8 passing **with no EP
built at all** is now a meaningful signal rather than a vacuous one — and among those 8 is the
runtime half of C1, confirming that `com.microsoft::NotARealOp` takes the ordinary decline path.
**C1 is therefore enforced from both ends**: Tank's static ban on the domain as a *value* in
`layering.rs` (no `==`, `!=`, `matches!`, `if let`, `starts_with` can express it) and Trinity's
runtime assertion that an unregistered contrib op declines like any other unregistered op. A
constraint checked only statically can be satisfied by code that never runs; a constraint checked
only at runtime can be reintroduced in a path no test reaches. C1 now has neither hole.

#### 9.1.2 Execution status — what has actually run, as of 2026-07-30T05:48:29-07:00

This document describes a design and a partially-implemented crate. It must not be read as
describing a working GPU pipeline, and the following is stated here so that no reader has to infer
it. **This section changed materially three times on 2026-07-29 — read the boundaries, not the
headline.**

- **45 op rows are `Live` and executing on the GPU through ONNX Runtime**, up from one that morning.
  Coordinator-verified on **both** local devices, with identical results on each: `test_elementwise`
  33 passed / 3 failed, `test_op_table` 49 passed / 28 failed. The claimed nodes run via
  `Compile` → `OrtNodeComputeInfo` → dispatch on ORT-owned tensors, attributed to
  `VulkanExecutionProvider` in the profiling JSON, matched against the ORT CPU EP.
- **Both barrier backends have now executed, and they agree bit-exactly.** Barrier parity: 46 passed
  / 28 skipped (the skips are `Staged` rows) on each device, under the default `synchronization2`
  path and under `ep.force_legacy_barriers=1`. **This is the first execution of the legacy backend
  we carry for ~31% of Android** (§7.3), and until it ran, that entire compatibility argument was
  untested code.
- **Intel and NVIDIA produce identical results across all of it.** The stricter implementation
  raising nothing extra is the informative half: it is evidence that these kernels are not leaning
  on anything the specification leaves undefined. It is not evidence about Adreno, Mali or MoltenVK.
- **A 2.2 GB production model has been through the EP end to end.** *2026-07-29T21:14:03-07:00.*
  Phi-3.5 (external data, fp16, an `If` in its prologue) loaded on the RTX 4060, ran, produced 65
  outputs, declined **all 363** nodes with machine-readable reasons, fell back to CPU cleanly, and
  was bit-identical across two sessions. **Zero nodes claimed** — so this is not an execution result
  at all; it is the *conservative-claiming machinery* (§1.3, C6) working at a scale nothing else has
  tested, and the decline histogram is what it produced (§8.8, §10.0.1 R8). Nothing about this run
  put a Phi-3.5 node on the GPU.
- **What this is not.** Windows only. Real hardware only. **This desk only.** The Linux lane has
  never executed a claimed node, the lavapipe CI lanes have not run since these changes landed, and
  CI still has no GPU hardware. **A result obtained only on this desk is not a result this project
  has** — that line was written when the news was thin and it applies unchanged now that the news is
  good, which is the only way a standard is worth anything.
- **Still entirely unexecuted:** every contrib op, all quantized paths, every fused multi-node
  island beyond what the elementwise tests build, and every platform other than Windows-on-this-
  machine. 28 rows remain `Staged` by design.
- **The failures are load-bearing and must not be tidied away.** The differential harness refuses to
  score a CPU-fallback run as a pass — *"VulkanExecutionProvider did not execute any node of this
  model — the CPU-match check would be a vacuous pass"*. **A weaker harness would report every one
  of those as green**, because a graph that silently falls back to CPU matches the CPU EP perfectly.
  That refusal is this section's discipline implemented in the test layer rather than asserted in
  prose, and it is worth more than the prose: **a differential test that does not verify the EP ran
  is a test of ORT, not of us.** It has already paid for itself once — see §10.0.1 R7, where a
  declared liveness flag would have bought exactly such a vacuous pass on `Add-i32`.
- Every "green" count reported to date at the ~300 level still measures mostly **host-side logic**:
  claim predicates, registry invariants, the layering lint, decline paths, and the harness itself.
  A growing minority now reach a device.

**The failure mode has inverted twice, and this section's job with it.** Until 2026-07-29 the risk
was **overclaiming execution we had not performed**. It then became letting *"a kernel dispatches"*
stand in for *"the EP works"*. It is now the subtler version of the same thing: letting **45 Live
rows on two GPUs on one desk** stand in for *coverage*. The gap is every contrib op, every quantized
path, Linux, CI, and every platform in `PLATFORMS.md` we have never touched. Anyone citing today's
numbers must state the qualifiers — Windows, local hardware, no CI, 45 of ~174 inventoried rows.

The rule this encodes, and it applies to every document in `docs/`: **a test count is a claim about
what was executed, and it must not be allowed to imply more execution than occurred.** The same
discipline that produced the RAI-003 platform disclosure in `README.md` and Link's
unverified-usability statement in `PLATFORMS.md` §8 applies to our own test numbers, and it applies
hardest to good news.

**UPDATE 2026-07-30T05:48:29-07:00 — the third inversion, and it is not a subtler version of the
last two. It is the opposite one.** The previous two inversions were both about *overstating
execution*. This one is about having understated nothing and still published a false statement.
Coordinator-verified on **both** local devices, real 2.2 GB Phi-3.5, VulkanEP output against a
CPU-only run of the same session:

```
cpu logits : [-13.0859, 13.0312]   argmax 30751
vk  logits : [  0.0000,  0.0000]   argmax 0        top-10 overlap 0/10
```

with `compile_calls: 1, subgraphs_live: 161, compute_calls: 161, compute_failures: 0,
dispatches_executed: 161, islands: 161`, all 161 `com.microsoft::MatMulNBits` nodes offered **and
accepted by ORT**, output #64 (a KV-cache output) differing from CPU by 25.27 and therefore **not**
uniformly zeroed, identical on Intel Iris Xe and RTX 4060 — a deterministic logic fault, not a race
and not a driver quirk — **and the entire test suite green.**

**Every qualifier this section has ever demanded was honoured, and the disclosure was still wrong.**
Nobody said "45 Live rows" without saying Windows-only, this-desk-only, no-CI. What was said, and
what nothing in this section forbade, was that **161 nodes executing on the GPU was progress toward
a working model.** It was progress away from one: before this change Phi-3.5 was correct via CPU
fallback, and after it Phi-3.5 is wrong via GPU. The failure is recorded as §10.0.1 **R9**, and the
rule it produces is the strongest one in the register: **for every claim, name the instrument that
would go red if the claim were false.**

The consequence for this section, binding from today: **§9.1.2 is an execution-status disclosure and
must never be read as a correctness disclosure.** Every statement in it above this line answers
*"did our code run?"*. Not one of them answers *"was the answer right?"*. Those are different
sections of this document and from today they are different paragraphs — see §9.1.3.

#### 9.1.3 What the execution counters are licensed to support — RULING on `compute_failures`

*Added 2026-07-30T05:48:29-07:00, on the R9 event.*

`counters.rs` exports six process-wide counters (`compile_calls`, `subgraphs_live`,
`subgraphs_stub`, `compute_calls`, `compute_failures`, `dispatches_executed`). They are the
project's only reliable answer to *"did anything execute"* and they must stay exactly that. The
ruling has two halves because the question has two halves.

**Half one — the documented meaning, constrained.** `compute_failures` is an **execution-status
counter**. It counts the times `Compute` returned a non-null `OrtStatus` — that is, the times our
own code detected a fault and reported it. Its licensed reading is exactly one proposition:

> `compute_failures == 0` means: **no dispatch reported an error it was able to detect.**

It is **not** licensed to support any of the following, and each of these readings has appeared in
our own summaries: "the kernels are correct"; "the graph produced the right answer"; "the run is
usable"; "nothing went wrong". A kernel that writes zeros into every output buffer, submits, waits
on the fence, and returns null is a **complete success** by this counter, by `compute_calls`, by
`dispatches_executed`, and by `subgraphs_live` simultaneously. That is not a hypothetical — it is
the 2026-07-30 reading, verbatim.

The general form, which applies to all six and to every counter added later: **an execution-status
counter's zero is a statement about the detector, not about the computation.** Its silence set —
the propositions it cannot be false about — is *everything downstream of "the dispatch returned"*.

**Half two — prose cannot close this, and I am not pretending otherwise.** Constraining the
documented meaning is necessary and I have just done it. It is not sufficient, for a reason this
project has already proved twice: R6 shows a written rationale carrying a false number for weeks
wearing the authority of documentation, and R7 shows a hand-declared fact drifting from the machine
fact it duplicated. A prose constraint on how a number may be read is a **declaration** — and
"derive, do not declare" (R7) applies to our own document as much as to the harness. The reading
`compute_failures: 0 ⇒ it works` is not closed by forbidding it; it is closed by **putting a red
instrument next to it that goes red when the composite claim is false.**

The mechanism, therefore, and it is a mechanism rather than a paragraph:

1. **A correctness verdict is emitted next to the counters, from the same run, or the run reports
   `UNMEASURED`.** The verdict is `model_output_equivalence` ∈ {`MATCH`, `DIVERGENT`, `UNMEASURED`}
   against a CPU-only execution of the *same session on the same artifact* — see §10.0's metric
   ruling. `UNMEASURED` is a first-class value and it is the **default**: a run that did not compare
   does not get to be silent about not comparing (§7.9 rule 1, third state; R7, absence is not a
   negative).
2. **No counters summary may be quoted without its verdict.** `epctl --check-counters` reports the
   verdict field alongside the six counters; a counters file with no verdict field reports
   `UNMEASURED` explicitly rather than omitting it. Owner: Switch for the emission, Trinity for the
   comparison, Niobe for `PERF.md`.
3. **The counter is not renamed and the C ABI struct does not change.** `VulkanEpCounters` is a
   published C ABI surface consumed by `epctl`, `probe_allocator.py` and `test_phi35.py`; renaming
   `compute_failures` to something more honest would break every one of them to fix a documentation
   problem. **Compatibility outranks API elegance** (standing user ruling). The verdict is an
   *addition*, never a mutation — new optional field, absent means `UNMEASURED`.

**The precedent, named because it is the same defect one layer down.** `Compute` signals success to
ORT by returning `null`. **The absence of a report is the success report.** That is the identical
shape as `compute_failures: 0`: in both cases the "everything is fine" reading is what you get when
nothing looked. An ABI whose success value is the null pointer cannot distinguish *succeeded* from
*never checked*, and a counter that counts detections cannot distinguish *no faults* from *no
detector*. We do not control the ORT ABI and are not proposing to; we control what we conclude from
it, and the conclusion is: **in this codebase, every "no error" signal on the compute path is
treated as `UNMEASURED` until a positive control or a differential comparison converts it.** This
is criterion 3's ruling (§10 M0) and §7.9 rule 1, applied a third time, on the execution path
rather than on the validation layer or the capability probe.

#### 9.1.4 Model discovery must never guess, and provenance must be a contract, not prose (issue #11)

`tests/ops/test_phi35.py` hardcoded the Foundry Local cache path for Phi-3.5-mini:
`Microsoft\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32\...`. By 2026-08-05 the real
cache held the same model at `Microsoft\Phi-3.5-mini-instruct-cuda-gpu-2\v2\...` — Foundry's own
on-disk naming is versioned by its CLI's internal catalog revision, which is not something this
repo controls or is notified of. Nothing in the test noticed; the fixture just skipped, and the
day's investigation ended with a manually-created directory junction bridging the two paths as a
workaround. A hand-made junction that papers over a version mismatch is exactly the kind of
undocumented, single-machine fix that reappears as a mystery the next time someone clones the
repo, and it is now deleted — the discovery mechanism below finds the real path without it.

**The fix is `rust/tools/foundry_discovery.py`, and its one rule is that it is never allowed to
guess.** `resolve_model_path(spec)` tries the Foundry CLI's own bookkeeping first —
`foundry cache list --verbose --variants -o json`, which reports `variantName`,
`executionProvider`, `cached`, and `cachePath` per cached variant — and falls back to a
constrained filesystem search only if the CLI itself could not be asked. The search is restricted
to an *exact* `<variant_name>` or `<variant_name>-<suffix>` prefix under
`<cache_root>/Microsoft/`, so a hit is guaranteed to be a different **version** of the requested
model, never a different **model**, and Foundry's own convention of folding the execution
provider into the variant directory name (e.g. the `-cuda-gpu` suffix) means a provider mismatch
at this layer surfaces as "missing," not as a silent substitution. There are exactly four ways
`select_variant`/`resolve_model_path` refuse to proceed, each raising `FoundryDiscoveryError`
with the exact identity requested and the remedy command, never a bare `None` or an arbitrary
pick: **missing** (nothing matches the variant name at all), **ambiguous/duplicate** (more than
one cached entry matches — the error lists every candidate rather than picking the first or the
newest), **stale** (the catalog says `cached: true` but the file the manifest points at is not on
disk, or the catalog entry itself is not `cached`), and **wrong execution provider** (the variant
name is cached, but under a different `executionProvider` than the one requested — checked as an
explicit field on the CLI-manifest path, since the manifest could in principle report any
provider string against any name). Negative tests for all four cases, at both the pure
CLI-manifest layer (synthetic JSON, no Foundry install needed) and the filesystem-fallback layer
(real `tmp_path` trees), live in `tests/ops/test_foundry_discovery.py`. `test_phi35.py`'s
Phi-3.5 and gpt-oss-20b fixtures now build a `FoundryModelSpec` (variant name, execution
provider, onnx filename, download alias) and call `resolve_model_path`, catching
`FoundryDiscoveryError` and converting it to `pytest.skip(str(exc))` — the resolver's job is to
fail loudly when asked directly; the fixture's job is the ordinary "this environment doesn't have
a confidently-known-good model" skip semantics every other model-gated test already has.

**Provenance became an enforced contract, not prose.** `bench/results/model_provenance.json` had
recorded MobileNetV2-12's and BERT-SQuAD-12's URL/SHA-256/byte-count since 2026-08-03/04, but
nothing read it programmatically — it was cited in `docs/OP_COVERAGE.md` and nowhere else. A
downloaded model file and its provenance record could silently diverge and nothing would notice.
`rust/tools/model_provenance.py` makes it a contract: `load_provenance()` reads the JSON,
`verify_file(path, entry)` streams a SHA-256 (chunked, not slurped — BERT-SQuAD-12 is ~436MB) and
raises `ProvenanceMismatch` with the expected-vs-actual size and hash on any disagreement, size
checked first so a truncated download is reported as a truncated download rather than waiting on
a full hash pass. The contract now also carries **MNIST-12**: URL
`https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx`
(the `github.com/.../raw/<branch>/...` form, not `raw.githubusercontent.com`, is required for
every LFS-tracked entry in this file — the latter serves a ~130-byte pointer, not the model),
26,143 bytes, SHA-256
`5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd` — computed directly from the
downloaded artifact, since no pinned hash for this exact file existed anywhere upstream or in
this repo before this entry. MNIST-12 is the smallest real model this project validates against:
12 nodes, 26KB, no external data, and it is the intended fast post-build smoke gate — if it
disagrees or dispatches nothing, nothing larger (MobileNetV2-12's ~14MB, BERT-SQuAD-12's ~436MB)
is worth running yet.

**The non-vacuous dispatch/agreement gates are now automated, not manual.**
`tests/ops/test_small_model_provenance.py` promotes the differential validation this section has
always required into a repeatable pytest suite rather than a one-off manual run: a fast tier
pins the exact provenance-contract content and verifies any locally-cached model file against it
(skips cleanly if the file is not present — this suite never downloads anything), and a
`@pytest.mark.slow` tier shells out to the **existing** `probe_model_op_census.py --run` and
`probe_model_output_agreement.py` — never reimplementing their guards — asserting
`dispatches_executed > 0` and `verdict == "AGREE"` (with `VulkanExecutionProvider` re-confirmed
present in the session's providers, so a weakening of the probe's own guard would still be
caught here). Output is written to pytest's own `tmp_path`, never into the tracked
`bench/results/` directory: re-running `probe_model_op_census.py` with the same `--name` was
found, during this same investigation, to silently overwrite two already-committed tracked
artifacts (`op_census_mobilenetv2.json`, `_claim_log_mobilenetv2.jsonl`) — a test suite that
writes there by default would reintroduce that exact defect on every CI run.

**The proven Windows build recipe is now documented** (`rust/README.md`, "Building" →
"Windows: MSVC environment via `vcvars64.bat`"): capture `vcvars64.bat`'s environment with
`cmd /c "call vcvars64.bat >nul && set"`, apply every `KEY=value` line natively in the *same*
PowerShell process via `[System.Environment]::SetEnvironmentVariable(..., 'Process')`, then set
`VULKAN_SDK`/`LIBCLANG_PATH` the same way, then invoke `cargo build --release` directly from that
PowerShell session. Chaining `vcvars64.bat && set X=... && cargo build` inside one `cmd /c "..."`
string is fragile on this toolchain, but **not** because a later child process fails to inherit a
`set` mutation made earlier in the same chain — it does; a `set X=1` followed later in the same
`cmd /c "..."` invocation by a genuine child process (another `cmd /c echo %X%`, or `cargo
build`) sees `X` correctly, confirmed by direct test. The two real, verifiable hazards are
parse-time `%VAR%` expansion and PowerShell/cmd quoting. `cmd.exe` expands every `%VAR%` token in
a single command line once, before executing any part of that line, so `set
PATH=%VULKAN_SDK%\Bin;%PATH%` chained after `call vcvars64.bat` in the same line reads the
pre-vcvars values of `%VULKAN_SDK%`/`%PATH%` (or a literal, unexpanded token if undefined) —
confirmed: `set X=1 && echo %X%` in one line prints the literal text `%X%`, not `1`. Delayed
expansion (`setlocal enabledelayedexpansion` + `!VAR!`) works around this but is easy to omit.
Separately, building a long `cmd /c "..."` argument from PowerShell requires getting nested
double-quotes (for paths containing spaces) and `$`/backtick escaping exactly right, so PowerShell
does not interpolate before `cmd` ever sees the string; a small quoting mistake drops or corrupts
arguments with no error message, only a confusing build failure or a wrong environment. Capturing
vcvars' output and applying it natively inside the current PowerShell process — using
PowerShell's own `$env:`/`[System.Environment]` mechanisms instead of `%VAR%`/chained-`cmd`
syntax — sidesteps both hazards entirely, since nothing is expanded or quoted through `cmd.exe`
more than once, and every applied variable is inspectable (`$env:INCLUDE`/`$env:LIB`) in the same
shell that runs `cargo build`. That determinism and inspectability, not a false claim about child
processes losing `set` mutations, is why it is the only recipe this document recommends for
Windows.

**Issue #19 closed the remaining gap: every hardcoded cache path outside the fixture PR #15
migrated, and a standing guard against a new one.** PR #15's resolver reached `test_phi35.py`
alone; a repo-wide inventory found ~34 more sites — 11 live tools (`bench/exec_census.py`,
`bench/island_attribution.py`, and nine `rust/tools/probe_*.py`/`roofline_split.py` scripts) that
called a hardcoded path directly, and 23 archived one-off investigation scripts under
`bench/results/` whose default path is the exact historical artifact the recorded result was
actually measured against (an earlier pass through this inventory undercounted the archival group
at 22; a precise grep for the `os.environ.get("PHI35_MODEL"` pattern during PR #31 review found
the exhaustive, verified count is 23 — `bench/results/probe_planted_kv.py` names the path fragment
only in a docstring and is neither of these groups). These two groups are migrated differently,
deliberately:

  - **Live tools** now build a `FoundryModelSpec` and call `resolve_model_path`, exactly like
    `test_phi35.py` — the same fail-loud four-way `FoundryDiscoveryError` from §9.1.4 applies,
    surfaced as each tool's own pre-existing error-handling idiom (`SystemExit`, a printed
    `ERROR(instrument)` plus non-zero return, or a graceful `SKIP` for tools that already
    tolerated an absent model).
  - **Archival scripts are never pointed at the live resolver.** A resolver call picks whatever
    is cached *today*; an archived script's job is to reproduce what was measured on the day it
    ran, which may be a different cache revision than the one currently installed. Each archival
    script instead reads `os.environ.get("PHI35_MODEL", <the exact historical literal path>)` —
    the override lets a maintainer point it at a different cache layout on purpose, but the
    default keeps failing loudly against the one specific path the archived numbers came from,
    never silently substituting a newer cache revision's model. `bench/results/probe_planted_kv.py`
    was found to name the same fragment only in prose (a narrative docstring, not executable
    code) and needed no change.

  No dead duplicates were found among these ~34 sites: every file's content is unique (no two
  hash-identical, and no pair scores above a 0.55 line-similarity ratio against any other), so
  issue #19's "delete dead duplicates" acceptance criterion is satisfied vacuously here rather
  than by a deletion.

  **The standing guard is `ci/check_hardcoded_foundry_paths.py`.** It is a static, source-text
  screen — no GPU, no Foundry install, no cached model required — that greps every `*.py` file
  for the literal fragment `.foundry/cache/models` (either path-separator spelling) and fails on
  any hit outside an explicit allowlist (`bench/results/**`, `rust/tools/foundry_discovery.py`'s
  own defect-documentation, and the check's own test surface). Deliberately narrow: it does not
  match a model *identity* string used as a resolver key, nor a `pathlib` join built from
  separate literal segments, only a single literal that already spells out the on-disk
  hierarchy — the shape a hardcoded lookup takes, and the shape a resolver call does not.
  `ci/negative_control_hardcoded_foundry_paths.py` proves the screen is load-bearing with a
  REPLAYED arm (the real `bench/exec_census.py` as it stood at `ea427fd`, immediately before
  this migration — the actual defect, not an invented shape) alongside PLANTED and LIVE arms.
  Both are registered in `ci/lane_inventory.py` under `hostfree.hardcoded_foundry_paths` /
  `hostfree.hardcoded_foundry_paths_negative_control`. The allowlist stands at **33** occurrences
  (24 archival `bench/results/*.py` files/lines, plus 3 in the check's own docstring/source, 1 in
  `ci/lane_inventory.py`, 2 in `ci/negative_control_hardcoded_foundry_paths.py`, 2 in
  `ci/test_lane_checks.py`, and 1 in `rust/tools/foundry_discovery.py`'s own defect-documentation)
  — not the "28" an earlier revision of this migration's evidence quoted, corrected after review.

  **The result-identity contract: every archival record names the exact bytes it was computed
  from.** An `os.environ.get("PHI35_MODEL", ...)` override lets an archival script be pointed at
  a different cache layout on purpose (the whole reason these scripts are not migrated onto the
  live resolver, above) — but an override that changed *which file ran* while the emitted JSON
  kept quoting only the historical default path string would let a maintainer silently swap in a
  different artifact and misattribute the result to the one the archived numbers actually came
  from. The same failure shape exists with no override at all, if a stale or corrupted re-download
  ever lands at the exact historical path. Review of PR #31 (issue #19 follow-up) closed this: each
  of the 23 archival scripts that accept `PHI35_MODEL` now also defines a lazily-evaluated
  `_result_identity()` — computed only when a result record is actually written, never at import
  time, so a script's existing graceful-missing-model behavior (an early `.exists()` return, or a
  later `onnx.load`/ORT exception) is unchanged — returning

  ```json
  {"onnx_file": "<the exact path resolved for this run>",
   "onnx_sha256": "<streaming SHA-256 of that exact file>"}
  ```

  merged into every JSON record the script writes. All 23 reuse one existing helper,
  `rust/tools/model_provenance.py`'s `sha256_of(path)` (already exercised for the MobileNet/BERT
  provenance contract), rather than each defining its own hasher. `ci/test_lane_checks.py` proves
  both the structural contract (every script accepting `PHI35_MODEL` defines `_result_identity`
  and stamps both keys, and reuses the shared hasher — not a hand-maintained file list, discovered
  by the same override pattern the fix targets) and the functional one (pointing `PHI35_MODEL` at
  a fixture file changes both `onnx_file` and `onnx_sha256` in lockstep; substituting different
  bytes at the *same* path — no override — changes `onnx_sha256`, proving a silent substitution
  cannot be invisible in the evidence).

  **The three live tools flagged in the same review — `rust/tools/probe_phi35_claim_reading.py`,
  `rust/tools/probe_silent_cpu_rebuild.py`, `rust/tools/roofline_split.py` — had each kept a
  `PHI35_MODEL` environment check ahead of the resolver call**, left over from an earlier
  migration pass; an override present in the environment (from an unrelated archival script run,
  say) would silently skip `resolve_model_path`'s own exact variant+execution-provider validation
  for what are otherwise live tools with no legitimate reason to replay a historical artifact.
  All three now match the other 8 migrated live tools exactly: an unconditional
  `resolve_model_path(_PHI35_SPEC)` at import time, with no environment override of any kind, and
  the same fail-loud `FoundryDiscoveryError` → `SystemExit("ERROR(instrument): ...")` contract.
  `ci/test_lane_checks.py` asserts both the absence of a functional `os.environ.get("PHI35_MODEL"`
  read (a comment naming the variable to explain why there is deliberately no override is fine)
  and the presence of the direct resolver call and its fail-loud exception handling.

  **The contract is not scoped to `bench/results/*.py`, and its own test coverage is not either.**
  A second review round found the same silent-substitution gap in two siblings a
  `bench/results/`-only glob structurally cannot reach: `tests/ops/probe_validation_phi35.py`
  reads `PHI35_MODEL` directly and writes its own `validation_phi35_probe-dev*-*.json` record, and
  `bench/results/probe_push_constants_written.py` never reads `PHI35_MODEL` itself but inherits it
  — via `dict(os.environ)` — into a subprocess of `probe_validation_phi35.py --child`, then writes
  `push_constants_written.json`/`push_constants_sensitivity.json` with only a DLL hash and no model
  identity at all. Both now carry the same stamp: `probe_validation_phi35.py` defines its own
  `_result_identity()` (tolerating a `None` model — resolution itself can fail here, unlike the
  archival scripts, so the failure is reported as data rather than raised out of a writer that
  must still land its record); `probe_push_constants_written.py` imports that same
  `_result_identity` directly rather than re-resolving, since its subprocess is spawned from
  this process's own `os.environ` after `probe_validation_phi35`'s module-level `MODEL` has
  already resolved against that identical environment — the two therefore always name the same
  file.

  **Third review round (PR #31 rejected at `60f0ae7`): the discovery itself was the defect, and
  is now semantic.** Discovery had been a set of source-text regexes in `ci/test_lane_checks.py`,
  and `subprocess\.run\(\s*\[\s*sys\.executable` recognised exactly one spelling — the argv
  written inline at the call site. Both remaining offenders build the argv into a variable first
  (`cmd = [sys.executable, str(PROBE), ...]; subprocess.run(cmd, env=env, ...)`), so the screen
  walked past `rust/tools/device_loss_gate.py`, which spawned `probe_kv_chain_phi35.py` and wrote
  `bench/results/device_loss_gate.json` with no model identity, and
  `bench/results/probe_device_memory_kv.py`, which spawned `probe_kv_bytes_earned.py`, *read* the
  child's record for its byte totals, and discarded the `onnx_file`/`onnx_sha256` in it while
  writing `device_memory_kv_lanes.json`. The same regexes could match their own source text, and
  `dict(os.environ)` as the inheritance test missed `os.environ.copy()` and scored the strongest
  case — passing no `env=` at all — as no inheritance.

  Discovery now lives in **`ci/phi35_identity_audit.py`**, which parses every `*.py` file in the
  tree and reasons over the AST: environment reads through any alias (`os.environ.get`,
  `os.environ[...]`, `os.getenv`, `from os import environ`, `import os as o`), subprocess argv and
  `env=` built inline *or* through variables, script targets named through module-level path
  constants, and JSON *records written to disk* distinguished from a `json.dumps` merely printed.
  The "the model reaches this file" relation is closed to a **fixed point**, not one hop, so a
  wrapper of a wrapper is caught by construction. It is a module rather than a `ci/check_*.py`
  entry point on purpose — it is driven from the already-registered `lane_checks_suite` and adds
  no new verification subject. It reports **25 identity-bearing producers**, all clean; it exits
  `4 ERROR(instrument)` on an unparseable file rather than skipping it into a green.

  Both offenders now **propagate** the child's identity rather than re-deriving it — the child is
  the process that opened the file, so its hash is of the bytes that actually ran. Neither writes
  a blank on a success path: `device_loss_gate.py` stamps per-repetition identity, refuses with
  `ERROR(identity=children_disagree)` when repetitions consumed different models (a pooled loss
  rate over two models is not a rate) and exits `ERROR(instrument=model_identity_unknown)` rather
  than publish an unattributable rate; `probe_device_memory_kv.py` refuses lanes that measured
  different models by the same argument its existing DLL check makes about the binary, and its
  `--reuse` path records `ERROR(identity=reused_records_named_no_model)` instead of inventing one.
  `bench/results/probe_lane_logits_identity.py`, which is derived entirely from the gate's record,
  propagates the identity forward; record-consumption is a **stated limit** of the audit (its
  relations are environment read and spawn), declared in the module rather than left implicit.
  `ci/test_lane_checks.py` drives the audit LIVE over the real tree and pairs it with planted arms
  for every shape — inline and variable-built argv, all four `env=` spellings including its
  absence, the alias/import variants, child-output field discarding, two-hop reachability, a
  `json.dumps` that is only printed, and source that merely *describes* the defect — each asserted
  against the rejected regex as well, so "the new screen is green" is not the only evidence that
  it works.

### 9.2 Benchmarking — Niobe

- **Baselines are versus the ORT CPU EP on the same machine, same model, same ORT build.** Any
  other comparison is marketing.
- `bench/` reuses `tests/ops/_models.py` builders so the benchmark cannot drift from what is
  tested.
- Reported per case: median wall time on Vulkan, median on CPU, ratio, **and** the claim
  diagnostics — island count, largest fused region, node count claimed. A speedup number without
  those three is not accepted. **From 2026-07-29 the metric of record is the triple
  `(claimed_op_coverage, island_count, largest_island_flops)`, reported together and per producer at
  version** (§10.0) — no member of it may appear alone. **From 2026-07-30 the triple is gated on a
  correctness verdict and may not be reported without it** — see §10.0's `model_output_equivalence`
  ruling. A benchmark of a wrong answer is not a benchmark.
- GPU-side timing uses `VkQueryPool` timestamp queries once the engine exposes them, so we can
  separate submit overhead from actual GPU time. Sub-millisecond cases are dispatch-bound and will
  be slower than CPU; that is expected, must be labelled, and must not be hidden.
- `bench.yml` posts an informational base-vs-PR table. **It does not gate**, because shared-runner
  timings are noise. It flags a regression as a prompt to re-measure locally.
- **No performance claim leaves this repo before the corresponding op is green in `tests/ops/` on
  at least one real GPU.**

### 9.3 Real-model validation without a Python interpreter — Tank

**The gap this closes.** Every real-model reading this project has ever taken came from a Python
probe under `rust/tools/`. Those probes need `onnx`, `onnxruntime`, `numpy` and a reachable package
index. On a host where the index is blocked — air-gapped runner, corporate egress policy, a fresh
container — none of them can run, and the project's whole-model evidence becomes unavailable at
exactly the moment somebody wants to check a claim. `rust/modelrunner` is the same question asked in
a binary that has no interpreter and no wheel behind it.

**Dependency-free is a load-bearing property, not an aesthetic.** SHA-256, JSON parsing/printing and
the input PRNG are implemented in the crate and tested against published vectors. The only non-`std`
dependency is `libloading`, used solely to `dlopen`/`LoadLibrary` ONNX Runtime — the EP crate itself
never links against `onnxruntime`, and neither does this. A runner justified by "works where PyPI is
blocked" that needed `cargo fetch` to reach a registry would fail in the same conditions for the
same reason.

**The claim it makes.** `--check-model-agreement <model>` reports `PASS` only when all six of these
hold, each written into the evidence document with its reason whether it held or not:

1. `model_identity_pinned` — the file's size **and** SHA-256 match `bench/results/model_provenance.json`.
2. `vulkan_ep_device_present` — the plugin registered an `OrtEpDevice`.
3. `vulkan_ep_in_session` — that device was appended to *this* session's options.
4. `vulkan_executed_nodes` — **ONNX Runtime's own profile** attributes at least one executed node
   to `VulkanExecutionProvider`.
5. `vulkan_dispatched_work` — the EP's counters snapshot reports `dispatches_executed > 0`.
6. `outputs_agree` — outputs match the CPU EP's for the same bytes, within a tolerance written in
   advance.

**Guard 4 is the one that did not previously exist.** `rust/tools/probe_model_output_agreement.py`
documents a dispatch guard and implements a provider-list membership test —
`"VulkanExecutionProvider" in session.get_providers()` — and that list is fixed at session-create
time. It is `True` whenever the EP was *requested*, including a run where every node fell back to
CPU. That is the §10.0 fabricated-speedup shape in a correctness harness.

**Witness rank is asymmetric, and the asymmetry is the design.** ORT's profile is the **primary**
witness because it is produced outside the frame under question: ORT decides what ran where and has
no stake in this EP's claims. The EP's own counter is **corroborating only**, because it is inside
that frame. When the two disagree the evidence records `split_frame` and the run does not pass — the
disagreement is reported, not arbitrated in the EP's favour. This is the same rule
`tests/ops/_verdict.py` applies and the same one §8.9.21 imposed on the proof ledger.

**Tolerance is reviewed policy.** `compare.rs` carries a table of per-model `rtol`/`atol` with a
written rationale per entry. A model absent from the table is **refused**, not compared against a
default: an unreviewed default is how a genuine numerical regression becomes a green run.
`--rtol`/`--atol` override the policy, must be supplied together, and are stamped into the evidence
as `tolerance.source = "cli"` so a reader can see the comparison was loosened and by whom.

**Four outcomes, not two.** `PASS` (0), `FAIL(condition=…)` (1) for a false claim about the EP,
`ERROR(instrument=…)` (2) when the harness could not ask the question, and `UNSUPPORTED(reason=…)`
(3) when the model is outside what the runner can drive. `UNSUPPORTED` is a first-class state: a
model whose inputs are interdependent (KV cache, `seqlens_k`, tokenised text) cannot be driven from
generated inputs, and saying so is the honest answer. The model's resolved path and SHA-256 are
still stamped into the evidence on that path — issue #19's lesson was that a document with a blank
identity cannot be checked later, so identity is recorded even when the comparison is not made.

**Library discovery refuses ambiguity.** `--ort-lib`, `ORT_MODEL_RUNNER_ORT_LIB`, `ORT_HOME`,
`ONNXRUNTIME_DIR`, the repository `.venv`, then the loader path — and two distinct libraries found
is `ERROR(instrument=ort_library_ambiguous)`, never a pick. The API version is gated against the
same `ORT_API_VERSION_MIN`/`ORT_API_VERSION_EXPECTED` constants the EP pins, so the runner cannot
drift from the plugin it is validating. On Windows this is immediately load-bearing:
`C:\Windows\System32\onnxruntime.dll` is 1.17.1 on many machines and wins the loader search; the
runner refuses it by version and names the remedy rather than failing later in a way that reads as
an EP defect.

**Extent.** It is not a benchmark and measures no speed. It does not replace `tests/ops`, which
reaches a per-op granularity no whole-model run does. Its host-free lane —
`cargo test -p ort-model-runner`, registered as `build.model_runner` in `ci/lane_inventory.py` on
both build lanes — proves the arithmetic, the pin refusals, the discovery arbitration and the
comparator, and can see **none** of the device guards. Those are only claimed by a real run on a
real GPU, and such runs are committed under `bench/results/rust-model-runner/` with an artifact
frame naming the commit and the device, per §8.9.26.

---

## 10. Milestones

> **STANDING DIRECTIVE (Justin, 2026-07-30):** 「要确保我们性能是非常高 一致向高性能推进」 —
> *ensure our performance is very high; push toward high performance continuously.* **Recorded as
> standing**, alongside 「兼容性是最好」 (compatibility outranks elegance, §7) and cross-platform
> generality. Read together with §10.0's disclosure obligations, which are what keep it from being
> satisfied by a number rather than by a machine.
>
> **RULING ON WHAT IT CHANGES — 2026-07-30T22:13:37-07:00. It changes the calendar and it does not
> change a single gate, and I want both halves said in the same breath.**
>
> - **It does not overturn the M0 performance ruling, and the day it was issued is the day that
>   argument got its second proof.** *Slowness is loud, wrongness is silent*, and **the cheapest way
>   to pass a ratio criterion is always to do less GPU work.** A directive to be fast is precisely
>   the condition under which a speed *gate* becomes dangerous, because a gate is a thing people are
>   rewarded for passing and this one is passable by claiming nothing. **The directive raises the
>   value of the interlocks, not the case for the gate.** No performance criterion enters M0.
> - **It does change sequencing, in the direction the coordinator states: performance work runs
>   continuously and in parallel with correctness, not queued behind M0.** This is not a concession —
>   it is what I already ruled at 19:05 (*"sequencing governs declarations, not calendars"*, third
>   time of saying it), now with a user mandate behind it. **Continuous** is the operative word:
>   `一致向高性能推进` is a *rate* obligation, so the thing it forbids is a week in which no
>   performance number moves, not a milestone that declares without one.
> - **The one clause I am adding on my own authority, because a directive of this shape always
>   acquires one eventually: no timing figure is quotable from a run whose verdict is not `MATCH`,
>   and every benchmark asserts EP presence and a non-zero claimed count before it starts a clock.**
>   The coordinator broadcast exactly this caveat before asking me; it is right, and it is now
>   architecture rather than a broadcast. **A fast wrong number is not partial credit toward this
>   directive. It is the failure mode this directive creates.**

Each milestone's exit criteria are verifiable by a command, not by an opinion.

### 10.0 What admitting contrib ops does to the milestones — stated without smoothing

M0–M3 were originally written under an `ai.onnx`-only assumption. The user ruling of
2026-07-28T20:54:42-07:00 changes that assumption, and the honest accounting is:

**M0 and M1 are unaffected.** Neither contains a contrib op. M0 is one `Add`; M1 is the 87-op T1
elementwise/shape/indexing surface, all `ai.onnx`. Nothing about the contrib ruling accelerates or
delays them, and nothing about it should be allowed to *reprioritize* them — the kernel-template
infrastructure that M1 gates on is what makes tiers 1–2 weeks-scale, and skipping ahead to a contrib
kernel because it is closer to the visible goal is the single most expensive mistake available to
this project.

**M2 gains one obligation and remains the critical path.** Its exit criterion is deliberately
BERT-base with *primitive* attention and no contrib ops, because M2's job is to prove the device
allocator and the transfer path, not to prove an attention kernel. The added obligation is §1.4 C2's
drift alarm: `graph_census.py` in CI, with pinned artifacts and per-op claim rates, must exist
before the first contrib op lands — which means it is built in M1/M2, not in M3 when it is needed.
M2 remains the critical path for everything after it, because a KV cache that round-trips to host
every token is not an inference engine.

**M3+ is where the whole cost lands, and it does not compress.** Tiers T3–T5c are gated on three XL
kernels with no template leverage — `GroupQueryAttention`, `MatMulNBits`, `LinearAttention` — plus
`CausalConvWithState`, `QMoE` and the vision path. These are each one person's deep work and they
are **not parallelizable away**; three people on three kernels is the maximum useful parallelism and
we do not have three people free. Rai's 🟢 on OQ-M6 (§8.4) is the one real accelerant: reading
llama.cpp's MIT-licensed Vulkan shaders for tiling and subgroup strategy is authorized, and my
estimates assume it is used. Without it I would widen T3/T4/T5a rather than hold them.

**The reconciliation with the ambition, stated plainly.** Justin's target is high coverage on a
weeks-to-months horizon with Qwen3.5 end-to-end as the real goal. Mouse's split is honest and I
ratify it unchanged: **tiers 0–2 (121 ops) are weeks-scale; Qwen3.5 end-to-end is months-scale.**
Admitting contrib ops does not shorten the second number — it is what makes the second number
*reachable at all*, which is a different and more important thing. Before the ruling, "Qwen3.5
end-to-end" had no completion path; now it has a long one. I am not going to present those as the
same kind of progress.

The guard against presenting them as the same kind of progress is `largest_island_flops` (§9.2).
The op count will look excellent for weeks before any LLM runs, and a milestone report that leads
with op count is a milestone report that is hiding something. Every milestone from M1 onward reports
`largest_island_flops` on the corpus artifacts alongside whatever else it reports — including on the
Qwen artifacts, where for most of M1 and M2 it will be near zero and *should* be, because that
number going up is the only thing that means the named target is getting closer.

**METRIC AMENDMENT — 2026-07-29T15:02:55-07:00. `largest_island_flops` alone is insufficient; the
metric of record is now a triple.** Mouse's partition simulation against two real Foundry Local
graphs settled this empirically rather than by argument, and I am changing my own metric on the
evidence:

- **Phi-3.5 sits at 34–35 islands from T1 through T3**, and `MatMulNBits` alone then collapses it to
  **one island of 364 nodes**. Partial coverage of that graph is worth *exactly nothing* — the
  useful transition is a cliff, not a slope.
- **On gpt-oss, claiming `Cast` moves coverage 28% → 54% while moving island count 52 → 125.** More
  ops claimed, strictly worse partitioning. **That is death-by-fallback observed, not argued** — and
  a coverage percentage would have scored it as a 26-point win.

**The metric of record is therefore `(claimed_op_coverage, island_count, largest_island_flops)`,
reported together, per producer at version, never separated.** Reasons, in order of how much they
cost us if ignored:

1. **`largest_island_flops` alone can rise while the graph gets worse.** The `Cast` step could
   plausibly raise the largest island slightly *and* double the fragment count; only the pair makes
   the regression visible. A single number is a number that can be gamed by accident.
2. **Island count is the transfer-boundary count**, and each boundary is a device↔host round trip
   under §6 until the allocator lands. It is the quantity §5.4's `retain_viable` exists to control,
   so it must be the quantity we report — a rule enforced by a mechanism nobody measures is a rule
   in name only.
3. **Coverage stays in the triple deliberately, as the thing being disciplined.** Dropping it would
   be dishonest in the other direction; it is a real signal about breadth. It just may never appear
   *alone*, and any report showing coverage without the other two is incomplete on its face.

**Binding consequence for the partitioner:** an op whose claiming raises island count without
raising largest-island FLOPs is a **candidate for declining even though we implement it**. §5.4's
`retain_viable` already models this per island; the `Cast` result says the same question must be
askable per *op* against a real corpus, and that the answer can be "we support this op and decline
it in this graph". Owner: Mouse, in the census and the MVS policy. That is not a contradiction of
§8.1 — claimed can never outrun translatable — it is the reverse direction, and the reverse
direction is a legitimate optimisation.

Niobe reports the triple in `PERF.md`; Mouse reports it in the census; I will not accept a milestone
report that shows one member of it.

**SECOND METRIC AMENDMENT — 2026-07-30T05:48:29-07:00. The triple has no correctness term, and on
2026-07-30 it rewarded a regression. It gains a gate, not a fourth number.**

The evidence is the R9 event (§10.0.1). On Phi-3.5 at the pinned producer-at-version, coverage went
**0 → 161 nodes** and island count went **0 → 161** — and the model got *less correct*: it moved
from **correct via CPU fallback** to **wrong via GPU**, `argmax 30751 → 0`, top-10 overlap 0/10,
identical on both devices. Every member of the triple moved, two of them in the direction we
usually call progress, and the number that would have told us the truth **was not in the metric**.

I considered making it a quadruple and rejected that, because a correctness term reported *alongside*
coverage invites exactly the trade the triple was invented to prevent — "coverage up 161, correctness
down" is not a mixed result, it is a failure with a decoration. **A wrong answer does not discount
the other three numbers; it voids them.**

**The metric of record is therefore the triple `(claimed_op_coverage, island_count,
largest_island_flops)`, per producer at version, *gated* on `model_output_equivalence` — a tri-state
verdict that must accompany the same run:**

| Verdict | Meaning | What may be reported |
|---|---|---|
| `MATCH` | Every model output agrees with a CPU-only run of the same session on the same artifact, within the §9.1 tolerance policy; argmax and top-k agree on every logits-shaped output — **and the run that produced those outputs executed at least one node on this EP, evidenced by an execution attribution taken from an instrument we do not own** (amendment of 2026-07-31T07:45:10-07:00 below) | The triple, as a result |
| `DIVERGENT` | Any output disagrees | **The triple may not be reported as progress at all.** The run reports the divergence: which outputs, max-abs-diff per output, argmax and top-k agreement |
| `UNMEASURED` | No CPU-only comparison was performed on this artifact in this run | **The triple may not be reported as progress at all.** It may be reported as a claim-path diagnostic, labelled `UNMEASURED` |
| `UNATTRIBUTED` | The comparison was performed and agreed, **and this EP executed zero nodes in the run that produced the outputs** — CPU-vs-CPU. Added 2026-07-31T07:45:10-07:00 | **Nothing.** Not the triple, not the ratio, not a correctness claim. The comparison is arithmetically correct and it is about a different world (R12). This is a lane **failure**, and it is a *different* failure from `DIVERGENT` — the model was not wrong, the subject was |

Four things about this, in the order they will be argued with:

1. **`UNMEASURED` is the default and it is not a soft `MATCH`.** A run that did not compare reports
   that it did not compare. This is R7's rule and §7.9's third state, arriving for the fourth time —
   device capabilities, the claim probe, lavapipe's `supportedStages`, and now the metric itself.
2. **The verdict is per artifact at producer-at-version (§8.5), like everything else in the metric.**
   `MATCH` on an `Add` graph says nothing about Phi-3.5; the verdict travels with the artifact it
   was measured on and never generalises across artifacts.
3. **It is a gate rather than a term precisely because it is not commensurable.** You cannot trade
   correctness against coverage, so it must not be printed in a row that invites the arithmetic.
4. **The comparison is against a CPU-only run of the same session on the same artifact**, not
   against a stored golden vector — for the reason §9.1.1 pins `accuracy_level`: an oracle that
   drifts with the machine is not an oracle, and a golden file drifts with every ORT bump instead.
   This is the model-level counterpart of C6's per-layer verification (§1.4): C6 says a wrong kernel
   must be *locatable* without comparing logits; this says the logits must nevertheless be
   *compared*, because C6 tells you where a defect is and only this tells you that there is one.

**Coverage that rises while the answer becomes wrong is worse than no coverage number**, because it
is a number that recruits effort in the wrong direction and it does so with the full authority of a
metric of record. Owner: Trinity emits the verdict, Niobe carries the gate in `PERF.md`, Mouse
carries it in the census, and I will reject any milestone report, benchmark table or coverage figure
that arrives with the verdict missing rather than assuming it.

**THIRD METRIC AMENDMENT — 2026-07-31T07:45:10-07:00. `MATCH` is not a verdict about this EP unless
it carries what executed the model. The verdict gains a frame, a fourth state, and a constructor
that cannot be called without the evidence.**

*Asked directly whether `model_output_equivalence` needs a frame tag. It does, and the specimen is
the strongest one this project has produced, because nothing was broken.*

**The specimen.** On 2026-07-30 the EP failed at run time on Phi-3.5's zero-length KV-cache inputs.
ORT printed `EP_FAIL … Falling back to CPUExecutionProvider`, re-ran the whole graph on CPU, and
**raised nothing**. `get_providers()` still listed `VulkanExecutionProvider`, because the provider
list is fixed at session-create time and the fallback happens inside `run()`. The comparison gate
then compared a CPU run against a CPU run and returned **`MATCH`**. Every gate in the lane passed.
The verdict was **wired, invoked, correctly named, and arithmetically correct** — and it certified a
run in which this EP contributed **zero nodes**.

**That is R12 exactly, arriving at a verdict rather than at a counter.** `alloc_device_upload_bytes:
0` was correct about the allocator's `VkDevice`; `MATCH` was correct about *some execution of this
graph*. Neither says which world it lives in, and in both cases the reader supplies the frame from
their expectations. The difference is that a counter's frame is a device and a verdict's frame is
**an executor** — so the rule generalises without weakening: *a reported quantity carries the
identity of its frame, and for a correctness verdict the frame is **what ran the graph**.*

**The verdict stops being a string.** `model_output_equivalence` becomes a record, and the four
things it carries are all things some mechanism computed on this run:

```json
"model_output_equivalence": {
  "verdict": "MATCH | DIVERGENT | UNMEASURED | UNATTRIBUTED | SPLIT-FRAME",
  "executed_by": { "VulkanExecutionProvider": 3, "CPUExecutionProvider": 24 },
  "attribution_source": "ort_profile",
  "attribution_witnesses_present": ["ort_profile", "ep_counters"],
  "attribution_witness_agreement": "AGREE | DISAGREE | UNOBSERVABLE",
  "attribution_witnesses": {
    "profile_node_events": 3, "profile_node_events_total": 27,
    "profile_path": "<path>", "profile_digest": "sha256:…", "profile_mtime_ns": 0,
    "counters_dispatches_executed": 1066,
    "counters_witness_reason": ""
  },
  "artifact": "<producer-at-version + file digest>",
  "device_index": 0, "device_name": "<physical device>"
}
```

**CORRECTED 2026-08-01T23:36:43-07:00 (Trinity's finding, and she was right not to edit this
document).** The example above previously showed **two** witness keys where the record emits the
shape now printed — `attribution_witnesses_present` and `attribution_witness_agreement` hoisted to
the top level, `counters_witness_reason` carrying *why* the counter witness is what it is, and
`counters_dispatches_executed` typed **int-or-`"UNOBSERVABLE"`** rather than nullable, so a witness
that could not report fails arithmetic loudly instead of reading as zero. **A schema example is a
claim about the record's extent**, and R11's first obligation — *declare the extent of what you are
reporting* — binds a document's example exactly as it binds a producer's output: an example that is
a strict subset of the record tells a reader the record is complete when it is not. **Two live keys
were missing from it for a day.** The remedy is the one this document already requires of everyone
else: the example is regenerated from an artifact, not written from memory —
`bench/results/criterion10-dev0.json` is the one it is now taken from.

Five binding clauses. Each closes a way of writing the words without the meaning, which is the test
every criterion in this document has to survive:

1. **The attribution comes from an instrument we do not own.** ORT's profiling trace — one `Node`
   event per executed node, carrying `args.provider` — is the primary witness. **Our own
   `dispatches_executed` may not be the primary witness**, because it lives inside the frame whose
   existence is in question: an EP that never ran leaves a stale or default counter, and an EP that
   ran leaves the same shape of number. Independent corroboration is R12 obligation 3 and it is the
   only thing that has caught anything this week.
2. **Both witnesses are recorded, and disagreement is red.** If the profile says the EP executed
   nodes and our counters say zero dispatches, or the reverse, the run reports `SPLIT-FRAME` and
   reports no triple. Two witnesses that can only ever agree are one witness.
3. **`MATCH` is unrepresentable without a non-zero own-provider count.** This is a constructor
   obligation, not an assertion beside the value: the function that writes the verdict takes the
   parsed attribution as a required argument derived from a profile *path*, and refuses to emit
   `MATCH` when the count is zero — it emits `UNATTRIBUTED` and says which providers did execute.
   **A caller may not pass a literal.** That is R10 amendment 1 applied to a verdict: the value must
   be one a mechanism computed on this run, never a flag its author set.
4. **`UNATTRIBUTED` is not `DIVERGENT` and must never be folded into it.** Merging them would lose
   the whole finding. `DIVERGENT` says *our kernels compute the wrong answer*; `UNATTRIBUTED` says
   *our kernels did not run and the answer is therefore not about them*. They have different owners,
   different fixes, and different next questions, and a lane that prints one red for both is a lane
   with R13's defect (§10.0.1). It is the fifth member of the family with `UNMEASURED`, `UNWIRED`,
   `UNOBSERVABLE` and `SPLIT-DEVICE`, for the same reason as the other four: **every way of not
   knowing gets a name a machine can print, because prose is where knowledge of a caveat goes to
   die.**
5. **The guard is an input to the verdict, not a neighbour of it.** Trinity's Guard D is the right
   observation in the wrong place. As a separate assertion it can be skipped, `xfail`ed, deleted,
   or crash before it observes anything — all four of which have now happened to controls on this
   project — and the verdict is emitted regardless, into the counters JSON that Niobe, Mouse and
   `epctl` read and that no pytest caveat travels with. **A caveat that lives in a different artifact
   from the number it qualifies is not attached to it.** So Guard D's *observation* becomes the
   verdict's constructor argument; Guard D's *assertion* remains in the lane as a fast, legible
   failure. The observation is load-bearing; the assertion is a convenience.

**What does not change.** The gate is still a gate and not a fourth number (`UNATTRIBUTED` voids the
triple exactly as `DIVERGENT` does); the verdict still travels with its artifact at
producer-at-version and never generalises; the comparison is still against a CPU-only run of the same
session. **What changes is that the verdict now says whose result it is** — and it turns out that a
verdict about *output equivalence* was never, on its own, a verdict about *this EP*.

**The cheapest thing that satisfies the words without satisfying the intent**, since I require that
question of every criterion: a caller that hardcodes `executed_by: {"VulkanExecutionProvider": 1}`.
Clause 3 closes it structurally — the argument is a profile path, parsed here, with the event count
recorded — and clause 2 makes a fabricated count disagree with the counters and go `SPLIT-FRAME`.
The second cheapest is a profile parsed from a *previous* run; the artifact digest and the
per-session profile path close that. **Owner: Trinity for the constructor and the profile parse,
Switch for the counters-side record and the `SPLIT-FRAME` emission, Tank for `epctl --check-counters`
failing on `UNATTRIBUTED` and on a missing `executed_by`.**

**Retroactive consequence, stated plainly because it is the expensive half.** Every `MATCH` this
project has recorded was recorded without an attribution. The 2026-07-30 verdicts on both devices are
**not `MATCH` under this rule; they are `UNATTRIBUTED` until re-emitted with a profile beside them**,
and the M0 table below moves accordingly. I am not grandfathering them. A rule whose first act is to
exempt the evidence that motivated it is a rule written to be passed.

**DISCLOSURE OBLIGATION — 2026-07-30T19:05:03-07:00. The triple gains no fourth number and no
second gate. It gains a figure that may not be omitted.**

On 2026-07-30 the EP became measurably **3.1× slower than the ORT CPU EP on Intel Iris Xe and 3.7×
on RTX 4060** with `model_output_equivalence = MATCH`. I ruled (§10, M0) that no performance
criterion belongs in M0 and that M2 carries the first threshold. That ruling is about *gates*. It
would be indefensible to conclude from it that the number may go unmentioned:

> **Every milestone report, benchmark table and coverage figure from M0 onward carries the measured
> end-to-end wall-clock ratio against a CPU-only run of the same session on the same artifact, per
> device, alongside the gated triple — including, and especially, when it is worse than 1.0. A
> report without it is incomplete on its face, exactly as one without the verdict is.**

Three properties, stated because each closes a way of complying with the words and not the intent:

1. **End to end, whole model, wall clock.** Never GPU-kernel time, never the claimed subgraphs in
   isolation, never a per-island figure presented as the headline. Kernel time falls toward zero as
   claiming falls toward zero; **whole-model wall clock cannot be improved by claiming less — an EP
   that claims nothing scores exactly 1.0 and never better.** That property is the whole reason this
   is the number we report.
2. **From a `MATCH` run, in the same process, on the same session objects.** `DIVERGENT` or
   `UNMEASURED` voids the ratio exactly as it voids the triple. The cheapest speedup available at any
   moment is computing the wrong answer faster, and we have now demonstrated we can do that at full
   speed with zero reported failures.
3. **Reported with the triple, and a run whose `claimed_op_coverage` or `largest_island_flops` fell
   against the last published run reports `REGRESSED-COVERAGE` rather than a speedup.** §7.0.2 makes
   declining a supported op legitimate; that legitimacy is exactly what makes it available as a
   cheat, and this is the interlock.

**A disclosure is a weaker instrument than a gate and it is the honest one at this stage.** A figure
nobody may hide is not the same as a threshold nobody may fail, and pretending otherwise would be
softening in the opposite direction — inventing a gate to look rigorous about a milestone that was
never scoped for speed. The disclosure's own falsifier is trivial and therefore must exist: a report
lacking the ratio is rejected, by me, in the same sentence in which I reject one lacking the verdict.

**THE OBLIGATION STANDS AS WRITTEN, AND FOUR HOURS OF EVIDENCE STRENGTHENED IT.** *Asked directly at
2026-07-30T20:58:11-07:00 whether it needs adjusting; it does not, and the reason is worth more than
the answer.* Between 19:05 and 21:00 our **phase decomposition was found to be wrong by roughly 50×**
on its largest row — 68.3% attributed to command-buffer recording was 95.8–98.4% host upload
(§10.0.1 R11) — while **the ratio in the paragraph above, 3.1× and 3.7×, was and remains correct.**
The ratio survived because **it has no internal structure to misattribute.** It is one interval
measured by one clock over one whole thing, with **zero naming decisions between the measurement and
the reader**. The decomposition had four names, one of them wrong, and the error was invisible
because the four summed to 99.0%.

Generalise it, because this is now a drafting rule and not an anecdote:

> **A metric's robustness is inversely proportional to the number of naming decisions standing
> between the measurement and the reader.** Decompose to *diagnose*; report the coarse invariant.
> Every subdivision is a place to put a cost under the wrong name, and every name is a place a
> reader cannot check.

Two consequences, both binding:

4. **A phase decomposition may accompany the ratio. It may never replace it, and it may never be
   the headline.** A report whose performance section is a phase table and whose wall-clock ratio is
   absent or in a footnote is rejected under this obligation, whatever the table's quality.
5. **A phase decomposition is publishable only with its R11 decomposition identity**: every phase
   declared inclusive or exclusive of its children, the parts summed against a whole measured by a
   *different* instrument, and the residual published. Absent that, the table may be circulated as a
   working note and may not be quoted as a finding — which is precisely the step that failed today,
   because the number was broadcast to the whole team before anyone had summed it against a wall
   clock.

**This is an argument for coarse honest metrics over fine misleading ones, and I want it on the
record as a preference and not only as a rule.** The fine metric is more useful *when it is right*
and there is no cheap way to know that it is. The coarse one is less useful and is nearly always
right. At this stage of the project we are optimising for not being wrong in public, so the coarse
one is the one that carries the obligation.

**THE THIRD DISCLOSURE OBLIGATION — ADDED 2026-07-30T22:13:37-07:00, AND I WOULD NOT HAVE ADDED IT
THIS MORNING.** *Asked directly whether tonight produces one that this morning would not have.* It
does, and it is not about numbers being wrong. It is about numbers being **about something else**:

> **6. Every reported performance quantity carries its frame — which selector index and physical
> device, which `VkDevice`, which instrument produced it — and a configuration in which two
> instruments observe different frames is disclosed as `SPLIT-DEVICE` rather than reconciled,
> averaged, or reported as a total.** (§10.0.1 R12, §6.5.)

Three reasons this is tonight's and not this morning's, in increasing order of importance:

- This morning I believed a device label was a label. It is an **index into an ordering that the
  printout did not carry** (R6 amendment 4), so half the frames written today were wrong before any
  measurement was taken.
- This morning the two upload accountings had not yet been shown to disagree by 15.2 seconds while
  both being correct.
- And this morning I would have written it as *"say which device"*, which is advice. **Advice does
  not survive transit** — the phase table's caveats did not — so it is written as a state the
  artifact emits, alongside `UNMEASURED`, `UNWIRED` and `UNOBSERVABLE`. That is the fourth member of
  the same family, and by now the family is the method: **every way of not knowing gets a name that
  a machine can print, because prose is where knowledge of a caveat goes to die.**

**A seventh, which is the corroboration rule and is a positive one for once.** Two independent
instruments agreeing on a number is the strongest evidence available to us, and it is the only thing
that has caught anything this week. **A performance claim that has been corroborated by a second,
independently-authored instrument says so; one that has not, says that too.** Tank's counter-derived
95.8–98.4% and Switch's span-derived 98.0% for the same quantity is the shape I want, and it should
be visible in the artifact rather than reconstructed by whoever remembers both.

**AN EIGHTH — THE DEVICE-STATE COMPANION. ADDED 2026-08-01T13:19:00-07:00, AND IT WITHDRAWS A
SENTENCE OF MINE.** *On Switch's `gpu_steady_tail` finding (§10.0.1 R9 amendment 5). Confirmed as
proposed, with three amendments, all of them tightenings that his version's wording left open.*

When wall clock proved contention-bound, Niobe and I moved the performance criteria onto the
**device clock**, on my sentence that *contention inflates host work but cannot touch the GPU clock*.
**That sentence is withdrawn.** It is false twice over: a second process submitting GPU work inflates
our device-busy figure directly (measured: `STEADY` at 10.99×), and the board's own clock governor
varies the device by **14.8× between its 210 MHz idle and its 3105 MHz boost** with nothing foreign
running at all (measured: `STEADY` at 21.4×, verified sole-tenant). The device clock is not a
contention-immune measurement surface. It is a *better* one than wall clock, and it needed the
interlock that wall clock got and it did not.

> **8. A device-clock figure is quotable only when a device-state record covers the same window as
> the statistic it accompanies, and that record carries both a tenancy verdict and a clock record.**
> The window is **the suffix the statistic was computed over**, not the run. The clock record carries
> the observed minimum, median and maximum **against the board's own advertised maximum**, because a
> clock number without its ceiling is an index without its ordering (R11). Absent the record, the
> figure's verdict is a fourth state — not `STEADY`, because "steady" is read as "quotable" and this
> record is the demonstration that it is not sufficient for that; and not `ERROR`, because the
> statistic did compute. `bench/`'s device-state probe becomes a **required companion, not a
> diagnostic**: a wrapper, never an afterthought.

**CONFIRMED. Three amendments, and each one exists because I asked my own drafting question — what
is the cheapest thing that satisfies these words without satisfying their intent?**

1. **Stated as a record, never as a tool, and generality is checked here and not later.**
   `nvidia-smi` is one vendor's implementation of this obligation and this project is
   cross-platform by mandate (§1.1). The obligation names the *content* — tenancy verdict, clock
   min/median/max, board maximum, over the statistic's own window — and any platform that can
   produce that content satisfies it. Intel, MoltenVK, Android and lavapipe each need their own
   producer, and Link owns finding out which of them can.
2. **The absence of the companion is never a waiver.** The cheapest way to satisfy obligation 8 as
   Switch worded it is to take the measurement on a platform that has no clock telemetry, where the
   requirement is vacuous and the figure is therefore unqualified. **A platform that cannot produce
   a device-state record produces `STEADY_UNCERTIFIED` forever**, and that is a true statement about
   what we can currently know there, not a penalty. Note the direction this cuts: the Intel iGPU
   shares its power budget with loaded CPU cores, so it is *more* exposed to this failure than the
   discrete board and would have been the platform most rewarded by the loophole.
3. **Three tokens, per R13, and one of them is the missing probe.** A device-state probe that is
   absent, unparseable or times out is `ERROR(instrument)` and never a finding of `SOLE_TENANT`. The
   absence of evidence and the evidence of absence come out of one code path here, and Switch
   already named them apart in his implementation; it is stated here so the obligation survives a
   reimplementation on another platform.

**And a fourth clause, which is the one the finding actually forces.** Obligation 8 makes a figure
*quotable*. It does not make two figures *comparable*:

> **8b. Two device-clock figures may be compared only if their device-state records agree** — same
> tenancy verdict, and overlapping clock during each statistic's own window. A before/after pair
> whose "before" predates the companion requirement is **not a pair**, and the improvement it would
> show is `UNMEASURED` until the "before" is retaken. This is not satisfied by both figures being
> `STEADY`; it is the whole content of the finding that `STEADY` does not carry it.

Switch has already applied 8b to his own barrier before/after and ruled it inadmissible — the
11.589/11.525/11.524 ms runs carry the record and the 12.183 ms baseline they would score against
does not. **I uphold that ruling.** He is right that it is probably sound at ~1.05× and right that
probably-sound is not the standard. The retake is cheap and it is item 1 of his resume note; it
should be taken in the first quiet window and it does not need my approval to be taken.

**THE RULING ON THE 40.201 ms REGIME-SEPARATION RESCUE — 2026-08-01T13:19:00-07:00. THE ARGUMENT
FAILS; THE FIGURE IS RE-QUALIFIED RATHER THAN WITHDRAWN, AND THOSE ARE DIFFERENT OUTCOMES.**

The rescue, as offered: the two clock regimes are 21× apart and do not overlap (idle ~247 ms,
boosted ~11.5–41 ms), so a run's regime is recoverable from its magnitude after the fact, with no
new instrument; therefore `PERF.md`'s **NVIDIA 40.201 ms at RSD 0.033%** cannot have been an
idle-clock run and stands, and comparisons within Switch's own 40.390 → 11.525 series stand with it.

It is a rescue argument, it arrives at the conclusion that nothing anyone published has to be
touched, and R13's second clause says a result that confirms gets more scrutiny than one that
contradicts. **Four objections; the third is fatal and the fourth is independent of all of them.**

1. **The band is a universal claim resting on two samples of one build.** "No idle-clock run of this
   workload lands in 11–41 ms" is supported by 246.720 and 246.735 ms — two observations, one
   pre-barrier build, one board. That is enough to establish *that* an idle-clock regime exists and
   nothing like enough to establish where its boundary lies for a different build.
2. **The 21× is not the margin that protects 40.201 ms.** The 21× is measured between 11.5 and 247,
   both from Switch's own recent builds. The figure being rescued is 40.201 ms, whose separation
   from the idle band is **6.1×, not 21×** — and it sits at the *top* edge of the boosted band,
   which is the least protected position in it. Quoting the widest ratio in the record to defend the
   number nearest the boundary is the shape of argument this project has repeatedly got wrong when
   it favoured us.
3. **There are not two regimes. Switch's own record says the board ranged 210 → 2490 MHz *within a
   single run*, against a 3105 MHz maximum.** A boost governor is continuous, not bimodal. "The two
   regimes do not overlap" is really *"the two clock states I happened to sample do not overlap"*,
   generalised into a claim about the device. A run held at an intermediate clock — 800 MHz, say —
   produces an intermediate figure, and **40.201 ms is exactly what a partially-boosted run of this
   workload would look like.** Nothing in the record excludes it, because no clock record covers
   that run. The regime-separation inference recovers the regime *from the magnitude*, and the
   magnitude is the quantity under certification: **the same-source falsifier, one level up.**
4. **The rescue is about clock and the finding has two halves.** Failure mode 1 — foreign GPU work —
   inflates the figure *continuously*, with no regime structure at all and therefore nothing for a
   magnitude argument to grip. The 40.201 ms run has no tenancy verdict. A run can be boosted and
   contended, and that combination is untouched by every sentence of the rescue.

**What survives, and it survives by arithmetic rather than by argument.** Every environmental
perturbation catalogued on this project — host contention, foreign GPU work, a low clock — has a
known **non-negative** sign on elapsed time. So `observed = true + delay`, `delay ≥ 0`, and any
timing figure we hold is an **upper bound** on the quiet, boosted, sole-tenant figure. Therefore:

- **40.201 ms is re-qualified, not withdrawn**, and is quotable only in that form: *≤ 40.201 ms,
  device state unrecorded*. Nothing shows it wrong; nothing shows it right; withdrawing it would be
  hardening a criterion to punish a bad week, and certifying it would be softening one because the
  alternative is inconvenient. An upper bound is what we have and it is worth having.
- **RSD 0.033% loses its certifying role and keeps its descriptive one.** It may be reported as a
  dispersion fact. It may never again appear as the reason a figure is quotable. It is the *best*
  RSD in the project's history and the finding is that this is not the recommendation it was read as.
- **Comparisons within the 40.390 → 11.525 series are not certified either, and for the reason
  Switch himself supplied.** Two upper bounds on the same side do not bound a difference from below
  (§10.0.4). He applied that correction to his `min()` claim on `vulkan.record` and did not carry it
  across to the GPU-busy series; it applies identically there. The *direction* of those changes is
  supported wherever a count backs it and is an estimate wherever only a clock does.
- **`PERF.md` is Niobe's file and the edit is hers.** This ruling establishes the label the figure
  must carry, not the sentence it must be written in; the re-qualification is a one-line change and
  it does not delete a number or a run.

One observation about the ledger this came in on, offered because it is worth more than the ruling.
Switch marked **his own** before/after ⛔ inadmissible on this exact reasoning, and marked **Niobe's**
figure ✅ rescued by an argument he did not hold his own numbers to. The strict standard applied to
one's own work and the lenient one to a colleague's is a generous instinct and it is still an
asymmetry, and this register exists because asymmetric standards are invisible from inside them.

### 10.0.1 Milestone risk register — "op works" ≠ "model works"

*Added 2026-07-28T22:28:08-07:00.* §10.0 warns that op count and target progress are different
kinds of progress. This is the register of *specific* places where a milestone could be declared
complete while the named target remains out of reach. Each entry names the cheap check that
resolves it and when it must run.

**R1 — `GroupQueryAttention` may arrive with fused Q/K normalisation, and that is a different
kernel.** Mouse found while verifying arity against the pinned 1.28 schema that GQA inputs 14/15
(`q_norm_weight` / `k_norm_weight`) fuse **per-head RMS normalisation of Q and K into the attention
kernel itself**, and that the ORT GenAI model builder sets `q_norm`/`k_norm` for **every
Qwen3-family decoder** (`builders/qwen.py`), emitting a 16-input GQA node when its fused-QK-norm
path is enabled. ORT's schema documentation states that an EP without support for that input must
reject the node when it is set, so Mouse's predicate declines inputs 10–15. **That is conformance
with the schema, not caution, and I ratify it as written** — the alternative is claiming a node
whose semantics we do not implement, which is C6's forbidden failure mode in its purest form.

The risk is not the decline. The risk is what the decline *means for the milestone*: the builder
emits separate `SimplifiedLayerNormalization` nodes for EPs that lack support, which is the form we
want — **but that is the builder's decision, not ours, and it is not a decision we control or have
observed.** If the fused form is what actually lands in front of us, then **"GQA works" and "Qwen3
works" are separated by a Q/K-norm variant of the hardest kernel in the project**, and T3 contains
either one XL kernel or two depending on an answer nobody has looked up. A T3 that silently doubles
is exactly the surprise §10.0 exists to prevent.

**RULING — this is an M1 verification item, not M2, and not a T3 precondition.** Three reasons:
1. **The artifact already exists.** Trinity built a GenAI Qwen3-0.6B graph for the oracle work
   (§9.1.1). The check is: load it, find the `com.microsoft::GroupQueryAttention` nodes, count their
   inputs, and record whether slots 14/15 are populated or whether `SimplifiedLayerNormalization`
   appears separately around each attention block. That is minutes of work today.
2. **The tooling it belongs in is already scheduled for M1.** `tools/graph_census.py` lands in M1
   (§10 M1 work table) and already walks pinned corpus artifacts producing node histograms. Node
   *arity* and populated-optional-input presence are a small extension of what it already extracts,
   and putting the check there means it is re-run on every artifact refresh and every ORT bump
   rather than being a one-time investigation someone remembers doing.
3. **Cheap now, expensive at T3.** Deferring it to when the kernel is being written means
   discovering a second XL kernel at the moment the schedule has the least slack. A precondition
   that can be checked before the work starts should be checked before the work starts — the entire
   value of Mouse's VERIFIED/UNVERIFIED discipline is refusing to plan against unobserved forms.

**Owner: Trinity produces the artifact fact; Mouse interprets it and records the consequence in
`OP_COVERAGE.md`.** Deliberately split that way — Trinity owns "what is in the file", Mouse owns
"what that means for T3" — because the failure mode here is an inference about a schema dressed up
as an observation about a graph.

**M1 exit criterion (added):** the census reports, for every corpus artifact containing GQA, the
input count of each GQA node and which optional slots are populated. Not "we looked once".

**What each outcome means, pre-committed so the result is not re-litigated:**

- **Separate `SimplifiedLayerNormalization` nodes.** Best case. T3 is one XL kernel; the norm is a
  T2 op we already have. Record the builder flag and version that produced this form, because it is
  a builder default we are now depending on — and a dependency on someone else's default is a
  tracked assumption, not a fact.
- **16-input fused nodes.** T3 contains a second variant of the hardest kernel and **T3 widens; I
  will say so rather than absorb it.** The mitigation to evaluate first is whether the fused Q/K
  norm can be handled as a prologue within our own attention kernel — it is per-head RMS over the
  head dimension, which is structurally the T2 RMSNorm we will already have written — rather than as
  a genuinely separate kernel. That is a real possibility and it is why this is a schedule risk to
  size rather than a certainty to fear.
- **Both forms in the corpus.** Then the decline predicate is load-bearing for correctness on real
  user models and the fused variant is required for T5a, not optional. This is the outcome that
  moves the item from "risk" to "scope".

**R1 UPDATE, 2026-07-29T08:13:58-07:00 — half of R1 is now answered, and the question was
mis-scoped.** Mouse's `mobius` finding (§8.5) settles it definitively **for that producer**: the
`mobius` builder emits Q/K norm as separate `ai.onnx::RMSNormalization` nodes before attention,
always; the choice is **not** conditioned on execution provider; and the 16-input fused GQA form
**never occurs**, because `mobius` never emits `GroupQueryAttention` at all. For the ORT GenAI
producer the hazard stands entirely unchanged — it does populate `q_norm_weight`/`k_norm_weight` at
inputs 14/15 when its fused path is enabled.

So R1 narrows to **"the ORT GenAI producer"**, and the M1 census item is **re-scoped from per-model
to per-producer**: the census reports GQA arity and populated optional slots *per producer per
artifact*, and a producer that emits no GQA reports that fact explicitly rather than being absent
from the table. An empty cell and a "this producer does not emit this op" are different findings and
must not look the same.

The pre-committed outcomes above stand, read as being about the ORT GenAI column only. Note that
`mobius` landing on the good side does **not** retire the risk: the ORT GenAI path is the one most
external users hit, so the widening exposure is undiminished — what has changed is that we now have
a producer we can iterate against while that answer is outstanding.

**R4 — a silent capability-probe failure is indistinguishable from an incapable device.** Recorded
in full as §7.9 rather than duplicated here, because the mechanism belongs with the baseline. The
milestone consequence: **no capability-derived behaviour is trusted until it has run on one
integrated and one discrete device**, and `caps.rs` changes carry that as a review requirement from
M0 onward. Both bugs that produced this rule were invisible on a single device and on lavapipe, so
CI would not have caught either.

**R5 — our errors are not randomly signed. Four of the five census contradictions were wrong in the
permissive direction, and that is the pattern, not the individual bugs.** *Added
2026-07-29T15:02:55-07:00.* Mouse's census of two real Foundry Local graphs surfaced five places
where a predicate or a fingerprint disagreed with the artifact. Individually each is a fix; together
they are a finding, because a randomly-signed error distribution would not land four-to-one.

The most serious: **packed QKV was a permissive hole.** The `GroupQueryAttention` predicate never
read inputs 1 and 2, so it would have **claimed** a packed node and handed the kernel a fused QKV
tensor where it expected a query. **Both real models pack on every layer**, so this was not an edge
case — it was the normal path, and it produces wrong numbers rather than a decline. Two others of
the same sign: `do_rotary=1` is universal in both graphs with no separate rotary node anywhere, so
the planned "claim GQA first, add fused rotary later" sequencing would have claimed **zero** nodes
while appearing to make progress; and contrib ops appear in the **default domain**
(`SimplifiedLayerNormalization` with `domain == ""`), a third registry category where `SinceVersion`
is meaningless and only a fingerprint can detect drift.

**Why the sign matters more than the count.** A too-strict predicate declines, and a decline is
loud: it shows up as a claim-rate drop, an island-count rise, and a CPU-fallback line in the claim
log. A too-permissive predicate claims, and a wrong claim is silent by construction — it produces
numbers. **This is now the third independent observation of the same asymmetry** (C2 item 7's
permissive fingerprint, §7.9's probe-failure-reads-as-no-capability, and this), which is enough to
stop treating it as a coincidence and start treating it as the direction our mistakes lean.

**The consequence, which is a review rule rather than a mechanism:** when auditing a claim predicate
or a fingerprint, **the question is not "is this right?" but "in which direction is this wrong?"** —
and an audit that finds nothing must say which permissive failures it specifically looked for. A
predicate that does not read an input cannot reject on it, so **every optional input a schema
defines is enumerated in the predicate and explicitly accepted or declined** — silence about an
input is acceptance of it, and that is exactly how packed QKV got through. Owner: Mouse for the
predicates, Fact Checker as the second reader whose brief is specifically the permissive direction.

**R6 — a decision can be right, reached by reasonable steps, and rest on evidence our own tooling
manufactured.** *Added 2026-07-29T16:00:55-07:00. This is the cleanest specimen in the register and
it deserves to be read before the others.*

The episode: lavapipe reported `VkPhysicalDeviceSubgroupProperties::supportedStages = 0`. That
number said our CI lanes would be excluded by gate criterion R5 (subgroup `BASIC` in compute), so
R5 was demoted from the **frozen** §7.2 device gate to a probed capability, and the lavapipe reading
was recorded as the reason in `DESIGN.md` §7.2, in `ENGINE.md`, in `PLATFORMS.md` quirk LVP2, and in
`caps.rs` / `instance.rs` comments. **Mesa 26.1 lavapipe does support subgroup `BASIC` in compute.**
The `supportedStages = 0` reading was almost certainly §7.9 Bug 1 — our own `push_next` chain never
being sent. **We changed a frozen architectural decision on the strength of a number our own bug
produced.**

**Every individual step was reasonable.** Read a capability. Notice it excludes a platform we care
about. Weigh the requirement against the exclusion. Change the requirement. Write down why. Nobody
was careless, no shortcut was taken, and the outcome is **correct** — R5 belongs out of the gate
under §7.0 for reasons that never mentioned lavapipe. That is exactly what makes it the useful
specimen: **this failure is invisible to diligence.** More care at any step would not have caught
it, because the flawed input entered at the only point nobody thought to question — a number our
own code printed.

**Three rules, and the first is the one that generalises.**

1. **A decision record names its load-bearing reason, and a number is never load-bearing alone.**
   §7.2's R5 removal was justified twice over — by §7.0's principle and by the lavapipe reading —
   and only the *observation* was written prominently. Had it been the load-bearing reason, this
   would be a reversal rather than a correction. **When a decision is supported by both a principle
   and a measurement, record which one would have to fail for the decision to change.** If the
   answer is "the measurement", that decision is provisional and must be labelled so.
2. **A number produced by our own tooling is evidence about our tooling until it is corroborated.**
   A device capability read through our probe, a claim rate from our partitioner, a green count from
   our harness: each is a measurement of the instrument as much as of the thing. Corroboration means
   a second instrument (`epctl --probe-loader` raw values, the vendor's own tooling, a second
   device) — not a second reading from the same code.
3. **A correction that leaves the conclusion standing must still be published.** The temptation is
   to fix the comment quietly since nothing changes. **A right answer reached through false evidence
   is an unaudited answer**, and the next decision that leans on the same reading — here, `PLATFORMS.md`
   LVP2 and every downstream statement about lavapipe's subgroup support — inherits the error
   silently. Owner: Link to re-observe LVP2 with the fixed probe and restate it as observed or
   retracted; Switch to re-point the `caps.rs` / `instance.rs` / `ENGINE.md` comments at §7.0.

**Where this sits among the others.** R4 says a failed probe reads as an incapable device. R6 is
what R4 costs when it is not caught: the false reading does not stay in a log, it propagates into a
frozen decision and into four documents, and it is still there weeks later wearing the authority of
a written rationale. **The characteristic failure of this project is checking a claim against a
description of an artifact rather than the artifact** (§8.5) — R6 is the version where the
description was generated by us, which is the hardest version to notice and the easiest to trust.

**R6 AMENDMENT 4 — an implausible result is a free instrument check, and we spent ours on a
celebration.** *Added 2026-07-30T20:58:11-07:00 on the device-label inversion (R11's second
specimen).* Because `epctl --probe-loader` prints unsorted enumeration order while `select_device`
indexes a sorted list, every device label written on 2026-07-30 was backwards: **`DEVICE=0` is the
RTX 4060, `DEVICE=1` is the Iris Xe.** The consequence is that *"the Intel integrated GPU beats the
discrete RTX 4060"* — a result treated all day as a finding worth explaining — **dissolves.** NVIDIA
is faster, which is what physics said before we measured.

The propagation is ordinary R6 and needs no new rule: a number our own tooling produced, believed,
and carried into prompts and tables. What is new is the missed opportunity, and it is the cheapest
check in this register:

> **A result surprising enough to be a discovery is first a reason to check the instrument.
> Implausibility is a falsifier that costs nothing, and it is the only one available before any
> extra work is done.** The correct first move on *"the integrated GPU beat the discrete one"* is
> not to explain it — it is to confirm which device was which.

We had a reading that contradicted the strongest prior available (a discrete GPU with an order of
magnitude more bandwidth is faster than an integrated one) and we spent the contradiction on a
hypothesis. R6 rule 2 already says a number from our own tooling is evidence about our tooling until
corroborated; this names the moment at which that rule is cheapest to apply and most likely to be
skipped, which is precisely the moment the number is *interesting*. **Corroborate the surprising
number first, not the boring one.**

**One documentation defect follows and I am recording it as owed work rather than as a rule**: a
printed index has no definition without its ordering (R11). `epctl --probe-loader` must print the
ordering it used and say that it is not `select_device`'s, or print `select_device`'s index. Owner:
Tank, with Switch on `instance.rs`. It is small and it is the kind of small that cost a whole team a
day of inverted labels.

**R7 — our instruments fabricate negatives, and a negative is the answer nobody questions.**
*Added 2026-07-29T19:42:07-07:00. R6's twin: there the tooling manufactured a number, here it
manufactured a **negative result**, which is worse because a negative asks for no explanation.*

The barrier-parity criterion (M0 #8) skipped every case while criterion 2's test passed on the same
machine. The anatomy was three layers, each **caused by the previous** and each individually
reasonable:

1. **A dead instrument read as a claim result.** `claim_log::path()` cached the environment variable
   in a `OnceLock`. A pytest process loads the EP **once** and runs hundreds of tests while setting
   that variable **per call**, so the first decision latched `None` and every later probe found no
   file — and a missing file was read as *"not claimed"*. The instrument was not merely wrong, it
   was switched off, and being switched off is indistinguishable from a true negative.
2. **A workaround around a phantom.** Believing the probe unimplemented, the harness moved to
   profiling JSON, which reaches `Compile` — and hit a hard access violation on Intel. A second
   failure, investigated on its own terms, caused entirely by the first.
3. **A declaration that recreated the hazard it replaced.** The harness then used a hand-declared
   `live` flag: a duplicate of `OpStatus::Live` in a second language, and **per-op while our claims
   are per-form**. `Add-i32` carried `live=True` against an f32-only predicate, so parity would have
   compared two CPU-fallback runs and agreed trivially — **the vacuous pass the flag existed to
   prevent** (§9.1.2).

**Two rules, both binding beyond the incident.**

- **Absence of an instrument must not read as a negative result.** A probe that cannot find its data
  **raises**; it does not return `False`. `is_vulkan_claimed` with no claim log is an error, not a
  "no". This is §7.9 rule 1 (*not determined* is a third state) applied to the test harness rather
  than to the capability probe, and it is the **third distinct layer** at which the same confusion
  has bitten: device capabilities, the claim probe, and lavapipe's `supportedStages`.
- **Derive, do not declare** (Mouse's formulation, adopted as a rule of record). Any fact the code
  already knows is **computed from the code**, never restated in a second place — harness liveness
  now comes from `epctl --dump-capabilities --json`, **per form**, because our claims are per form.
  A hand-written duplicate of a machine-known fact is a fork that drifts, and it drifts in the
  permissive direction (R5): a stale `live=True` buys a vacuous pass, a stale `live=False` merely
  skips.

**Why layer 2 is the part worth remembering.** Layer 1 was a bug. Layers 2 and 3 were *competent
engineering applied to a false premise*, and each made the system worse — the workaround introduced
a crash, and the workaround's workaround reintroduced the original hazard in a form that would have
produced green numbers. **When a fix produces a second, unrelated-looking failure, suspect the
diagnosis before extending the fix.** The cost of a fabricated negative is never the negative; it is
everything built on top of it.

**R8 — we planned against the ops the model contains, having never measured why its nodes are
declined.** *Added 2026-07-29T21:14:03-07:00. §8.5's lesson landing a fourth time, in a new place.*

The roadmap was built around three kernels — `MatMulNBits`, `SkipSimplifiedLayerNormalization`,
`GroupQueryAttention` — on the premise that missing kernels are what stand between us and a model.
The first end-to-end run of a real model says otherwise: **258 nodes declined on symbolic shapes and
100 on missing kernels**, and because the codes are first-match with status checked before the
predicate (§8.8), the 258 is a floor and the 100 is a ceiling. Landing all three kernels tomorrow
would leave most of the graph declined.

**The inventory was never wrong.** Phi-3.5 does contain those ops, in those counts; §8.5's census
was accurate. The census answered *"which ops does this graph contain?"* and the roadmap needed
*"why does this graph's nodes get declined?"* — **a different question, asked of the same artifact,
with a different answer.** That is the fourth instance of the same shape: the wrong producer, the
wrong revision, the unread model file, and now the unasked question. Each time the artifact was
available and each time we reasoned about it instead of measuring it.

**The rule.** **Coverage planning is driven by the decline histogram of a real graph, not by its op
histogram.** An op census tells us what to build; only a decline census tells us what to build
*first*. Every tier plan and every milestone that names a model now carries the decline histogram
for that model alongside the op counts, and the metric triple (§10.0) is reported next to it.

**And read the histogram as first-match.** A decline code names the **first** failing check, not the
only one. A category checked early absorbs nodes that would also have failed later — so counts for
early codes are ceilings and counts for late codes are floors, and **the two are not comparable
without the check order**. Mouse should additionally report, for each declined node, the **full set**
of checks that would have failed, not only the first. Until that exists, no plan may be sequenced on
a difference between two decline counts without stating which is the floor.

**One thing this incident also shows working.** A 2.2 GB fp16 production model with external data
and an `If` prologue loaded, ran, declined all 363 nodes with machine-readable reasons, fell back to
CPU, and was bit-identical across sessions. **The conservative-claiming machinery (§1.3, C6) did
exactly what it was designed to do at a scale nothing else has tested.** The roadmap was wrong; the
safety net was not.

**R9 — every instrument was individually correct, and the set of them was jointly silent on the only
property that mattered.** *Added 2026-07-30T05:48:29-07:00. R7's mirror image, and the most
important entry in this register.*

R7 is about instruments that **lie**: a dead probe returning `False`, a fabricated negative nobody
questions. This is the opposite event and R7 does not reach it. **Nothing lied.**

- `dispatches_executed: 161` really counted 161 dispatches — coordinator-verified after Switch fixed
  the probe-contamination bug that had been under-reporting it.
- `compute_failures: 0` really counted zero reported failures.
- `subgraphs_live: 161`, `compute_calls: 161`, `compile_calls: 1`, `islands: 161` — all correct.
- ORT really did accept all 161 offered `com.microsoft::MatMulNBits` nodes.
- The test suite really was green.

Every one of those instruments was sound, and several of them had just been *repaired* to be sound.
The composite reading taken from them — **"161 nodes execute on the GPU"** — was true. It was then
used as a **correctness claim**, and **not one instrument in the set measured correctness at all.**
The actual state of the model: `vk logits [0.0000, 0.0000]`, `argmax 0` against CPU's `30751`,
top-10 overlap `0/10`, on both an Intel Iris Xe and an RTX 4060, deterministically.

**The rule of record.** Sharpening the coordinator's draft, because the draft's second clause is the
part that does the work and it deserves to be stated as a mechanism rather than as a warning:

> **A set of individually sound instruments can be jointly silent on the property that matters, and
> their agreement raises confidence without raising evidence. Therefore: for every claim, name the
> instrument that would go red if the claim were false. If no such instrument exists, the claim is
> not evidenced — however much telemetry surrounds it.**

Call the second sentence **the red-instrument test**. It is a question asked of a *claim*, not of a
system, and it takes about ten seconds: *what would have to be broken for this number to change, and
is the thing I am asserting on that list?* On 2026-07-30 the answer for "the EP works on Phi-3.5" was
that **no counter in the set would have changed if every kernel wrote zeros** — which is exactly what
happened.

**R9's THIRD GENERALISATION — the red-instrument test applied to a *criterion* rather than to a
claim. NOT a new amendment, and I am saying so explicitly because the register grew twice today.**
*Added 2026-08-01T23:36:43-07:00. Three specimens arrived within one day and the coordinator is
already briefing agents with the sentence, which is the point at which it needs to exist here in a
form he can cite.*

> **A criterion is discharged by an observable that changes when the claim is false, never by one
> that is true whatever happens.**

That is the red-instrument test with *criterion* substituted for *claim*, and it inherits R9's
remedy unchanged — **a different instrument**, or the same one given a reachable state in which it
disagrees. The scope extension is the whole content: R9 was written about instruments certifying a
number, and a milestone criterion is a claim like any other, evaluated by a reading, and therefore
falsifiable or decorative on exactly the same terms.

**Three specimens, one day, three costumes.**

| specimen | the observable | why it cannot go red |
|---|---|---|
| RAI-011 — *the gate is always evaluated, with no branch in front of it* | `net_benefit_gate = EVALUATED`, `net_benefit_gate_bypasses = 0` | an unconditional early return **inside** the gate satisfies every word; `bypasses` is `0` forever whether or not the gate decides anything (§5.4.1) |
| Link's screen reading `ONNXRUNTIME_EP_VULKAN_TRACE_FILE` | `OPTIONAL-UNWIRED` | nothing defines that variable, so the screen reported the same value on every run it ever made **and would have done so had the tracer been deleted** |
| Switch's assertion comparing two values both exactly `0.0` | the assertion passes | `0.0 == 0.0` has no reachable failing state on the path that produces both sides |
| **`model_output_equivalence` on the criterion-10 series** *(added 2026-08-02T04:30:29-07:00)* | `verdict = MATCH` over 65 outputs | the CPU oracle reads `vk_out[0]` only; the other 64 are checked **against each other across runs**, so a wrong-but-stable KV write has no reachable failing state in either gate |

**And note that the first and third are *green* checks while the second is a *negative* one.** The
class is indifferent to polarity — *an always-false screen and an always-true screen are equally
blind*, which is already this document's sentence about the census, arriving now as a general
property rather than an incident. The operational form, which is cheap and is the thing to actually
do: **before recording a criterion met, name the run that would have failed it, and say whether that
run is reachable.** If it is not reachable, the criterion is not met — it is unfalsifiable, and the
tally should say so rather than say `MET`.

**COVERAGE DOES NOT COMPOSE — a fourth specimen, recorded 2026-08-02T04:30:29-07:00, and DELIBERATELY
NOT NUMBERED as an amendment or a generalisation.** *I am declining to number it because the remedy is
unchanged — R9's remedy is a different instrument and that is exactly what is owed here — and because
I wrote a self-check into this register that if the next finding also landed as a generalisation I
should check whether I had found a softer way of declining. This is that check being run. The content
below is real and belongs in R9; the number is not.*

Criterion 10 was closed on a series carrying two gates, each individually honest:

| gate | outputs covered | property proven |
|---|---|---|
| cross-run bit-identity (`outputs_bit_equal`) | **all 65** | determinism only |
| CPU oracle (`_compare_run_to_cpu`) | **1 of 65** — `vk_out[0]`, the logits | correctness |

> **Two gates whose extents differ compose to the weaker extent and the stronger name.** The union of
> what they are called reads *all outputs correct*. The union of what they check is *one output
> correct, sixty-five outputs stable*. Nothing in the tree ever compared a KV output to CPU.

The property that matters — *all 65 outputs correct* — is the conjunction, and **both instruments are
silent on it**, which is R9's own sentence arriving one level up: not two instruments describing
different worlds (that is R12) but two instruments each sound, jointly blind. **A deterministically
wrong KV write is green on both.** That is not hypothetical here: `test_phi35.py`'s Guard 1 already
documents the mechanism in this codebase — *"the output buffer falls outside the descriptor set and is
never written… zero-initialised by both Intel Iris Xe and NVIDIA drivers for security, reads back as
all-zero"* — and the guard built against it is applied to output 0, **the same one output the oracle
compares.** The 2026-07-30 failure mode has one falsifier and it is pointed at the one tensor that
already has another.

**Why nobody saw it, and this half is R11 rather than R9.** The artifact records
`outputs_compared: 65` in the same per-run dict as `argmax_cpu`, `top10_overlap` and `max_abs_diff`.
Every one of its neighbours is an oracle fact; it is a cross-run fact; and its name asserts the
conjunction that was never measured. **That is R11 obligation 4 — name–content agreement — and R11
obligation 1, extent, undeclared per gate.** Had each gate declared its own extent the composition
would have been visible on the face of the record without anyone reading a test. **The general
operational form: a record with two gates owes two extents.** One `outputs_compared` for two
different coverages is a decomposition that appears to close.

**And the reader it caught was me.** I quoted `max_abs_diff = 0.0625` into criterion 10's row without
stating over what, and read `outputs_compared: 65` as sixty-five comparisons against the oracle,
three hours after diagnosing exactly this obligation in criterion 12's row against someone else.
Recorded here rather than only in the criteria table, because a register whose author's own misreads
are kept out of it is a register being curated.

**THE DUAL — A READING THAT MOVES WHEN THE SUBJECT IS FINE. Recorded 2026-08-02T15:15:12-07:00, and
again DELIBERATELY NOT NUMBERED.** *Self-check, run in the open because this is the second consecutive
finding I have declined to number and the register carries a standing instruction to suspect that: the
remedy here is R9's remedy unchanged — a different instrument, and I name the replacement below — so it
is not an amendment. That the **dual** of the third generalisation keeps arriving is itself the argument
that R9 is its home rather than the argument for a new number.*

Everything ruled this session concerned a check whose reading **does not move when its subject is
wrong**. Criterion 10's tolerance gate is the mirror image:

> **`max_abs_diff` against a fixed `atol`, applied along a 32-layer chain, rises with depth for a
> perfectly correct implementation.** Its reading moves with something that is not its subject.

The two failures are one class, and the sentence that covers both is: **the reading must be a function
of the claim and of nothing else.** An observable that is true whatever happens cannot convict; an
observable that degrades whatever happens cannot acquit. **Both are readings of something other than
the claim**, and R9's remedy — a different instrument — is the remedy for both.

**The specimen, and it is arithmetic rather than judgement.** Of the 65 per-output residuals in
`criterion10-dev0.json`, **64 are exact negative powers of two** and the 65th is `3 × 2⁻⁹`. They are
small integer multiples of the **fp16 ULP** at the magnitude each tensor carries. A transformer's KV
activations grow with depth; the ULP grows with them; **the absolute residual therefore rises with depth
whether or not anything is wrong.** The "monotone accumulation curve" is, on this evidence, a plot of
tensor magnitude. **`atol` is an absolute bound applied to tensors of growing scale — §10.0.4's second
form, *prefer the ratio*, arriving as a defect rather than as advice.**

**The replacement instrument, stated with its prediction so that it is falsifiable before it is built:**
record the residual **in ULPs** — `max_abs_diff / ulp(|value at the differing element|)` — per output.
**Prediction: flat, order 1–3, across all 32 layers.** If it is flat there is no accumulation defect and
there is no curve to have a tolerance argument about. If it **steps** at some layer, that layer holds a
real defect and the step localises it. **The instrument is strictly better in both outcomes, and unlike
the present one it can be wrong** — which is the whole test.

**A caution about the series that was quoted.** The *absolute* residual is broadly monotone in depth;
`max_rel_diff` is not — layer 2's key reads `0.4559`, above every layer from 3 to 30 and level with
layer 31's `0.4917`, while its absolute residual is unremarkable at `2⁻⁸`. `max_rel_diff` is attained at
near-zero elements, so its denominator is unstable and it is not a depth series at all. **Quote the
absolute series or the ULP series; never `max_rel_diff`.** One wrong denominator was already corrected
in this instrument by its author; this is the second, and it is in the criterion rather than in the
probe.

**And the comparison has no ground truth in it.** The CPU EP is not a reference implementation; it is a
second fp16 implementation with a different summation order. **Elementwise disagreement between two
correct fp16 implementations grows with depth by construction.** So *"how much accumulated error do we
accept"* is not the question either — the question is *how much implementation disagreement is
consistent with both being correct*, and the ULP unit answers it directly: one to three ULPs is two
correct implementations, a hundred is a defect.

**THE CLASSIFIER'S OWN FAILURE MODE — RULING 2026-08-02T01:42:02-07:00: DECLINED AS A NEW RULE,
because R13's second clause already is it, and the coordinator was asked for the remedy without
being told which rule it came from.** *The specimen he brought against himself: he named "union
defects" as a pattern that afternoon, then reported clippy's `manual_contains` as "the fourth union
defect today" inside a table of five. Mouse checked instead of accepting — `-D warnings` produced
**five** errors, not one, and `git show origin/main:<file>` on each showed **four of the five present
on `origin/main` verbatim**. Clippy was already red on main, independently of any merge. Only
`registry.rs:2261` was a union defect. His sentence is the one to keep: "worth knowing before the
fifth union defect gets attributed to a merge."*

**Why it is not new.** R13's second clause reads: *a result that confirms a prediction deserves more
scrutiny than one that contradicts it — **quote the failure text, never the failure count.*** The
specimen is that clause with nothing added. A pattern was named; a subsequent observation confirmed
it; the confirmation was **counted** (five) rather than **quoted** (four of which were older than the
merge). And the remedy Mouse actually applied is R13's remedy performed literally — he retrieved each
failure's text from its own source rather than trusting the tally. **The register individuates by
remedy, and this remedy is already written down.**

**What is new, and it is a scope note rather than an obligation.** Every prior R13 specimen was a
*mechanism* mis-reporting. This one is a **person** — the classifier, not the check. That widening is
worth one sentence and no new machinery:

> **A newly named pattern begins attracting cases that do not belong to it, and the cost is borne by
> the real instances, which are diluted by the false ones.** The remedy is R13's, applied to
> yourself: *quote each instance's own evidence, never the count of them*. A class assembled from a
> tally has the same evidential weight as an instrument that cannot go red.

**And this is where the register stops growing today.** I have declined three rules and written two
amendments and three generalisations in one session; the coordinator explicitly asked to be declined
rather than have the register grow a rule per incident, and that request is correct **and is also the
shape of the error above** — a register that grows by one entry per named pattern is a register
attracting cases to its own new categories. **The test remains the only test: does the remedy differ?
Here it does not.**

**R12's FOURTH GENERALISATION — for a test result, the frame is the binary that ran it.** *Two
specimens, both Mouse's, both caught by their author.* (i) A build that linked a sibling agent's
in-flight `registry.rs` from a shared worktree and produced a **false `ALL-DECLINED` he nearly wrote
up as a finding**. (ii) **`Copy-Item` preserves `LastWriteTime`, so cargo's fingerprint does not
notice a restore-from-backup and re-runs the *mutated* binary** — which produced a persistent false
failure he came close to "fixing" by **weakening a correct assertion**, the most expensive possible
outcome and the one this document exists to prevent.

> R12 already reads *a reported quantity carries the identity of its frame* — for a counter the frame
> is a **device**, for a correctness verdict an **executor**, for a rationale a **date**. Add: **for a
> test result the frame is the binary that produced it, and a mutation harness that restores sources
> by timestamp-preserving copy has no claim about which binary that was.** Remedy is R12's unchanged:
> record the frame. Concretely — **a mutation-testing harness touches or hashes the restored file,
> and asserts the rebuild happened, before reading any result as a control.**

**Any harness on Windows that backs up and restores sources is subject to this**, which is a
cross-platform generality note and not a Mouse-specific one: the same trap exists wherever a build
system fingerprints on mtime alone. **A false red that gets "fixed" by softening the assertion it
fired on converts a working control into a decoration in one commit**, and it is worth noticing that
the failure arrived disguised as the thing we most want — a check that goes red.

**Why agreement is worthless here, stated precisely, because this is the counter-intuitive part.**
Six instruments agreeing feels like six independent confirmations. It is not. Confidence scales with
the number of **agreeing** instruments; evidence scales only with the number of **falsifying** ones —
instruments that had a reachable state in which they would have disagreed with the claim. A set with
zero falsifiers has zero evidential weight no matter how large it is, and **the larger it is, the
more confident the wrong conclusion becomes.** That is the mechanism by which this failure gets
*worse* as the telemetry gets *better*, and it is why R9 could not have been prevented by more
instrumentation. Switch's counter fixes were correct, necessary, and made the false conclusion more
persuasive.

**The silence set.** Every instrument has one: the set of propositions it cannot be false about.
`dispatches_executed`'s silence set contains everything downstream of "the dispatch returned".
`compute_failures`'s contains everything our own code did not detect (§9.1.3). `islands`' contains
every question about values. A test suite's contains every case it does not have — and
`test_matmulnbits.py` mentions `f16` exactly twice while Phi-3.5 is entirely fp16, so the suite's
silence set contained the defect. **When an instrument is added, its silence is recorded with it.**
An instrument documented only by what it detects is an instrument whose limits will be discovered
the way this one was.

**Four operational rules, binding from today.**

1. **Every claim in this repository carries a named falsifier.** In a milestone criterion, in a
   decision record, in a PR description, in `PERF.md`: the sentence "this would have gone red if the
   claim were false: ___" must be completable. If it cannot be completed, the claim is downgraded to
   `UNMEASURED` — not to "probably fine".
2. **A criterion whose falsifier does not exist yet is not met, however much evidence surrounds it.**
   This is applied to the M0 table below, and it reopens criteria I had already recorded as met.
3. **Positive controls are the standard mechanism, and we already had the pattern.** Criterion 7 —
   the layering lint must fail a *deliberately planted* violation — is a red instrument by
   construction, and it is the only criterion in M0 that was written that way from the first day.
   Criterion 3's ruling (§10 M0) reached the same design under duress. Generalise it: **a check that
   has never been observed to fail is a check of unknown polarity.**
4. **The composite is not a free instrument.** Combining sound readings produces a new claim with
   its own falsifier requirement; it does not inherit the soundness of its parts. "161 nodes execute"
   ∧ "0 failures" does not compose into "161 nodes compute correctly", and the gap between those two
   sentences is where this project spent a day.

**Where R9 sits among the others.** R6: our tooling manufactured a *number*. R7: our tooling
manufactured a *negative*. R9: **our tooling manufactured nothing at all, and was believed about a
question it had never been pointed at.** R6 and R7 are defeated by corroboration — a second
instrument, a second device. R9 is not: the second device agreed, both devices were right, and the
agreement of two correct instruments on the wrong question is still zero evidence. **R9 is the only
entry in this register that more diligence, more devices and more telemetry would not have caught.**
The only thing that catches it is asking, before the number is quoted, what would have made it
different.

**What it costs, immediately.** The M0 criteria table below is amended on R9's authority; the metric
of record is gated on a correctness verdict (§10.0); `compute_failures` is constrained to an
execution-status reading with a mechanism rather than prose behind it (§9.1.3); and §9.1.2 is
labelled, in its own text, as a section that answers *"did our code run?"* and never *"was the
answer right?"*.

**R9 AMENDMENT 5 — THE ANTI-CORRELATED FALSIFIER: a check whose confidence rises as its subject's
error rises. And the ruling that the register does not grow.** *Added 2026-08-01T13:19:00-07:00 on
Switch's `gpu_steady_tail` finding, reproducible from committed artifacts by
`python bench/results/probe_gputenancy.py` with no GPU. The coordinator's candidate was R11. My own
first reading was R11. Both are wrong, and R14 is wrong too — the reasoning is below, because on
this register the placement is the ruling.*

**The specimen.** `gpu_steady_tail` releases a device-clock figure when the relative standard
deviation over a suffix of ≥5 inferences is ≤2%. Measured, on real hardware, three ways:

```
soloA      [SOLE_TENANT]         STEADY   11.525 ms   RSD 0.8098%   n=33
contended  [FOREIGN_GPU_WORK×1]  STEADY   11.770 ms   RSD 0.1082%   n=8
contended3 [FOREIGN_GPU_WORK×3]  NO_STEADY_TAIL       — but only because the load generators
                                                        stopped before the measurement did
contended3 truncated to 20/28/34 inferences
                                 STEADY  126.647 ms   RSD 0.79–0.91%   — 10.99× wrong
board held at its 210 MHz idle clock, verified SOLE_TENANT
                                 STEADY  246.720 ms   RSD 0.1163%, zero discarded — 21.4× wrong
```

The board idles at **210 MHz against a 3105 MHz boost**. A run held the whole way at idle clock is
*perfectly steady*, so it produces the gate's **most confident possible verdict**. **In both failure
modes the wrong number carried the better RSD than the right one.** `contended3` refused only
because someone else's job finished first; truncated to a length that is entirely the other
project's schedule and not ours, it passes confidently.

**Why this is not R11, and I am rejecting my own reading of it.** The attraction of R11 is one
sentence of R11's *explanation*: an identity whose two sides are computed from the same source is a
falsifier that cannot fire. A variance test over a biased series has that shape — the check is
computed from the series it certifies, so a uniform error is invisible by construction. But that
sentence is borrowed in R11's own text ("R10's uninvoked falsifier arriving in a different
disguise"); R11 does not own it, and a register entry is not its epigram. **Apply R11's four
obligations to `gpu_steady_tail` and they certify it.** There is no decomposition, so there is no
identity to check against an independent whole. There is no flat percentage table asserting
disjointness. Nothing is inclusive of a child. And name–content agreement *passes*: the quantity
named "RSD over the steady tail" is exactly an RSD over the steady tail, denoting precisely what it
contains. **A rule that, applied faithfully, would have certified the specimen does not cover the
specimen** — that is the test I used to refuse folding R11 into R10, it was correct then, and it
disqualifies R11 here. Stretching R11 over this would make the naming census the place a reader
looks for a defect the naming census structurally cannot see, which is the error R11 exists to
name, committed against R11.

**Why this is not R12 or R13 either.** The gate was in the right frame — same device, same run, same
`VkDevice`, no split. It did not fail; it computed correctly and returned a true statement. It is
not R7: nothing lied. The series really was steady. **The gate's output was true and was consumed as
a different proposition than the one it asserts.**

**Why it is R9, exactly.** R9's question is the red-instrument test: *for the claim "11.525 ms is
this EP's uncontended device cost", name the instrument that would have gone red had the claim been
false.* Nothing in the set would have. Bias in the level of a series is in a dispersion statistic's
**silence set** — the set of propositions an instrument cannot be false about — and R9 already
requires that an instrument's silence be recorded when the instrument is added. It never was for
`gpu_steady_tail`. The remedy is R9's remedy and no other: **a different instrument**, measuring a
quantity the first one cannot see. The register individuates its entries by remedy, not by flavour
— R10's remedy is observing invocation, R11's is an independently-measured whole, R12's is frame
identity the writer cannot supply alone, R13's is three tokens — and this remedy is already spoken
for. **Minting R14 would put a second name on one failure class: two names for one measurement,
appearing to close.**

**What is genuinely new, and it is a mechanism inside R9 rather than a class beside it.** R9 as
written describes *plural* sound instruments that are **jointly silent**, whose agreement raises
confidence without raising evidence. This specimen is a **single** instrument whose confidence
measure is **anti-correlated with the error it is believed to bound**: the further the level is from
the truth, the steadier the device that produced it, and the tighter the interval the gate reports.
Silence is neutral. This is worse than silence, and it has a consequence R9's four rules do not
state:

> **R9 rule 5 — the anti-correlated falsifier.** Where a check's confidence measure is computed from
> the same series as the quantity it certifies, ask **which way the check moves when the quantity is
> wrong.** If it moves the same way as the reader's confidence, **the check cannot be repaired by
> tightening its threshold** — a tighter bound admits *more* of the failure, not less — and no value
> of the threshold makes it evidence. Such a check is downgraded from a gate to a *precondition*,
> and the claim it was believed to license is `UNMEASURED` until a **second quantity, from outside
> the series, records the state of the thing being measured.**

**Precision is not accuracy, and this register had never had to say so.** Dispersion bounds
repeatability; it says nothing about level. Every rule up to now concerned an instrument that was
absent, silent, misnamed, out of frame, or broken. **This one was present, correct, well-named, in
frame, and working — and its virtue was the symptom.** That sentence belongs in R9's section and it
does not need a number of its own to be true.

**The generalisation, because "steady" is not the only word this happens to.** Any statistic of
*shape* — variance, RSD, drift, monotonicity, agreement between replicates, test flakiness rate —
is silent about *level*, and any of them can be made to look better by a condition that makes the
level uniformly wrong. When one of ours is used as a release condition, its silence set is written
down beside it in the same artifact, per R9's silence clause, and rule 5 is applied to it before it
gates anything.

**R10 — a mechanism that exists in the source tree and not in the call graph is indistinguishable
from one that was never written, and review cannot tell them apart.** *Added
2026-07-30T19:05:03-07:00. R9's blind spot, found by R9's own machinery failing to see it.*

R9 says: for every claim, name the instrument that would go red if the claim were false. That test
has a hole, and 2026-07-30 drove a truck through it. **A falsifier that is never invoked is
indistinguishable from a falsifier that never fires.** R9 asks whether an instrument *would* go red;
it never asks whether the instrument *runs*. Counting falsifiers does not catch this, because the
uninvoked one counts.

**The specimens, all from a single day.** In every case **the code was correct.**

| Mechanism | Written by | What it would have been worth | How it was found |
|---|---|---|---|
| `ops/partition.rs` — `Island`, `island_count`, `largest_island_flops`, `boundary_bytes_per_inference`, the whole connected-subgraph mechanism | Mouse, days earlier | **3.7×.** `GetCapability` handed ORT one capability per node; wiring it collapsed Phi-3.5 from 321 islands to 33 and Intel from 2954.6 ms to 807.2 ms | Noticed by eye |
| The GPU tracer | Niobe, with the env vars wired | Every performance number we now have | **Niobe verified empirically that no trace file appeared** — the only specimen caught by its own author, and caught because he looked for the output rather than at the code |
| `model_output_equivalence` | Specified by me (§10.0) | It is M0 criterion 10 | Nobody owned implementing it; found when I went looking for the verdict |
| `retain_viable` / the net-benefit doctrine (§7.0.2) | Mouse and me | The §7.0.2 category | Still uninvoked outside `#[cfg(test)]`, **after** the day everyone agreed partition.rs was wired |
| The EP-side `VkDebugUtilsMessengerEXT` | — | It printed the exact root cause of the all-zero-logits defect *in one line* the moment it was attached | The validation layer **was** loaded; its output went to the layer's default stderr handler with no in-process listener. Wired to a destination nobody read |
| `ep_messenger_fires_for_planted_fence_leak` | Switch | Criterion 3's positive control | `#[ignore]`. A control that must be opted into is a control that is not in the lane |

**One item on the coordinator's list of five is not R10 and the distinction is load-bearing.**
`compute_failures` **is** called, on every dispatch, and counts correctly. Its problem is that it is
silent on the property anyone cared about — that is R9's silence set (§9.1.3), not R10. Keeping them
apart matters because the remedies are different: R9's remedy is *a different instrument*; R10's is
*the same instrument, invoked*. Conflating them produces the worst outcome available — writing a
second instrument to cover a gap that a first, correct instrument already covers and is simply not
being called.

**The rule of record.** Adopting the coordinator's formulation, with three amendments that are where
the work is:

> **A mechanism's existence is a claim about the call graph, not about the source tree. The
> falsifier for "X is wired" is an observation of an artifact X produced, whose content varies with
> X's input. It is never a reading of X's code, and never a flag X's author set.**

1. **The artifact must vary with the input.** "X produced output" is satisfied by a hardcoded banner,
   a constant, or a `wired: true` line — the cheapest thing that satisfies the words. The observation
   must be a *value X computed*: a count, a span, a verdict, a histogram. This is R7's *derive, do
   not declare*, applied to wiring rather than to liveness.
2. **The uninvoked state must be reportable and distinct from the empty state.** A mechanism with no
   observation in a run reports **`UNWIRED`** — which is not "produced nothing" and not "not
   applicable". This is §7.9's third state arriving for the sixth time (device capabilities, the
   claim probe, lavapipe's `supportedStages`, the harness liveness flag, `model_output_equivalence`,
   and now the call graph itself). **`partition.rs` survived precisely because a partitioner that
   never runs is indistinguishable from a graph with no islands to merge**, and nothing in the
   system could tell those apart.
3. **The identity case is a failing state, not a passing one.** This is the part that generalises
   furthest and it is the cheapest check in the register. A partitioner that emits one island per
   node **is the identity function**; `island_count == claimed_count` is one line, it was true for
   the whole life of the defect, and nobody had written down that it should be false. Generally:
   **every mechanism that transforms a quantity carries an assertion relating its input to its
   output, and the degenerate case in which the transform does nothing must be an explicit red
   state.** A transform that silently no-ops is the default failure mode of an unwired mechanism,
   because *doing nothing* is exactly what not being called looks like. `counters.rs` now carries
   this one as the *partition falsifier* (`islands_offered == claimed_nodes` with both `> 1`), which
   is the shape every other mechanism owes.

**Sub-rule: wiring is a property of an entry point, not of a file.** On 2026-07-30 "`partition.rs` is
wired" was asserted, believed, and half-true: `evaluate` is called from `GetCapability`;
`retain_viable` is called only from tests. A module-granular claim about the call graph is not a
claim about anything.

**The review consequence, which is the expensive one.** Every specimen above was **invisible to
review**, and not because review was careless — a diff shows a mechanism being added and shows it
correct, and the reviewer's job as usually practised is exactly to check those two things. Neither
answers whether anything calls it. So:

> **Review of a new mechanism is not complete until the reviewer has seen an artifact the mechanism
> produced.** A diff is evidence about code. Only output is evidence about the call graph. A PR
> introducing a mechanism attaches its first output; a PR that cannot is landing a mechanism in the
> `UNWIRED` state and says so in the description.

**Where R10 sits among the others.** R6: our tooling manufactured a number. R7: our tooling
manufactured a negative. R9: our tooling was sound, and jointly silent on the question. **R10: our
tooling was sound, and absent from the run entirely** — and unlike R9 it is not defeated by asking
what would go red, because the answer *"this would have gone red"* is true of code that never
executes. R9 asks whether an instrument would go red; R10 asks whether it goes at all. Ask R10's
question first: **it is upstream of every other rule in this register**, since a rule enforced by a
mechanism nobody calls is a rule in name only — which §10.0 already said about `retain_viable` on
2026-07-29, in a sentence I wrote and did not check.

**R11 — a measurement's name is not its definition, and a decomposition that appears to close is the
hardest kind of wrong.** *Added 2026-07-30T20:58:11-07:00. R10's companion, found within hours of
R10 landing, by a specimen R10 certifies as healthy.*

**Ruling first, because the coordinator asked whether this is a new rule or an amendment: it is a new
rule, R11, and folding it into R10 would be a mistake of exactly the kind it describes.** R10's
subject is **invocation** — did this mechanism run? R11's subject is **denotation** — does the name
this number is reported under describe its content? They fail differently, they are caught by
different mechanisms (R10 by an observation, R11 by an arithmetic identity), and merging them would
make the wiring census the place a reader looks for a defect the wiring census structurally cannot
see. **That is the specimen, applied to the register itself.**

**The specimen.** `Phase::Record` opens before `vkBeginCommandBuffer` and closes after
`vkEndCommandBuffer`. The host staging memcpy runs **inside** that window and reports through
`Tracer::record_transfer` into `phase_us[Upload]`, deliberately emitting **no `ph:"X"` span** so as
not to double-count. The phase table aggregated `ph:"X"` spans, so it **structurally could not see
upload**. Upload is a **child** of `record`, not a sibling — and the whole team, including me, spent
a day reasoning about a "68.3% command-buffer recording" cost that was **95.8–98.4% host upload** in
every cell on both devices. Real command-buffer recording is **87–229 ms, about 1–3% of wall.** The
actual defect it was hiding is far worse and far more fixable: **the EP re-uploads the entire weight
set on every inference** — 1997.6 MiB per inference against `device_upload_bytes` of 1997.2 MiB
(ratio 1.0002), exactly linear at one, two and three runs, in:out **2481:1** on a 1-token prefill.

**Now apply R10 to it, which is why R10 needs a companion rather than a patch.** `Phase::Record`
**is** wired. It **does** emit an artifact. The artifact's content **does** vary with its input. It
passes criterion 12's wiring census as specified, cleanly — **and the census would have certified a
number wrong by a factor of fifty.**

**The coordinator's candidate, and why I am changing its cut.** His draft frames the fault as
*invisibility*: a cost the aggregation cannot see must be named as unmeasured rather than absent.
That is a true sentence about a different failure. **Upload was never invisible.** It was measured,
correctly, to the microsecond, sitting in `phase_us[Upload]` the whole time. Nothing failed to see
it. What failed is that **`record` was read as an *exclusive* quantity — a leaf, disjoint from its
siblings — when it is an *inclusive* one, an interval containing children, and no artifact anywhere
said which.** Every profiler ever written distinguishes self time from total time; we published a
table with neither column and a name that implied the first.

> **R11 — the name of a reported quantity is not its definition. Every reported quantity declares its
> extent: the interval it spans, what it contains, and what it excludes. A set of quantities may be
> presented as a decomposition of a whole only if it is declared disjoint and exhaustive, and only
> alongside the identity `Σ parts + named_residual == whole`, where **the whole is measured by a
> different instrument than the parts.**

**The last clause is the whole rule and it is R9 applied to arithmetic.** Our table read
68.3 + 16.3 + 14.1 + 0.3 = 99.0%. **It appeared to close.** It appeared to close *because the missing
cost was inside one of the rows*, so the residual was zero by construction and no amount of staring
at the table could have revealed it. **An identity whose two sides are computed from the same source
is a tautology with no reachable red state** — it is a falsifier that cannot fire, which is R10's
uninvoked falsifier arriving in a different disguise. The whole must be wall clock, taken from a
clock, independently of every part.

**Four obligations, which is what a census must additionally emit.**

1. **Extent, declared as data and not as a doc comment.** Each phase declares `inclusive` or
   `exclusive` and, if inclusive, its children. `Phase::Record` declares itself inclusive of
   `Upload`. This is derived-not-declared (R7) pointed at structure: the tracer already knows the
   nesting, so a reader must never have to reconstruct it.
2. **The decomposition identity, computed in the artifact, against an independent whole.** Not by
   the reader, not in a spreadsheet, and never with the whole defined as the sum. **A residual that
   is zero by construction is not a residual.**
3. **A flat percentage table is an assertion of disjointness.** Publishing one is making the claim;
   if the parts nest, the artifact is a tree with self and total columns, or it is not published.
4. **Name–content agreement is checkable and therefore checked.** When a quantity's content is
   dominated — I will fix the threshold at **more than half** — by something other than what its
   name denotes, that is a **misnomer defect** and the name changes. A phase that is 95.8–98.4%
   upload is not "recording" in any sense a reader can act on, and it recruited a day of the team's
   planning toward `recorded.rs` while the cost sat in the allocator.

**Two further specimens, recorded because R11 is a class and one instance reads as an anecdote.**

- **`record_dispatches()` and the counters file: two writers, one path.** It called a dump function
  that wrote a **subset** of the counters to the **same path**, so any run ending in a dispatch
  silently lost 27 keys **and reset `model_output_equivalence` to `UNMEASURED`**. The artifact's
  identity is its path, and where two producers write one path, **the name denotes whichever ran
  last** — a naming failure, not a wiring one; both writers ran, both were correct, and the file was
  a lie about which of them made it. **One thing worked and it is worth naming, because it is a
  design choice earning out**: the value was destroyed *into the state that refuses to be reported*,
  not into `MATCH`. §10.0 point 1 — `UNMEASURED` is the default and is not a soft `MATCH` — is the
  reason a silent data loss cost us a missing number instead of a false one. **Make the default the
  refusing state, and corruption degrades to a refusal.**
- **`DEVICE=0`: one label, two index spaces.** `enumerate_capable_devices()` sorts best-first and
  `select_device` indexes the sorted list; `epctl --probe-loader` prints unsorted enumeration order.
  **`DEVICE=0` is the RTX 4060 and `DEVICE=1` is the Iris Xe**, and every device label in every
  prompt and every table written on 2026-07-30 was backwards. **An index is a name whose definition
  is an ordering, so a printed index that does not carry its ordering has no definition at all.**
  The propagation is R6's (see R6 amendment 4 below); the defect is R11's.

**A negative specimen, recorded here because this is where the next reader will come looking.**
`gpu_steady_tail` reporting `STEADY` at 10.99× and at 21.4× (see R9 amendment 5 above) reads as R11
and is not R11. It has R11's *silhouette* — a falsifier computed from the same source as the thing
it falsifies — but none of R11's obligations bite on it, and all four of them certify it. **The
shared-source sentence in R11 is R11's explanation, not R11's extent.** A rule is what its
obligations require, and a specimen belongs to the rule whose obligations would have caught it. That
one belongs to R9, whose remedy — a different instrument — is the one that actually fixes it.

**Where R11 sits, and the sentence I want kept.** R7: the instrument **lied**. R9: the instruments
were sound and **jointly silent**. R10: the instrument was **never called**. **R11: the instrument
was called, was correct, measured exactly what it was written to measure — and was reported under a
name that meant something else.** It is the hardest of the four to see because **every check we have
passes**: it is wired, it is invoked, it is sound, it agrees with itself, and its table sums to 99%.
The only thing that catches it is refusing to let a set of numbers be called a breakdown until
something outside the set says what the total was.

**R12 — two instruments can each be correct about a different world, and a counter that reads zero
may be structurally incapable of reading anything else.** *Added 2026-07-30T22:13:37-07:00 on
Switch's two-`VkDevice` finding. R11's sibling: R11 is a fault of **naming**, R12 is a fault of
**frame**, and R12 is the one prose cannot fix.*

**The specimen.** On a run where `vulkan.cmd_upload` measured **15.2 s**, the allocator reported
`alloc_device_upload_bytes: 0` and `alloc_staged_bytes: 0`. **Both numbers are correct.** They are
correct *about the allocator's own `VkDevice`*, which is not the device the session uploads through
(§6.5). Neither instrument is misnamed; neither is unwired; neither is silent. They describe
different worlds and the artifact does not say so, so a reader who adds them gets nonsense and a
reader who quotes either gets a caveat they did not receive.

> **R12 — a reported quantity carries the identity of the frame it was measured in: which device,
> which queue, which process, which instrument. Two quantities may be compared, summed, or read as
> agreeing only if their frames are identical and the artifact says so. And a counter whose value
> cannot vary — because the mechanism it observes cannot occur in its frame — reports
> `UNOBSERVABLE`, never `0`.**

**The second sentence is the load-bearing one and it is why this is a rule and not a bug report.**
`alloc_device_authoritative_spans` is 0 today. It will be 0 tomorrow. It will be 0 after a fix that
works and after a fix that does not, for as long as the device split exists, **because the event it
counts cannot happen in the frame it lives in.** A zero that cannot become nonzero is not a
measurement, and reading it as a negative is R7's fabricated negative arriving from a cause that is
not a bug in the instrument. **`UNOBSERVABLE` is to R12 what `UNWIRED` is to R10 and `UNMEASURED` is
to §10.0: the third state that keeps a structural absence from being read as an empirical zero.**

**`GATED_NEVER_RUN` JOINS THAT FAMILY, AND IT IS R7 — endorsed 2026-08-02T21:24:34-07:00.** *Link found
that a red CI step skips the seven Linux steps behind it, so `device.op_correctness` was never "never
observed to fail" — **it has never run**. He introduced `GATED_NEVER_RUN` and deleted the entry's
`observed` date.* Both moves are correct and the second is the better one: **a date is a claim that
something happened**, and a check behind a gate that never opened has no date to carry. *"Never observed
to fail"* obtained by never looking is R7's fabricated negative arriving from CI's control flow rather
than from a probe, and it is distinct from `UNWIRED` — the mechanism is wired, the gate in front of it
never opened — so it earns its own token by the same argument every other third state in this register
earned one. **A green suite whose green depends on a step that never executed is reporting the
scheduler, not the software.**

**And a second `misnamed` specimen, noted rather than acted on.** The Linux step that dies on eleven
`i32`/`u32` errors in `ep.rs` is called **`Clippy (all warnings as errors)`**. It is a **portability**
failure wearing a lint's name, and it read as low priority all day because of it. Link's line is the
one to keep: the previous `misnamed` specimen *"was wrong by 50× in a number"* — `Phase::Record` — and
**"this one by a whole platform in a priority."** That is R11 arriving somewhere R11 was not written
for: **a step's name sets its triage priority, so a misnamed step is a mis-prioritised one, and the cost
is paid in delay rather than in a wrong number.**

**Three obligations:**

1. **No criterion may name as its instrument a counter that is `UNOBSERVABLE` in the configuration
   the criterion will be assessed in.** Concretely: M1's weight-residency criterion is read against
   `device_upload_bytes` on the *session's* device, and **not** against
   `alloc_device_authoritative_spans` until §6.5 is closed. A criterion whose instrument is pinned
   is not a criterion — it is a sentence.
2. **Cross-frame comparison is an explicit act.** Any artifact presenting numbers from more than one
   frame labels every row with its frame, and a run detected in a multi-frame configuration reports
   `SPLIT-DEVICE` on the affected accounting rather than a total.
3. **Agreement across frames is the strongest evidence we have, and it must be claimed
   deliberately.** Tank measured upload at 95.8–98.4% of the record phase from the counters; Switch
   measured `vulkan.cmd_upload` at 98.0% from a span. **Two instruments, two authors, two
   mechanisms, one number** — that is the good case, and it is worth as much as the bad case above
   costs, provided the artifact says the frames were the same. Independent corroboration is the only
   thing that has caught anything this week.

**Why R12 is not R11.** R11's remedy is to rename the quantity or declare its extent; a writer with
enough care could have got it right alone. **R12's remedy is not available to the writer at all** —
Tank's counter is exactly right in his frame and no wording he could choose would make it describe
the run. The fix is structural (§6.5, one device) and the rule's job is to keep the number from
being believed until then. **That is the class: not a mistake anyone made, an artefact of two people
being correct in different places.**

**Where R12 sits.** R6: our tooling manufactured a *number*. R7: it manufactured a *negative*. R9:
sound instruments, *jointly silent*. R10: *never called*. R11: called, correct, *misnamed*. **R12:
called, correct, correctly named — and about a different world than the one the reader is in.**

**R12's SHARPEST SPECIMEN, AND IT IS A FAIL-OPEN IN THE PROOF LEDGER — RULED 2026-08-02T21:24:34-07:00.
No new rule; this is R12 with the frame written down and not read.** *Link demonstrated it in one
command: on Windows with `DEVICE=1` the EP prints "proven … on **device0**" and claims the form anyway.*

Verified in source before ruling. `LedgerEntry` carries `device`, `ort_build` and `tolerance` — **the
entry records its frame in full** — and `parse_ledger` populates `device` from the line. **No predicate
reads it.** There is not one use of an entry's `.device` anywhere in `registry.rs`: not in `get`, not in
`lookup_key`, not in `ledger_contains`. The frame is present on 74 of 75 entries and is inert.

> **Recording a frame is not carrying it. A field no predicate reads is not a guard — it is a comment
> with a schema.**

**That is the third costume of the defect this document has been chasing all session**, beside
RAI-011's unconditional early return inside the gate it was supposed to guard and `'<absent>'`'s
always-absent screen: not an instrument that goes red and changes nothing (R9's decoration sentence),
but **an instrument that cannot go red at all, while looking exactly like one that could.** The
asymmetry is what makes it urgent rather than merely wrong: **a digest disagreeing fails safe; an entry
matching on a device that proved nothing fails open, and nothing watches it.**

**THE QUESTION — is a proof a property of a form, or of a form on a device? — AND WHY THE DICHOTOMY IS
FALSE.** Both horns offered are correctly costed and both are unacceptable. Per-device proofs mean a new
GPU cannot run the EP until someone proves ninety-five forms on it, which for a cross-platform EP is
close to fatal; device-independent proofs assert that a form proven anywhere is correct everywhere,
which the `timestampPeriod` class already falsified once on Intel. **The dichotomy is false because the
ledger entry is doing two jobs and being asked to answer for both with one bit.** It is evidence that
*the form is implemented correctly* — largely a property of the kernel and the op semantics — and
evidence that *the form is correct here* — a property of subgroup width, fp16 rounding and driver
behaviour. **The answer is: a proof is a property of a form on a device, and the remedy for that is not
re-proof. It is R12's remedy — carry the frame, and give the extrapolation a name.**

**Three states, replacing a two-state predicate that cannot express what it is doing:**

| state | condition | claimable |
|---|---|---|
| `PROVEN` | an entry exists, `MATCH`, witnesses present, **and its `device` matches the running device** | yes |
| `PROVEN-ELSEWHERE` | an entry exists and is sound, **but was obtained on another device** | **yes — and counted, disclosed, and named in the run record** |
| `UNPROVEN` | no entry, or a demoted one | no |

**`PROVEN-ELSEWHERE` is claimable on purpose and this is not a softening.** Refusing it is the fatal
horn; and **today that extrapolation already happens on every non-`device0` run, silently, and is
indistinguishable in every artifact from a proof obtained here.** Naming it costs nothing at bring-up
and removes the silence — which is §10.0's standing obligation that **every way of not knowing gets a
name a machine can print**, applied to the one way of not knowing this ledger had no word for. **The
test that this is a strengthening: after the change, a divergence on a new device arrives with a named
suspect list; today it arrives with 74 entries all claiming to be proofs.**

**And the promotion path is cheap, which is what actually dissolves the cost argument.** A full
differential per form per device is expensive. **A ULP-scale residual check per device is not** — and
§10.0.1 R9's dual established what device-dependence looks like when it is real: subgroup width, fp16
rounding and driver differences move a residual by *ULPs*, and the ULP series is precisely the
instrument that sees them. **`PROVEN-ELSEWHERE` is promoted by the cheap per-device instrument, not by
re-running ninety-five differentials.** That is the architectural answer: **the expensive proof
establishes the form; the cheap invariant establishes the port.**

> **AMENDED 2026-08-03 — §8.9.18 part 1. The paragraph immediately above is WITHDRAWN.** Fact Checker
> refuted it and the refutation holds: the instrument I called "the cheap per-device instrument" is the
> **model-level** ULP series, whose subject is the composed graph, and it therefore reaches **no proof
> key at all** — not the exercised ones and certainly not the unexercised ones. `ProofKey::from_node`'s
> own doc comment states the rule I broke: *"the lookup is by key, so evidence about one path cannot be
> returned for another."* The arithmetic is in-tree: `wiring_census-dev1.json` records
> `proven_key_lookups=6` against `ledger_entries=95`, so a model run on the second device touches **six
> keys and leaves eighty-nine untouched**. **`PROVEN-ELSEWHERE` keeps its licence to disclose and loses
> its licence to promote**; the replacement promotion mechanism is ruled in §8.9.18.

**THE TOOLCHAIN COUPLING — separately wrong, ruled R13 rather than R12, and NOT to be fixed by
narrowing the digest.** `shader_digest_for` hashes the SPIR-V **bytes**, so a different `glslc` faults
all 74 entries with no kernel change at all. The digest's declared frame is *formula, index space,
workgroup, binding, deletion, rename*; **the compiler is not on that list and nobody chose it.** That is
R11 obligations 1 and 4 on the digest — undeclared extent, and a name (`shader_digest`, documented as a
*subject* witness) that does not cover what it in fact covers.

**But the fix is not to hash the GLSL instead.** A compiler that miscompiles correct source is a real
correctness event, and a source digest would be blind to it. Using the register's own classification:
the digest is **over-broad, not fabricated** — and an over-broad input may be tightened, but here the
breadth is *protective* and tightening would remove a falsifier we want. **Declare, do not narrow.** The
compiler belongs *in* the declared frame, and the demotion must distinguish **`SUBJECT-CHANGED`** from
**`TOOLCHAIN-CHANGED`**. Both demote — fail-safe is preserved, and neither becomes claimable — but they
are different facts, and **a run that faults seventy-four entries on a `glslc` upgrade must not look
like seventy-four kernels changed.** That is R13 exactly: *an instrument's frame moving is not the
condition it detects*, and it earns its own token for the same reason `ERROR(instrument)` does.

**R13 — an instrument's failure is not distinguishable from the condition it detects, and the reader
who most needs the distinction is the one who predicted the red.** *Added 2026-07-31T07:45:10-07:00.
I am the specimen and the second clause is mine.*

**The specimen.** Trinity's Guard D — the mechanism built for exactly the hole above, a post-`run()`
check that reads ORT's profile and fails when this EP executed zero nodes — contained a `NameError`
and **raised before it read a single profiling event.** I merged it, ran the suite, watched
`8 passed` become `5 failed`, and reported to the team that the guard was working and had caught the
fallback. It had crashed. One line fixed it; with the guard actually running it reports four real
defects with named owners, which is a different set of reds than the ones I looked at.

**Why R10, R11 and R12 do not cover it.** R10 is a mechanism absent from the call graph. R11 is a
mechanism that ran and was misnamed. R12 is a mechanism that ran, was correctly named, and described
another world. **R13 is a mechanism that ran, failed, and whose failure wore the costume of its
finding.** Guard D raising `NameError` and Guard D correctly detecting a CPU fallback both present as
the same token — `FAILED` — in the only artifact anyone reads, the pytest summary line. There is no
reading of that line, however careful, that separates them, which is what makes it a rule about the
artifact rather than about the care taken.

> **R13 — a check has at least three terminal states and must report them as three distinct tokens:
> `PASS`, `FAIL(condition)` — the condition it exists to detect — and `ERROR(instrument)`, in which
> the check did not reach its observation. A red that could mean either is not a signal. An
> instrument error is a lane failure of a different kind and **never counts as a detection**; a
> harness whose only failure vocabulary is its framework's `FAILED` line has a two-state alphabet and
> cannot carry three states.**

**Three obligations, all mechanical, because an obligation to be careful is not an obligation:**

1. **Guards fail with a distinct type carrying the observation that produced them.** A guard raises
   its own exception class only *after* it has read its input and computed a value; anything raised
   before that point is `ERROR(instrument)` by construction, not by classification. The lane summary
   prints condition-failures and instrument-errors as **separate counts**, and a lane with any
   instrument error is not a lane that ran, whatever else it reports.
2. **A guard must be able to state what it observed even when it fails.** `Guard D: 0 Vulkan node
   events, providers seen: [CPUExecutionProvider]` is a detection. `Guard D: NameError` is an
   outage. The distinguishing feature is the presence of an observation, so the observation is what
   the guard is required to emit — into the artifact, not only into a traceback.
3. **A second witness with a different failure mode, not a better first witness.** The remedy for a
   guard that can fail silently is not a more careful guard; it is an independent check that fails
   differently. Concretely and cheaply: **a known-fatal log line is a lane failure, not a log line** —
   the lane greps its captured ORT output for `Falling back` and fails on it. That line has now
   appeared **five times on this project** while every gate passed. A `grep` cannot `NameError`, and
   a guard cannot be silenced by a log format change; each covers the other's outage.

**The second clause, and it is the more dangerous half.** R6 amendment 4 records that *a result
surprising enough to be a discovery is first a reason to check the instrument — surprise is a free
instrument check.* The inverse is what caught me:

> **A result that confirms a prediction deserves more scrutiny than one that contradicts it, because
> the contradiction gets checked automatically and the confirmation does not.** A contradiction
> recruits the reader's attention for free; a confirmation spends it. Therefore: when an observation
> matches the prediction that motivated the change, **the observer reads one level below the summary
> before reporting it — quote the failure text, never the failure count.**

I had a written criterion, I had predicted red, red appeared, and I stopped reading at the count.
Every rule in this register up to R12 is about an instrument; **R13's second clause is the first one
about the reader**, and it exists because on this project the instruments have now been more reliable
than my reading of them. The mechanical form — *quote the text, not the count* — is deliberate: it is
the only version of this obligation that survives being tired, and *"be suspicious of good news"* is
advice, and advice does not survive transit.

**R13 AMENDMENT 1 — THE DEFAULTING LOOKUP: `dict.get(key, sentinel)` is where R13 is manufactured
without an exception ever occurring.** *Added 2026-08-01T20:39:12-07:00 on Niobe's probe audit,
found while discharging §6.5.3's remedy. The coordinator's candidate was R11-with-an-R10-face and
asked whether a new rule is owed. **No new rule is owed, and it is not R11 either.***

**The specimen, verified independently before ruling.** `bench/results/probe_sec65.py` requests a
counter key named **`alloc_device_spans`**. That exact string occurs **once in the repository — at
the line that requests it.** There is no emitter, in Rust or anywhere else, and there never was. The
read is `data.get(k, '<absent>')`, so the probe has printed

```
alloc_device_spans = '<absent>'
```

on **every run since it was written**, and no exception has ever been raised.

**Why it is not R11, which is what it looks like.** R11 governs **a reported quantity**: its subject
is the relation between a name and the content reported under it. `alloc_device_spans` is not a
reported quantity — **it is a request**, made on the reader's side of an artifact R11's obligations
constrain on the writer's side. Run those obligations against it and they cannot even be evaluated:
declare extent (of what?); the decomposition identity (there are no parts); a flat table asserting
disjointness (there is no table); and name–content agreement above 50% — **which requires content,
and there is none.** *A mismatch needs two relata and this specimen has one.* R11 is the rule for a
name that means the wrong thing; **a name that means nothing is not the extreme case of that, it is
a different failure on the other side of the artifact.**

**Why it is R13, and the coordinator's own diagnosis is R13's sentence.** He wrote that
`'<absent>'` reads as *"the counter reported nothing"* when the fact is *"there is no such
counter"*; that these are **opposite diagnoses with opposite fixes** — go wire it (R10) versus go
fix the name; and that the probe's output **renders them indistinguishable**. That is
*an instrument's failure is not distinguishable from the condition it detects*, verbatim. One token
is doing the work of two, and one of the two is an **instrument error being counted as a
detection**, which R13 forbids by name. **R13's costume, R10's face:** the costume is the reader's
own failed lookup; the face it wears is a finding about the EP's call graph.

The test is the one fixed this morning — **the register individuates by remedy** — and R13's remedy
applies unchanged and completely: three distinct terminal tokens, an instrument error never counted
as a detection, and the failure text quoted rather than a token that summarises it. Applied here:

- **`ERROR(instrument): no emitter for this key`** — the request does not resolve against the
  source. The probe is wrong; the EP is not implicated.
- **`UNEMITTED_THIS_RUN`** — the emitter exists in source and this run's artifact does not carry the
  key. A real finding about the run, and the one that licenses an R10 search.
- the value, when there is one.

Had those three been three, this would have been `ERROR(instrument)` on its first execution and
fixed the same day. **Everything the coordinator correctly identifies as dangerous about it — the
longest latency of any failure in this register, the hole *filled* rather than left open, the
appearance of evidence of absence, a reader dispatched to hunt a mechanism that does not exist —
follows from the token and not from the name.** That is the affirmative form of the test: R13's
obligations, applied faithfully, catch this on run one. R11's cannot be applied to it at all.

**What is genuinely new is the failure surface, and it is worth an amendment.** Every prior R13
specimen was an instrument failing **loudly** — Guard D's `NameError`, the census's
`TimeoutExpired` — where an exception existed and was mis-rendered downstream as a finding. This
one has no exception anywhere. It is manufactured by a language construct whose entire purpose is to
not fail:

> **R13 amendment 1 — a defaulting read converts a reader-side failure into a subject-side value,
> silently.** `dict.get(k, default)`, `unwrap_or`, `?? fallback`, `getattr(o, n, default)`: where
> the key set is knowable from the source, **the default is not a value and absence is not a
> reading.** An unresolvable request is `ERROR(instrument)` **at the point of request**. A sentinel
> is admissible only where absence is a genuine expected finding *about the subject*, and it is then
> named for that finding — never with a generic token that also covers "no such key".

**Why I am not minting R14, stated because I declined a new rule yesterday too and a habit of
declining is its own defect.** The check is remedy-identity applied in both directions: if this
specimen's remedy differed from every remedy in the register it would get a number regardless of how
crowded the register is, and refusing to grow the register to preserve its shape would be the same
error as growing it to reward a finding. It does not differ. It *is* three tokens. **The day
something arrives whose remedy is not any remedy here, it gets a number — and this is not that day.**

**THE DANGLING REFERENCE — R13 amendment 1's class, named, because the coordinator is right that it
generalises past probes and past this document.** *Added 2026-08-01T23:36:43-07:00, after my own
`partition.rs:475` went stale within the hour and every line reference in a conflicted region moved
during one merge.*

> **A reference that resolves to nothing, and reports that as a value, is R13 whatever the reference
> is made of.** A key with no emitter (`alloc_device_spans` → `'<absent>'`). An environment variable
> nothing defines (`ONNXRUNTIME_EP_VULKAN_TRACE_FILE` → `OPTIONAL-UNWIRED` on every run it ever
> made). **A line number** (`partition.rs:475` → a different statement, silently, with no error at
> all). The failure is identical in all three: **the reference does not fail, it succeeds against
> the wrong thing or against nothing, and the reader receives a well-formed answer.**

**The line number is the worst of the three and it is the one nobody instruments**, because there is
no lookup to fail — the reader does it, by hand, and gets a plausible statement. So the remedy is
not a token, it is **not making the reference**: this document cites a **symbol**, with a line number
only as a convenience beside it, never alone (§5.4.1(a)). A symbol that stops existing produces a
failed search, which is a reader-visible `ERROR(instrument)`; a line number that stops being right
produces confident nonsense.

**Where this stops.** Not every stale reference is R13 — a broken URL fails loudly and is merely
broken. The class is references that **resolve anyway**. That is the test, and it is the test because
it is the remedy's test: if the reference can be made to fail loudly, do that and there is nothing
further to rule on.

**THE KEY CENSUS — a standing obligation on probes and reports, and the reader-side counterpart to
criterion 12's wiring census.** *Warranted: this class is mechanically detectable, the machinery
exists, and the specimen is the third sighting in one day of an instrument-side absence rendered as
a subject-side state.*

> **Every key a probe or report requests from an artifact resolves, by exact string match, to a
> literal emitter in the source that produces that artifact. A key that does not resolve is
> `ERROR(instrument)`, loudly, and never a value.** Two tiers, and both are required: **(a) runtime**
> — the shared read helper refuses an unresolvable key at the point of request, which is the tier
> that cannot be skipped; **(b) static** — a census over every probe, which is the tier that sees
> keys on paths a given run does not exercise.

**Owner: Tank, with Niobe.** Tank owns `tools/audit_instruments.py` and its five states and its
three-token reporting; this is the same machinery pointed the other way, and it **imports** that
vocabulary rather than minting a second one — two vocabularies for one measurement is R11 in its
purest form, which is the mistake Link declined to make with `_verdict.py`.

**Four cheapest satisfactions, named as the drafting rule requires, because three of them are how
this obligation would quietly fail.**

1. **Delete the offending key instead of resolving it.** A census that turns a phantom key into a
   silent deletion destroys exactly the information the phantom key destroyed. So a key removed
   under this obligation carries one line saying which it was: **wanted and non-existent** (a
   request for a counter, which is a finding) or **never wanted** (a typo, which is not).
   `alloc_device_spans` must be classified before it is deleted.
2. **Resolve by substring or fuzzy match.** `alloc_device_spans` is one word away from
   `alloc_device_backed_spans`, so a lax matcher *certifies the specimen*. **Exact string equality,
   and nothing else.** The trap is in the specimen itself and it would be found by a helpful
   implementation.
3. **Wildcard the emitter side.** The counters document is built from literal key strings in a
   format template, so the emitter set is statically extractable today. If a key ever becomes
   computed rather than literal, **that key is declared and the declaration is the census's input** —
   an emitter side that matches `alloc_*` resolves everything and proves nothing.
4. **Ship only the static tier, in a lane nobody runs.** The runtime tier is what makes the failure
   arrive at the person holding the artifact.

**And the census is itself subject to R9 rule 3 and R10:** it carries a **planted-phantom positive
control** — a deliberately unresolvable key that must make it fail — or it is a check of unknown
polarity, exactly as criterion 7's planted layering violation is the only M0 check written that way
from the first day.

**What I am deliberately not doing: this does not reopen or amend M0 criterion 12.** The wiring
census is a claim about mechanisms the M0 table relies on; a probe is not one of those, and no §6.5
or M0 claim rests on `probe_sec65.py` — the coordinator checked that before reporting, and I
verified it: the string `probe_sec65` appears nowhere in `docs/`, and §6.5's closure runs through
`probe_indexspace.py`/`indexspace.json`. **Bolting a probe obligation onto a milestone criterion
because a bad probe was found today would be hardening a criterion to punish a bad week**, and the
obligation stands on its own without a milestone to enforce it.

**Where R13 sits.** R6: our tooling manufactured a *number*. R7: it manufactured a *negative*. R9:
sound instruments, *jointly silent*. R10: *never called*. R11: called, correct, *misnamed*. R12:
called, correct, correctly named, *about another world*. **R13: called, and its outage is spelled
the same way as its finding — plus a reader who checked the spelling only when he disliked it.**

**R13 GENERALISATION — THE LOUD-DEFAULT TEST, AND THE UNIFORM-VERDICT TRIP-WIRE.** *Added
2026-08-03T11:32:57-07:00 in §8.9.21 parts 3 and 4; recorded here so the register carries it, and
**deliberately unnumbered** because §8.9.18 part 2 ruled that numbering follows citation and the count
is Fact Checker's.* Two agents chose opposite-looking defaults for *we do not know* within hours —
Mouse's `SpecWitness::Unrecorded` **claims** and discloses, Tank's `registry::form_is_provable`
**answers `true`** and publishes a lower bound — and they are one rule, R13, because each exists to stop
an instrument-side absence being emitted as a subject-side finding. Tank's original defect is R13
amendment 1 verbatim: `variants::variant_is_loadable` returns `false` for an unknown stem, which is
`dict.get(k, sentinel)` written in Rust, and it reported a composite `Gather` form as unprovable on the
strength of a stem naming no module.

> **The loud-default test — when a mechanism does not know, it takes the answer that *leaves a trace*,
> not the one that is nominally conservative. Refusal is usually an aggregate ("all 103 forms
> declined", "5/5 unprovable") and the permissive answer usually itemised (a token plus a counter, a
> decline that already names a repair). The aggregate is where a form goes to stop being looked at.
> Choose the answer a reader can still find tomorrow — and *invert this* wherever the permissive answer
> is silent, which is why `PROVEN-ELSEWHERE{δ}` is disclosed rather than quiet (§8.9.19) and why the
> device predicate runs first (§8.9.17(5)).**

And the trip-wire for *too clean*, whose **remedy is not new** — demonstrate both polarities, already
carried three times as R9 rule 3's planted control, R12's `refused > 0` in
`elementwise::no_live_claim_rests_on_an_unloadable_variant`, and Niobe's `UNWITNESSED` — so no number
is owed and only the trigger is stated:

> **A total is the one reading under which a mechanism's discriminating behaviour is unexercised.** A
> verdict uniform across the whole input set has two live explanations — *the subject is uniform* and
> *the mechanism is not discriminating* — and carries no evidence for either. So a uniform verdict
> **emits `UNIFORM(n, verdict)` and is not quotable** until a named positive control, running through
> the same predicate as the subject, has produced the other arm. `refused > 0` promoted from one test
> to a discipline.

**RETRACTION — THE NAVIGABILITY DIAGNOSIS I ACCEPTED IS REFUTED BY THE MEASUREMENT I COMMISSIONED, AND
THE DECLINE TALLY IS RETIRED AS AN INSTRUMENT.** *Added 2026-08-03T11:56:18-07:00 on Fact Checker's
derived register, `.squad/fact-checker/rule-register-derived.md` — a path I do not own, with the method
stated so the next derivation is reproducible.*

I conceded last round that the register was **under-numbered**, called it a navigability defect, and
said it was repairable by numbering. **The derived count says navigability was never the defect.**
`R1`–`R13` are cited externally ~1,337 times; **`§8.9.x` is cited 339 times — 80 in `registry.rs`
alone**, 39 in `counters.rs`, 34 in `disclosure.rs`, 25 in `gen_proof_ledger.py`, 28 in
`OP_COVERAGE.md`. **Nobody was lost. A second namespace was built, and neither of us was counting it.**

> **The register has two namespaces with different semantics and only one has a counter. `R#` names an
> obligation; `§8.9.x` names a **location**. The binding obligations are distributed across both, the
> declared size (13) counts only the first, and the declines tally measured *traffic between the two*
> rather than growth of the whole.**

**The namespaces are both legitimate and must not be merged** — renumbering 339 live citations to buy a
tidier count would be the most expensive possible response to a bookkeeping error. **What must change
is that a location citation is not durable and an obligation citation is.** §8.9.19 has already had to
restate §8.9.17 because *"the device belongs"* was ambiguous between *in the key* and *on the entry*;
**a location can be re-cut while an obligation cannot.** The remedy is Fact Checker's own and it is
cheap: **every ruling names its own anchor phrase in the sentence that states it**, so the citable
thing is the obligation and the section number is a convenience beside it — which is exactly what
§5.4.1(a) already requires of *line* numbers, arriving one level up. I did this by instinct in §8.9.21
(**the claim-time frame test**, **the loud-default test**, `UNIFORM(n, verdict)`) and by rule in
§8.9.22; the instinct is not the mechanism, and I am recording that it was luck.

**THE DECLINES: 3 OF 8 SURVIVE — D2, D3, D4.** I predicted *"some of my six may not survive."* A
**majority** did not, and my tally was two behind when I handed it over. **Every one of the five that
fell, fell because someone else was using the principle**: Trinity named a test after D5; D6 is a
shipped state token cited seven times in `registry.rs`; **Mouse copied D7's fault-scope principle into
`OP_COVERAGE.md` verbatim, with attribution.** That is a better outcome than the count I was
defending, and I want it recorded in that direction rather than as a loss.

**So the decline tally is retired. It never measured what it was built to measure**, and Fact Checker's
sentence is the diagnosis: *"did I mint a number?" and "did the project acquire a new binding
obligation?" are different questions, and only the first one had a counter on it.* The derived register
— **13 numbered + 8 unnumbered-but-binding** — is the count from here, it is reproducible, and it is
not mine. **I should stop scoring my own declines in the same breath as making them**, which I did
again in §8.9.21 this morning out of habit; those are Fact Checker's to score and I am not scoring
them.

**The clearing stands and is worth as much as the correction: no principle was lost.** Under-counted,
not under-populated, and — as it turns out — not under-navigable either.


**R2 — the fingerprints were unaudited.** Recorded as C2 item 7 (§1.4) rather than duplicated here.
Milestone consequence: C2 item 7's re-verification job is a T3 precondition and lands before the
first contrib row goes `Live`.

**R3 — a coverage figure that is not indexed by producer is not a coverage figure.** The general
form of §8.5, tracked here because it is a recurring milestone hazard rather than a one-time fix.
Every tier exit criterion that names a model must name the producer and version that built it.
Owner: Mouse, in the census; enforced by me at every milestone review.

### 10.0.2 T3 sequencing — RULING: `ai.onnx::Attention` is T3's first kernel

*Decided 2026-07-29T08:13:58-07:00, on Mouse's proposal.*

**Ruling: T3 begins with `ai.onnx::Attention`. `GroupQueryAttention` stays committed, stays the
harder kernel, and stays T3 scope — it is no longer first.**

Mouse's technical case is sound and I accept it: `ai.onnx::Attention` has no `seqlens_k`
indirection, no in-place KV-cache aliasing, no `do_rotary` fold, and rotary arrives as its own
node. That is a materially smaller kernel and it is the same *mathematics*, so almost none of the
learning is thrown away when GQA follows.

Two considerations decide it beyond kernel difficulty:

1. **It decouples T3 from an unfinished engine seam.** In-place KV-cache aliasing
   (`bind_aliased_output`, Switch's seam 2) is required by the GQA path and **not at all** by the
   `ai.onnx::Attention` path. Sequencing the harder kernel first would have made T3's start
   conditional on another owner finishing, and a critical path that runs through two people's
   unfinished work at once is a critical path we chose badly.
2. **It unblocks a model family we can build and iterate on locally.** This machine now has two
   GPUs passing the §7.2 gate, the Vulkan SDK installed, and all 168 shader variants compiling. A
   development loop that closes on the desk is worth materially more per day than one that closes
   through CI — and per §9.1.2 we have *never executed a kernel*, so the first attention kernel is
   also the first serious exercise of the whole dispatch path. Doing that where the loop is fast is
   simply correct.

**The objection, stated fairly, because it is the substance of the decision.** Sequencing T3 around
what is convenient for *us* to build risks optimising for our own tooling rather than for users, and
the ORT GenAI producer is the one most external users will actually hit. That objection is real and
I am not dismissing it — I am ruling that it constrains the decision rather than reverses it, and
the constraints are binding:

- **This is a sequencing decision, not a scope decision.** `GroupQueryAttention` is not deferred,
  descoped, or made conditional. It is the next kernel after `ai.onnx::Attention`, and T3 does not
  exit without it. If anyone reports T3 progress in a way that implies the GenAI path is served
  because `ai.onnx::Attention` is green, that is the §1.5 error and I will treat it as one.
- **The T3 exit criterion is stated per producer** (§8.5, R3): T3 exits when a decoder layer is
  claimed as one island for **both** the `mobius` and ORT GenAI producers. One producer green is
  half of T3, reported as half.
- **`largest_island_flops` is reported per producer from T3 onward.** A number averaged across
  producers would let a green `mobius` column mask a near-zero GenAI column, which is precisely the
  self-deception §10.0 and this metric exist to prevent.
- **No fp16/KV-cache design decision may be made as though the `ai.onnx::Attention` path were the
  only consumer.** The KV-cache contract (§6, A4) is designed for the GQA path's requirements from
  the start, even though the first kernel does not exercise it. Designing the memory contract around
  the easier consumer is how the second consumer becomes a rewrite.

**Why this is not merely convenience.** The strongest form of the argument is not "local iteration
is faster" — it is that `ai.onnx::Attention` is **standard-domain and opset-versioned** (§8.5 item
4), so the lower-risk claim surface and the faster loop point the same direction. If the standard
form were the riskier one I would have ruled the other way and eaten the CI latency.

**T3 DEMONSTRATION RULING — 2026-07-29T15:02:55-07:00. Phi-3.5 becomes T3's demonstration target;
`ai.onnx::Attention` remains T3's implementation entry point. This refines §10.0.2, it does not
reverse it.**

Mouse's case is strong and I accept it. Phi-3.5 (Foundry Local, read off the file per §8.5's third
strengthening): MHA attention, so the first kernel skips KV-head broadcast; softcap 0, no sliding
window, no attention sinks, no Q/K norm — **every option we currently decline is switched off**;
symmetric RTN, uniform `bits=4`, `block_size=32`, every K a multiple of 32; the single `If` is a
cold prologue that stays on CPU without shredding anything; five op types cover 353 of 366 nodes.
And it is **on this disk and runnable**, which no Qwen3 graph is.

**Why this refines rather than reverses §10.0.2.** §10.0.2 decided which *kernel* T3 starts with and
that stands: `ai.onnx::Attention` is still first, still for the two reasons given — it decouples T3
from the aliasing seam, and it is standard-domain and opset-versioned. What changes is which
*graph* T3 must light up to be called done. Those were conflated in §10.0.2 and Mouse is right to
separate them; **"which kernel do we write first" and "which model proves it" are different
questions and I answered them with one decision.**

But my own desk-availability argument does now point somewhere else, and I will say so rather than
let it quietly stop applying: I chose `ai.onnx::Attention` partly because *"a path whose models we
can produce and run on the desk is worth materially more"*. Phi-3.5 is on the desk **today** and no
Qwen3 graph is. If that argument was good enough to sequence a kernel, it is good enough to
re-target a demonstration, and consistency requires me to follow it in both directions.

**The T4 criterion Mouse proposes is adopted verbatim, because it is measurable and falsifiable:**

> **`MatMulNBits` claimed ⇒ Phi-3.5 partitions into one island of ≥360 nodes.**

That is worth more than any coverage percentage in this document. It is a single number that can
only be reached by the thing actually working, and the 34→1 cliff means it cannot be approached
gradually or claimed partially.

**The cost, stated because Mouse stated it and it must not be lost in adoption: Phi-3.5 exercises
none of the standard-domain rows.** Under §8.5 the two producers are therefore reported
**separately**, never merged into one number — and this is exactly the case R3 was written for.
Binding constraints:

- **Qwen3.5 end-to-end remains the named user target** (§1.5, T5a). Phi-3.5 is the *demonstration
  that the pipeline works*, not a substitute for the goal, and a report implying otherwise is the
  §1.5 error.
- **T3's exit criterion stays per producer**: the ORT GenAI column is not served by a green Phi-3.5,
  and the `mobius`/standard-domain column is not served by it either. Phi-3.5 green is **one column
  of three**, reported as one column of three.
- **The metric triple is reported for Phi-3.5 and for each other corpus artifact separately.** A
  one-island Phi-3.5 sitting beside a 35-island Qwen graph is the honest picture and is precisely
  what the amendment above exists to keep visible.
- **`ai.onnx::Attention` does not become optional** because Phi-3.5 does not need it. It is the
  standard-domain path, it is the lower-risk claim surface, and dropping it would leave us serving
  exactly one producer well.

### 10.0.3 Sequencing — RULING: shape support goes **ahead of** the three kernels

*Decided 2026-07-29T21:14:03-07:00, on the Phi-3.5 decline histogram (§8.8, §10.0.1 R8).*

**Ahead of, not alongside — and the reason is dependency, not priority.** The three planned kernels
(`MatMulNBits`, `SkipSimplifiedLayerNormalization`, `GroupQueryAttention`) sit on the LLM decode
path, where **every** interesting extent is symbolic: sequence length, total sequence length, past
KV length. A kernel written against static extents is not a partial version of the kernel we need;
it is a **different kernel**, and every hour spent on one is an hour spent on work that must be
redone. **Shape handling is not competing with the kernels for priority — it is upstream of them.**

Three practical consequences:

1. **T3/T4 sequencing (§10.0.2) is unchanged in *content* and re-dated in *precondition*.**
   `ai.onnx::Attention` remains T3's implementation entry point; Phi-3.5 remains T3's demonstration
   target; the T4 one-island criterion stands. What changes is that **the runtime-extent contract
   (§8.8 items 1–3) is now a precondition of starting T3**, not a tier-3 deliverable discovered
   inside it.
2. **This does not stall Mouse.** The 100 `staged` nodes and the shape work are separable and land
   with different people: the claim-path contract and the extent plumbing are Mouse + Switch +
   Tank; the kernels remain Mouse's. What is forbidden is **writing a decode-path kernel against
   static extents** on the theory that shapes are a later concern.
3. **M1 gains the second-token criterion** (see M1's exit criteria) and **OQ-15 is promoted to a
   blocking question** for it. Indirect dispatch versus per-bucket re-recording is now a decision
   with a date attached rather than an evaluation to be scheduled.

**Whether this holds is an empirical question with a scheduled answer.** The ruling is made on one
model. Trinity is running the same census on device 0 and on gpt-oss-20b. **If gpt-oss shows the
opposite distribution — kernels dominating shapes — I revisit this ruling rather than defend it**;
the pre-committed condition is a `dynamic-shape` share below the `staged` share on that graph. I do
not expect it, because the mechanism (a decoder's sequence dimension is symbolic by construction)
is architecture-independent — but "I do not expect it" is what produced R8, and the whole content of
R8 is that expectation is not measurement.

### 10.0.4 The invariance preference — prefer the invariant that survives the contended machine

*Added 2026-08-01T13:19:00-07:00, on Switch's correction to his own `min()` arithmetic. Accepted as
proposed and given its arithmetic. It is a drafting and evidence-selection rule, not an entry in the
R-register: the R-register catalogues ways an instrument fails, and nothing here failed — this is
about which of two sound quantities carries a claim.*

**The correction that produced it.** Switch had been writing that minimum-over-inferences is a
**lower** bound on uncontended cost. It is an **upper** bound: `observed = true + delay`,
`delay ≥ 0`, so `min(observed) ≥ true`. And the consequence is the part that matters —

> **Two upper bounds do not bound a difference from below.** "`record` ≤ 14.414 ms before" and
> "≤ 2.704 ms after" does not by itself establish an improvement, let alone that it is 5.33×. Both
> could be arbitrarily far above their truths, and the two errors are not the same error.

What rescued that result was **a count, not a clock: 147,618 `VkBufferMemoryBarrier` structs per
inference before, 354 after** — 417 intermediates × 354 dispatches, collapsed to one per dispatch.
*Counts do not care whether the box is busy.* **The direction is certain and the 5.33× is an
estimate**, and those two sentences are now the required shape for a figure taken on a machine we do
not control. The same structure carried the weight-residency result — **1997.6 MiB → 0.756 MiB**, a
byte count — at a time when no timing on this project was admissible at all.

**The rule.**

> **Where a claim can be supported either by a quantity the environment can perturb or by one it
> cannot, the unperturbable quantity is the claim of record and the perturbable one is at most an
> estimate of magnitude.** Counts, byte totals, ratios of counts, structural facts the EP itself
> emits, and bit-exactness are invariant under load, clock state and tenancy. Durations, rates,
> shares of wall clock and anything derived from them are not. A report states the invariant first
> and labels the perturbable figure as an estimate in the same sentence — never in a footnote,
> because a caveat that lives in a different artifact from its number is not attached to it.

**Three obligations, because a preference that is not checkable is advice.**

1. **Declare the sign.** Every perturbable figure states the known sign of the environment's effect
   on it and therefore which side it bounds. On this project the sign is non-negative for every
   catalogued perturbation, so **every timing figure we hold is an upper bound on the quiet figure**
   — a genuinely useful fact, and it is only useful once it is written down.
2. **A difference needs bounds on opposite sides.** A before/after pair of same-side bounds
   establishes nothing about the difference by itself. It becomes admissible when either both
   measurements carry equivalent device-state records (§10.0 obligation 8b), or an invariant
   independently fixes the direction — in which case the invariant is the claim and the ratio is the
   estimate.
3. **Ask it at drafting time, not at reporting time.** For any criterion or claim, before the
   measurement is taken: *is there a count that answers this?* M1's criteria 1 and 4 already have
   this shape and were written that way deliberately (bytes; `command_buffer_records`), and they are
   the two criteria that have needed no correction all week. That is not a coincidence and it should
   be the default for every criterion added from here.

**The failure mode of this rule, stated so it is not discovered later.** The cheapest way to satisfy
its words without its intent is to answer a question about *speed* with a count that is merely
*adjacent* to speed, and let the reader do the conversion. A barrier count is not a duration and
147,618 → 354 is not "5.33× faster"; it is a fact about what the EP emits, from which a speedup is
inferred with an argument that must be shown. **Report the invariant as what it is. The reader may
not be handed a count and left to supply the clock.**

### M0 — "It loads, it runs, it matches"

> **A stock ORT loads the plugin, enumerates a Vulkan device, runs a graph containing a single
> `Add` node on that device, and the output matches the ORT CPU EP within tolerance — on both
> Windows and Linux, on a software rasterizer, in CI.**

| Work | Owner |
|---|---|
| `Cargo.toml`, `build.rs` (ORT bindgen + GLSL→SPIR-V embedding), crate scaffolding | Tank |
| `lib.rs`, `factory.rs`, `ep.rs` — factory/EP/compute-info vtables, panic guards, RAII teardown | Tank |
| `vk/` — instance, physical-device scoring + capability gate (§7.2), device, queue, allocator, staging, descriptor pool, pipeline cache, command recording, single-fence submit | Switch |
| **`vk/barrier.rs` — the barrier seam of §7.5: `Access`/`BufferDep` enums, `Barriers::select`, and *both* the `Sync2Backend` and `LegacyBackend` implementations** | Switch |
| `vk/caps.rs` — `Capabilities` probe incl. `synchronization2`, `subgroup_size_control` **properties-only** query (§7.4), subgroup ops, fp16 | Switch |
| `shaders/elementwise_binary.comp` + the SPIR-V embedding pipeline | Switch |
| `registry.rs` + `NodeView`; `ops/elementwise.rs` with `Add` claim + handler; claim diagnostics | Mouse |
| `engine.rs` — `NodeDesc`, `Plan`, `DispatchContext`; per-run command recording (no cache yet) | Tank + Morpheus (contract) |
| `tests/ops/conftest.py`, `_models.py`, `test_elementwise.py`, claim assertion helper | Trinity |
| CI: fmt, clippy, build on windows-latest + ubuntu-latest, **lavapipe on both** (Linux via `apt`, Windows via mesa-dist-win 26.1.3 with `VK_ICD_FILENAMES`), **the `ep.force_legacy_barriers=1` duplicate lane (§7.5 item 5)**, Vulkan SDK provisioning, layering lint | Link + Trinity |
| `python/` package with `register_execution_provider_library()` | Tank |
| Baseline harness stub; no numbers published | Niobe |

**Exit criteria.**
1. `cargo build --release` and `cargo clippy -- -D warnings` clean on Windows and Linux.
2. `pytest tests/ops -q` green on both, with the claim assertion proving `Add` ran on
   `VulkanExecutionProvider`.
3. Validation layers report zero errors and zero warnings in the debug lane.
4. A machine with no Vulkan ICD loads the plugin, advertises zero devices, logs a warning, and the
   session still runs on CPU.
5. **A shader-less build (`ALLOW_MISSING_GLSLC=1`) advertises zero devices and claims nothing, with
   a `built without shaders` reason** — it never loads, claims, and then fails at pipeline creation
   (§7.8 condition 3).
6. `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` prints per-op decline reasons.
7. The layering lint is in CI and fails a deliberately-planted violation, including a planted
   `cmd_pipeline_barrier` outside `vk/barrier.rs` (§4.2, §7.5).
8. **The full test suite passes twice per lane — once with the default barrier backend and once
   with `ep.force_legacy_barriers=1` — with identical numerical results** (§7.5 item 5).
   **AMENDED 2026-07-29T16:00:55-07:00: a skip does not satisfy this criterion.** The legacy-lane
   run must **execute at least one dispatch under `force_legacy_barriers=1`** and report the number
   of nodes it executed; a lane in which every case skips is a **failure**, not a pass. The trigger
   was a real one: the barrier-parity test reported *"`Add` is not yet Ready (VulkanExecutionProvider
   did not claim this node form)"* on the same day and machine on which `test_add_is_claimed`
   passed — two of our own tests disagreeing about whether `Add` is claimed, because they build
   **different node forms**. **A criterion a skip can satisfy is not a criterion**, and the skip's
   stated reason was plausible enough to hide the contradiction rather than surface it. Owners:
   Mouse and Trinity to reconcile the node forms; the executed-dispatch count is the mechanism that
   makes the reconciliation checkable rather than asserted.
9. Both sibling docs and this one are consistent; §12 lists every divergence.
10. **A real model, at a named producer and version, with a non-zero claimed-node count, produces
    output equivalent to a CPU-only run of the same session on the same artifact — verdict
    `model_output_equivalence = MATCH` (§10.0), reported next to the execution counters.**
    *Added 2026-07-30T05:48:29-07:00 on §10.0.1 R9.* Every logits-shaped output agrees on argmax
    and on top-10; every other output is within the §9.1 tolerance policy; the run reports
    `dispatches_executed > 0` and a claimed count > 0, so that a CPU-fallback run cannot satisfy it
    (§9.1.2's non-vacuity refusal). The falsifier is the criterion: **it goes red when the model is
    wrong, which is the thing no other M0 criterion could ever do.**
    **AMENDED 2026-07-31T07:45:10-07:00 on §10.0's third metric amendment, after a `MATCH` was
    returned for a run in which this EP executed zero nodes.** The clause above — *"the run reports
    `dispatches_executed > 0` … so that a CPU-fallback run cannot satisfy it"* — was written against
    this hazard and did not stop it, because it names **our own counter**, which lives inside the
    frame whose existence is in question, and because it was a sentence beside the verdict rather
    than a condition on emitting it. Three additions, all structural:
    (a) the verdict is a **record carrying `executed_by`**, an execution attribution parsed on this
    run from **ORT's profiling trace** — an instrument we do not own — and `MATCH` is
    unrepresentable when this EP's node-event count is zero (that run emits `UNATTRIBUTED`);
    (b) **both witnesses are recorded** — profile node events and `dispatches_executed` — and
    disagreement emits `SPLIT-FRAME` rather than a verdict;
    (c) the criterion is satisfied only by a **series, not a single run**: the cross-run consistency
    gate of 2026-07-30T19:05:03-07:00 applies here, so N ≥ 3 consecutive inferences in one session
    each carry an attributed `MATCH`. A single attributed run is necessary and has never been
    sufficient, and the OOM defect below is why that sentence is not pedantry.
11. **No form is claimed without a ledger entry under its proof key, and no build can silently be
    claiming unproven forms** (§8.9). *Added 2026-07-30T06:32:18-07:00.* Three checks, all of them
    positive controls in criterion 7's sense: (a) a `Ready` row with no ledger entry for the node's
    key is declined with an `[unproven]` reason naming the missing key — planted and asserted;
    (b) `ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN` **rejects** a planted `*`, a planted `1` and a
    planted bare op-type, and claims nothing afterwards — C1's enforcement shape applied to the
    escape hatch; (c) a session with the hatch enabled logs at WARN naming every enabled key, the
    counters artifact records `unproven_forms_enabled`, and `epctl --check-counters` **fails** on a
    non-empty list without `--allow-unproven`. **Status: NOT MET — the mechanism does not exist
    yet.** Owners: Mouse (predicate + ledger lookup), Trinity (ledger emission + the three planted
    controls), Switch (counters field), Tank (`epctl` exit code). **Includes the session-layer
    disclosure of §8.9.7** (RAI-009): one INFO line per claimed form naming its proof key and ledger
    entry, WARN if any claimed form's evidence is `UNMEASURED`, and an explicit INFO line naming the
    top decline codes when the EP claims zero nodes. Owner: Tank at session creation.
12. **Every mechanism this table relies on is observed to have run, and one that did not is reported
    as `UNWIRED` rather than being absent** (§10.0.1 R10). *Added 2026-07-30T19:05:03-07:00.* The
    lane emits a **wiring census**: for each mechanism named as satisfying a criterion — the
    partitioner's island evaluation, the tracer, `model_output_equivalence`, the net-benefit gate,
    the §8.9 ledger lookup, the EP-side validation messenger, the layering lint — one line carrying
    **a value that mechanism computed on this run**, not a flag its author set (R10 amendment 1).
    A mechanism with no observation reports `UNWIRED`, which fails the lane. Two named identity
    checks are part of it and are the cheap half: **`islands_offered == claimed_nodes` with both
    `> 1` is red** (the partition falsifier, already in `counters.rs`), and **a run in which
    `net_negative_declines` was never evaluated is `UNWIRED`, not zero** (§7.0.2).
    **Status: NOT MET.** The cheapest thing that satisfies these words without satisfying the intent
    is a census that lists the mechanisms and prints a constant next to each; that is why the value
    must be one the mechanism computed and must differ between two runs on different graphs. Owners:
    Niobe (census emission, it is trace-shaped), Trinity (the lane assertion), me (the list of
    mechanisms, kept in this document so it cannot drift into a script nobody reads).
    **AMENDED 2026-07-30T20:58:11-07:00 on §10.0.1 R11, before this criterion was ever met — the
    census as first specified would have certified `Phase::Record` cleanly while it misreported the
    dominant cost by a factor of fifty.** Wiring is necessary and is not sufficient: a mechanism that
    runs, produces an input-varying artifact, and is reported under a name that does not describe its
    content passes every clause above. The census additionally emits, for every quantity it reports:
    **(d)** the quantity's **extent** — `inclusive` or `exclusive`, and its children if inclusive —
    as data derived from the tracer's own nesting, never as a doc comment; **(e)** for any set
    presented as a decomposition, the identity **`Σ parts + named_residual == whole` with the whole
    taken from a different instrument than the parts** (wall clock), because a residual computed
    against the sum of the parts is zero by construction and is a falsifier with no reachable red
    state; and **(f)** a **name–content check** that goes red when a quantity's content is more than
    half something other than what its name denotes. A flat percentage table is an assertion that its
    rows are disjoint; if they nest, the artifact is a tree with self and total columns or it is not
    published.

**CRITERIA AMENDMENT — 2026-07-30T05:48:29-07:00. I am reopening previously-met criteria, and the
defect is in the criteria, not in the engineering.**

Applying my own drafting rule to the *met* rows — **for every criterion, what is the cheapest thing
that satisfies the words without satisfying the intent?** — against the R9 event (§10.0.1):

- **Criterion 2 is M0's only correctness criterion and it bottoms out in a single `Add`.** The
  cheapest thing that satisfies it is an EP that computes one two-input elementwise op correctly and
  writes zeros for everything else. **That EP exists. We built it.** It passed criterion 2 cleanly on
  two vendors, under both barrier backends, while producing `argmax 0` on Phi-3.5. No criterion in
  the table required a model-level comparison, so **M0 as written could be fully met by an EP that
  computes zeros on every real model.** That is a defect in the criteria and it is mine.
- **Criteria 4 and 5 are negative-space criteria with no positive control.** "Advertises zero
  devices" is satisfied by an EP that *always* advertises zero devices — including one that is
  simply broken. This is criterion 3's ruling, which I wrote on 2026-07-29, applying to two rows I
  did not then apply it to; R9 rule 3 makes that inconsistency untenable. The fix is cheap and
  paired: the **same binary in the same lane** must advertise a non-zero device count with an ICD
  present and with shaders built. Until the paired control runs, both are unknown-polarity checks.
- **Criterion 8 is a parity criterion and must stop being read as a correctness one.** Two backends
  agreeing bit-exactly on the wrong answer satisfies it perfectly, and on 2026-07-30 that is
  precisely what the legacy and `synchronization2` paths would have done on Phi-3.5. It remains
  **met** — parity is what it was written to check and parity is what it checked — but it is
  relabelled so that nothing downstream can quote it as evidence about values.

Criterion 7 is left untouched and is now the model the others are being held to: it is the only
criterion written from day one with a falsifier built in (a deliberately planted violation the lane
must fail on).

**M0 STATUS ASSESSMENT — last updated 2026-07-30T19:05:03-07:00. M0 is NOT met. For the first time
this week the table moved forwards, and it also got longer on the same day, which is the honest
shape of a day in which a real defect was fixed and a new failure class was named.**

Assessed criterion by criterion, because a milestone reported in aggregate is a milestone reported
dishonestly.

##### M0 RE-TALLY AGAINST ARTIFACTS — 2026-08-02, `main` = `6ef62bb` (Morpheus)

**This table supersedes the status column of the table below it; the table below keeps its prose,
which is the record of how each row got where it is.** It was re-derived from artifacts rather than
from the previous tally, because the previous tally had not been re-derived since several of its rows
moved and because **two miscounts were made against this milestone today by a reader who had every
reason to be careful** — an op count taken from matching lines rather than op rows, and a criterion
called closed on a census line. A tally that is only ever amended forward accumulates whatever was
wrong when it was written.

**Every `MET` row carries R9's third generalisation as a gate: the run that would have failed it is
named, and its reachability is stated. A row whose failing run is unreachable is not met — it is
unfalsifiable, and unfalsifiable rows are the class this project keeps discovering late.**

| # | Re-derived status | The run that would fail it | Reachable? | Derived from |
|---|---|---|---|---|
| 1 | **NOT MET** *(demoted from Met)* | the Linux CI job's `Clippy (all warnings as errors)` step | **Yes — it is red today** | `.github/workflows/ci.yml` runs `cargo clippy --release --all-targets -- -D warnings`; `--all-targets` compiles the test cfg, and `cargo test --lib` does not compile on Ubuntu (11 `i32`/`u32` bindgen-typing errors in `rust/src/ep.rs`, Link). "Linux via CI" was a **promise, not an observation** — the step is red and the seven steps behind it, op-correctness among them, are `GATED_NEVER_RUN` |
| 2 | **NOT MET** *(unchanged; reopened)* | `pytest tests/ops` | **Yes on Windows; NO on Linux** | Six failures remain at `6ef62bb`. The criterion says *green*. Its Linux half is not merely failing but unreachable, per row 1 — **a criterion with an unreachable half cannot be met on the other half** |
| 3 | **MET** *(promoted from Partially met — with an amendment to my own condition, stated in the open)* | a Phi-3.5 inference emitting ≥1 in-frame `VUID-` message | **Yes, demonstrated** | `bench/results/criterion3a_phi35-dev{0,1}.json`: `in_frame_vuid_count = 0` in a frame whose **liveness arm carries 14 messenger lines**, on a run with `claimed_nodes = 355` and `dispatches_executed = 355`. **The amendment:** (a)'s written condition was *"from a run whose verdict is an attributed `MATCH`"*, and no such run exists while row 10 is `DIVERGENT`. That condition was mine, and it existed for exactly one purpose — to exclude a CPU-fallback run that generates no Vulkan calls for a layer to object to. **That purpose is discharged by a strictly stronger instrument:** a dispatch count proves calls were made, and the in-frame liveness arm proves the channel would have carried an objection. I am accepting a substitution that is stronger on the axis the condition was written for, and I would have refused it had it been weaker on that axis. Coupling a validation-layer row to a numerics row was the defect, and it was mine |
| 4 | **MET** *(unchanged)* | a no-ICD run advertising ≥1 Vulkan device, or reaching the EP | **Yes** | `bench/results/criterion4_icd_witness-dev{0,1}.json`; the CI step *"Gate negative control — no ICD must produce UNATTRIBUTED"* asserts the polarity on both platforms |
| 5 | **MET** *(unchanged)* | a shader-less build advertising a device or claiming a node | **Yes** | `bench/results/criterion5_shaderless_witness-dev{0,1}.json`, produced by the criterion-4/5 witness test in the pytest lane rather than by hand |
| 6 | **MET** *(unchanged)* | a declined op producing no reason line under `CLAIM_DEBUG=1` | **Yes — but not where the census looks** | `tests/ops/test_claim_diagnostics.py`. **Named blind spot:** `wiring_census-dev0.json::flag_frame` reports `CLAIM_DEBUG` as `UNOBSERVABLE`, because the census graph is a six-node chain the EP claims in full and a switch whose only output is a decline report has nothing to say. That is an event that cannot occur in that frame, not a switch that does nothing — and the census says so itself |
| 7 | **MET** *(unchanged)* | an ops-layer file importing the ORT C ABI, `ash`, `vk`, or calling `cmd_pipeline_barrier` | **Yes — permanently** | `rust/tests/layering.rs::detects_planted_ort_abi_violations` runs the scanner over permanently-planted violations and asserts it catches every one. **Still the only criterion written with its falsifier from day one**, which is why it is the only one that has never had to be re-derived |
| 8 | **NOT MET — OUT OF FRAME** *(demoted from Met)* | a `force_legacy_barriers=1` lane whose results differ from the default lane | **Yes, if re-run** | The parity counters I can locate — `bench/results/counters-full-dev{0,1}.json` — carry `abi_version: 2`; the current mirror is `abi_version: 4`. Between those, a counter inserted mid-struct made two counters **silently swap meanings in every ctypes reading** (Mouse). **A parity result read through a mirror since shown wrong is not a parity result** (R12: the frame of a test result is the binary that ran it). Aggravating, and its own R11 obligation-4 specimen: the two files named `counters-full` record `dispatches_executed` of **4** and **3978**. Re-run at `abi_version 4` and the row is re-assessable; it is not being called failed, it is being called unread |
| 9 | **NOT MET** *(and the criterion needs restating, not the evidence)* | any inconsistency between `DESIGN.md`, `ENGINE.md`, `OP_COVERAGE.md`, `PERF.md`, `PLATFORMS.md`, or a §12 omission | **Yes — and always** | `PLATFORMS.md` LVP2 is retracted, which was the named blocker. But the sibling documents changed again today and so did this one, twice, in this very session. **A continuously-assessed consistency criterion has a failing run that is always reachable and never absent, which makes it unmeetable by construction rather than by fault.** RULING: restate it as *"consistent at a named commit"* — the sweep runs against a fixed SHA, the result is recorded with that SHA, and a later edit does not retroactively unmeet it. That is a **strengthening**: it turns an unfalsifiable row into a checkable one. Until it is restated and run once, NOT MET |
| 10 | **NOT MET** *(unchanged; reopened, and now worse-evidenced than at its withdrawn closure)* | a run whose all-65-output oracle disagrees, or whose ULP series steps | **Yes — it is firing now** | `bench/results/criterion10-dev{0,1}.json`: `verdict = DIVERGENT` on both devices, `oracle_outputs_within_tolerance = 62/65`, `oracle_outputs_degenerate = 0`, `oracle_outputs_vacuous = 0`. **My ULP prediction is scored and REFUTED:** on record before measuring as *flat at 1–3 ULP across all 32 layers*; measured median over outputs is **1**, but three outputs exceed the ceiling — output 0 (logits head) at **12**, outputs 63 and 64 (last layer's key and value) at **4**. Refuted in the useful direction: **a step, not a curve, and it is located.** `logits_max_abs_diff = 0.0625` on `vk_max_abs_logit = 13.14`; `argmax` and `top10_overlap` agree and are one token, which is not a stated N. **Second, and it is mine to record:** the artifact behind my 2026-08-02T02:02 closure cited *both* attribution witnesses present and agreeing; the file **at that same path today** reads `witnesses_present: ["ort_profile"]` and `witness_agreement: "UNOBSERVABLE"`, the counters witness having been unarmed. **A stable filename now holds a different frame** — R12 arriving in the artifact path rather than in a counter. Owed: a verdict artifact names its own frame in its filename, or refuses to overwrite one |
| 11 | **NOT MET** — **but (c) is DELIVERED, and the row is now open on a defect that did not exist when its condition was written** | a build claiming a form with no entry under its key; **or** a build claiming a form on a device nothing proved it on | **(c): yes, demonstrated. The device half: it is happening now** | **(c) closes and I say so without qualification.** `bench/results/census/criterion11c-ledger-arms-dev{0,1}.json` reads four arms in two pairs differing in exactly one proof-key component each, with opposite outcomes (`ALL-PROVEN`/`HIT` vs `ALL-DECLINED`/`KEY-ABSENT`); `criterion11c_mutations-dev{0,1}.json` records three mutations all `CAUGHT`, including the identical-file control arm that makes the other two **detections rather than a check that fails on everything**. Whole-model reading: `phi35_claim_reading_summary.json` — 355 claimed, 355 hits, 3 `unproven_declines`, **0 `unproven_forms_claimed`**. **Why the row nonetheless stays open, and the ground is my own ruling of today, not a new condition:** the criterion's second clause is *no build silently claiming unproven forms*, and §8.9.17 establishes on measured evidence that no predicate reads an entry's device. The specimen is in this repository: `wiring_census-dev1.json` reads `ALL-PROVEN ... ledger_hits=6` on the Iris Xe, against a ledger whose every one of 97 entries records `device0` — which `wiring_census-dev0.json` shows is the RTX 4060. **Six forms read as proven on a device on which nothing has ever been proven.** That is the clause failing, measured, not a condition added because the row was about to pass. It closes when the device predicate lands (§8.9.17 (3)) — and on nothing else |
| 12 | **NOT MET** *(unchanged — three of four conjuncts open)* | a mechanism this table relies on with no observation, or an extent claim that does not match the independent whole | **Yes** | The census reports its own row open: `wiring_census-dev0.json::criterion_12.closes_row = false`. **Re-measured 2026-08-05 (issue #33):** the independent whole grew, in production Rust, to `ci/check_census_completeness.py`'s **63 surfaces** — 13 of them (7 `counters.rs` fields plus 6 env switches across `session.rs`, `ep.rs`, `host_device_memory.rs`, `logging.rs`, `factory.rs`) had **no entry at all** in `ci/census_surface_map.json` and failed the screen with `FAIL(condition=unmapped_surface)`, not merely uncensused (the issue's own prose undercounted this at 12; the screen's own enumeration was 13, and that is the number that is now checked in, not asserted). Each of the 13 now carries a named owner and a reasoned entry — one (`ONNXRUNTIME_EP_VULKAN_DEBUG_CONSTANTS`) turned out to be `not_a_mechanism`: a name that exists only inside a doc comment and is never read by any `std::env::var` call in the tree. The screen now reads **PASS** on unmapped-surface (33 censused, 24 uncensused, 3 out of frame, 3 not mechanisms) — this closes zero of criterion 12's four conjuncts (Trinity owns row 12's tally; supplying the map entry and closing the row must not be the same act, Morpheus's criterion-11 ruling applies here unchanged) and the row stays **NOT MET** on the other three. **New, and it weakens the pair rather than a conjunct:** `wiring_census-dev0.json` reads `ledger_entries=97` and `wiring_census-dev1.json` reads `ledger_entries=95`. **The two-device census is not a pair — it is two censuses of two different binaries**, and the later one was never re-run. R12's fourth generalisation, arriving in the criterion built to catch exactly this |

**Re-derived count: five met (3, 4, 5, 6, 7), seven not met (1, 2, 8, 9, 10, 11, 12).** The previous
tally read seven met. **Two rows moved down and one moved up, and the two that moved down had both been
carrying a promise where an observation was required** — "Linux via CI" on row 1, and a parity result
read through a since-corrected ABI mirror on row 8. That is the same defect twice, and it is the defect
this milestone table was written to prevent.

| # | Criterion | Status | What remains |
|---|---|---|---|
| 1 | build + clippy clean, Windows & Linux | **Met** on Windows; Linux via CI | Nothing; hold it |
| 2 | `pytest tests/ops` green with the claim assertion proving `Add` ran on `VulkanExecutionProvider` | **REOPENED 2026-07-31T07:45:10-07:00 — partially met** | Reopened on two independent grounds, and I want both on the record because either alone would suffice. **(i) It was closed on criterion 10, and criterion 10's closure is withdrawn** — I wrote that this row "closes when criterion 10 closes", so when that closure turns out to have rested on an `UNATTRIBUTED` run, the ground under this row goes with it. **(ii) It fails on its own terms today**: the suite is not green. With Guard D actually executing, the lane reports four real defects with named owners. Ground (ii) is the one that matters, because it needs no argument about promises — *the criterion says green and the suite is red.* This row re-closes automatically when 10 does and the suite is green; it does **not** acquire new conditions on the way, which is the fault I reopened it for in the first place |
| 3 | Validation layers clean in the debug lane | **Partially met — and one of its two owed items is now discharged** | (b) is **done**: the planted control fires **in the pytest lane** rather than behind `#[ignore]`, which is what "a control that must be opted into is not in the lane" asked for. (a) is still owed and is now owed more precisely: **a re-run after the binding-arity fix showing zero errors with the messenger armed, on both devices, from a run whose verdict is an *attributed* `MATCH`** — the prior clean reading is void and so is any reading taken from a fallback run, because a graph that ran on CPU generates no Vulkan calls for a validation layer to object to. **A silent validation lane and a lane with nothing to validate are the same reading**, which is R13's shape arriving in criterion 3. Then lavapipe |
| 4 | No-ICD machine advertises zero devices, session runs on CPU | **Met 2026-08-01 — witnessed (Trinity)** | Witness: `bench/results/criterion4_icd_witness-dev{0,1}.json`, produced by `tests/ops/test_criterion_4_5_witness.py::test_criterion4_icd_polarity_witness`. **Both polarities, same binary (`library_sha_prefix` identical in both rows), one lane.** With an ICD: `ep_devices_advertised=1`, `vulkan_node_events=1`, no zero-device warning. With `VK_DRIVER_FILES`/`VK_ICD_FILENAMES` pointed at a nonexistent ICD: `ep_devices_advertised=0`, `vulkan_node_events=0`, the zero-device warning present, and the session still returned the bit-exact expected output (`output_exact_match=true`) on CPU. **The negative control is shown to have fired rather than asserted to**: Link's own `ci/check_icd_suppression.py::classify` is imported (not re-implemented) and returns two different tokens for the two rows — `icd_suppression_ineffective` vs `suppressed`, the latter carrying `ERROR_INCOMPATIBLE_DRIVER` and `epctl_exit_code=3`. **Link's trap confirmed and avoided**: `gate_line_present` is `true` in *both* rows, because a suppressed run's error block quotes the very phrase — a substring test on it discriminates nothing, so it is recorded as a field and never used as the gate. R13: a child that produced no record raises `InstrumentError`, so an outage cannot be spelled like "advertised zero devices". Remaining gap: the second OS |
| 5 | Shader-less build advertises zero devices and claims nothing (§7.8 condition 3) | **Met 2026-08-01 — witnessed (Trinity)** | Witness: `bench/results/criterion5_shaderless_witness-dev{0,1}.json`, produced by `tests/ops/test_criterion_4_5_witness.py::test_criterion5_shaderless_polarity_witness`; the shader-less artifact is built in-lane by `tests/ops/_shaderless.py`. Shaders compiled: `ep_devices_advertised=1`, `vulkan_node_events=1`, `shaderless_reason_emitted=false`. Shader-less: `ep_devices_advertised=0`, `vulkan_node_events=0`, `shaderless_reason_emitted=true` — the reason string is emitted **on the failing path**, which is what R13 asked for — and the session still returned the bit-exact expected output on CPU. The negative control is shown to have fired: the reason string is present in one polarity and absent in the other, from two different `library` paths. **Stated limit, owed to Tank**: `ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1` is *unreachable* on any host with the SDK installed, because `build.rs::find_glslc()` falls through to `installed_sdk_glslc()`, an unconditional scan of `C:\VulkanSDK` that honours no env var, before the flag is ever consulted. The witness therefore builds from an emptied shader source set, which executes the identical `write_shader_modules(out_dir, &[])`; the evidence relied on is the artifact's own emitted reason string, not that equivalence, since §7.8's guard reads `shaders::has_any()` and its subject is emptiness, not the route to it. Fix owed: make the env var a hard override checked *before* the search |
| 6 | `CLAIM_DEBUG=1` prints per-op decline reasons | **Met** | Nothing. Reasons are first-match (R8) — the criterion asks for reasons, not complete reasons, and that is deliberate |
| 7 | Layering lint in CI, fails a planted violation incl. a planted `cmd_pipeline_barrier` | **Met** | Nothing. Still the only criterion written with a falsifier from day one; criteria 11 and 12 are both built in its image |
| 8 | Full suite twice per lane, default and `force_legacy_barriers=1`, identical results, **non-zero executed-dispatch count in each lane** | **Met — as a *parity* criterion, which is all it ever was** | Unchanged, and today reinforces the relabelling rather than softening it: **both barrier backends agreed bit-exactly while the model produced all-zero logits**, which is exactly the failure mode the 2026-07-30 relabelling predicted in the abstract and has now been observed in the concrete. Link's lavapipe run adds a third independent barrier implementation (58/0 parity, `subgroup_size = 8`); that lane is **operational, not green**, so it is not quoted here as satisfying the criterion. Open: the twice-per-lane run on the lavapipe lanes, carrying criterion 10's gate |
| 9 | Sibling docs consistent; §12 lists every divergence | **Partially met — the named blocker is discharged, the criterion is not** | `PLATFORMS.md` LVP2 is now **retracted** with the corrected reading from the fixed probe, which was the specific thing blocking this row (§10.0.1 R6 rule 3, executed). But `DESIGN.md`, `ENGINE.md`, `OP_COVERAGE.md`, `PERF.md` and `PLATFORMS.md` **all changed today**, and a consistency criterion assessed against yesterday's documents is not assessed. **Two items added 2026-07-30T20:58:11-07:00, both of which make every document written today wrong in a specific, mechanical way:** (a) every occurrence of the "68.3% command-buffer recording" figure is a misnomer defect (§10.0.1 R11) and must be restated as host upload — I have corrected this document's own occurrences below and the same sweep is owed in `PERF.md` and anywhere Niobe's phase table was quoted; (b) **every device label written on 2026-07-30 is inverted** — `DEVICE=0` is the RTX 4060, `DEVICE=1` is the Iris Xe (R6 amendment 4). Owner: Link for `PLATFORMS.md`, Niobe for `PERF.md`, me for §12 and the cross-references |
| 10 | **Real model at producer-at-version, non-zero claimed count, attributed `model_output_equivalence = MATCH` against a CPU-only run of the same session, over N >= 3 consecutive inferences** | **REOPENED 2026-08-02T04:30:29-07:00 (was MET 2026-08-02T02:02:23-07:00 — my closure, and it was wrong on a reading I could have made and did not)** | **The evidence that closed this row is void, and the evidence that replaces it is genuine and incomplete.** Void: the 2026-07-30 `MATCH` on both devices came from runs in which ORT had already fallen back to CPU inside `run()` — this EP executed zero nodes, so the verdict was `UNATTRIBUTED` under §10.0's third amendment, not `MATCH`. Genuine: after Switch fixed `Allocator::alloc(size=0)` returning `None` for Phi-3.5's `[1,32,0,96]` KV-cache inputs, **ORT's own profiling — an instrument we do not own — reports 1 `VulkanExecutionProvider` node event (one fused island of 354 nodes) and 10 `CPUExecutionProvider` events, exactly Mouse's ten declined edge ops, with `logits argmax 30751` matching CPU.** That is the first attributed execution this project has ever recorded and it is worth more than the reading it replaces. Incomplete: **the multi-run picture is red** — the weight cache exhausts device memory across runs (`gpu-allocator failed to allocate 14155776 bytes: Out of memory`, followed by silent CPU fallback) and 50 KV-cache outputs are never written, giving cross-run divergence on a dirty arena. **I am reopening this on the voidness of the old evidence and the incompleteness of the new, not on the badness of the news** — the test of that claim is cheap and I am stating it in advance: **when one session produces three consecutive attributed `MATCH` runs on this artifact, this row closes the same day, with no new conditions.** Owners: Switch (arena lifetime and the 50 unwritten outputs), Trinity (the attributed verdict constructor and the N >= 3 series)<br>**CLOSED 2026-08-02T02:02:23-07:00, on the condition I wrote in advance and with no new conditions, which is the whole reason the condition was written in advance.** Verified by me, from the artifacts rather than from the report: `bench/results/criterion10-dev{0,1}.json`, both devices, `verdict = MATCH`, `comparison = AGREE`, `series.runs = 3`, `per_run_comparison = ['AGREE','AGREE','AGREE']`, `uniformly_attributed = true`, `executed_by = {VulkanExecutionProvider: 3, CPUExecutionProvider: 24}` from **ORT's own profiling**, `attribution_witnesses_present = ['ort_profile','ep_counters']`, `attribution_witness_agreement = AGREE`, `counters_dispatches_executed` 1066 (dev0) / 1186 (dev1), and per run `argmax_vk == argmax_cpu == 30751`, `top10_overlap = 10`, `max_abs_diff = 0.0625`, **`cross_run_identical_to_run1 = true` on all three** — which is the cross-run divergence that reopened this row, gone, on its own terms. **It closes stronger than the bar it had to clear, and the extra strength is not mine:** when I set the condition the counters witness could be `null` while the verdict still read `MATCH`, and Trinity established this session that `split_frame == False` was spelling both *two witnesses agreed* and *there was only one witness*; separating them into `AGREE`/`DISAGREE`/`UNOBSERVABLE` means **a one-witness `MATCH` would now confess to being one**. That is a vocabulary in which this row's closure could not have been stated when the row was reopened. **On the independence objection, which the coordinator raised against his own evidence and which I am overruling on the record:** the shape I have refused all session is *the party who supplies the artifact also moving the tally*. That is not this. He supplied it and **explicitly declined to close the row**; the verdict logic is Trinity's, the attribution is from an instrument we do not own, and the tally is mine. **The separation is already where it needs to be, and demanding a re-run by a third party would be adding a condition after seeing that the news is good** — which is the mirror of the rescue argument I rejected on the 40.201 ms figure, and no better for pointing the other way. **A criterion may not be hardened because it is about to pass.** **What closing does NOT mean, recorded so this row is not later read as covering things it never measured:** Defect 2's KV write path is still unwitnessed and the arena/weight-cache exhaustion is a live item; **criterion 10 was never the instrument for either**, and folding them in now would be exactly the new condition I promised not to add. They keep their own owners and their own falsifiers. **Both named owners delivered and the row was reopened on their work being incomplete, so it is recorded here: Switch (arena lifetime and the `Allocator::alloc(size=0)` fix), Trinity (the attributed verdict constructor, the witness-agreement vocabulary, and the N ≥ 3 series).** **Standing falsifier, not a condition:** the next independently produced artifact — anyone who did not run this one — either confirms it or this row reopens the same day, on the same terms and with no argument from me.<br>**AND IT REOPENED IN THREE HOURS, ON A SOURCE READING I VERIFIED MYSELF AFTER FACT CHECKER FOUND IT IN DEVIL'S ADVOCATE MODE** *(2026-08-02T04:30:29-07:00)*. **`model_output_equivalence` compares one output out of sixty-five.** `_compare_run_to_cpu` takes `vk_out[0]` and `cpu_out[0]` — the logits — and derives `argmax`, `top10_overlap` and `max_abs_diff` from those and nothing else; `test_phi35.py`'s oracle is the same shape, asserting `len(vk_out) == len(cpu_out)` (a **structural** check) and then comparing `[0]`. **No KV output is compared against CPU anywhere in the tree.** The all-65 gate is `outputs_bit_equal(vk_runs[0], run)` — **cross-run identity, which proves determinism and cannot prove correctness, because a deterministically wrong KV write passes it by being consistently wrong.** The two gates compose to the weaker extent and the stronger name, ruled at §10.0.1 R9 as a fourth specimen of the red-instrument test and explicitly **not** given a number. **This is not incomplete coverage; it is the absence of a falsifier for the exact defect this row was reopened for.** The reopening ground on 2026-07-31 was *"50 KV-cache outputs are never written, giving cross-run divergence on a dirty arena"* — and divergence is the symptom of a **dirty** arena. On a clean or zero-initialised one the same unwritten output is *stable*, and every gate we own goes green. **The codebase already knows this mechanism and already built the guard, for one output:** `test_phi35.py` Guard 1 says in terms that an output outside the descriptor set *"is never written… zero-initialised by both Intel Iris Xe and NVIDIA drivers for security, reads back as all-zero"*, and that guard is applied to output 0 — **the one tensor that already has an oracle.** So the closure certified that the symptom is gone; it never established that the defect is fixed. **I reject the escape offered to me — that the criterion's words were always about logits and KV belongs in another row.** The measurement is named `model_output_equivalence`; R11's first sentence is *a measurement's name is not its definition*; the model has 65 outputs. If it always meant logits the honest name is `logits_equivalence`, and adopting that name **now, having seen that the broad reading fails**, is narrowing a criterion because it has just failed — the exact mirror of the move I refused three hours ago in this same cell when I declined to add a re-run requirement because the news was good. **The rule has to run both ways or it is not a rule: a criterion may not be hardened because it is about to pass, nor narrowed because it has just failed.** **What survives, stated so the reopening does not punish delivered work:** the session structure, the attribution, the three-run series, the logits agreement and the 65-output determinism are all still true and still measured; what was never measured is the conjunction the criterion names. Switch's and Trinity's delivery above stands unchanged — **the missing arm was never assigned to anyone**, which is the actual defect. **DISCHARGE NOW REQUIRES, and these are stated in full so that no further condition can be added later:** **(a)** a CPU-oracle comparison over **all 65 outputs**, each output's tolerance **stated with its justification** rather than assumed, and the record carrying **two extents — `oracle_outputs_compared` and `cross_run_outputs_compared` as separate keys — with the present ambiguous `outputs_compared` renamed or removed**, because it is the key that misled me and *names outlive doc comments*; **(b)** a **planted control that is wrong *and stable***, specifically the all-zero case, since an unstable plant is caught by cross-run identity and would prove only what we already know; **(c)** a **non-triviality guard on both sides** — 64 pairs of all-zero tensors would satisfy (a) perfectly, and *an oracle comparison that passes on the absence of data* is Switch's `0.0 == 0.0` assertion wearing a fourth costume; **(d)** the existing attribution, session and cross-run evidence **re-emitted unchanged**, not re-argued. **If arm (b) cannot be constructed, the gate cannot detect what this row was reopened for and the row must not rest on it** — that is the coordinator's sentence and it is correct. **Owner: Trinity for the comparison and verdict constructors; Switch for whether the KV write path writes at all, which is still unwitnessed.** **OPEN, AND NOT A REOPENING GROUND:** Fact Checker also argues attribution is session-aggregate and that ORT profile events may represent *failed* attempts, so `VulkanExecutionProvider: 3` counts events rather than proven successful executions. He ran no tests; that half is analysis, and by R13's second clause a result pointing the way I am already going deserves more scrutiny, not less. It does not change this disposition and is **not** being folded in as a condition. It is owed an artifact: **plant a failing island execution and observe whether a Node event is still emitted.** He confirmed the two things the closure leaned on — zero attribution cannot produce `MATCH`, and the three runs are one session. **On method, and this is the part I most want kept:** I verified every field of the artifact and still closed wrongly; the coordinator, who supplied the evidence, put it to an adversary *because* he had supplied it. **Content verification by the party ruling is weaker than adversarial review by a party with no stake in the outcome, and this row is the demonstration.**<br>**THE REOPENING GROUND IS ABSENT AND THE ACCUMULATION QUESTION HAS A FALSE PREMISE** *(2026-08-02T15:15:12-07:00)*. **First, the arms are built and the defect this row was reopened for is not reproducing:** `oracle_outputs_compared = 65`, `cross_run_outputs_compared` recorded separately, `oracle_outputs_degenerate = 0`, `oracle_outputs_vacuous = 0`, `outputs_cpu_only = 0`, bit-identical across three runs, with `vk_degenerate`/`cpu_degenerate` per output and a planted control in `bench/results/planted_kv_probe.json` in which 64 of 65 outputs were zero and the gate refused. **Arms (b), (c) and (d) are discharged and arm (a) is built.** **Second, and this is the ruling I was asked for: `should f16 kernels accumulate in f32?` — they already do, everywhere, and have.** Verified in source by symbol rather than by line: `q_gemv.comp`'s accuracy note states *"Accumulation is fp32 regardless of storage, which is also what ORT's `SQNBIT_CompFp32` path does"*; `simplified_layer_norm_f16.comp` — *"Sum-of-squares, tree reduction, rsqrt and the gamma multiply are all fp32"*; `skip_simplified_layer_norm_f16.comp` — *"All arithmetic … is fp32 … the same choice `q_gemv.comp` makes for accumulation"*; and `gqa_f16.comp` declares `float q[128], k_new[128], v_new[128], acc[128]`, runs its dot products and its Milakov–Gimelshein online softmax in `float`, and converts on load through `f16_qkv`/`f16_pk`/`f16_pv`. **fp16 is a storage format in this EP and has never been an accumulation format.** So there is no cost decision, no occupancy trade, and — the part that matters most — **no invalidation of the 74 re-proved ledger entries. Had I ruled on the economics as the question was framed, I would have authorised a real cost to obtain a property the tree already has.** **Third, the residual is not accumulation error.** Of the 65 per-output residuals, **64 are exact negative powers of two and the 65th is `3 × 2⁻⁹`** — small integer multiples of the **fp16 ULP** at each tensor's magnitude. KV activation magnitude grows with depth; the ULP grows with it; **the absolute residual rises with depth for a correct implementation.** The curve is magnitude, not error. **Fourth, therefore the tolerance argument is the wrong argument for a second reason, and not the one offered.** It is not that the pass/fail line falls mid-curve — that is true and well found — it is that **`atol` is an absolute bound applied to tensors of growing scale**, which is §10.0.4's *prefer the ratio* appearing as a defect. **The unit is wrong, not the number.** One wrong denominator was already corrected inside this instrument by its author; this is the second, and it sits in the criterion. **Note what follows: changing the unit may make this gate TIGHTER, and that is why this is not a relaxation.** **The replacement, with its prediction stated before it is built so that it can fail:** record the residual **in ULPs**, `max_abs_diff / ulp(magnitude at the differing element)`, per output. **Prediction: flat, order 1–3, across all 32 layers.** Flat ⇒ no accumulation defect and no curve to argue about. A **step** at layer L ⇒ a real defect, localised. Ruled at §10.0.1 R9 as the **dual** of the third generalisation — *an observable that is true whatever happens cannot convict; an observable that degrades whatever happens cannot acquit* — and **not given a number**, fifth decline, because the remedy is R9's remedy unchanged. **Fifth, an instrument caution that changes what may be quoted:** the *absolute* series is broadly monotone, but `max_rel_diff` is not — layer 2's key reads `0.4559`, above every layer from 3 to 30 and level with layer 31's `0.4917`, on an unremarkable absolute residual of `2⁻⁸`, because `max_rel_diff` is attained at near-zero elements and its denominator is unstable. **The depth series must be quoted absolute or in ULPs, never as `max_rel_diff`.** **Sixth, on `argmax = 30751` and `top10_overlap = 10/10`: I decline to take comfort from it, and the reason is not scepticism but arithmetic — it is one token.** A rank invariant over a single decode step at ctx 0 is a coin that came up heads. §10.0.4 is right that the rank invariant is the one that survives; the sample size is not there. **The task-level gate is top-1 agreement over a stated N, and N = 1 is not a stated N.** **Seventh, GQA's 1.37× margin is untouched by any of this** — GQA already accumulates in fp32, so a thin margin there is not an accumulation-width question and nothing here explains it. It stays open and it stays thin; **Switch flagged it before he was asked, which is the behaviour, and it should not be closed by the fact that its proposed remedy turned out to be already in place.** **THE ROW REMAINS OPEN, for a smaller and different reason than it appears:** not a KV defect — that is measured absent — but a gate whose unit cannot distinguish a correct implementation from a defective one at depth. **It closes when the ULP series exists and is either flat or has a located step.** **`verdict = DIVERGENT` is honest and stays until the unit is fixed; it must not be flipped by moving `atol`.**<br>**THE ORACLE ANSWERED, THE DEPTH SERIES ANSWERED, AND THE ROW STAYS OPEN — §8.9.25** *(2026-08-04T19:40:00-07:00)*. **Both of the things this row was waiting on exist, and neither closes it.** *(i) The oracle.* Trinity's model-scale float64 chain (`bench/results/criterion10_chain-dev{0,1}.json`; 355 nodes, 32 layers each proven live, both devices, both reference variants, reading **initialisers and `input_ids` only** with `assert_chain_never_reseeded` digesting every boundary, so neither EP enters its own derivation) answers §8.9.24(4): output 0 → **`cpu`**, ORT's own CPU EP is the further side from true, unanimous on five discriminators in both variants on both devices, **83 vs 70 element-ULP**; outputs 63 and 64 → **`direction: null`**, discriminators conflicting inside a variant and the variants disagreeing across it. **The answer is not uniform and that is what disposes of it as a budget: a tolerance motion resting on "the reference is the further side" would admit three outputs on a direction measured for one, and would have to *default* on the other two — permissively, on exactly the outputs that would go green.** Discharging a block permits motions to be made; it does not grant one. **`atol` and `rtol` do not move — fifth round.** *(ii) The depth series, which returned the branch that convicts.* `kv_depth_curve` is flat at 0–2 median ULP for layers 0–30, `kv_depth_largest_step` is **1.0**, and the sole `kv_depth_exceedances` entry is **layer 31, key and value, 4 ULP** — and layer 31's key and value **are outputs 63 and 64**. My own words were *flat ⇒ no accumulation defect; a step at layer L ⇒ a real defect, localised*: the instrument I demanded, in the unit I demanded, has **located a real defect at the one layer whose two outputs are two of the three that fail.** *(iii) And the sentence that would have closed this row was mine.* *"It closes when the ULP series exists and is either flat or has a located step"* reads as **sufficient**; it was written as the discharge of a blocking objection about the gate's unit. Read as sufficient it closes this row today on an artifact whose own `verdict` is `DIVERGENT` and whose `per_run_comparison` is `['DISAGREE','DISAGREE','DISAGREE']` — closing a criterion whose text requires an attributed `MATCH` on a series that is not the criterion. **§8.9.25 rules it UNBLOCKING, not SUFFICIENT: this row closes on its own text and on nothing else, and every close condition from here declares which of the two it is, because an undeclared one is read as sufficient and therefore permissive.** *(iv) What an AGREE is worth, measured.* Both EPs sit far further from a weight-only reference than from each other — **70/83 vs 12 apart** on the logits, **12/12 vs 4** and **6/7 vs 4** at layer 31 — so **an agreement bounds only the difference and never the distance from truth**; `compare_all_outputs_to_cpu` is a consistency instrument and an `AGREE` means CONSISTENT-WITH-CPU, never CORRECT. **The asymmetry is what keeps this row red on the merits: a shared error cancels in a difference, so the finding weakens an AGREE and leaves a DISAGREE exactly where it stood.** *(v)* The "roughly 6×" figure is the **logits'**, not the model's — it is 3.0× at layer 31's key and 1.5–1.75× at its value, and the ratio is quoted per output or not at all. **THE ROW REMAINS OPEN with a fully characterised cause, and that is a better artifact than the same row turned green.** |
| 11 | **No form claimed without a ledger entry under its proof key; no build silently claiming unproven forms** (§8.9) | **Not met — scaffolding only. Reverted from MET 2026-08-02T00:15:00-07:00 (Morpheus, over Mouse's write-up; the tally is not the artifact-supplier's.)** | The ledger exists, is generated, and is consulted. Artifact: **`evidence/proof_ledger.jsonl`, 9 entries, header digest `331003e0ff88df3f` (was `e4436e93c19c8744`; regenerated 2026-08-02 when provenance became mandatory per entry)**, produced by `rust/tools/gen_proof_ledger.py` and baked in with `include_str!` so a build cannot claim a form whose proof is absent from the binary doing the claiming. The census line that replaced `UNWIRED`: **`ledger_lookup: ALL-PROVEN proven_key_lookups=6 ledger_hits=6 ledger_entries=9 unproven_declines=0 unproven_forms_enabled=[]`**, and `hits` is **typed** — `'UNOBSERVABLE'` / `'UNWIRED'` / `int` — so R12's three states cannot collapse into one `0` the way *bypassed* and *all-rejected* once did. **Rai's planted control runs in the lane, not behind `#[ignore]`**: `mul_f16_unproven` is deliberately never proven while its sibling `mul_f32` is, so the pair is two-armed and the arms are asserted to differ (`distinct_forms_have_distinct_keys`); the unproven arm yields `ledger_hits=0 ledger_gate=ALL-DECLINED unproven_declines=1`. **§8.7's expression-vs-path distinction is now mechanical** and the key is vindicated by a real defect that is *in the ledger as a pair*: `MatMulNBits` with vs without `zero_points` — `.../f16,u8,f16>f16/.../scales` and `.../f16,u8,f16,u8>f16/.../scales+zero_points` — different `populated_optional_input_set`, therefore different keys, therefore the 2026-07-30 all-zero-logits proof could never have been returned for the other form. **The price was paid and not softened: Phi-3.5's claimed count is 355 → 0**, exactly the number Morpheus accepted when he ruled this, and predicted in writing before the run (`bench/results/proof_ledger_prediction.json`, P4, confirmed exactly). **The fall is temporary**: the 355 nodes reduce to **8 distinct proof obligations**, mechanically discoverable because every claim-log audit line now carries `proof_key`. **Two real defects the controls caught while landing this**, both recorded because a mechanism that finds nothing on its first day is the one to distrust: (i) proof keys contain `,` and the `CLAIM_UNPROVEN` hatch split on `,`, shredding every key — the list was correctly discarded, the run claimed nothing, and the comparison still said `MATCH` because it was CPU-vs-CPU; only the **attribution** requirement caught it, separator now `;`; (ii) the regression test for (i) found `ai.onnx::Add/7+/f32` *passed* `ProofKey::validate` — a truncated key matches nothing and reads like a key that matches something. One `ERROR(instrument)` and it was never a detection (R13): `sqrt_f32` returned `DIVERGENT` with `worst_rel: 0.0` because `standard_normal` inputs made `Sqrt` NaN on **both** sides — fixed with an `ERROR` verdict for non-finite *reference* output plus an `INPUT_DOMAIN` table. Escape hatch is a list of keys and nothing else — no `1`, no `*`, no wildcard — with a session WARN and `unproven_forms_enabled` in the counters artifact. **WHY THE ROW IS OPEN DESPITE THE ARTIFACT EXISTING.** *“The cheapest satisfaction is a ledger generated from the claim table — derive the ledger from the same enumeration that produces the claims and the criterion is true by construction, `ledger_hits == proven_key_lookups` forever, and the check can never fail. That is an identity whose two sides come from the same source, and `6/6` looks identical under both readings.”* (Morpheus). The shipped ledger is **not** that shape — it is produced by `rust/tools/gen_proof_ledger.py` from executed differential runs — but nothing in the artifact *distinguished* the two shapes, and a reader could not tell them apart. **Four discharge conditions, with owners:** **(a) provenance — DONE 2026-08-02 (Mouse).** Every entry records the witnesses of a proof run that the claim table cannot produce: `claimed_nodes`, `dispatches_executed`, `worst_rel`. A dispatch count exists only after a session executed; an enumeration cannot forge one. Enforced on **both** sides and in the same direction — `gen_proof_ledger.py` raises rather than writing an unattributed entry, `--check` fails on one, and `parse_ledger` **faults** it, so it grants nothing. *Absent is treated exactly like zero*, and a **quoted** count is treated like absent, because a writer that stringified its counters did not read a counter. Control: `an_entry_without_attribution_proves_nothing_however_well_formed` — four ledgers differing only in the attribution fields, four different outcomes (R10); mutation-tested red at *“a run that dispatched nothing proves nothing, whatever it compared”*. Plus `every_shipped_ledger_entry_carries_its_proof_run` over all 9 shipped entries. **(b) three planted controls in the lane.** (i) `mul_f16_unproven` — **in the lane since 2026-08-01**, never `#[ignore]`. (ii) a key differing only in `opt_inputs`/`shape` — the `MatMulNBits` `zero_points` pair, **in the ledger** and asserted distinct by `distinct_forms_have_distinct_keys`, which doubles as the regression test for the 2026-07-30 all-zero-logits defect. (iii) **a build whose baked digest disagrees with the ledger on disk refuses to claim rather than warning — DONE 2026-08-02 (Mouse).** `ONNXRUNTIME_EP_VULKAN_LEDGER_FILE` names the on-disk ledger; a digest disagreement, **or a named file that cannot be read**, is pushed into `Ledger::faults`, and non-empty faults makes every lookup return `Faulted`, so **every form declines**. A WARN would leave the run claiming from evidence nobody can read. **This is a second, distinct threat from the header-vs-body digest**: that one catches a hand-edit *before* the build; this one catches the file changing *after* it, which is the case where the artifact a reviewer reads is not the artifact the binary claimed from. R9 amendment 5 — the check moves **against** the reader's confidence: a mismatch can only remove claims, never add one, which is why it cannot be repaired by tightening and why it is safe for it to be strictly optional. Control: `a_disk_ledger_that_disagrees_with_the_baked_one_refuses_to_claim`, three arms (identical → no fault; one line added → fault naming **both** digests; named-and-absent → fault), mutation-tested red. **(c) `ledger_hits` shown to move with its input** — Trinity's tally. Open. **(d) a three-token miss path (R13) — DONE 2026-08-02 (Mouse).** *Key absent from the ledger*, *ledger failed to parse or its digest disagreed*, and *key never attempted* are **three findings with three different repairs** — regenerate this form, fix the ledger file, and nothing at all — and a `bool` spells all three `false`. `LedgerLookup::{Hit,KeyAbsent,Faulted,NeverAttempted}` with a token apiece; `record_ledger_lookup` now takes the outcome rather than a `bool`; the counters artifact carries `"ledger_miss"`. Precedence is R13's order — **`LEDGER-FAULTED` outranks `KEY-ABSENT`**, because a run with no reading about any form must not spell an instrument outage the way it spells a detection. `NeverAttempted` is *derived* from `proven_key_lookups == 0` and is never counted, since recording it would be a lookup, which is exactly what it asserts did not happen. Control: `the_ledger_miss_token_names_which_of_three_things_happened`, four states driven, four tokens asserted distinct. Owner: Mouse ((a), (b)(iii), (d)); Trinity ((c) and lane membership)<br>**RESTORED 2026-08-02T01:42:02-07:00, having been lost in a merge where neither side was a superset and the coordinator correctly declined to splice my prose. It is the part of this ruling I would least want dropped: WHAT IS WITHHELD IS THE TALLY, NOT THE WORK.** The row is open because discharge needs an observable that moves, and for no other reason. **Mouse's evidence is not rejected, not doubted, and not diluted by being uncounted** — the ledger, the two digests, the two-armed `mul_f16_unproven` control, the `MatMulNBits` ± `zero_points` pair, the `355 → 0` price paid unsoftened and predicted in writing beforehand. **I do not want this register becoming a way of declining people's findings**, and a lead who can only ever withhold is running a different instrument from the one he thinks he is. **Three of the constructions above meet the standard I set, and I want that said in the row rather than in a decision file nobody re-reads:** (a) **provenance that cannot be forged by an enumeration** — a dispatch count only exists after a session executed, and the part I did not think to require is the better half, *absent is treated exactly like zero, and a **quoted** count exactly like absent, because a writer that stringified its counters did not read a counter*; (b)(iii) **the identical-file arm**, which is what makes the other two arms detections rather than a check that fails on everything — that is R9's falsifier-polarity discipline arriving unprompted in someone else's control; and (d) **`NeverAttempted` derived and never counted, since recording it would be a lookup, which is what it asserts did not happen** — which is the cleanest statement of R13's instrument/subject boundary anyone on this project has written, mine included. **Row 11 closes on (c), Trinity's, and on nothing else.** |
| 12 | **Wiring census: every mechanism this table relies on is observed to have run; a mechanism with no observation reports `UNWIRED`** (§10.0.1 R10) **— plus extent, the decomposition identity against an independent whole, and the name–content check** (§10.0.1 R11) | **Not met — added 2026-07-30T19:05:03-07:00, amended 20:58:11-07:00** | The census, the identity checks, the extent declarations and the lane assertion. **Amended within four hours of being written, by a specimen it would have certified** — `Phase::Record`, wired, invoked, correct, input-varying, and misnamed by 50×. **The tally does not move and that is the whole benefit of the row having been open**: a criterion strengthened while it is still unmet costs nothing and retracts nothing. Had I recorded it met this morning I would be reopening it tonight, on the seventh consecutive day of reopening a met criterion. **AMENDED AGAIN 2026-07-31T07:45:10-07:00, again by a specimen it would have certified, and this time the specimen is a mechanism whose own outage the census would have recorded as an observation.** Two additions: **(g)** every mechanism's census line carries the **frame** it observed in — for `model_output_equivalence` that is `executed_by` (§10.0 third amendment); a census that reports a verdict without its executor reports a value from a world it has not identified; and **(h)** the census distinguishes three states per mechanism, `OBSERVED` / `UNWIRED` / `INSTRUMENT-ERROR` (R13), because a census whose vocabulary is *observed or not observed* records a crashed mechanism as an absence and an absence as a crash **CENSUS RUN AND WITNESSED 2026-08-01 (Trinity)**: `bench/results/wiring_census-dev{0,1}.json`, from `tests/ops/test_wiring_census.py::test_wiring_census`, now reports twelve mechanisms and every line carries a value that mechanism computed on the run. Added this round: **`net_benefit_gate`** — `EVALUATED clusters_seen=1 evaluations=1 bypasses=0 sole_island_overrides=1 viable_islands_retained=0`, the three states that used to share one `0` now separate fields (R12, RAI-011); **`broken_commitment_warn`** (Tank) — read from two counters children differing only in `ONNXRUNTIME_EP_VULKAN_FORCE_COMPUTE_FAILURE`, planted `channel='ORT_SINK' broken_commitments=1 fault_injection='ACTIVE' ort_sink_warn_lines=1` against clean `channel='UNOBSERVABLE' broken_commitments=0`, and a reading that did **not** move between the two is reported `UNWIRED` however green it looked; **`device_state_guard`** (Link) — his `ci/check_device_state.py` imported and run over two inputs, a planted companionless duration returning `FAIL(condition=STEADY_UNCERTIFIED)` exit 1 against this run's own evidence; and **`instrument_census`** — Tank's `rust/tools/audit_instruments.py` via `main_guarded`, `CENSUS VERDICT: PASS`. **No second census was built**; his six states (`absent → uninvoked → unfalsified → unreachable → out-of-frame → misnamed`) and R13's three terminals are the vocabulary throughout. **A screen defect of the same family as Link's ICD probe was found and fixed in the census itself**: the `gpu_tracer` line read `ONNXRUNTIME_EP_VULKAN_TRACE_FILE`, a variable nothing defines (`trace.rs::ENV_TRACE` is `ONNXRUNTIME_EP_VULKAN_TRACE`), so it had reported `OPTIONAL-UNWIRED` on every run it ever made and would have done so had the tracer been deleted — an always-false screen and an always-true screen are equally blind. It now arms the tracer itself and reports `28 trace event(s), phases=['C','M','X','i'], distinct_names=16`. **No mechanism reports `UNWIRED` as of 2026-08-01T21:15:16-07:00.** The last one, `ledger_lookup`, was closed by criterion 11 (Mouse) and now reads `ALL-PROVEN proven_key_lookups=6 ledger_hits=6 ledger_entries=9 unproven_declines=0 unproven_forms_enabled=[]`; the `xfail(strict=True)` on `test_ledger_lookup_wired` was **replaced by assertions rather than deleted**, because an expectation that is merely dropped leaves no record that the thing it expected has happened<br>**AND THE CENSUS/CRITERION CONFUSION AROSE AGAIN WITHIN EIGHT HOURS, IN THE OTHER DIRECTION, REPORTED BY THE PERSON WHO MADE IT — WHICH IS THE ONLY REASON IT IS FIXABLE** *(2026-08-02T02:02:23-07:00)*. The coordinator told the team repeatedly that **criterion 12 was closed**, having run the census himself on both devices and obtained `unwired: []`. **The table says *Not met*, and the table is right.** He held a **witness** and read it as a **discharge** — the distinction written two rows above, walked past by its own author's reader. **This criterion is a conjunction and the census is one conjunct of four**, so they are now enumerated in the cell rather than left recoverable only from prose, because a criterion whose parts must be reconstructed invites closure on whichever part someone happens to be holding: **(i) the wiring census — every mechanism observed to have run, three terminal states, each line carrying the frame it observed in — DONE, `unwired: []` on both devices, on a rebuilt cdylib; (ii) extent declared for every quantity this table reports (R11 obligation 1) — OPEN; (iii) the decomposition identity checked against an independently-measured whole (R11 obligation 2) — OPEN, and the boundary-byte estimator's residual ~16,268× against the instrumented figure is a live instance of precisely this; (iv) the name–content agreement check (R11 obligation 4) — OPEN, with `Island::MEASURED_PHI35_DEV0` an outstanding named specimen. The row closes on all four and on no proper subset.** **The diagnosis is R11's first obligation, turned on the reader instead of the writer:** *declare the extent of what you are reporting.* One conjunct was verified and the conjunction was reported — **a decomposition presented as closed**, which is R11's own sentence about the hardest kind of wrong, arriving in a status report rather than in a measurement. **The aggravating detail is his and it is the part worth keeping:** *the thing I verified myself was the thing I over-weighted.* Personal verification raises confidence in a part and does exactly nothing for the whole; the general form is already in the register — **confidence scales with agreeing instruments, evidence scales only with falsifying ones** — and it applies to one's own hands no differently than to a telemetry set. **No rule is minted, for the second time tonight and on the same grounds:** the remedy is R11 obligation 1 plus R13's second clause, both already written and both already binding, and a register that grows an entry every time an existing rule is walked past has stopped cataloguing failure modes and started counting its own traffic |

**RULING ON CRITERION 3 — it does not discharge, and the reason is R7, not pedantry.** *"Ran with
validation enabled and no errors surfaced"* is the same observation a run with the layer **not
loaded** produces. That is precisely R7's fabricated negative and §7.9's *"a failed probe is
indistinguishable from a device with no capabilities"*, and it would be incoherent to write both of
those rules this week and then accept a silent instrument as a clean result. **Criterion 3 closes
when the validation lane carries a positive control**: either a deliberately-planted violation that
the lane catches and fails on (the mechanism criterion 7 already uses for the layering lint), or the
layer's own startup banner asserted in the log. Cheap, one-off, and it converts *"we saw nothing"*
into *"we would have seen something"*. Owners: Switch and Trinity.

**TALLY AT 2026-07-31T07:45:10-07:00 — four met, six partial, two not met, of twelve. Two rows moved
backwards and I am the reason one of them was ever forward.** *Superseding the 20:58 and 19:05
readings below, which are preserved because a table that only ever shows its current state is a
table nobody can audit.*

| | 19:05 · 20:58 | Now | Why |
|---|---|---|---|
| Met | 6 | **4** | 1, 6, 7, 8 |
| Partial | 4 | **6** | 2 and 10 reopened; 3, 4, 5, 9 unchanged in status, three of them advanced in substance |
| Not met | 2 | **2** | 11, 12 |

**The question I was actually asked was whether to reopen criterion 10 or record it met with scope,
and the answer turns on which of two things is true of the old evidence.** *Met with recorded scope*
is the right instrument when evidence is **sound and narrow** — that is what I did on 2026-07-30 for
one artifact, one producer, two devices, one OS. It is the wrong instrument when the evidence is
**void**, and this evidence is void rather than narrow: those runs did not measure a narrower version
of the thing, they measured **a different thing** — a CPU run compared against a CPU run. Scope
narrows a true statement; it cannot repair one whose subject was absent. So: **reopened.**

**And I am aware of the mirror.** I have twice refused to let good news close a criterion for the
wrong reason, and the symmetric failure is refusing to let bad news reopen one for the wrong reason —
reopening as penance, which is the same defect wearing the opposite sign and is precisely the
*"do not harden a criterion to punish a bad week"* prohibition. Three things keep this from being
that:

1. **The reopening is caused by the old evidence, not by the new defects.** Even if the OOM and the
   50 unwritten outputs did not exist, the 2026-07-30 `MATCH` would still be `UNATTRIBUTED` and this
   row would still reopen. The bad news is not doing the work.
2. **The row's price is stated in advance and it is low.** Three consecutive attributed `MATCH` runs
   in one session on this artifact closes it, on the day they arrive, **with no new conditions
   attached on arrival**. A reopening that specifies its own cheap closure is not a punishment; a
   reopening whose conditions are written after the next attempt would be.
3. **The multi-run requirement is not new.** The cross-run consistency gate was recorded on
   2026-07-30T19:05:03-07:00, before any of this. I am applying an existing criterion to a row that
   had not been assessed against it, which is the opposite of inventing one.

**What did not move, and the discipline is the same as on the good day.** Criteria 3, 4 and 5 all
advanced in substance — the planted messenger control now fires in the pytest lane, and Trinity's
paired positive controls landed in one lane — and **none of them changes status, because I have not
seen the artifacts.** *Review of a mechanism is not complete until the reviewer has seen an artifact
it produced* (R10) is a rule I applied to other people's work all week; it applies identically when
the mechanism is one I asked for and the news is good. **The single strongest piece of evidence this
project has ever produced also arrived today** — an execution attribution from ORT's own profiler,
354 nodes on the GPU in one island, argmax matching CPU — **and it moves no row either**, because it
is one run and the row asks for three. Good news and bad news are being held to the same standard on
the same day, which is the only day that test is worth anything.

**What the tally does not say.** It does not say the project went backwards. **The EP executed a real
model on the GPU today for the first time**, and per-inference upload fell from 1997.6 MiB to
0.756 MiB, measured on bytes. Both are real and neither is a criterion. **A milestone table is not a
progress report** — it is a list of claims we are prepared to defend, and today we learned that one
of them was undefendable. That is the table working.

**SUPERSEDED — the 20:58 and 19:05 readings, preserved for the audit trail:**

**Six met, four partial, two not met, of twelve — against four met, four partial, three not met, of
eleven, this morning.** The table moved forwards for the first time this week, and it grew by a row
on the same day. Both facts belong in the same sentence. What closed closed on a **measurement**
(criterion 10 went `DIVERGENT` → `MATCH` on two devices), which is the only way a criterion in this
document is allowed to close. What did not close did not close for reasons **today's good news does
not touch**: criteria 4 and 5 are negative-space checks with no positive control, and an EP that is
correct on Phi-3.5 passes them exactly as an always-broken EP does. **A correct model does not
retroactively give an unknown-polarity check a polarity.** I record that explicitly because the
temptation on a good day is to let the good news wash over the table, and the amendment of this
morning exists because the same thing happened in reverse. *[The `MATCH` in this paragraph is now
known to be `UNATTRIBUTED`. The paragraph's reasoning about criteria 4 and 5 survives intact and was
right for reasons that had nothing to do with the verdict.]*

**TALLY UNCHANGED AT 20:58:11-07:00, and the reason is worth more than the number.** The R11 event
(§10.0.1) invalidated a headline figure, corrected a 50× misattribution, inverted every device label
in the project's documents, and strengthened criterion 12 — **and moved no row.** That is not luck.
Criterion 12 was still open, so hardening it retracts nothing; criterion 9 was already partial, so it
absorbs two new owed items without a status change; and **no criterion in this table was ever
supported by the phase decomposition**, because the only performance obligation I placed on M0 is a
*disclosure*, not a gate, and disclosures do not certify. A milestone table that survives the
invalidation of the day's most-quoted number without moving is a table whose rows were resting on
the right things. **That is the first evidence I have that the criteria are load-bearing rather than
decorative, and it arrived from a defect rather than from a success**, which is the only way such
evidence ever arrives.

**What criterion 10 closes, and what it does not.** *Written 2026-07-30T19:05:03-07:00 and
**withdrawn 2026-07-31T07:45:10-07:00**: the run it describes was `UNATTRIBUTED`, so nothing in this
paragraph closed anything. **One sentence in it survives, and it survives in a stronger form than it
was written in.*** It closes criterion 2, on the promise I made
when I reopened it. It closes the question R9 was written about: **there is now one instrument in
this project that goes red when the model is wrong, it has been observed in both states, and the
transition between those states was caused by a real defect.** That is the strongest form of
evidence a criterion can have and no other criterion in this table has it. It does **not** close the
M0 sentence's tail, it does not travel to another artifact or another producer (§10.0 point 2), it
does not touch criteria 3, 4, 5, 9, 11 or 12, and it says nothing whatever about speed.

**The surviving sentence, restated correctly.** *There is now one instrument in this project that
goes red when the model is wrong, and it has been observed in both states* — that remains true, and
today it acquired a third state and a second witness. What was false in the original was the
implication that its red and green were **about this EP**; they were about whatever executed the
graph. **An instrument with a real red state that measures the wrong subject is not a weaker version
of a good instrument — it is a different instrument**, and that is the whole content of §10.0's
third amendment.

**A note on the defect, because it is the best evidence §8.9 will ever get.** The uncovered axis was
**binding arity**, not dtype — everyone, including me, reasoned toward the f16 kernel. `MatMulNBits`
with and without `zero_points` differ in `populated_optional_input_set`, which is a component of the
§8.9 proof key, so they are **different keys and a proof of one could never have been returned for
the other.** The gate was designed against a hazard of exactly that shape before the hazard was
diagnosed, and it was designed there because the key was made granular on principle rather than on
an incident. That is the argument for granularity that no amount of arguing produces, and it is also
the reason criterion 11 does not get to be deprioritised now that the model is correct.

**RULING — no performance criterion belongs in M0. The reasoning is the part that binds; the verdict
is the easy half.** *2026-07-30T19:05:03-07:00, on the coordinator's question, with the first honest
measurement in hand: **3.1× slower than the ORT CPU EP on Intel, 3.7× on NVIDIA**, with
`model_output_equivalence = MATCH`.*

Four reasons, in the order that would survive being argued with:

1. **The criteria are written in advance so that the day's most salient number cannot rewrite them.**
   That is the entire function of writing them down. A performance criterion added on the day
   performance was first measured is a criterion selected by salience, and the next salient number
   would have equal claim. If the answer would have been "no" yesterday when we could not measure,
   the availability of a measurement changes what we can *report*, not what M0 *is*.
2. **The asymmetry is the real argument, and it is R5's asymmetry.** *An EP that is correct and slow
   is a milestone; an EP that is fast and wrong is not* — and the reason is not taste. **Slowness is
   loud.** It is self-reporting, monotone, visible in the first benchmark anyone runs, and it cannot
   be believed to be absent. **Wrongness is silent by construction**: it produces numbers, it
   survives six agreeing instruments, and it cost this project a day. We add criteria against
   failures that hide. A 3.1× regression needs no criterion to be noticed — we noticed it without
   one, which is itself the proof.
3. **M0 is not a release, and the counter-argument is about releasing.** *"A Vulkan EP nobody would
   enable is not a shipped Vulkan EP"* is true and is about shipping. M0's sentence contains no
   product claim: it says the plugin loads, claims, executes correctly, cross-platform. The place
   where "you should enable this" is first asserted is **M2**, whose exit criterion already says
   *"a measured speedup vs the ORT CPU EP on at least one real discrete GPU, published with
   methodology"*. The criterion the counter-argument wants **already exists, one milestone away**,
   and moving it forward would not make it arrive sooner — it would make M0 unreachable while the
   allocator, which is the actual precondition for being fast, is still M2 work.
4. **Applying my own drafting rule kills every M0-shaped version of it.** *What is the cheapest thing
   that satisfies the words without satisfying the intent?* For "the EP is within N× of CPU on the
   `Add` graph": claim one node and fall back for the rest — the ratio goes to ~1.0 by claiming
   nothing, and the cheapest way to pass a ratio criterion is always **to do less work on the GPU**.
   At M0's op floor that is not even a cheat, it is the honest reading. A performance criterion at
   M0 would reward exactly the behaviour §7.0.2 has to be *measured* to authorise.

**What M0 gains instead: a disclosure obligation, not a gate, and it lives in §10.0 where the metric
of record lives — not in this table.** We can now measure, so we may no longer omit:

> **Every milestone report, benchmark table and coverage figure from M0 onward carries the measured
> end-to-end wall-clock ratio against a CPU-only run of the same session on the same artifact, per
> device, next to the §10.0 gated triple — including when it is worse than 1.0, and especially
> then.** Absent the number, the report is incomplete on its face, exactly as it is when the verdict
> is missing. A figure nobody may hide is a different instrument from a threshold nobody may fail,
> and it is the one that is honest at M0: it puts 3.1× in front of every reader without pretending
> M0 was ever about speed.

**The next milestone's performance criterion, and the instrument that makes it non-gameable.**

**CORRECTED 2026-07-30T20:58:11-07:00 on §10.0.1 R11, and the correction changes the criterion, not
only the number.** What follows below was written on a phase table in which
*"command-buffer recording is 68.3%"*. **That is false.** `Phase::Record` is an *inclusive* interval
containing the host staging copy, which reports separately into `phase_us[Upload]` and emits no
`ph:"X"` span, so the aggregation could not see it. **Upload is 95.8–98.4% of the `record` phase in
every cell on both devices; real command-buffer recording is 87–229 ms, about 1–3% of wall.** The
dominant cost is **the EP re-uploading the entire weight set on every inference** — 1997.6 MiB per
inference, ratio 1.0002 against `device_upload_bytes`, exactly linear across one, two and three
runs, in:out **2481:1** on a 1-token prefill, with a transfer ceiling of **~94.8% of wall on
discrete and ~44.0% on UMA**. The unchanged part is the headline: **85.9% of runtime with no GPU
work happening** was right, and only the name of the 68.3% was wrong.

Under R6 rule 1, which asks which support is load-bearing: M1's pre-existing criterion — *"a shape
change re-records once and then replays"* — was justified by `ENGINE.md` §6.1 **before** any trace
existed, so it survives its corroboration turning out to be wrong. But it is **demoted from headline
to secondary**, because it now addresses 1–3% of wall, and **a new criterion takes the lead: weight
residency.** Both are stated in the M1 section. I am recording the demotion rather than quietly
reordering the list, because a criterion that arrived on a false number and stays at the top of a
list is exactly R6 rule 3's unaudited right answer.

*Original text of this ruling, superseded in its numbers and preserved for the audit trail:*
M1 already contains the right mechanism and states it as a behaviour rather than as a number — *"a
shape change re-records once and then replays"*. Today's trace says why that is the one that matters:
of 19460+4648+4018+84 ms on device 0 across 661 subgraph invocations, **command-buffer recording is
68.3% and GPU kernels are 14.1%**; **85.9% of runtime has no GPU work happening**; and `vkQueueSubmit`
is **0.3%**, which **falsifies the fixed-per-submission hypothesis outright** — the cost is fixed per
`Compute` call, not per dispatch (a 1-dispatch island records in 34.43 ms, a 20-dispatch island in
24.25 ms). M1's criterion is therefore amended in the M1 section to add, with the falsifier named:

- **A counter, not a timer, first.** `command_buffer_records` over N ≥ 30 consecutive inferences in
  one session must be **≤ the number of distinct shape buckets observed, and strictly less than
  `compute_calls`**. A timer can be improved by a faster machine; a counter that equals
  `compute_calls` is red on any machine, and it is red for the right reason. This is R10 amendment 3
  in advance: **the identity case — one record per call — is the explicit failure state.**
- **Then the share, from the same trace that produced today's numbers**: steady-state recording
  share of wall clock, reported per phase, below 5%.
- **And the ratio, reported, not thresholded at M1**: end-to-end wall clock against a CPU-only run
  of the same session, per device.

*End of superseded text. The device label in it — "device 0" — is also wrong: `DEVICE=0` is the
RTX 4060, not the Iris Xe (R6 amendment 4). The three surviving numbers from that paragraph are
85.9% no-GPU-work, 0.3% submit, and the falsification of the fixed-per-submission hypothesis.*

**The corrected M1 lead criterion — weight residency, stated here and carried in the M1 section:**

- **`device_upload_bytes` per inference, in steady state, is less than 1% of the model's constant-
  initializer bytes.** A ratio, not a byte count, so it is device- and model-independent, and its
  falsifier is the measurement we already have: today it is **1.0002** — we upload the entire weight
  set every single inference. The identity case (upload once per inference == upload once per
  session) is the explicit red state, exactly as R10 amendment 3 requires.
- **It is reported with the first-inference upload beside it**, so that "we uploaded nothing" and
  "we uploaded once and kept it" are distinguishable — otherwise the cheapest way to pass is to
  claim nothing and upload nothing. **This criterion is admissible only from a run whose
  `claimed_op_coverage` is at or above the last published figure, with `MATCH`.** Without that
  interlock it is the single most gameable criterion in this document: zero claims, zero bytes,
  perfect score.
- **Recording amortisation stays as a criterion and moves below it**, on its own merits and its own
  justification (`ENGINE.md` §6.1), addressing 1–3% of wall rather than 68%.

**Why the ratio is non-gameable given §10.0's gated triple — this is the design, not a caveat.**
Three interlocks, and each closes a different cheat:

1. **The denominator is the whole model, end to end, in wall clock — never GPU-kernel time, never
   the claimed subgraphs alone.** Kernel time trends to zero as you claim less; whole-model wall
   clock does not. **An EP that claims nothing scores exactly 1.0 and can never score better**, so
   declining work can defend a ratio but can never improve one. That single choice removes the
   entire class of "get fast by getting narrow".
2. **The figure is admissible only from a run whose §10.0 verdict is `MATCH`, computed in the same
   process on the same session objects.** `DIVERGENT` or `UNMEASURED` voids it exactly as it voids
   the triple — because the cheapest speedup available at any moment is to compute the wrong answer
   faster, and today is the day we learned we can do that at full speed with zero reported failures.
3. **It is reported *with* the triple from the same run, and a run whose `claimed_op_coverage` or
   `largest_island_flops` fell against the last published run reports the change as
   `REGRESSED-COVERAGE`, not as a speedup.** §7.0.2 makes declining a supported op a legitimate
   move; that legitimacy is precisely what makes it available as a cheat, and this is the interlock
   that keeps a discretionary decline from being laundered into a performance win.

**M2 keeps the threshold**, sharpened so it cannot be met by selection: end-to-end ratio **< 1.0 on
at least one real discrete GPU with `MATCH`**, with the ratio **reported for every device in the
matrix**, including the ones above 1.0. Publishing only the device that won is the last cheap trick
available and it is closed by making the matrix, not the maximum, the deliverable.

M0's sentence is *"a stock ORT loads the plugin, enumerates a Vulkan device, runs a graph containing
a single `Add` node on that device, and the output matches the ORT CPU EP within tolerance — **on
both Windows and Linux, on a software rasteriser, in CI**"*. Everything before the dash is satisfied
for the sentence's literal subject, *one `Add` node*, **and as of 2026-07-30 also for a real model of
364 nodes at a named producer-at-version, on two devices** — which is criterion 10, the property
anyone actually cared about, and it took adding a criterion to find out we had never asked for it.
**Everything after the dash is untouched.** The Linux lane has now executed a claimed node under
WSL2 and is *operational*, not green; the lavapipe lanes do not yet carry the verdict field; CI has
no GPU hardware and never will under the current plan, which is why the software-rasteriser clause
exists.

**RULING ON SEQUENCING — SUPERSEDED IN PART, 2026-07-30T19:05:03-07:00.** Criterion 10 is now
`MATCH`, so the ordering below has *executed* rather than been revised: the tail was correctly held
behind a defect that has since been fixed, and it is now the front of the queue. The reasoning is
preserved because it is the reasoning I will re-apply the next time a correctness criterion goes red,
and reason 2 in particular still binds every lane Link brings up. **The one thing that changes is
that reason 1's premise — "we have a result that reproduces perfectly and is wrong" — is no longer
true, which is exactly the condition under which the tail resumes.** The new ordering is the
seven-item list below. Original ruling follows unedited.

**RULING ON SEQUENCING — model-level correctness now outranks the M0 tail in order, and does not
replace it as a gate.** *2026-07-30T05:48:29-07:00.* The tail — Windows **and** Linux, software
rasteriser, CI — stays in M0's sentence unchanged and unsoftened; cross-platform generality is a
standing user constraint checked continuously, not a clause I get to trade. But it moves **behind**
criterion 10 in sequencing, for three reasons:

1. **Every hour Link spends turning the Linux and lavapipe lanes green is an hour spent certifying a
   defect onto three more platforms.** The tail's entire purpose is to prove a result reproduces off
   this desk. We currently have a result that reproduces perfectly off this device — it reproduced
   on the RTX 4060 and on the Iris Xe, bit-for-bit — and it is wrong. **Reproducibility is only
   valuable applied to something true.**
2. **A green CI lane is itself an R9 composite.** Those lanes run the same suite that was green
   during the R9 event. Bringing them up *before* criterion 10 exists adds three more agreeing
   instruments to a set with no falsifier in it, which by R9 makes the wrong conclusion more
   persuasive, not less. When the lanes do run, **they must carry criterion 10's gate**, or they are
   measuring the same silence in a new location.
3. **Criterion 10 is where the current defect lives and where the three engineers already are.**
   Mouse on fp16, Switch on descriptors/readback at N=161, Trinity on the gate. Sequencing the tail
   ahead of them would be sequencing against the work in flight.

Link's parallel dispatch on the Linux lane is **not** blocked by this — parallel work on a lane that
has never executed a claimed node is exactly right, and the lane's *existence* is a prerequisite for
running criterion 10 anywhere but here. What is sequenced is the **declaration**: criterion 10 goes
`MATCH` before the tail is worth closing, and the tail closes before M0 is declared. Neither
substitutes for the other.

**What specifically remains, in order — REVISED 2026-07-30T19:05:03-07:00. Seven items, none
skippable. Items 1 and 2 of the previous list have collapsed into one another: the defect is fixed
and the gate is not built, so the gate no longer sits ahead of a fix, it sits ahead of a
declaration.**

1. **The claim gate of §8.9 lands** (Mouse, Trinity, Switch, Tank) — criterion 11. **Still first,
   and for a harder reason than this morning's.** This morning it was first because `main` shipped an
   EP computing zeros at full speed. That argument is gone — the EP is correct on Phi-3.5 — and the
   gate's priority is unchanged, because **the gate was never about today's defect.** Its price has
   gone up (claimed count 321 → 0 on an EP that is now known correct) and its justification has got
   stronger: the defect that occurred was a **binding-arity** defect, `populated_optional_input_set`
   is a component of the proof key, and the key would have refused the exact form that broke. A gate
   whose value is demonstrated by an incident is a gate that gets built after the incident and
   deferred before the next one. Not this time.
2. **Criterion 3 discharged properly** (Switch + Trinity): re-run with the messenger armed after the
   binding fix on both devices, **and the plant run in the lane rather than behind `#[ignore]`.**
   Promoted from third to second because the messenger has now demonstrated, on a live defect, that
   it is the highest-yield instrument we own — it printed the root cause of the day's worst bug in
   one line, the first time it was attached.
3. **Paired positive controls for criteria 4 and 5** (Trinity + Switch) — same binary, same lane,
   non-zero device count and non-zero claim count. Unchanged, still the cheapest items on the list,
   still undone.
4. **The wiring census** (Niobe + Trinity) — criterion 12, with the identity checks, the extent
   declarations and the name–content check (R10 + R11). New today, amended today.
5. **Doc consistency pass** (Link for `PLATFORMS.md`, Niobe for `PERF.md`, me for §12) —
   criterion 9. LVP2 is retracted; what remains is that five documents changed today, that every one
   of them quoting the phase table carries a 50× misnomer, and that **every device label written on
   2026-07-30 is inverted.** Bulk-correctable and therefore cheap, which by the record of criteria 4
   and 5 is the best predictor available that it will not be done.
6. **The CI lanes run**: `test_add_is_claimed`, the elementwise suite, the twice-per-lane barrier
   parity, **and criterion 10's gate on each lane's gate artifact**, green on lavapipe, on **Windows
   and Linux** (Link + Trinity). The whole tail of M0's sentence. Link's lavapipe lane is
   **operational** — a claimed node executes under WSL2, 196 tests pass, `subgroup_size = 8`, barrier
   parity 58/0 as a third independent implementation — and it is **not green**, because it does not
   yet carry the verdict field. That is the gap between item 6 and where he is, and it is a small
   one.
7. **M0 is declared, in one line, without qualification.**

**RULING — the dominant non-GPU cost does not enter M0, does not outrank the tail, and starts now
anyway. The three are not in tension and it is worth being precise about why.** *Asked directly;
answered in three parts. **Restated 2026-07-30T20:58:11-07:00: the ruling is unchanged and its
subject is not.** It was made about "the 68.3% recording cost"; the cost is 85.9% non-GPU and it is
**per-inference weight re-upload**, not recording (§10.0.1 R11). Every clause below survives the
substitution because none of them depended on which host activity it was — which is the correct
outcome for a sequencing ruling and would have been a warning sign in a technical one.*

- **It does not outrank the tail, by my own sequencing.** The tail is a *generality* obligation and a
  standing user constraint; transfer cost is neither correctness nor generality. Nothing about
  85.9%-idle changes on a second platform, and the lanes it would be traded against run on a software
  rasteriser where the figure is meaningless. The order above stands.
- **It is not a sequencing conflict at all, and treating it as one would be a scheduling error
  dressed as a principle.** Items 1–6 are Mouse, Trinity, Switch and Link. **The owner has changed
  with the diagnosis**: it was `recorded.rs`, Tank's; it is now **persistent weight residency in the
  allocator and transfer path — Switch's file**, handed over by Tank on the day he measured that his
  own device-backed allocation is a *mirror* rather than a move (staging stays authoritative and is
  read on every input, so the mirror is an *additional* upload, and
  `alloc_device_authoritative_spans` is still **0**). **Tank reported his own feature as not
  capturing the prize, prominently, and routed the fix to someone else's file. That is the behaviour
  this project needs most and rewards least, and I am recording it by name.** Either way the point
  stands: **sequencing governs declarations, not calendars**; it always has, and this is the third
  time I have had to say so.
- **One hard constraint on it, which is a correctness constraint on performance work and not the
  reverse — and the diagnosis makes it stronger, not weaker.** Both candidate fixes are the same
  hazard: **a thing computed once and reused across calls.** Command-buffer caching reuses a binding
  table; weight residency reuses an *upload*, which means the device copy must be proven to still be
  the copy the kernel reads after any path that could have invalidated it. That is exactly today's
  defect generalised — `push_dynamic_kernel` computed a 4-token binding table at Compile time, the
  translate path computed 5 with concrete shapes, they diverged, and both drivers silently dropped
  the write. **Therefore no residency or recording-amortisation change lands without criterion 10's
  gate run on it, on both devices, and without the counter that says the reuse happened** —
  `device_upload_bytes` per inference for residency, `command_buffer_records` for recording. The
  same run answers both questions, so this costs a flag, not a schedule.

And the honest sentence underneath all of it: **an EP that is correct and 3.1× slower than CPU is
not something anyone would enable, and I am not pretending otherwise.** It is a milestone, which is a
claim about a checkpoint in a plan, not a claim about a product. The distinction survives only as
long as we keep publishing the number — which is what the §10.0 disclosure obligation above is for,
and why I made it an obligation rather than a threshold.

**RULING ON THE STANDING PERFORMANCE DIRECTIVE AND THE TAIL — 2026-07-30T22:13:37-07:00. The tail
is unchanged and stays at the front. Saying so is the ruling; the reasoning is the part worth
reading.** *Asked directly, and the coordinator was right to ask rather than assume.*

- **The tail is not competing with performance work for a slot, so a directive about performance
  cannot move it.** The tail — *on both Windows and Linux, on a software rasteriser, in CI* — is a
  **generality** obligation and a standing user constraint, owned by Link, running on a software
  rasteriser where every timing number is meaningless by construction. Weight residency is Switch's,
  in the transfer path, on real hardware. **They contend for nothing**: not a person, not a file, not
  a machine. Re-ordering them would be theatre. The order stands as written this evening.
- **A standing directive is a reason to re-examine a placement and never on its own a reason to
  change it, and the difference matters more than this instance does.** If a directive moved every
  item that could be argued to serve it, the ordering would be a record of the most recent
  instruction rather than of the dependencies. I re-examined; the placement is load-bearing for a
  reason the directive does not touch. **Note also which way the evidence points: the tail is
  cheap insurance against a class of defect we have hit five times this week, and performance work
  is where cross-platform assumptions get quietly baked in. The directive is, if anything, a
  mild argument for keeping the tail early, not for moving it late.**
- **What the directive does change here is the reporting cadence, and that is real.** `一致` — the
  rate obligation — means the performance number moves, and is published with its frame and its
  verdict, on a cadence rather than at a milestone. **The instrument for a rate obligation is a
  series, not a value**: `device_upload_bytes` per inference, the wall-clock ratio, and the claimed
  triple, published each time they change, so that "we are pushing continuously" is falsifiable by
  a flat line. That is the criterion-shaped thing the directive deserves, and it belongs in the
  cadence, not in a gate.

**RULING — the ranked performance order survives, rank 1 keeps its place and changes its content,
and every wall-clock figure this project holds is withdrawn.** *2026-07-31T07:45:10-07:00, asked
directly whether the ordering still stands now that the timings behind it are known to have been
taken during CPU fallback.*

**The ordering, unchanged in sequence:**

| Rank | Item | Basis, and what it rests on now |
|---|---|---|
| 1 | **Weight residency** — `device_upload_bytes` per inference | **Landed and measured on bytes: 1997.6 MiB → 0.756 MiB per inference, sweep flat instead of linear.** Bytes are not a clock and were never taken from a timed run, so this result is untouched by the withdrawal below. **It is also now leaking**: the cache exhausts device memory across runs and the EP drops silently to CPU |
| 2 | **Net-benefit declines** — `retain_viable` in the partitioner | Wired by Mouse. Rests on the island-count decomposition (321 → 33), a **count**, not a clock. **RE-QUALIFIED 2026-08-01 (§5.4.1) — this row is two mis-attributions and it drops.** (a) The 321 → 33 collapse was produced by **wiring the clustering mechanism**, not by `retain_viable`; the R10 specimen table in §10.0.1 attributes it correctly and this row did not. (b) `retain_viable` has produced **zero declines on a production graph**: Phi-3.5's island is anchor-bearing, so stage 3b returns `Claim` before the economics arithmetic reads anything, and the only real-input rejection on record is the census lane's one-node chain at `TOO_SMALL` (stage 3a). The remaining opportunity here is real but it is **not this mechanism** — it is the byte estimator that stage 3c is fed (§7.12.1, 104,116× over-count), and that is a correctness item ahead of any optimisation. **Ranked position withdrawn; the row stays as the record of the withdrawal** |
| 3 | **Fence-wait GPU idle** — ~16% of wall, 53.6% of `fence_wait` is not kernel | Rests on a *share*, i.e. a ratio internal to one trace. Ratios within a trace survive the trace being from the wrong run only as **relative structure**, and that is exactly and only what they are being used for here |
| 4 | **Kernels** — ≤15% of wall, 98% of it one kernel | Same basis as 3, plus the concentration figure, which is a property of the graph rather than of the run |

**Why it survives.** The ordering was never derived from wall clock. It was derived from **the phase
decomposition and the byte-level residency result**, and the byte result is the load-bearing one:
2481:1 in:out, ratio 1.0002, exactly linear over one, two and three runs. **A count and a ratio do
not care what the absolute number was**, which is the same property that let the 3.1×/3.7× ratio
survive the R11 misattribution — one clock, one whole thing, no naming decisions — arriving now in
its dual form. **An ordering is a claim about relative magnitude and is falsified only by a relative
result**, and no relative result changed today.

**What is withdrawn, without hedging.** Every wall-clock figure this project has published —
**3.1× on Intel, 3.7× on NVIDIA**, and every derived per-phase millisecond — was measured on a run
in which this EP executed zero nodes. They are timings of the ORT CPU EP against the ORT CPU EP.
They are not "roughly right" or "an upper bound": they are `UNATTRIBUTED` and they are void. The
22:13 clause I added on my own authority — *no timing figure is quotable from a run whose verdict is
not `MATCH`* — is what catches this, and today it acquires its teeth: **`MATCH` now means attributed
`MATCH`, so the clause retroactively voids the very figures it was written beside.** A rule that
first bites its author is a rule that was aimed at the right thing.

**Therefore §10.0's disclosure obligation currently has no admissible value, and it reports that
rather than the old number.** The ratio is not omitted and it is not stale — it is
**`UNATTRIBUTED`**, published as that token, until a run with three consecutive attributed `MATCH`
verdicts produces a new one. This is the family doing its job for the fourth time: `UNMEASURED`,
`UNWIRED`, `UNOBSERVABLE`, `SPLIT-DEVICE`, and now a disclosure that names its own absence rather
than quietly keeping the last number that looked like one.

**Rank 1's content changes and its position does not, and the distinction matters.** The work was
*"make the weights resident"*; it is now **"make residency bounded"** — arena lifetime, eviction, and
a cross-run invariant on device memory. That is not a demotion of the achievement; the upload
mechanism is right and the measurement proving it is right is the best-evidenced performance result
this project has. It is a statement about what remains. **And it is worth naming what the OOM defect
actually is: a performance mechanism that fails into silent CPU fallback is a correctness defect
wearing a performance costume.** A cache miss should cost milliseconds; this one costs the entire
device. That places it above ranks 2–4 on correctness grounds even if it had no performance value at
all, which resolves the ordering question twice over.

**One interlock restated because the directive points everyone at rank 1.** No residency figure is
admissible except from a run at or above the last published coverage with an **attributed** `MATCH`,
with first-inference upload reported beside the steady-state figure. Today's 0.756 MiB satisfies the
byte side and does **not** yet satisfy the interlock, because the run that produced it is one run and
the multi-run behaviour is the OOM. **The number is real and it is not yet admissible**, and I would
rather write that sentence than round it in either direction.

**RULING — M1's weight-residency criterion stands exactly as written, and the tempting change is
the one to refuse.** *Asked whether it survives contact with a directive that will point everyone
at it. It does, and the interlocks are the reason, so they are restated rather than assumed.*

- **The threshold does not move.** `device_upload_bytes` per inference below **1%** of constant-
  initializer bytes. Today it is **1.0002**. There is no version of "we made progress" between 1.0
  and 0.01 that deserves a criterion: the mechanism either uploads weights once per session or it
  uploads them once per inference, and every intermediate number describes a bug rather than a
  design. **A threshold with no honest intermediate value is the rare case where a round number is
  the rigorous choice.** **UPDATE 2026-07-31T07:45:10-07:00: the ratio measured after Switch's
  persistent-residency change is **0.756 MiB against 1997.6 MiB — 0.0004, forty times inside the
  threshold** — and the criterion is **not** met, because the interlock below is not satisfied and
  because the mechanism producing that number exhausts device memory on later runs. **This is the
  first time a criterion's headline number has been comfortably passed while the criterion stays
  open, and it is the clearest evidence yet that the interlocks are the criterion.***
- **The two interlocks are what the directive puts under pressure, and neither may be relaxed.**
  (a) The figure is admissible only from a run whose `claimed_op_coverage` is **at or above the last
  published figure**, with an **attributed** `model_output_equivalence = MATCH` (§10.0 third
  amendment — a `MATCH` from a run this EP did not execute is `UNATTRIBUTED` and admits nothing).
  Without it the criterion is the most
  gameable sentence in this document — claim nothing, upload nothing, score perfectly, and the
  directive rewards you for it. (b) The first-inference upload is reported beside the steady-state
  figure, so *"we uploaded once and kept it"* and *"we uploaded nothing"* stay distinguishable.
- **One clarification the directive forces, and it is R12's:** the criterion is read against
  `device_upload_bytes` **on the session's device**, and explicitly **not** against
  `alloc_device_authoritative_spans`, which is `UNOBSERVABLE` while §6.5's device split exists.
  **A criterion whose instrument is structurally pinned at zero is not a criterion**, and the fix
  for that is §6.5, not a weaker criterion. This is the second time in two days that the honest move
  was to strengthen the instrument rather than the words.
  **AMENDED 2026-08-01T18:59:38-07:00 — the pin is lifted and the instrument choice is unchanged,
  for a new reason.** §6.5 is closed (§6.5.1), so in the armed lane the counter is a measurement
  rather than a pinned `UNOBSERVABLE` — and it measures **0 at a ceiling of 0** (§6.5.3), because
  every device-backed span is still host-staged. In the **default** lane no device-memory provider
  exists at all (`alloc_device_frame = OFF`), so the counter has no frame to be observed in. Either
  way it is not the instrument for this criterion: **it was unusable because it was pinned, and it
  is unusable because it is at its ceiling on one lane and absent on the other.** `device_upload_bytes`
  stands, and the day the counter's ceiling rises is the day this clarification gets re-read.
- **What I will not do is accelerate it into M0.** The directive is about rate, and M0 is about
  declaring. Moving residency into M0 would convert a rate obligation into a gate, which is the
  precise transformation §10's ruling exists to prevent — and it would put a speed criterion in the
  milestone whose subject is *"the EP loads, claims, executes correctly, cross-platform"*. **Work on
  it tonight; do not gate M0 on it. Those are consistent and I want the consistency on the record.**

**RULING ON LINK'S LANES — the gate is a precondition for a lane being declared *green*, and not a
precondition for a lane being brought *up*.** *2026-07-30T06:32:18-07:00, asked directly and answered
directly so it is not discovered by shipping three more green-and-silent lanes.* The distinction is
between two different things we have been calling the same word:

- **Operational** — the lane exists, executes, and reports. Link may declare this without the gate,
  it is a real deliverable, and it is a **prerequisite for running criterion 10 anywhere but this
  desk**. Bring-up is unblocked and should continue at full speed in `ep-vulkan-link`.
- **Green** — the lane's result is admissible as evidence, satisfies an M0 criterion, or is quoted
  in a status report. **This requires the gate.** A lane without it measures the same silence in a
  new location, and by R9 three agreeing silent lanes are worse than one, because the agreement
  raises confidence without raising evidence.

Made unrepresentable rather than merely required: **a lane's pass condition includes the verdict
field.** A run producing `UNMEASURED` reports `UNMEASURED` — it does not report PASS, and it does not
report a failure either. That is §7.9's third state arriving in the CI lane, and it means a lane
cannot be accidentally green; it can only be explicitly unmeasured.

**One feasibility ruling attached, because it would otherwise block Link on something impossible.**
Criterion 10's gate is *the mechanism, not the model*: a 2.2 GB fp16 Phi-3.5 on a software rasteriser
in a shared CI runner is not a reasonable per-lane requirement and I am not imposing it. Each lane
carries a **gate artifact**: the smallest real producer-at-version model that (a) claims a non-zero
node count on that lane, (b) contains at least one island of two or more nodes, and (c) exercises at
least one proof key in every dtype that lane claims. Owners: Trinity chooses and pins it, Link wires
it into the lanes. A lane whose gate artifact claims nothing reports that fact explicitly — "this
lane claimed zero nodes" is a finding, not an absence (§8.5, R7).

When those seven land, M0 is met and I will say so in one line without qualification — which is what
writing the criteria down in advance was for, and what rewriting them today and yesterday was for.

### M1 — "A useful elementwise EP" (`OP_COVERAGE.md` tier T1 — 87 ops)

**Gate: M1 does not begin until the kernel-template infrastructure is merged** (§8.4 A2) —
`indexing.glsl`, the `build.rs` dtype/capability variant generation, the `ops!` table macro, and the
shared claim helpers. Hand-writing ops #1–#20 before the templates exist forfeits the entire
leverage thesis and, with it, the schedule.

The T1 op set, claimed, tested, and documented; shape-keyed command-buffer recording cache
(`recorded.rs`); convex clustering with multi-node fused subgraphs; the generated `OP_SUPPORT.md`
coverage table as the authoritative contract.

| Work | Owner |
|---|---|
| **Kernel-template infrastructure (gate), then** op families + claim predicates + the `ops!` table | Mouse |
| Shader variants, spec constants, broadcast indexing, descriptor layout per family | Switch |
| Convex clustering, `recorded.rs`, arena-based temporaries | Tank |
| Per-family differential tests, tolerance policy, `tests/backend/` node tests | Trinity |
| `bench/` harness + first published baselines with island counts | Niobe |
| macOS/MoltenVK lane; first real-GPU lane if a runner is available; driver quirk log | Link |
| `tools/graph_census.py` + node histograms for all 7 corpus artifacts (`OP_COVERAGE.md` §2.2), **indexed by producer (§8.5), including GQA node arity and populated-optional-input reporting per producer (§10.0.1 R1)** | Mouse + Trinity |

**Exit criteria.** Every T1 op green vs CPU on ≥2 platforms; **a pure-elementwise graph of ≥20 nodes
compiles to one island, one submission**; a shape change re-records once and then replays;
**a node is claimed and correctly executed on a graph carrying a *symbolic* dimension, and the same
session produces correct results for two different concrete values of that dimension without
re-compiling the session** (§8.8 — added 2026-07-29T21:14:03-07:00);
**the Phi-3.5 decline histogram is re-measured and the `dynamic-shape` count has fallen**, reported
as a histogram and not as a coverage percentage (§10.0.1 R8);
`OP_SUPPORT.md` is generated from the registry and matches it by construction; **`graph_census.py`
exists, is indexed by producer, and has produced histograms for the full corpus**; **the census
reports GQA input counts and populated optional slots per producer for every artifact, resolving
§10.0.1 R1 for the ORT GenAI column**; and
**the ops-per-hand-written-kernel ratio is reported and is ≥ 8** (§8.4 A2).

**The symbolic-dimension criterion is deliberately worded to be unsatisfiable by a static
workaround.** *Two different concrete values in one session* cannot be met by resolving shapes ahead
of time, which is exactly the substitution §8.8 item 4 warns against. It is the second-token test,
written as an M1 exit criterion because a decoder that works only on the first token is not a
decoder.

**M1 EXIT CRITERION — ADDED 2026-07-30T19:05:03-07:00, CORRECTED AND REORDERED
2026-07-30T20:58:11-07:00 (§10.0.1 R11).** M0 gets no performance criterion (ruling in the M0
section above); M1 gets these, because M1 is where the transfer path and `recorded.rs` land.

*The version added at 19:05 led with recording amortisation, on a phase table reading* **"68.3%
command-buffer recording"**. *That figure was a misnomer: `Phase::Record` is an inclusive interval
containing the host staging copy, which reported separately into `phase_us[Upload]` and emitted no
`ph:"X"` span. Upload is **95.8–98.4% of the `record` phase**; real recording is **1–3% of wall**.
The surviving numbers are **85.9% of runtime with no GPU work happening**, **0.3% `vkQueueSubmit`**,
and the falsification of the fixed-per-submission hypothesis. The device labels in that version are
also inverted (R6 amendment 4). The full correction is in the M0 section above.*

**Lead criterion — weight residency.** In this order:

1. **`device_upload_bytes` per inference, in steady state, is below 1% of the model's constant-
   initializer bytes.** A ratio, so it is device- and model-independent. **Today it is 1.0002** — we
   re-upload the entire weight set on every inference (1997.6 MiB/inference, exactly linear over
   one, two and three runs, in:out 2481:1 on a 1-token prefill). The identity case is the explicit
   red state, per R10 amendment 3.
2. **Reported with the first-inference upload beside it**, so "uploaded nothing" and "uploaded once
   and kept it" are distinguishable — and **admissible only from a run whose `claimed_op_coverage`
   is at or above the last published figure, with `model_output_equivalence = MATCH`.** Without that
   interlock it is the most gameable criterion in this document: claim nothing, upload nothing,
   score perfectly.
3. **Host↔device transfer time is reported as a share of wall clock, per device, next to it** — the
   measured ceiling is **~94.8% of wall on discrete and ~44.0% on UMA**, and those two were reported
   separately and never compared, which is why the gap survived a day.

**CONFIRMED UNCHANGED 2026-07-30T22:13:37-07:00** under the standing performance directive (§10
head), with one instrument clarification from §10.0.1 R12: the criterion is read against
`device_upload_bytes` **on the session's device**, never against `alloc_device_authoritative_spans`,
which is `UNOBSERVABLE` while the two-`VkDevice` split of §6.5 exists. The threshold, both
interlocks and the first-inference companion figure all stand; the full ruling is in the M0 section
above. *(§6.5 closed 2026-08-01; the instrument choice is unchanged and its reason is restated in
the M0 clarification above and in §6.5.1–6.5.3 — the counter is no longer pinned, it is at a
measured ceiling of zero in the armed lane and has no frame at all in the default one.)* **The corrected decomposition, now with upload as a declared sibling rather than an
undeclared child** (Switch's `vulkan.cmd_upload` span, selector 0 = RTX 4060, 661 invocations,
`MATCH`): subgraph 21424.2 ms, of which record 15510.0 ms, of which **`cmd_upload` 15197.8 ms —
98.0% of record and ~71% of wall**; fence wait 5322.5 ms; GPU kernels ~3300 ms (~15%); submit
127.4 ms; **pipeline lookup 91.4 ms and descriptor allocation 67.1 ms, which is 0.4% and 0.3% of
wall and kills both of the standing hypotheses about per-island recording cost.** Switch's span-
derived 98.0% and Tank's counter-derived 95.8–98.4% are **two independently-authored instruments
agreeing on one quantity**, which is the corroboration §10.0's seventh disclosure point now asks
every performance claim to state.

**Secondary criterion — recording amortises, checked with a counter before a clock.** It survives
its corroboration turning out to be wrong because its justification never depended on it
(`ENGINE.md` §6.1, R6 rule 1); it is demoted because it addresses 1–3% of wall.

4. **`command_buffer_records` over N ≥ 30 consecutive inferences in one session is ≤ the number of
   distinct shape buckets observed, and strictly less than `compute_calls`.** A counter, not a
   timer — a timer improves on a faster machine, this does not. **The identity case (one record per
   call) is the explicit red state**, and it is the falsifier for "the cache works".
5. **Steady-state recording share of wall clock, below 5% — published only with the R11
   decomposition identity**: every phase declared inclusive or exclusive, the parts summed against a
   wall clock measured by a different instrument, and the residual published.
6. **End-to-end wall-clock ratio against a CPU-only run of the same session, per device, reported —
   not thresholded at M1.** The threshold is M2's. The ratio is admissible only from a run whose
   §10.0 verdict is `MATCH`, is measured on the whole model end to end (never GPU-kernel time, never
   the claimed subgraphs alone, so that claiming less can never improve it), and is reported with the
   §10.0 triple from the same run. **It is the criterion that survived tonight untouched**, because
   it has no internal structure to misattribute (§10.0 disclosure obligation).

**M1 INTERLOCK AMENDMENT — ADDED 2026-08-01T13:19:00-07:00 on §10.0 obligation 8 (the device-state
companion) and §10.0.4 (the invariance preference).**

*Asked directly whether M1's criteria need restating. **They do not.** No threshold moves, no
criterion is rescoped, and no wording changes. What changes is that the criteria which rest on a
clock gain the interlock the criteria which rest on a count never needed — the same move made when
the residency criterion gained its `MATCH` and coverage interlocks, and for the same reason.*

- **Criteria 1, 2 and 4 are untouched and this is the finding, not an aside.** They are bytes and
  counts: `device_upload_bytes` per inference as a ratio of initializer bytes, the first-inference
  companion figure, and `command_buffer_records` over N ≥ 30 inferences. **They are invariant under
  contention, tenancy and clock state**, they needed no correction through a week in which every
  timing figure this project held was withdrawn twice, and §10.0.4 exists because of them. Criterion
  4's own text already says it: *a timer improves on a faster machine, this does not.* It is now
  clear that a timer also improves on a **slower** one, if the slowness is steady.
- **Criteria 3, 5 and 6 gain §10.0 obligation 8 as an admissibility interlock**, alongside the
  interlocks they already carry (`MATCH`, attributed; coverage at or above the last published run;
  the R11 decomposition identity for 5). A figure offered against any of them without a device-state
  record covering the statistic's own window is `STEADY_UNCERTIFIED`, which is not a failure of the
  criterion — it is the criterion having no reading yet.
- **Criterion 6 is not exempt because it is wall clock rather than device clock.** It was written as
  the metric with no internal structure to misattribute, and that property is intact and is about
  *naming*, not about the environment. This EP is host-bound, so contention inflates the Vulkan arm
  and the CPU-only arm by different factors and the ratio is not protected by being a ratio.
  Criterion 6 therefore carries the machine-quiescence verdict on both arms, as `bench/` already
  requires, and states it. Neither measurement surface is now unqualified: **wall clock carries the
  quiescence verdict, device clock carries the device-state record, and there is no third surface to
  retreat to.**
- **Nothing here softens a criterion to make it reachable, and nothing hardens one to punish a bad
  week.** The three thresholds — below 1% of initializer bytes, below 5% recording share, the
  reported end-to-end ratio — are exactly as they were this morning. A criterion whose reading is
  `STEADY_UNCERTIFIED` is in the same position as one whose reading is `UNMEASURED`: not met, for a
  reason that names the missing instrument rather than the team.

**What the cheapest satisfaction of these words would be, asked as the drafting rule requires.** For
criterion 5 — steady-state recording share below 5% — the cheapest pass is now visible and it is
this week's finding turned into an attack: **run on a board stuck at its idle clock.** Device time
inflates ~21×, host recording time does not, the recording *share* collapses far below 5%, the
series is perfectly steady, and every gate reports its most confident verdict. That is not a
hypothetical; it is `246.720 ms at RSD 0.1163%` with the arithmetic run backwards. Obligation 8 is
what closes it, which is the argument for making the companion mandatory rather than advisory: **a
share-of-a-total criterion can be satisfied by inflating the total, and a device-state record is the
only thing in the building that would notice.**

### M2 — "Real memory, real compute" (`OP_COVERAGE.md` tier T2 — 33 ops, cum. 121)

Device allocator + data transfer (§6.3), `GetDefaultMemoryDevice` returning a real device, and T2's
reductions / `MatMul` / `Gemm` / norms / softmax. This is the first milestone that can honestly
claim a speedup. **It is also the critical path for everything after it** — the LLM tiers are
worthless without the device allocator, because a KV cache that round-trips to host every token is
not an inference engine (`OP_COVERAGE.md` §6.1 precondition 1). That dependency is Tank's and
Switch's, not Mouse's, and it should be resourced accordingly.

| Work | Owner |
|---|---|
| `allocator.rs`, `transfer.rs`, **reserved-VA span registry (§6.4, OQ-3 resolved)** incl. platform probe-and-halve sizing, `OrtMemoryDevice` wiring | Tank |
| Device-local arena, staging ring, coherence handling, barrier batching | Switch |
| Reductions / `MatMul` / `Gemm` / `Softmax` / norms — semantics, claim, tiling; the fused Softmax and RMSNorm kernels from the §5.6 allowlist | Mouse + Switch |
| MVS transfer-cost calibration at device init + its claim-debug dump (§8.4 A3) | Mouse + Switch |
| Timestamp queries and the trace exporter | Switch + Niobe |
| Numerical tolerance policy for accumulation-order-sensitive ops (OQ-10) | Trinity |
| GPU CI lane; per-vendor result matrix | Link |
| First published speedup table, with island counts and CPU baseline | Niobe |

**Exit criteria.** A small MLP **and a BERT-base encoder using primitive attention (no contrib
ops)** run end-to-end on Vulkan; with `IoBinding`, per-inference host↔device traffic is **zero** for
a fully-claimed graph; a measured speedup vs the ORT CPU EP on at least one real discrete GPU,
published with methodology; and the MVS constants are re-derived from Niobe's measurements rather
than left at their provisional values (§8.4 A3).

**M2's speedup criterion, sharpened 2026-07-30T19:05:03-07:00 so it cannot be met by selection.**
*"A measured speedup on at least one real discrete GPU"* is satisfied by measuring until one device
wins and publishing that one. It now reads: **end-to-end wall-clock ratio against a CPU-only run of
the same session on the same artifact, `< 1.0` on at least one real discrete GPU with
`model_output_equivalence = MATCH`, with the ratio reported for every device in the matrix —
including every device above 1.0.** The matrix is the deliverable, not the maximum. This is **the
first milestone that carries a performance threshold**, and it is the right one: it is the milestone
whose subject is the device allocator and the transfer path, which is what makes speed possible at
all, and it is the first point at which "you should enable this" is a claim we make.

### M3+ — "Breadth and platforms" (`OP_COVERAGE.md` tiers T3–T6)

Sequenced by `OP_COVERAGE.md` §6, not re-sequenced here. **Entry precondition for the whole of M3:**
`tools/graph_census.py` runs in CI against pinned `.onnx` artifacts and reports per-op claim rates
(§1.4 C2). No contrib op is claimed before that alarm exists. The shape of M3+:

| Tier | Target | Gating item |
|---|---|---|
| T3 | **Qwen3-0.6B fp16, decoder layer as one island for *both* the `mobius` and ORT GenAI producers**, KV cache, correct tokens end-to-end, ≤2 islands. **Demonstration target: Phi-3.5 (Foundry Local) — §10.0.2** | `ai.onnx::Attention` **first** (§10.0.2), then `GroupQueryAttention` (XL); M2's allocator; fp16 (OQ-14); push-constant shapes + OQ-15 |
| T4 | Qwen3-1.7B int4, correct tokens, ≤2 islands, beats ORT CPU on ≥2 vendors. **Measurable criterion: `MatMulNBits` claimed ⇒ Phi-3.5 partitions into one island of ≥360 nodes** | `MatMulNBits` (XL) + weight prepacking |
| T5a | **Qwen3.5 hybrid end-to-end — the named target of the directive** | `LinearAttention` `gated_delta` (XL) + `CausalConvWithState` |
| T5b | Qwen3-MoE int4 with the expert block on Vulkan | `QMoE`; likely needs indirect dispatch (OQ-15) |
| T5c | Qwen-VL vision tower + projector feeding the decoder in one session | `Conv` (patch-embed form), `MultiHeadAttention` |
| T6 | ResNet-50 / MobileNetV3 end-to-end, beating ORT CPU | General `Conv`/pooling breadth |

Android hardware validation (§11.1's OQ-12 experiment) runs in parallel with T3–T4 and is gated only
on devices, not on op coverage. The three XL kernels are **not parallelizable away** — each is one
person's deep work — and §1.5's months-scale claim rests on them. **Every tier row above names a
model; from now on it must also name the producer that built it** (§8.5, §10.0.1 R3), and
`largest_island_flops` is reported per producer so that one green producer column cannot mask
another that is near zero.

---

## 11. Open questions

| # | Question | Decided by | Blocks |
|---|---|---|---|
| **OQ-1** | ~~How many real devices report Vulkan 1.1/1.2 **without** `VK_KHR_synchronization2` or `VK_EXT_subgroup_size_control`?~~ **RESOLVED 2026-07-28T19:16:08-07:00.** Link measured it (`PLATFORMS.md` §8, vulkan.gpuinfo.org 2026-07-28): `VK_KHR_synchronization2` is missing on **31.43% of Android** and **12.22% of Windows**; `VK_EXT_subgroup_size_control` on **14.12% of Android**, and its *feature flag* is `VK_FALSE` on all of macOS/iOS. **Ruling (§7.2–§7.5): both are dropped from the hard requirement.** `synchronization2` becomes a probed capability selecting one of two barrier backends behind a single seam (`vk/barrier.rs`); `subgroup_size_control` is consulted as a *properties query* only and never as a required feature. Link's layer-shim option is **rejected** — the AOSP loader cannot discover a layer we ship from a plugin `.so`, and the cited wgpu/Dawn/Godot precedent turned out to be legacy-barrier-only in all three. | Link investigated → **Morpheus decided** | — (§7 is frozen) |
| **OQ-2** | ~~Do llama.cpp and ExecuTorch's stated version floors survive verification?~~ **RESOLVED 2026-07-28T17:59:54-07:00.** Fact Checker claims 1–2: both "requires 1.3" claims **contradicted**. llama.cpp base shaders target `vulkan1.2` (only `_cm2` variants target 1.3); ExecuTorch hardcodes `VK_API_VERSION_1_1`. Claim 4 (Android share) remains *unverified but plausible*. | **Fact Checker** (done) | — |
| **OQ-3** | ~~The ORT allocator's pointer problem (§6.3): ORT allocators return `void*`, a Vulkan allocation is a `(VkBuffer, offset)` pair.~~ **RESOLVED 2026-07-28T22:28:08-07:00 — see §6.4.** `Alloc` returns a span of **reserved, never-dereferenceable virtual address space** (`VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` / `mmap(PROT_NONE, MAP_NORESERVE)`), resolved to `(VkBuffer, offset)` through an opaque-handle registry once per descriptor binding. **`VK_KHR_buffer_device_address` is not carried at all** — Tank's argument that BDA is a second *shader architecture* rather than an optimization is correct and superseded my "registry primary, BDA on top" framing. Reserved VA makes ORT's pointer arithmetic correct by construction and turns a stray dereference into an MMU fault instead of silent corruption. Android's narrower address space is handled by **probe-and-halve at construction**, not by a platform constant — a tuning parameter, not a blocking dependency on Link. | Tank proposed → **Morpheus decided** | — |
| **OQ-13** | **Zero-copy IO binding via `OrtEpFactory::CreateExternalResourceImporterForDevice`.** *New, 2026-07-28T19:16:08-07:00.* Verified by Fact Checker: the public vtable member is `CreateExternalResourceImporterForDevice` (the `…Impl` suffix is a local static in test code, not API), it landed in **ORT 1.24** — not 1.28 — and Tank has already set `ORT_API_VERSION_MIN = 24` with version negotiation, so **it costs us no ABI floor movement.** It is **orthogonal to OQ-3**: it is an OS-handle external-memory path in which the *caller* exports their `VkDeviceMemory` via `vkGetMemoryWin32HandleKHR` / `vkGetMemoryFdKHR` and we re-import it, answering "how does an external caller hand us their buffer as a graph input/output", not "what does our `Alloc()` return". Tank and Fact Checker independently reached this and Tank has recorded it as evaluated-and-rejected for OQ-3; **it is not to be re-proposed there.** Tracked here on its own merits: it is real, supported upstream, has an in-tree reference (`onnxruntime/test/providers/nv_tensorrt_rtx/nv_vulkan_test.cc`), and is the complete answer to zero-copy IO binding. **Scope: post-M2**, because it presupposes the device-memory tensor path exists. Known constraint to design around: the caller's memory must have been allocated with `VkExportMemoryAllocateInfo` up front — it cannot be retrofitted onto an ordinary allocation, so this is an integration contract we must document, not a transparent optimization. | **Tank** designs → Morpheus reviews | post-M2 |
| **OQ-14** | **What fraction of target devices support `shaderFloat16` + `storageBuffer16BitAccess`?** *Escalated from Mouse's OQ-M2, 2026-07-28T19:16:08-07:00.* Under the frozen §7.2 both are probed, not required. An fp32-upcast LLM path is a memory-footprint failure, not a slow path (§8.4 A4), so a low Android number means the LLM story is **desktop-first as a product boundary**, regardless of op coverage. This decides a product scope, which is why it is not a shader-variant detail. | **Link** measures → **Morpheus** rules on scope | tier 3 / M3 |
| **OQ-15** | **Indirect dispatch. PROMOTED TO BLOCKING, 2026-07-29T21:14:03-07:00 (§8.8, §10.0.3).** *New, 2026-07-28T19:16:08-07:00.* Shape-agnostic push-constant kernel parameters (§8.4 A5) make the *shader* length-agnostic but not the **workgroup count**, which still depends on sequence length — so either we re-record per shape bucket anyway, or we use `vkCmdDispatchIndirect` with a device-computed count. The same mechanism is what `QMoE`'s data-dependent expert routing needs on a pre-recorded command buffer. One evaluation should serve both. Evaluate, do not assume. **No longer a tier-3 evaluation: it now gates M1's second-token exit criterion**, because symbolic extents were measured to be the dominant decline cause on a real model. | **Switch** evaluates → Morpheus decides | **M1**, tier 3, tier 5b |
| **OQ-4** | ~~Shader compilation: build-time `glslc` vs checked-in pre-generated SPIR-V vs both with SDK preferred. Provisionally: build-time with checked-in fallback.~~ **RESOLVED 2026-07-28T22:28:08-07:00 — §7.8. The provisional decision is *changed*, not implemented: the Vulkan SDK is a hard build prerequisite and there is no checked-in SPIR-V fallback.** Found by the coordinator building on a machine without the SDK — `build.rs` panics with an escape hatch, which contradicted this row. The doc was wrong, not the code: a checked-in `.spv` that drifts from its `.comp` changes what runs with no signal, "reviewable diffs" is not true of SPIR-V binaries, and the variant table already generates 168 modules. Five binding conditions in §7.8, of which one is new work for Switch: **a shader-less artifact must advertise zero devices and claim nothing**. If the prerequisite ever proves to block contribution, the remedy is vendoring `shaderc`/`naga`, not checked-in binaries. | Switch proposed → coordinator found the divergence → **Morpheus decided** | — |
| **OQ-5** | `gpu-allocator` vs a hand-rolled suballocator. `ENGINE.md` §3.1 picks `gpu-allocator`; I concur provisionally. Confirm it cross-compiles cleanly for Android and works under MoltenVK. | **Switch** owns → Link validates | M0/M3 |
| **OQ-6** | What vendor ID does the factory report when it advertises zero devices, or before a device is bound? ORT calls `GetVendorId` on the factory, not per device. | **Tank** proposes → Morpheus decides | M0 |
| **OQ-7** | Do we need a real GPU CI runner for M2's exit criteria, and if so, self-hosted or a cloud GPU lane? Software rasterizers cannot validate a speedup claim. | **Link** proposes → Justin decides (cost) | M2 |
| **OQ-8** | ~~Is `com.microsoft` contrib-op support ever in scope?~~ **RESOLVED — and re-resolved at a higher level. 2026-07-28T20:54:42-07:00: Justin ruled directly that the `com.microsoft` domain is in scope** (`.squad/decisions/inbox/copilot-directive-contrib-ops.md`), superseding both the original `ai.onnx`-only non-goal and my narrower 2026-07-28T19:16:08 formulation of "nine named ops, never the domain". The admitted v1 set is still those nine; what changed is that a tenth is now a scoping decision inside an in-scope domain rather than a re-opened non-goal. The engineering constraints (§1.4 C1–C7) are unaffected by the elevation — they were always about safe claiming, not about permission. The deciding fact, verified by Mouse from the ORT GenAI model builder source: the builder *emits* these ops directly, so declining `com.microsoft` means a Qwen graph cannot run at all. | **Justin** (domain) / **Morpheus** (constraints) | — |
| **OQ-9** | Threading model: one `VkDevice` per session (chosen) vs a process-shared device with a mutex. Sharing saves memory and pipeline-cache warmth for multi-session hosts. | **Tank + Switch** propose → Morpheus decides | post-M2 |
| **OQ-10** | Tolerance policy for accumulation-order-sensitive ops (GEMM, reductions) across vendors, where fp32 associativity differs. Needs a stated, derived rule before M2's ops land, not after. | **Trinity** proposes → Morpheus ratifies | M2 |
| **OQ-11** | ~~Ratification of `OP_COVERAGE.md` (§8.1).~~ **RESOLVED 2026-07-28T19:16:08-07:00: ratified with five amendments** (§8.4). It supersedes §8.2/§8.3; §8.1's seven principles stand. **Its central question — whether to admit `com.microsoft` — was then settled above my level by Justin's ruling of 2026-07-28T20:54:42-07:00** (see OQ-8); A1 is revised accordingly and survives as the *discipline* rather than as the permission. | Mouse proposed → **Morpheus ratified** → **Justin ruled on the domain** | — |
| **OQ-12** | Does carrying the legacy barrier backend (§7.3) actually buy *usable* devices, or does the Adreno 5xx / Mali Bifrost population fail for some other reason? **The 31.43% figure is a database claim, not a usability claim, and until the experiment in §11.1 runs, that is exactly how much of it is unverified: all of it.** *Concurrence noted 2026-07-28T21:01:56-07:00: Link's `PLATFORMS.md` §8 rewrite now states the same position in his own words — the gpuinfo data proves those devices lack `VK_KHR_synchronization2`, not that a legacy barrier path makes them usable. The two documents agree on the honest position rather than each implying the other verified it.* The experiment, its pass/fail bar, and what would reverse the decision are specified in §11.1. Needs real hardware, which we do not have. | **Link** measures → Niobe benchmarks → Morpheus reviews | M3 Android scope |


| **OQ-16** | **When do `LinearAttention` and `CausalConvWithState` appear in a *released* ORT, and does their schema change between our main-branch fingerprint and that release?** *New, 2026-07-28T22:28:08-07:00.* Four of the eleven admitted contrib rows (`LinearAttention`, `CausalConvWithState`, `QMoE`, `MoE`) carry `MAIN_BASELINE` — they exist only on ORT main. All four are `Staged`, so nothing is claimed against an unreleased schema, and C2 item 6 now makes that structural rather than intentional. But `LinearAttention` and `CausalConvWithState` gate **T5a, the named Qwen3.5 target**, so we are partly gated on an *upstream schema stabilizing* — a different risk from "the Vulkan kernel is hard", with different mitigations, and it must be reported as such rather than folded into a Vulkan schedule slip. Practically: the T5a kernels may be written twice, and every main-baseline fingerprint needs re-verification the moment its release exists. | **Fact Checker** watches upstream → Mouse re-verifies → **Morpheus** rules on T5a scope | T5a |

### 11.1 OQ-12 — the minimum decisive experiment

Written now, so it can be executed the hour hardware exists rather than designed then. Until it
runs, **the entire 31.43% Android claim behind §7.3 is unverified as a *usability* claim** — Link's
data proves those devices lack `VK_KHR_synchronization2`, not that they can run a model. §7.3's
Windows argument (12.22%) is unaffected and stands on its own; this experiment is only about how
much of the Android half is real.

**Devices — four, chosen to be decisive rather than representative.** Two must be from the
sync2-missing population and two are controls:

| Slot | Device class | Why this one |
|---|---|---|
| A | **Adreno 5xx** — e.g. Snapdragon 660 (Adreno 512) or 636 (Adreno 509), Android 8–10, stock OEM blob | The single largest sync2-missing bloc. Frozen pre-2021 drivers. If any class fails for other reasons, it is this one. |
| B | **Mali Bifrost on MediaTek** — e.g. G52 (Helio G85) or G76 (Helio G90T), stock ROM | The second bloc, and a different vendor's driver bugs. MediaTek specifically, because the same Mali IP on a Samsung/Exynos ROM has a different update history. |
| C | **Adreno 6xx on Android 12+** — e.g. Snapdragon 865/888 | Control: *has* sync2. Isolates "is the legacy backend correct" from "is this device usable". |
| D | **Mali Valhall on Android 12+** — e.g. G78 / G710 | Second control, second vendor. |

Two physical units of A and B are worth more than four of C and D. If only two devices can be
obtained, take **A and C**.

**Workload — three stages, each of which can independently fail the experiment.**

1. **Gate check (minutes).** Does the device pass §7.2? Report `vkEnumerateInstanceVersion`,
   `maxComputeWorkGroupInvocations`, `maxComputeSharedMemorySize`, memory types, subgroup
   `supportedOperations` and `subgroupSize`, and — separately tracked, this is OQ-14's data point —
   `shaderFloat16` and `storageBuffer16BitAccess`. A device that fails §7.2 outright is the
   cleanest possible negative result.
2. **Correctness (hours).** The **full M1 differential suite** — the §8.2 elementwise/shape floor —
   run on device against ORT CPU, plus the M2 reduction/GEMM/softmax/norm set if it exists by then.
   Run it **twice**: once normally, once with `ep.force_legacy_barriers=1` on devices C and D, so
   any failure can be attributed to the legacy barrier backend rather than to the device. This is
   the same parity harness §9.1 already requires; no new test infrastructure is needed.
   Validation layers on. Record every failure with its op, dtype, shape and driver version.
3. **Usability (hours).** One bandwidth-bound elementwise chain and one GEMM-anchored subgraph,
   sized realistically, timed against **that device's own CPU** through the ORT CPU EP. Not against
   a desktop GPU — the question is whether the Vulkan path beats the CPU already in the phone.
   Report Mouse's `boundary_time_fraction` alongside wall time (`OP_COVERAGE.md` §7.3).

**Pass bar.** For the legacy backend to be vindicated on Android, devices A **and** B must:
(i) pass the §7.2 gate; (ii) pass the full differential suite with **zero** numerical failures and
zero validation-layer errors; and (iii) beat their own device's ORT CPU EP by **≥ 1.5×** on the
GEMM-anchored subgraph. 1.5× is the threshold below which the integration cost, the memory
footprint and the thermal cost are not worth a user's trouble.

**What reverses the decision, stated in advance so it is not rationalized afterwards.**

- **If A and B both fail stage 1 or stage 2** — i.e. the sync2-missing population cannot run correct
  compute at all, for reasons unrelated to barriers — then the Android half of §7.3's justification
  is void. The legacy backend **still stays**, on the strength of the 12.22% Windows gap alone, but
  Android's §7.2 gate should then be tightened per device class in `PLATFORMS.md`, and M3's Android
  scope narrows to the Adreno 6xx / Valhall tier. I would record that as a scope decision, not
  quietly.
- **If A and B pass stages 1–2 but fail stage 3 (< 1.5×)** — they are correct but not worth using.
  The legacy backend stays and the devices remain supported (correctness is free once written), but
  they are documented as "runs, not recommended" and get **no tuning budget**. This is the outcome I
  consider most likely.
- **If A and B pass all three stages** — §7.3 is fully vindicated and the Android tier gets a real
  tuning budget in M3.
- **If devices C or D fail the differential suite only under `ep.force_legacy_barriers=1`** — that
  is a bug in `LegacyBackend`, not a finding about devices, and it is the most valuable possible
  result of the whole experiment because it is the failure mode §7.5's parity lane exists to catch.

**Cost and owners.** Two to four used mid-range Android phones, obtainable second-hand for a modest
sum — this is the cheapest decisive experiment in the entire project and it retires the largest
unverified assumption in §7. **Link** owns device acquisition, the gate-check harness and the
`PLATFORMS.md` rows; **Trinity** owns running the differential suite on-device; **Niobe** owns
stage 3 and the `boundary_time_fraction` numbers; **Morpheus** rules on the outcome. Blocked only
on hardware, not on any other milestone — stages 1 and 2 can run the day M1 is green.

---

## 12. Divergences from the `onnxruntime-mlx` reference

Every deliberate difference, with its reason. Anything not listed here is intended to match the
reference.

| # | Divergence | Reason |
|---|---|---|
| D1 | **Real device allocator + data transfer** (`allocator.rs`, `transfer.rs`); MLX returns null stubs. | No unified memory. Without them, every partition boundary is a full host round-trip. Phased: null in M0/M1, real in M2 (§6.3). |
| D2 | **`GetSupportedDevices` advertises N devices, with a capability gate**; MLX advertises one. | Multi-GPU is normal on Windows/Linux, and an unusable Vulkan device must never be offered. |
| D3 | **A whole Vulkan engine layer (`vk/`) that has no MLX counterpart.** MLX supplies scheduling, memory, and op semantics; Vulkan supplies none. | Unavoidable. It is also why Rule 2 (§4.2) is enforced rather than encouraged. |
| D4 | **`recorded.rs` (command-buffer recording cache) replaces `compiled.rs` (`mlx_compile` tracing).** | Same intent — pay graph construction once, replay many. Different mechanism: Vulkan's unit of replay is a recorded `VkCommandBuffer`, following ExecuTorch's model rather than llama.cpp's re-record-every-eval. |
| D5 | **Weight prepacking is a first-class `Compile` step**, not a first-`Run` cache fill. | Vulkan uploads are explicit and expensive; ORT gives us initializer bytes at compile time; doing it lazily would put a staging copy on the first inference for no benefit. |
| D6 | **`shaders/` directory and a SPIR-V build step.** MLX explicitly *deleted* its `.metal` kernels. | We are the backend. This is the one place where the MLX project's history is an anti-pattern for us — its lesson was "don't hand-write kernels when a good backend exists", and for Vulkan no such backend exists. |
| D7 | **fp32-only v0; fp16 as a per-family gated variant.** MLX is dtype-generic for free. | MLX carries dtype through its ops with no per-dtype code. Every Vulkan dtype is a separate SPIR-V variant plus a device feature probe. |
| D8 | **`com.microsoft` contrib ops in scope; eleven named ops are the v1 admitted set (tiers 3–5).** MLX's highest-value ops are contrib ops. | *Revised 2026-07-28T20:54:42-07:00.* Originally "no contrib ops in v1"; reversed by the OQ-11 ratification and then settled directly by Justin's ruling that the domain is in scope — the ORT GenAI model builder emits them, so declining means no Qwen graph runs at all. The remaining divergence from MLX is *how*: we register ops **by name with a hand-written claim predicate each and no domain-wide opt-in in the code**, plus a graph census in CI as the drift alarm for a surface that has no opset number (§1.4 C1–C7). |
| D9 | **Vendor ID is read from the bound device**, not hardcoded to one vendor. | Cross-platform mandate. |
| D10 | **Validation layers are part of the definition of done.** No MLX equivalent. | Vulkan's error surface is enormous and mostly silent without layers; MLX's C API validates for us. |
| D11 | **`ash` (safe-ish Rust Vulkan bindings) rather than bindgen over `vulkan.h`.** MLX bindgens `mlx-c` directly. | `ash` is the ecosystem standard, handles the loader/extension-function-pointer problem correctly, and removes a large class of hand-written FFI bugs. The ORT side still uses bindgen, matching the reference. |
| D12 | **Two barrier backends behind one internal seam (`vk/barrier.rs`), selected once at device init**, plus a session option to force the legacy one. MLX has no counterpart — MLX's C API owns all synchronization. | §7.3. Requiring `synchronization2` would exclude 31.43% of Android and 12.22% of Windows; the compatibility-first directive makes that unacceptable. The cost is one file with two implementations and a doubled CI lane, which is bounded; the cost of the alternative is permanent device exclusion. wgpu, Dawn and Godot all ship legacy-only barriers, so this is the mainstream position, not an exotic one. |

---

## 13. References

- **Reference architecture:** `onnxruntime-mlx` — `docs/DESIGN.md`, `docs/OP_ARCHITECTURE.md`,
  `docs/COMPILED_CAPTURE.md`, `rust/src/{lib,factory,ep,engine,registry,compiled,sys}.rs`,
  `rust/{Cargo.toml,build.rs}`, `tests/`, `bench/`, `python/`.
- **Sibling docs:** [`ENGINE.md`](./ENGINE.md) (Switch), [`PLATFORMS.md`](./PLATFORMS.md) (Link),
  [`OP_COVERAGE.md`](./OP_COVERAGE.md) (Mouse), `OP_ARCHITECTURE.md` (Mouse, forthcoming),
  `BENCHMARKS.md` (Niobe, forthcoming).
- **Vulkan layer deployment (§7.3 ruling):** `KhronosGroup/Vulkan-Loader`
  `docs/LoaderLayerInterface.md` and `docs/LoaderApplicationInterface.md` (Android layer discovery;
  "The Android loader does not use manifest files"; elevated-privilege `secure_getenv` caveats),
  `loader/loader_environment.c`; `KhronosGroup/Vulkan-ExtensionLayer`
  `docs/synchronization2_layer.md` ("needs to be packaged inside the APK");
  `developer.android.com/ndk/guides/graphics/validation-layer`.
- **Barrier prior art (§7.3):** `gfx-rs/wgpu` `wgpu-hal/src/vulkan/command.rs`
  (`cmd_pipeline_barrier`, legacy only); `google/dawn`
  `src/dawn/native/vulkan/CommandBufferVk.cpp` (`fn.CmdPipelineBarrier`, legacy only);
  `godotengine/godot` `drivers/vulkan/rendering_device_driver_vulkan.cpp` (`vkCmdPipelineBarrier`,
  legacy only). None ships `VK_LAYER_KHRONOS_synchronization2`.
- **ORT plugin-EP C ABI:** `onnxruntime_ep_c_api.h`, `RegisterExecutionProviderLibrary`,
  `SessionOptionsAppendExecutionProvider_V2`, `CreateEpFactories`, `ReleaseEpFactory`.
- **Prior art — llama.cpp Vulkan backend:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
  `vulkan-shaders/vulkan-shaders-gen.cpp`. Per-node eager dispatch with graph-level fusion passes;
  re-records command buffers every eval; buffers only, no VMA; hard runtime floor Vulkan 1.2; base
  shaders compiled at `--target-env=vulkan1.2` with `vulkan1.3` reserved for cooperative-matrix-2
  variants; build-time SPIR-V with runtime binary patching.
- **Prior art — ExecuTorch Vulkan backend:** `backends/vulkan/`. Ahead-of-time partitioning
  (`vulkan_partitioner.py`), serialized graph, `prepack()` at init, command buffer recorded once
  and replayed, buffers **and** image textures, VMA, on-disk `VkPipelineCache`, hard floor
  Vulkan 1.1.
- **Decision records:** `.squad/decisions/inbox/morpheus-architecture-v0.md`,
  `.squad/decisions/inbox/morpheus-oq1-resolution.md`,
  `.squad/decisions/inbox/morpheus-oq11-ratification.md`,
  `.squad/decisions/inbox/link-oq1-extension-availability.md`,
  `.squad/decisions/inbox/mouse-op-coverage-plan.md`,
  `.squad/decisions/inbox/fact-checker-ort-1.28-api.md`,
  `.squad/decisions/inbox/rai-oq-m6-license-ruling.md`,
  `.squad/decisions/inbox/tank-m0-foundation.md`.
