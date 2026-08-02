# onnxruntime-ep-vulkan — Architecture Design

**Status:** v0 architecture of record — accepted for M0/M1 implementation. **§7 (Vulkan baseline) is frozen.**
**Date:** 2026-07-28T17:59:54-07:00 · **Last revised:** 2026-08-01T18:59:38-07:00 (**§6.5 CLOSED — AND THE CLOSURE IS OF A *CONDITIONAL*, WHICH IS STATED WITH ITS LANE OR NOT STATED.** Raised by the coordinator against his own earlier report, by scoring standing predictions rather than by new work. The two-armed artifact holds and is not withdrawn: `indexspace.json`, allocator factory index **`1` on selector 0 (NVIDIA) and `0` on selector 1 (Intel)**, matching each arm's offered index, both `SHARED`, both `alloc_device_buffer_binds = 6`, verdict `ONE_INDEX_SPACE` — the R10 artifact whose content varies with its input, which the earlier one-armed check was not (the pre-fix state read `SHARED` on selector 0 by **coincidence of two index spaces**). **But the probe *arms* the device-memory provider** with `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1`, and the ordinary inference lane reports `alloc_device_frame = OFF`, `binds = 0`. These do not contradict — §6.5 is the conditional *when both sides exist they are on one index space*, and the probe makes its antecedent true — **but a closure statement that omits the lane is not a shorter version of the true sentence, it is a different and false one**, because a reader who runs an ordinary inference finds the mechanism switched off and nothing told them to expect it. Canonical form now fixed in §6.5.1. **`OFF` is a third state and its existence is why this was catchable at all**: the standing prediction was `SHARED` xor `SPLIT-DEVICE`, and the instrument **declined to pick one of the two offered options rather than forcing itself into the nearer one** — the *every way of not knowing gets a name a machine can print* family paying out in the way that is hardest to arrange deliberately, since **a binary prediction met by a third token is a refutation the predictor cannot talk himself out of.** Had `OFF` been folded into `SPLIT-DEVICE` (both are "not SHARED") the prediction would have scored a clean pass and the scope gap would still be invisible. The scoring discipline is ratified as **R12 applied to predictions: a prediction is scored only against an artifact from the lane it described** — wrong lane is `UNSCORABLE`, no artifact is `UNSCORED`, both count as non-passes, and **the denominator never shrinks to flatter the numerator.** **§6.5.2 RULING — `alloc_device_frame = OFF` on the default path is INTENDED, it is NOT the `offer_shared_device` gap, and its recorded justification HAS EXPIRED.** Not the gap: `offer_shared_device` has a production caller (`vk/session.rs`), so §6.5 proves a property of a **wired seam on the only lane where the seam has two sides**, not of a path users never take. Intended: the allocator is opt-in behind `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY`, gated in `factory.rs`, already written down. **But the reason written down with it is no longer true** — it says advertising device memory is a package deal requiring an `OrtDataTransferImpl` without which *every session fails at `Run`*, and that the transfer *"cannot be written until the handle→`VkBuffer` seam is filled"*. **That precondition is discharged**: `CreateDataTransfer` is registered unconditionally, `transfer::VulkanDataTransfer` exists, and armed-lane sessions complete — nine spans, 9,437,184 bytes through the provider, and 427 allocations / 2.09 GB of Phi-3.5 at model scale. **The condition the switch was waiting for was met and nobody went back to the switch.** Recorded as **R12 with a date as the frame** — *for a counter the frame is a device; for a correctness verdict the frame is an executor; for a rationale the frame is a date* — and **a default whose stated reason has expired is indistinguishable from one that is still needed**, the `retain_viable` shape arriving in a justification instead of a call graph. **The default is not changed by this ruling and I am not ruling on whether it should be**: there is a *live* reason for `OFF` that the source does not give — the switch currently buys **host memory wearing a device handle**, risk with no measured benefit — and **a default defended by a reason its own documentation does not give is a default nobody has re-decided.** Re-justify on current evidence, dated, or re-decide; owner Tank with Switch, by M2 entry. Generalised: *a gate, flag, default or `Staged(why)` whose stated precondition has since been discharged is re-justified or removed*, with the precondition stated as something that has an artifact — cheapest abuse named, a precondition too vague to be observed discharged. **§6.5.3 RULING — `alloc_device_authoritative_spans = 0` in the closure artifact is a MEASURED ZERO AT A MEASURED CEILING OF ZERO, not R12.** Right question, wrong shape: the counter is emitted as the **string** `"UNOBSERVABLE"` out of frame and `"UNWIRED"` unrun, and as an **integer** only when measured — it printed `0`, a number, so **the three-state type discipline answered the question before it was asked**; its unconditional twin moved (`residency_evaluations = 9` from the same call site, so `0`/`9` is nine measured negatives, not a silent no-op); and `ceiling = backed − staged = 9 − 9 = 0`, so zero is **the only value it could correctly take** — every device-backed span is still host-staged because the engine reads through `host_backing_for`, and `binds = 6` with `authoritative = 0` is the consistent, honest description. **`UNOBSERVABLE` would be a stronger and false claim**: R12 is a question that *cannot be asked* in this frame; **a zero at a zero ceiling is contingent** — asked nine times, answered no nine times, and it flips the day a span is allocated device-only. **What does need fixing is the probe, not the counter:** `probe_indexspace.py` drops `alloc_staged_spans` and `alloc_device_authoritative_ceiling` from its extract — the two keys that make the zero interpretable — so it presents `backed=9, evaluations=9, authoritative=0` as a sufficient set when it is not, and a careful reader of that artifact **correctly** could not tell a measured zero from a pinned one. **R11's shape in a probe's *selection* rather than in a name.** Remedy, Niobe's and Switch's to make: the extract carries the ceiling and the staged count, or it does not carry the authoritative count at all. M1's residency clarification amended in place — **the pin is lifted and the instrument choice is unchanged for a new reason**: `alloc_device_authoritative_spans` was unusable because it was pinned, and is unusable because it is at its ceiling on the armed lane and has no frame at all on the default one; `device_upload_bytes` stands, and the day the ceiling rises is the day that clarification gets re-read.) · *prior revision 2026-08-01T13:19:00-07:00* (**§10.0.1 R9 AMENDMENT 5 — THE ANTI-CORRELATED FALSIFIER, AND THE REGISTER DOES NOT GROW.** `gpu_steady_tail` is a variance test over a suffix and **cannot see a bias**: measured `STEADY` at **126.647 ms, RSD 0.79%** with three foreign GPU processes outliving a truncated run (**10.99× wrong**), and `STEADY` at **246.720 ms, RSD 0.1163%, zero discarded** on a verified sole-tenant board held at its **210 MHz idle clock against a 3105 MHz boost** (**21.4× wrong**). **In both failure modes the wrong number carried the better RSD than the right one** — *a low clock does not raise RSD, it lowers it*, so a run that is uniformly wrong produces the gate's most confident possible verdict. **RULING: this is not R11 and it is not R14.** R11's four obligations, applied faithfully, *certify* the specimen — there is no decomposition, no flat table, no inclusive parent, and name–content agreement passes — and **a rule that would have certified the specimen does not cover it**, which is the same test used to refuse folding R11 into R10. It is **R9**: bias in a series' level is in a dispersion statistic's **silence set**, and the remedy is R9's and no other, *a different instrument*. **The register individuates by remedy, not by flavour**; a second name for one failure class would be two names for one measurement, appearing to close. What is new is a **mechanism inside R9**: R9 as written describes plural instruments **jointly silent**, this is a single instrument whose confidence is **anti-correlated with the error** — **R9 rule 5: ask which way a check moves when its subject is wrong; if it moves with the reader's confidence, it cannot be repaired by tightening its threshold** (a tighter bound admits *more* of the failure), it is demoted from gate to precondition, and the claim is `UNMEASURED` until a **second quantity from outside the series** records the state of the thing measured. **Precision is not accuracy and this register had never had to say so.** **§10.0 EIGHTH DISCLOSURE OBLIGATION — THE DEVICE-STATE COMPANION, and it withdraws a sentence of mine:** *contention inflates host work but cannot touch the GPU clock* is **false twice over** — foreign GPU work inflates device-busy directly, and the board's own governor varies it **14.8×** with nothing foreign running. **A device-clock figure is quotable only when a device-state record covers the same window as the statistic — the suffix, not the run — carrying a tenancy verdict and a clock min/median/max against the board's advertised maximum**; absent it, `STEADY_UNCERTIFIED`, a fourth state that is neither `STEADY` (read as quotable) nor `ERROR` (the statistic computed). Three tightenings on the drafting rule *what is the cheapest thing that satisfies the words without the intent?*: stated as a **record, never a tool** (cross-platform by mandate; `nvidia-smi` is one vendor's implementation); **the absence of the companion is never a waiver**, or the cheapest pass is to measure on a platform with no telemetry — and the Intel iGPU, which shares its power budget with loaded CPU cores, is the platform that loophole would most reward; and **a missing probe is `ERROR(instrument)`, never `SOLE_TENANT`** (R13). Plus **8b — two device-clock figures are comparable only if their device-state records agree**; a before/after whose "before" predates the requirement **is not a pair**, which upholds Switch's own ⛔ on his barrier result (*"probably sound is not the standard"*). **THE 40.201 ms REGIME-SEPARATION RESCUE FAILS; THE FIGURE IS RE-QUALIFIED, NOT WITHDRAWN.** The argument — two regimes 21× apart, so a run's regime is recoverable from its magnitude — dies on **there not being two regimes**: the board ranged **210 → 2490 MHz within a single run**, a governor is continuous, and *"the two clock states I sampled do not overlap"* was generalised into a claim about the device. Three more: the band rests on **two samples of one build**; the margin protecting **40.201 ms is 6.1×, not 21×**, and it sits at the *top edge* of the boosted band, the least protected position in it; and the rescue is about **clock** while foreign-GPU contention inflates **continuously**, with no regime structure to grip and no tenancy verdict on that run. What survives is arithmetic: every catalogued perturbation has a **non-negative** sign on time, so **40.201 ms is quotable as *≤ 40.201 ms, device state unrecorded*** — **RSD 0.033% loses its certifying role and keeps its descriptive one**, and the 40.390 → 11.525 within-series comparisons are **not certified either**, for the reason Switch himself supplied and did not carry across. **§10.0.4 THE INVARIANCE PREFERENCE — prefer the invariant that survives the contended machine.** Switch's correction to his own arithmetic: `min()` over inferences is an **upper** bound, not a lower one (`observed = true + delay`, `delay ≥ 0`), and **two upper bounds do not bound a difference from below** — "≤ 14.414 ms before" and "≤ 2.704 ms after" does not establish an improvement, let alone 5.33×. What rescued that result was **a count, not a clock: 147,618 `VkBufferMemoryBarrier` structs per inference before, 354 after** — *counts do not care whether the box is busy* — exactly as **byte counts (1997.6 MiB → 0.756 MiB)** carried weight residency when no timing was admissible. **Where a claim can be supported by a quantity the environment can perturb or by one it cannot, the unperturbable quantity is the claim of record and the perturbable one is at most an estimate of magnitude**; declare the sign, and a difference needs bounds on **opposite** sides. Its own cheapest abuse, named: *report the invariant as what it is — the reader may not be handed a count and left to supply the clock.* **M1 CRITERIA NEED NO RESTATING: no threshold moves, no criterion is rescoped.** Criteria **1, 2 and 4 are untouched and that is the finding** — bytes and counts, invariant under contention, tenancy and clock, and the only criteria that needed no correction through a week in which every timing figure was withdrawn twice. Criteria **3, 5 and 6 gain obligation 8 as an admissibility interlock**, and **6 is not exempt for being wall clock**: this EP is host-bound, so contention inflates the two arms by different factors and a ratio is not protected by being a ratio. **Neither surface is unqualified now and there is no third to retreat to.** Named attack on criterion 5: **run on a board stuck at idle clock** — device time inflates ~21×, host recording does not, the recording *share* collapses far below 5%, and every gate reports its most confident verdict; **a share-of-a-total criterion is satisfiable by inflating the total, and a device-state record is the only thing that would notice.**) · *prior revision 2026-07-31T07:45:10-07:00* (**§10.0 THIRD METRIC AMENDMENT — `MATCH` is not a verdict about this EP unless it carries what executed the model.** Specimen: ORT printed `EP_FAIL … Falling back`, re-ran the whole graph on CPU without raising, `get_providers()` still listed VulkanEP because the provider list is fixed at session-create time — and `model_output_equivalence` returned **`MATCH` for a run in which this EP executed zero nodes**. Wired, invoked, correctly named, arithmetically correct, **about a different world: R12 arriving at a verdict rather than a counter, where the frame is not a device but an executor.** The verdict becomes a **record** carrying `executed_by`, parsed on this run from **ORT's profiling trace — an instrument we do not own** — with `MATCH` **unrepresentable** at a zero own-provider count (constructor obligation, not an assertion beside the value), both witnesses recorded and disagreement emitting `SPLIT-FRAME`, and a fourth state **`UNATTRIBUTED`** that is emphatically **not** `DIVERGENT`: *the model was not wrong, the subject was.* **§10.0.1 R13 — an instrument's failure is not distinguishable from the condition it detects, and the reader who most needs the distinction is the one who predicted the red.** Trinity's Guard D — the fix for exactly the hole above — raised `NameError` before reading one profiling event; I watched the suite go `8 passed` → `5 failed` and reported the guard as working. **Three terminal tokens, always: `PASS` / `FAIL(condition)` / `ERROR(instrument)`; an instrument error never counts as a detection; a guard must state what it observed even when it fails; and the remedy is a second witness with a different failure mode, not a better first witness** — the lane now fails on the `Falling back` line itself, five sightings and every gate green. Second clause, and the more dangerous half, the inverse of R6 amendment 4: **a result that confirms a prediction deserves more scrutiny than one that contradicts it, because the contradiction gets checked automatically and the confirmation does not — quote the failure text, never the failure count.** First rule in this register about the reader rather than the instrument. **M0: criteria 2 and 10 REOPENED; four met, six partial, two not met, of twelve.** Criterion 10's closing evidence is **void, not narrow** — scope narrows a true statement and cannot repair one whose subject was absent — and the reopening is priced in advance: **three consecutive attributed `MATCH` runs in one session close it, same day, no new conditions.** Genuine and incomplete on the other side: **ORT profiling reports 354 of 364 nodes on the GPU in one fused island, 10 on CPU matching Mouse's declines exactly, `argmax 30751` == CPU** — the first attributed execution this project has recorded — against a multi-run picture that is red (weight cache OOM → silent fallback; 50 KV-cache outputs never written). Criteria 3, 4 and 5 advanced in substance and moved no row, because **I have not seen the artifacts** (R10), applied on a good day to mechanisms I asked for. **The ranked performance order stands — residency, net-benefit declines, fence-wait idle, kernels — because it was derived from counts and ratios, never from wall clock**; rank 1 keeps its place and changes its content from *make the weights resident* to **make residency bounded**, and **a performance mechanism that fails into silent CPU fallback is a correctness defect wearing a performance costume.** Residency landed on bytes: **1997.6 MiB → 0.756 MiB per inference, ratio 0.0004, forty times inside M1's threshold — and the criterion stays open**, which is the clearest evidence yet that the interlocks are the criterion. **Every wall-clock figure this project holds is withdrawn, 3.1× and 3.7× included** — taken during CPU fallback — so §10.0's disclosure obligation currently publishes **`UNATTRIBUTED`**, not a stale number; my own 22:13 clause, *no timing figure is quotable from a run whose verdict is not `MATCH`*, is what voids them, and **a rule that first bites its author was aimed at the right thing.**) · *prior revision 2026-07-30T22:13:37-07:00 (**STANDING DIRECTIVE (Justin): 「要确保我们性能是非常高 一致向高性能推进」** — *ensure performance is very high; push toward high performance continuously.* **RULING: it changes the calendar and not one gate.** It does **not** overturn the M0 performance ruling — a directive to be fast is exactly the condition under which a speed *gate* becomes dangerous, because the cheapest way to pass a ratio criterion is always to do less GPU work; it raises the value of the interlocks, not the case for the gate. It **does** make performance work continuous and parallel with correctness (`一致` is a **rate** obligation, so **the instrument for it is a series, not a value**, falsifiable by a flat line). Added on my own authority: **no timing figure is quotable from a run whose verdict is not `MATCH`, and every benchmark asserts EP presence and a non-zero claimed count before starting a clock — a fast wrong number is the failure mode this directive creates, not partial credit toward it.** **The tail (Windows + Linux + software rasteriser + CI) is unchanged and stays at the front**: it contends with residency for nothing — not a person, not a file, not a machine — and *a standing directive is a reason to re-examine a placement, never on its own a reason to move it.* **M1's weight-residency criterion stands exactly as written** (< 1% of constant-initializer bytes; today **1.0002**), both interlocks intact. **§6.5 RULING — exactly one `VkDevice` per (physical device, EP instance); the second one is a defect, not a design.** Tank's memory provider created its own device, so the session cannot bind its buffers; §2.3 already said `VkDevice` lifetime is EP-scoped, **so the document was right and the code diverged and nothing in between could tell**. No legitimate reason for two survives inspection (queue families, extension unions and external memory all fail to apply) and the split *costs* compatibility rather than buying it. **Seam owner: Switch** (the side that owns the lifetime, never the side that owns the caller); the allocator changes from *creating* to *receiving*. **§10.0.1 R12 — two instruments can each be correct about a different world, and a counter reading zero may be structurally incapable of reading anything else.** Specimen: `vulkan.cmd_upload` 15.2 s against `alloc_device_upload_bytes: 0`, both correct, different `VkDevice`s. **A quantity carries the identity of its frame; a counter whose event cannot occur in its frame reports `UNOBSERVABLE`, never `0`** — the fourth member of the family with `UNMEASURED`, `UNWIRED` and `SPLIT-DEVICE`, because **every way of not knowing gets a name a machine can print, since prose is where knowledge of a caveat goes to die.** **No criterion may name a pinned instrument**, so residency is read against `device_upload_bytes` on the session's device, never `alloc_device_authoritative_spans`. **R12 is not R11**: R11's remedy is available to the writer, R12's is structural. **Third disclosure obligation added — frame provenance — plus a seventh, positive one: independent corroboration is stated, not reconstructed** (Switch's span-derived 98.0% and Tank's counter-derived 95.8–98.4% for one quantity). Corrected picture, selector 0 = RTX 4060: **`cmd_upload` is ~71% of wall**, GPU kernels ~15%, and **pipeline lookup 0.4% / descriptor allocation 0.3% kill both standing hypotheses about per-island recording cost**) · *prior revision 2026-07-30T20:58:11-07:00* (**§10.0.1 R11 — a measurement's name is not its definition, and a decomposition that appears to close is the hardest kind of wrong.** R10's companion, found within hours of R10 by a specimen R10 certifies: `Phase::Record` is wired, invoked, correct, input-varying — **and misnamed by ~50×**. It is an *inclusive* interval containing the host staging copy, which reports into `phase_us[Upload]` and emits no `ph:"X"` span, so a span aggregation structurally could not see it: **upload is 95.8–98.4% of the "recording" phase; real command-buffer recording is 1–3% of wall.** The dominant cost is **the EP re-uploading the entire weight set on every inference** — 1997.6 MiB/inference, ratio **1.0002** against `device_upload_bytes`, exactly linear over 1/2/3 runs, in:out **2481:1**, transfer ceiling **~94.8% of wall discrete / ~44.0% UMA**. **The old table summed to 99.0% and appeared to close — because the missing cost was *inside* a row, so the residual was zero by construction.** R11 obliges: **every phase declares its extent (inclusive or exclusive of children); a flat table is an assertion of disjointness; the parts are summed against a whole measured by a *different* instrument and the residual published; and any row above 50% has its name checked against its content.** The register now reads **R6 our tooling manufactured a number · R7 a negative · R9 sound instruments jointly silent · R10 never called · R11 called, correct, and misnamed** — R11 is the hardest because *every check we have passes*. **The M0 tally does not move: six met, four partial, two not met, of twelve** — criterion 12 is *strengthened* rather than reopened, which is the whole benefit of its having been left open. **§10.0's disclosure obligation stands as written and is strengthened**: the phase decomposition was wrong by 50× while the wall-clock ratio (3.1× / 3.7× slower than CPU) was correct, **because the ratio has no internal structure to misattribute** — *a metric's robustness is inversely proportional to the number of naming decisions between the measurement and the reader; decompose to diagnose, report the coarse invariant.* A decomposition may accompany the ratio, never replace it, and is publishable only with its identity check. **R6 amendment 4 — the device labels were inverted team-wide**: `enumerate_capable_devices()` sorts best-first and `select_device` indexes the sorted list while `epctl --probe-loader` prints unsorted enumeration order — **`DEVICE=0` is the RTX 4060, `DEVICE=1` is the Iris Xe**, the "Intel beats the discrete GPU" finding dissolves, and **a result surprising enough to be a discovery is first a reason to check the instrument**. M1's lead performance criterion is corrected to **weight residency** — `device_upload_bytes`/inference below 1% of constant-initializer bytes, admissible only at or above last-published coverage with `MATCH` — with recording amortisation demoted to secondary) · *prior revision 2026-07-30T19:05:03-07:00* (**§10.0.1 R10 — a mechanism that exists in the source tree and not in the call graph is indistinguishable from one that was never written, and review cannot tell them apart.** R9's blind spot: *a falsifier that is never invoked is indistinguishable from one that never fires*. Five specimens in one day, all with correct code — `ops/partition.rs` (worth **3.7×**: islands 321 → 33, Intel 2954.6 ms → 807.2 ms when wired), the tracer, `model_output_equivalence`, `retain_viable`, and the EP-side validation messenger (loaded layer, no listener). **The falsifier for "X is wired" is an observation of an artifact X produced whose content varies with X's input — never a reading of X's code, never a flag its author set.** Uninvoked reports **`UNWIRED`**, distinct from empty; **the identity case is an explicit red state** (`island_count == claimed_count` was one line and was true for the whole life of the defect); **wiring is a property of an entry point, not of a file**; **review of a mechanism is not complete until the reviewer has seen an artifact it produced.** **§7.0.2 companion — a claim is a scheduling decision, not a capability statement: a correct claim can be a wrong claim**, net benefit is a property of the op *in a graph at a coverage level*, it lives in the partitioner and never in the registry, and it carries its own decline code. **M0 criterion 10 is MET — `model_output_equivalence = MATCH` on both devices** (argmax 30751 == CPU, top-10 10/10, max diff 0.031 / 0.035; **the non-identity is the correct answer** for fp16 accumulation order). Root cause was **binding arity, not dtype** — a 4-entry pipeline layout against a shader writing binding 4, silently dropped by both drivers — which is the strongest possible vindication of §8.9's `populated_optional_input_set` key component. **Criterion 2 closed on the promise made when it was reopened; criteria 4 and 5 stay partial — a correct model does not give an unknown-polarity check a polarity. Criterion 12 added (wiring census). Six met, four partial, two not met, of twelve.** **RULING: no performance criterion belongs in M0** — slowness is loud, wrongness is silent, and the cheapest way to pass a ratio criterion is always to do less GPU work; **M2 keeps the first threshold** (end-to-end ratio `< 1.0` on one discrete GPU with `MATCH`, every device in the matrix reported). **M0 gains a §10.0 disclosure obligation instead of a gate: the end-to-end wall-clock CPU ratio may never be omitted — currently 3.1× slower on Intel, 3.7× on NVIDIA.** **M1 gains a recording-amortisation criterion** checked with a counter before a clock, on the first honest trace: **68.3% command-buffer recording, 14.1% GPU kernels, 0.3% submit — 85.9% of runtime with no GPU work happening**, fixed per `Compute` call rather than per dispatch, which falsifies the fixed-per-submission hypothesis. **Sequencing: the tail resumes at the front; the 68.3% starts in parallel as Tank's M1 work and lands only through criterion 10's gate**, because a cached binding table is exactly the shape of today's defect) · *prior revision 2026-07-30T06:32:18-07:00* (**§8.9 RULING — unproven is a claim-path state; claiming is gated on evidence and `Live` stops being a thing we write down.** §7.0.1 companion: *evidence shortfalls degrade op coverage, not device availability, identically to capability shortfalls* — the frozen §7.2 gate is untouched. The table declares only `Staged(why)` / `Ready`; **claimability is derived per form from a harness-generated proof ledger**, keyed on `(domain, op_type, opset_bucket, every input/output dtype, kernel_variant_key, shape_class, populated_optional_input_set)` — so **§8.7's expression-vs-path distinction becomes mechanical: an expression difference leaves the key equal, a path difference changes it**, and an f32 proof can never be returned for an f16 node. **A `DIVERGENT` model verdict demotes every form that participated, automatically.** Escape hatch is **a list of proof keys and nothing else** — no `1`, no `*`, no wildcard, C1's shape — with WARN at session creation, `unproven_forms_enabled` in the counters artifact, and `epctl --check-counters` failing on a non-empty list. **Honest cost: Phi-3.5's claimed count goes 161 → 0** — and per §10.0's gate that 161 was already void. **M0 criterion 11 added; four met, four partial, three not met.** **Link's lanes: the gate is a precondition for a lane being declared *green*, not for a lane being brought *up*** — with a per-lane **gate artifact** rather than Phi-3.5 on a rasteriser) · *prior revision 2026-07-30T05:48:29-07:00* (**§10.0.1 R9 — a set of individually sound instruments can be jointly silent on the property that matters; *for every claim, name the instrument that would go red if the claim were false*.** Phi-3.5 on both devices: 161 `MatMulNBits` claimed **and accepted by ORT**, `compute_failures: 0`, `dispatches_executed: 161`, suite green — and `vk argmax 0` against `cpu argmax 30751`, top-10 overlap 0/10. **§9.1.3 RULING — `compute_failures` is an execution-status counter and may never be read as a correctness signal**; prose cannot close that reading, a verdict emitted next to the counters must. **Metric of record gated on `model_output_equivalence` ∈ {MATCH, DIVERGENT, UNMEASURED}**, default `UNMEASURED`. **M0 criteria amended: criterion 10 added (model-level correctness); criteria 2, 4 and 5 REOPENED; criterion 8 relabelled parity-only.** **Sequencing: criterion 10 outranks the Windows/Linux/lavapipe/CI tail in order, and does not replace it as a gate**) · *prior revision 2026-07-29T21:14:03-07:00* (**§8.8 RULING — dynamic shapes are a claim-path capability, not a kernel feature**, and move **ahead of** the three kernels, §10.0.3; measured on the first end-to-end real-model run: **258 nodes declined on symbolic shapes vs 100 on missing kernels**, and the decline codes are first-match so 258 is a *floor*; **§1.2's dynamic-shape non-goal reversed**; **M1 gains a second-token exit criterion** — one session, two concrete values of a symbolic dimension; **OQ-15 promoted to blocking**; **§10.0.1 R8 — we planned against the ops a model contains, having never measured why its nodes are declined**) · *prior revision 2026-07-29T19:42:07-07:00* (**M0 criterion 8 MET — both barrier backends executed, bit-exact on two vendors**; **45 op rows `Live`**; **criterion 3 ruled not discharged — a validation lane needs a positive control**; **criterion 9 not met** — `PLATFORMS.md` LVP2 still carries §7.2's false premise; **§10.0.1 R7 — our instruments fabricate negatives**, *derive, do not declare*; **§8.7 template evidence covers a different expression, never a different path**) · *prior revision 2026-07-29T16:00:55-07:00* (**`Add` executes through ORT — M0 criterion 2 MET**; **§7.2's R5 rationale corrected**, re-grounded on §7.0; **§10.0.1 R6**; criterion 8 amended so a skip cannot satisfy it) · *prior revision 2026-07-29T15:02:55-07:00* (**§8.5 third strengthening**; **metric triple `(coverage, island_count, largest_island_flops)`**; **T3 demonstration target is Phi-3.5**; **R5**) · *prior revision 2026-07-29T09:47:45-07:00* (**first shader dispatch**; **M0 assessed criterion by criterion**; **§7.9 capability probing**; §8.5 *producer **at version***; R4) · *prior revision 2026-07-29T08:13:58-07:00* (**§8.5 producer-relative**; **§8.6 crate evaluations**; **§10.0.2 `ai.onnx::Attention` first**; R1 narrowed + R3) · *prior revision 2026-07-28T22:28:08-07:00* (**OQ-4 §7.8**; **OQ-M6 ruling** §8.4; **OQ-3 §6.4** reserved-VA, no BDA; C2 **item 7**; `retain_viable` §5.4; eleven contrib ops; OQ-16; **§9.1.1 oracle validated**; **R1**)
**Author:** Morpheus (Lead / EP Architect)
**Repo:** `onnxruntime-ep-vulkan`
**Reference architecture:** `onnxruntime-mlx` (Justin Chu's MLX plugin EP for Apple Silicon)
**Sibling documents:** [`ENGINE.md`](./ENGINE.md) (Switch — Vulkan runtime & shaders), [`PLATFORMS.md`](./PLATFORMS.md) (Link — platform & hardware matrix), [`OP_COVERAGE.md`](./OP_COVERAGE.md) (Mouse — **authoritative op-coverage plan**, ratified), `THIRD_PARTY.md` (Rai — licence compliance)

---

## 0. TL;DR

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
   not silently change the meaning of a field five consumers already parse.

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
  "verdict": "MATCH | DIVERGENT | UNMEASURED | UNATTRIBUTED",
  "executed_by": { "VulkanExecutionProvider": 1, "CPUExecutionProvider": 10 },
  "attribution_source": "ort_profile",
  "attribution_witnesses": { "profile_node_events": 1, "counters_dispatches_executed": 354 },
  "artifact": "<producer-at-version + file digest>",
  "device_index": 0, "device_name": "<physical device>"
}
```

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

**Where R13 sits.** R6: our tooling manufactured a *number*. R7: it manufactured a *negative*. R9:
sound instruments, *jointly silent*. R10: *never called*. R11: called, correct, *misnamed*. R12:
called, correct, correctly named, *about another world*. **R13: called, and its outage is spelled
the same way as its finding — plus a reader who checked the spelling only when he disliked it.**

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
| 10 | **Real model at producer-at-version, non-zero claimed count, attributed `model_output_equivalence = MATCH` against a CPU-only run of the same session, over N >= 3 consecutive inferences** | **REOPENED 2026-07-31T07:45:10-07:00 — partially met** | **The evidence that closed this row is void, and the evidence that replaces it is genuine and incomplete.** Void: the 2026-07-30 `MATCH` on both devices came from runs in which ORT had already fallen back to CPU inside `run()` — this EP executed zero nodes, so the verdict was `UNATTRIBUTED` under §10.0's third amendment, not `MATCH`. Genuine: after Switch fixed `Allocator::alloc(size=0)` returning `None` for Phi-3.5's `[1,32,0,96]` KV-cache inputs, **ORT's own profiling — an instrument we do not own — reports 1 `VulkanExecutionProvider` node event (one fused island of 354 nodes) and 10 `CPUExecutionProvider` events, exactly Mouse's ten declined edge ops, with `logits argmax 30751` matching CPU.** That is the first attributed execution this project has ever recorded and it is worth more than the reading it replaces. Incomplete: **the multi-run picture is red** — the weight cache exhausts device memory across runs (`gpu-allocator failed to allocate 14155776 bytes: Out of memory`, followed by silent CPU fallback) and 50 KV-cache outputs are never written, giving cross-run divergence on a dirty arena. **I am reopening this on the voidness of the old evidence and the incompleteness of the new, not on the badness of the news** — the test of that claim is cheap and I am stating it in advance: **when one session produces three consecutive attributed `MATCH` runs on this artifact, this row closes the same day, with no new conditions.** Owners: Switch (arena lifetime and the 50 unwritten outputs), Trinity (the attributed verdict constructor and the N >= 3 series) |
| 11 | **No form claimed without a ledger entry under its proof key; no build silently claiming unproven forms** (§8.9) | **Not met — scaffolding only. Reverted from MET 2026-08-02T00:15:00-07:00 (Morpheus, over Mouse's write-up; the tally is not the artifact-supplier's.)** | The ledger exists, is generated, and is consulted. Artifact: **`evidence/proof_ledger.jsonl`, 9 entries, header digest `331003e0ff88df3f` (was `e4436e93c19c8744`; regenerated 2026-08-02 when provenance became mandatory per entry)**, produced by `rust/tools/gen_proof_ledger.py` and baked in with `include_str!` so a build cannot claim a form whose proof is absent from the binary doing the claiming. The census line that replaced `UNWIRED`: **`ledger_lookup: ALL-PROVEN proven_key_lookups=6 ledger_hits=6 ledger_entries=9 unproven_declines=0 unproven_forms_enabled=[]`**, and `hits` is **typed** — `'UNOBSERVABLE'` / `'UNWIRED'` / `int` — so R12's three states cannot collapse into one `0` the way *bypassed* and *all-rejected* once did. **Rai's planted control runs in the lane, not behind `#[ignore]`**: `mul_f16_unproven` is deliberately never proven while its sibling `mul_f32` is, so the pair is two-armed and the arms are asserted to differ (`distinct_forms_have_distinct_keys`); the unproven arm yields `ledger_hits=0 ledger_gate=ALL-DECLINED unproven_declines=1`. **§8.7's expression-vs-path distinction is now mechanical** and the key is vindicated by a real defect that is *in the ledger as a pair*: `MatMulNBits` with vs without `zero_points` — `.../f16,u8,f16>f16/.../scales` and `.../f16,u8,f16,u8>f16/.../scales+zero_points` — different `populated_optional_input_set`, therefore different keys, therefore the 2026-07-30 all-zero-logits proof could never have been returned for the other form. **The price was paid and not softened: Phi-3.5's claimed count is 355 → 0**, exactly the number Morpheus accepted when he ruled this, and predicted in writing before the run (`bench/results/proof_ledger_prediction.json`, P4, confirmed exactly). **The fall is temporary**: the 355 nodes reduce to **8 distinct proof obligations**, mechanically discoverable because every claim-log audit line now carries `proof_key`. **Two real defects the controls caught while landing this**, both recorded because a mechanism that finds nothing on its first day is the one to distrust: (i) proof keys contain `,` and the `CLAIM_UNPROVEN` hatch split on `,`, shredding every key — the list was correctly discarded, the run claimed nothing, and the comparison still said `MATCH` because it was CPU-vs-CPU; only the **attribution** requirement caught it, separator now `;`; (ii) the regression test for (i) found `ai.onnx::Add/7+/f32` *passed* `ProofKey::validate` — a truncated key matches nothing and reads like a key that matches something. One `ERROR(instrument)` and it was never a detection (R13): `sqrt_f32` returned `DIVERGENT` with `worst_rel: 0.0` because `standard_normal` inputs made `Sqrt` NaN on **both** sides — fixed with an `ERROR` verdict for non-finite *reference* output plus an `INPUT_DOMAIN` table. Escape hatch is a list of keys and nothing else — no `1`, no `*`, no wildcard — with a session WARN and `unproven_forms_enabled` in the counters artifact. **WHY THE ROW IS OPEN DESPITE THE ARTIFACT EXISTING.** *“The cheapest satisfaction is a ledger generated from the claim table — derive the ledger from the same enumeration that produces the claims and the criterion is true by construction, `ledger_hits == proven_key_lookups` forever, and the check can never fail. That is an identity whose two sides come from the same source, and `6/6` looks identical under both readings.”* (Morpheus). The shipped ledger is **not** that shape — it is produced by `rust/tools/gen_proof_ledger.py` from executed differential runs — but nothing in the artifact *distinguished* the two shapes, and a reader could not tell them apart. **Four discharge conditions, with owners:** **(a) provenance — DONE 2026-08-02 (Mouse).** Every entry records the witnesses of a proof run that the claim table cannot produce: `claimed_nodes`, `dispatches_executed`, `worst_rel`. A dispatch count exists only after a session executed; an enumeration cannot forge one. Enforced on **both** sides and in the same direction — `gen_proof_ledger.py` raises rather than writing an unattributed entry, `--check` fails on one, and `parse_ledger` **faults** it, so it grants nothing. *Absent is treated exactly like zero*, and a **quoted** count is treated like absent, because a writer that stringified its counters did not read a counter. Control: `an_entry_without_attribution_proves_nothing_however_well_formed` — four ledgers differing only in the attribution fields, four different outcomes (R10); mutation-tested red at *“a run that dispatched nothing proves nothing, whatever it compared”*. Plus `every_shipped_ledger_entry_carries_its_proof_run` over all 9 shipped entries. **(b) three planted controls in the lane.** (i) `mul_f16_unproven` — **in the lane since 2026-08-01**, never `#[ignore]`. (ii) a key differing only in `opt_inputs`/`shape` — the `MatMulNBits` `zero_points` pair, **in the ledger** and asserted distinct by `distinct_forms_have_distinct_keys`, which doubles as the regression test for the 2026-07-30 all-zero-logits defect. (iii) **a build whose baked digest disagrees with the ledger on disk refuses to claim rather than warning — DONE 2026-08-02 (Mouse).** `ONNXRUNTIME_EP_VULKAN_LEDGER_FILE` names the on-disk ledger; a digest disagreement, **or a named file that cannot be read**, is pushed into `Ledger::faults`, and non-empty faults makes every lookup return `Faulted`, so **every form declines**. A WARN would leave the run claiming from evidence nobody can read. **This is a second, distinct threat from the header-vs-body digest**: that one catches a hand-edit *before* the build; this one catches the file changing *after* it, which is the case where the artifact a reviewer reads is not the artifact the binary claimed from. R9 amendment 5 — the check moves **against** the reader's confidence: a mismatch can only remove claims, never add one, which is why it cannot be repaired by tightening and why it is safe for it to be strictly optional. Control: `a_disk_ledger_that_disagrees_with_the_baked_one_refuses_to_claim`, three arms (identical → no fault; one line added → fault naming **both** digests; named-and-absent → fault), mutation-tested red. **(c) `ledger_hits` shown to move with its input** — Trinity's tally. Open. **(d) a three-token miss path (R13) — DONE 2026-08-02 (Mouse).** *Key absent from the ledger*, *ledger failed to parse or its digest disagreed*, and *key never attempted* are **three findings with three different repairs** — regenerate this form, fix the ledger file, and nothing at all — and a `bool` spells all three `false`. `LedgerLookup::{Hit,KeyAbsent,Faulted,NeverAttempted}` with a token apiece; `record_ledger_lookup` now takes the outcome rather than a `bool`; the counters artifact carries `"ledger_miss"`. Precedence is R13's order — **`LEDGER-FAULTED` outranks `KEY-ABSENT`**, because a run with no reading about any form must not spell an instrument outage the way it spells a detection. `NeverAttempted` is *derived* from `proven_key_lookups == 0` and is never counted, since recording it would be a lookup, which is exactly what it asserts did not happen. Control: `the_ledger_miss_token_names_which_of_three_things_happened`, four states driven, four tokens asserted distinct. Owner: Mouse ((a), (b)(iii), (d)); Trinity ((c) and lane membership) |
| 12 | **Wiring census: every mechanism this table relies on is observed to have run; a mechanism with no observation reports `UNWIRED`** (§10.0.1 R10) **— plus extent, the decomposition identity against an independent whole, and the name–content check** (§10.0.1 R11) | **Not met — added 2026-07-30T19:05:03-07:00, amended 20:58:11-07:00** | The census, the identity checks, the extent declarations and the lane assertion. **Amended within four hours of being written, by a specimen it would have certified** — `Phase::Record`, wired, invoked, correct, input-varying, and misnamed by 50×. **The tally does not move and that is the whole benefit of the row having been open**: a criterion strengthened while it is still unmet costs nothing and retracts nothing. Had I recorded it met this morning I would be reopening it tonight, on the seventh consecutive day of reopening a met criterion. **AMENDED AGAIN 2026-07-31T07:45:10-07:00, again by a specimen it would have certified, and this time the specimen is a mechanism whose own outage the census would have recorded as an observation.** Two additions: **(g)** every mechanism's census line carries the **frame** it observed in — for `model_output_equivalence` that is `executed_by` (§10.0 third amendment); a census that reports a verdict without its executor reports a value from a world it has not identified; and **(h)** the census distinguishes three states per mechanism, `OBSERVED` / `UNWIRED` / `INSTRUMENT-ERROR` (R13), because a census whose vocabulary is *observed or not observed* records a crashed mechanism as an absence and an absence as a crash **CENSUS RUN AND WITNESSED 2026-08-01 (Trinity)**: `bench/results/wiring_census-dev{0,1}.json`, from `tests/ops/test_wiring_census.py::test_wiring_census`, now reports twelve mechanisms and every line carries a value that mechanism computed on the run. Added this round: **`net_benefit_gate`** — `EVALUATED clusters_seen=1 evaluations=1 bypasses=0 sole_island_overrides=1 viable_islands_retained=0`, the three states that used to share one `0` now separate fields (R12, RAI-011); **`broken_commitment_warn`** (Tank) — read from two counters children differing only in `ONNXRUNTIME_EP_VULKAN_FORCE_COMPUTE_FAILURE`, planted `channel='ORT_SINK' broken_commitments=1 fault_injection='ACTIVE' ort_sink_warn_lines=1` against clean `channel='UNOBSERVABLE' broken_commitments=0`, and a reading that did **not** move between the two is reported `UNWIRED` however green it looked; **`device_state_guard`** (Link) — his `ci/check_device_state.py` imported and run over two inputs, a planted companionless duration returning `FAIL(condition=STEADY_UNCERTIFIED)` exit 1 against this run's own evidence; and **`instrument_census`** — Tank's `rust/tools/audit_instruments.py` via `main_guarded`, `CENSUS VERDICT: PASS`. **No second census was built**; his six states (`absent → uninvoked → unfalsified → unreachable → out-of-frame → misnamed`) and R13's three terminals are the vocabulary throughout. **A screen defect of the same family as Link's ICD probe was found and fixed in the census itself**: the `gpu_tracer` line read `ONNXRUNTIME_EP_VULKAN_TRACE_FILE`, a variable nothing defines (`trace.rs::ENV_TRACE` is `ONNXRUNTIME_EP_VULKAN_TRACE`), so it had reported `OPTIONAL-UNWIRED` on every run it ever made and would have done so had the tracer been deleted — an always-false screen and an always-true screen are equally blind. It now arms the tracer itself and reports `28 trace event(s), phases=['C','M','X','i'], distinct_names=16`. **No mechanism reports `UNWIRED` as of 2026-08-01T21:15:16-07:00.** The last one, `ledger_lookup`, was closed by criterion 11 (Mouse) and now reads `ALL-PROVEN proven_key_lookups=6 ledger_hits=6 ledger_entries=9 unproven_declines=0 unproven_forms_enabled=[]`; the `xfail(strict=True)` on `test_ledger_lookup_wired` was **replaced by assertions rather than deleted**, because an expectation that is merely dropped leaves no record that the thing it expected has happened |

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
| 2 | **Net-benefit declines** — `retain_viable` in the partitioner | Wired by Mouse. Rests on the island-count decomposition (321 → 33), a **count**, not a clock |
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
