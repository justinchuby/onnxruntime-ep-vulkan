# Squad Decisions

<!-- Round 1: 2026-07-28T17:59:54-07:00 (design session)              → ARCHIVED -->
<!-- Round 2: 2026-07-28T22:28:08-07:00 (implementation round)        → ARCHIVED -->
<!-- Round 3: 2026-07-29T09:00:39-07:00 (first-hardware round)        → ARCHIVED -->
<!-- Round 4: 2026-07-30T19:05:03-07:00 (correctness-and-measurement) → ARCHIVED -->
<!-- Round 5: 2026-08-01T09:53:14-07:00 (execution-and-instrumentation) → ACTIVE -->

## Active Decisions

<!-- ═══════════════════════════════════════════════════════════════════════════════ -->
<!-- ARCHIVAL POINTER — 2026-08-01T09:53:14-07:00                                  -->
<!-- Round 4 (26 entries, the full correctness-and-measurement round) archived to:  -->
<!--   .squad/decisions-archive/2026-08-01T09-53-14-07-00-round4.md                -->
<!-- Trigger: decisions.md was 45,337 bytes (volume threshold: 40 KB) before this   -->
<!--   run's merge, and the 47-record inbox batch below is far larger than Round 4. -->
<!-- Floor: Round 4 was written 2026-07-30T19:05:03-07:00, ~38.8 hours before this   -->
<!--   run — past the 24-hour floor. Boundary: the Scribe-run boundary (this run's   -->
<!--   Round 5 marker), not a calendar date, per established policy.                -->
<!-- Supersession (why archiving all of Round 4 is safe): several Round 5 entries    -->
<!--   below correct Round 4 claims by name — device labels inverted (corrects the   -->
<!--   Round-4 "Intel beats the 4060" reading), Phase::Record misattributed ~50x     -->
<!--   (corrects the Round-4 "68.9% recording" figure), the fixed-per-submission     -->
<!--   hypothesis (dead, was never asserted as Round-4 fact but is now retired by    -->
<!--   name), and every wall-clock figure including 3.1x/3.7x (withdrawn). The       -->
<!--   corrections are all in Round 5 and stay live; only the things they correct    -->
<!--   move to archive — never the reverse.                                        -->
<!-- Previous archival check (2026-07-30T19:05:03-07:00): 0 entries, age gate only  -->
<!-- Previous archival (2026-07-30T20:58:11-07:00): Rounds 1-3, 122 entries         -->
<!-- ═══════════════════════════════════════════════════════════════════════════════ -->

<!-- =============================================================================== -->
<!-- ROUND 5 DECISIONS -- 2026-08-01T09:53:14-07:00 (execution-and-instrumentation)  -->
<!-- 47 inbox records merged (48 expected per spawn manifest; verified 47 present --  -->
<!--   see Scribe health report); consolidated into 18 entries below where two or    -->
<!--   more records described one event from independent angles (R9: corroboration,  -->
<!--   not redundancy -- both angles preserved rather than one discarded).           -->
<!-- =============================================================================== -->

### 2026-08-01T09:53:14-07:00: §6.5 closed on both selectors — exactly one VkDevice per (physical device, EP instance) (consolidated)

**By:** Switch (`two-vkdevice-flag`, `defects-1-2-3-and-offer-gate`, `sec65-closed`, `index-space-unified-by-single-offer`, `index-spaces-closed`), Tank (`frame-provenance`), Morpheus (ruling D-M26-03)
**Why consolidated:** one architectural saga told from three angles — the original flag raised by Switch, the ruling by Morpheus, the consumer-side interface built by Tank, and Switch's own three-stage implementation (gated → unconditional → both selectors fixed). All describe one closure.

**What:** Two independent Vulkan logical devices existed per process (the compute session's and the allocator's device-memory provider's) with no shared ownership plan — "a design decision nobody made deliberately." Morpheus ruled (D-M26-03) exactly one `VkDevice` per (physical device, EP instance); seam owned by Switch (owns the lifetime), Tank as consumer. Switch built `acquire_ep_device` returning a `&'static EpDeviceOwner`, deliberately leaked (`Box::leak`), so `VulkanSession` borrows rather than owns; three sequential sessions in one process now survive (previously a multi-session UAF: the offered device was session-scoped and destroyed at teardown while `HandleRegistry` is process-global — the fix was "stop destroying the device," not "stop offering it"). Tank's consumer-side surface: `SharedVkDevice` trait + `offer_shared_device(device_index, ctx)`, called once from `CreateEp` before the first ORT allocation, `Arc` retained for process lifetime — cross-owner diff on `vk/session.rs`/`instance.rs`/`cmd.rs`/`device.rs`/`trace.rs` is **empty by construction**.

**The two-selector wrinkle:** §6.5 first closed only on selector 0 (NVIDIA); selector 1 (Intel) still reported `SPLIT-DEVICE` because the offer was keyed by the raw `vkEnumeratePhysicalDevices` index while `ONNXRUNTIME_EP_VULKAN_DEVICE` indexes the best-first sorted list — a third instance of the two-index-space defect (after `epctl --probe-loader` and `dispatches_executed` vs `compute_calls`). Switch's first fix attempt made ORT's bound device authoritative but this **silently relocated** a run onto the wrong physical device while still reporting `MATCH` — "a silently relocated run is an unattributed result wearing a MATCH," caught only because the timing shape didn't match. Reverted. Final fix: `devices_to_advertise()` treats a pinned `ONNXRUNTIME_EP_VULKAN_DEVICE` as advertising **only** that device, so ORT cannot bind a device not advertised — the two spaces become one rather than being translated between. Verified on both selectors: `alloc_device_frame: SHARED`, correct device name per selector, `alloc_device_authoritative_spans: 0` (int, not the `"UNOBSERVABLE"` string) on both.

**Falsifier:** unit test built on the **inverted** pairing (discrete = enum 1 / selector 0 on this desk), so an identity-mapping regression fails it rather than passing silently.

---

### 2026-08-01T09:53:14-07:00: `alloc_device_authoritative_spans` — UNOBSERVABLE → UNWIRED → measurement, and R12's fifth census state (consolidated)

**By:** Tank (`frame-provenance`, `tank-residency-screen-and-r13-ruling` / D-T85), Switch (`leaked-device-validation-unobservable`), Morpheus (ruling, R12)
**Why consolidated:** the same counter's three-state life story, told once by the author who moved it through each transition, framed by the taxonomic rule that makes the transitions meaningful.

**The counter's three states, each a different JSON type so arithmetic on it fails loudly:** `"UNOBSERVABLE"` (str, R12 — the event cannot occur in this run's frame, e.g. `SPLIT-DEVICE`); `"UNWIRED"` (str, R10 — frame allows it but the increment point has no production caller, `evaluations == 0`); measured `0`/`n>0` (int). D-T85: `HandleRegistry::free` is the only point a span's residency state is terminal; `on_residency_evaluated` increments an unconditional evaluations counter plus a conditional authoritative counter from the same call site, so the count **cannot be forged** — only the unconditional twin moves without it. Measured: selector 0 → `authoritative=0 (int), evaluations=3` → genuine measurement; selector 1 → still `UNOBSERVABLE` even though `evaluations=3`, because **R12 outranks R10** (a wired instrument in the wrong frame is still not a measurement). `alloc_device_buffer_binds` remains 0 — the engine still resolves inputs through `host_backing_for` rather than binding the mirror buffer, so no honest implementation can report non-zero authoritative count yet; that is Switch's, not a counter change.

**R12 fifth census state — `out-of-frame`:** distinct from `unreachable`. The four original states (absent/uninvoked/unreachable/misnamed) are each repaired by their author touching their own file; `out-of-frame` has **no author-side repair** — you cannot fix this counter by editing `allocator.rs`; the repair is §6.5, in a different file. Tank re-derived the coordinator's proposed `quarantine_retired` specimen and found it is actually `unreachable` (event occurred, only the *reading* path was missing an API call), not `out-of-frame` (event structurally cannot occur) — "they look identical from a reader's chair and are opposites in the workshop." Ordering: `absent → uninvoked → unreachable → out-of-frame → misnamed`; R10's screen catches the first two mechanically, nothing but a second independent instrument catches the last two.

**Switch's companion finding:** "0 validation errors at shutdown" on the *production* path is itself `UNOBSERVABLE`, never `0` — the EP's `VkDevice` is deliberately leaked (never destroyed) per §6.5, so the validation layer's leak-report frame (which fires at `vkDestroyDevice`) never runs on production. Criterion 3 must not be certified by that gate; `ep_messenger_fires_for_planted_fence_leak` remains the correct positive control because it owns and destroys its own throwaway device.

---

### 2026-08-01T09:53:14-07:00: Device labels were inverted — three independent discoveries (consolidated)

**By:** Tank (`transfer-bound`, Defect A), Niobe (`phase-split-and-two-defects`, defect 4), Switch (corroboration via `bench/devices.py` fingerprinting in `kernel-spread-reconciliation-niobe`/`intel-gap-separated`), Morpheus (ruling D-M25-03, R6 amendment 4)
**Why consolidated:** Per R9, three independently-authored routes converged on one root cause — this is evidence, not restated redundancy, and destroying two of the three angles would destroy exactly what makes it trustworthy.

**Root cause:** `enumerate_capable_devices()` sorts best-first (discrete > integrated), and `ONNXRUNTIME_EP_VULKAN_DEVICE` indexes *that* sorted list, while `vulkaninfo`, `epctl --probe-loader` and raw `vkEnumeratePhysicalDevices` order use **enumeration** order — two index spaces under one label. On this desk: selector 0 = NVIDIA RTX 4060 (discrete), selector 1 = Intel Iris Xe — the opposite of every label the team had quoted, including the "Intel beats the 4060, 807.2 vs 1156.0 ms" finding, which dissolves once corrected (and was never quotable anyway, being a cross-device comparison with no quiescence record).

**Tank's route:** confirmed physically — the coordinator's own island result becomes plausible once corrected — and fixed his own consumer-side bug (Defect B: the device-backed memory mirror had independently landed on the wrong physical device via the same index confusion, one level up from the label bug).
**Niobe's route:** built `bench/devices.py::device_identity_check`, which labels each result row from the **timestamp fingerprint in that row's own trace** (Intel 52.0833 ns/36-bit; NVIDIA/lavapipe 1.0/64-bit) rather than from the selector index — "a label must travel with the evidence, not beside it." This is durable: it separately caught that a blanket re-label of already-correctly-labelled `bench/results/` files would have inverted correct rows, while `docs/PERF.md` prose (written before the check existed) was genuinely wrong and got fixed.
**Morpheus's ruling (R6 amendment 4):** filed under R6, not a new rule — "our own tooling manufactured the number." New standing rule: *a result surprising enough to be a discovery is first a reason to check the instrument; surprise is a free instrument check, and we spent it on celebration.* Owed: `epctl --probe-loader` must print the selection index alongside enumeration order, or neither.

---

### 2026-08-01T09:53:14-07:00: `Phase::Record` misattributed the upload cost by ~50x — R11 named (consolidated)

**By:** Niobe (`phase-split-and-two-defects`, `niobe-name-falsifiers`, `phase-containment-was-mine`), Switch (`phase-containment-niobe`, retracted claim in `weight-cache-recording-bottleneck`, `measurement-contamination`), Tank (`instrument-census`, misnamed finding #8), Morpheus (ruling D-M25-01, R11)
**Why consolidated:** the same defect discovered, fixed, and ruled on by four people in one day — the "68.9% command-buffer recording" figure that was actually 96-99% host memcpy.

**What:** `Phase::Record` brackets `vkBeginCommandBuffer`→`vkEndCommandBuffer`, and the host staging memcpy runs *inside* that window via `Tracer::record_transfer`, which reports through `phase_us[Upload]` but emits **no `ph:"X"` span** — so any span-based aggregation structurally cannot see it. Measured: upload is 95.8–98.6% of `record`; real command-buffer construction is 1–3% of wall, not 68.3/68.9%. Both Switch's and Niobe's independent instruments (span-derived vs counter-derived) agreed to within a few points — the corroboration itself is evidence. Switch additionally corrected `trace.rs`'s caveat ("host: command-buffer recording; amortised across replays"), which had been false since the weight cache landed and shipped inside every trace — every child of `record` was named, "which made the decomposition look closed," while ~93% of a warm `record` had no span of its own (the R11 case Switch found in his own trace).

**Niobe's phase_containment RED false alarm (same defect, opposite direction):** the containment checker went RED reporting "1 subgraph whose phases exceed their own duration" — but the defect was in `bench/phases.py` double-counting nested children (`cmd_upload`, `desc_alloc`, `pipeline_lookup`) as siblings of their own parent `record`, not in Switch's spans. Verified by five independent geometry checks before accepting blame (R13: confirming results get more scrutiny). Fixed with a two-tier check (siblings vs subgraph; children vs their own record) using R13's PASS/FAIL(condition)/ERROR(instrument) states.

**Morpheus's rule (R11):** *a measurement's name is not its definition, and a decomposition that appears to close is the hardest kind of wrong.* Distinct from R10 (never called) — here the artifact existed, was correct, varied with input, and was believed; the failure is at interpretation, not observation. New obligations for any phase/counter table: declare extent (inclusive/exclusive of children) at definition; a flat table asserts disjointness; check the decomposition identity against an independently-measured whole and publish the residual; any row above 50% gets its name checked against its content before quoting. Criterion 12 (wiring census) amended (not reopened) to add these three checks — it "would have certified `Phase::Record`" as written.

---

### 2026-08-01T09:53:14-07:00: Binding-arity + KV-cache + weight-cache-leak defects (1, 2, 3) closed on real Phi-3.5

**By:** Switch (`defects-1-2-3-and-offer-gate`, `kv-aliased-output-fix`)

**Defect 1 (weight-cache device leak, FIXED and wired):** `release_weight_cache` existed but had no caller — "a mechanism in the tree but not the graph." `VulkanSession::Drop` now drains every weight cache before dumping counters; predicted device high-water FLAT across N sessions (falsifier: 3 sessions ~3x one session's peak or OOM) — measured FLAT (~3.907 GiB after 3 sessions, same as 1), every device alloc freed. Original failure mode was **silent CPU fallback**, not a surfaced error — fifth instance on this project.

**Defect 2 (50 KV outputs never written, FIXED):** `bind_aliased_output`'s default impl ignored the `out` descriptor; `ShapeOnlyRecorder` had no override, so aliased KV outputs got a zero-byte placeholder and were silently skipped in `write_outputs_to_ort`. Root cause: empty-past decode feeds (`[1,32,0,96]`) sized the present-KV descriptor to zero. Fix (both files): `bind_aliased_output` gained a `desc: TensorDesc` parameter; `attention.rs::translate_gqa` gates on `empty_past` and binds present as REAL outputs when past is empty. Cross-run evidence: all 65 outputs bit-identical across 3 runs, differing-output count 0 (was 50).

**Defect 3 (`compute_calls=1` after 5 runs):** confirmed downstream of Defect 1 — once the OOM was fixed, `compute_calls == N` and `compute_failures = 0`, both single- and multi-session sweeps. "Confirmed by observation, not assumed."

**offer_shared_device index-space bug (fixed) + multi-session UAF (gated, later resolved — see §6.5 entry above):** offer was keyed on the sorted-capables selector index rather than the factory's advertised (raw enumeration) index; fixed, then found the offer's process-global consumer outlived per-session devices, causing `STATUS_ACCESS_VIOLATION` with ≥2 sessions — gated OFF by default until §6.5's architectural fix (later closed unconditionally, see above).

---

### 2026-08-01T09:53:14-07:00: Weight residency landed on bytes — and the halving that was already in the control (consolidated)

**By:** Switch (`weight-cache-recording-bottleneck`, `cb-cache-prediction`), Tank (`staging-bytes`, `transfer-bound`), Mouse (`transfer-recalibration`), Morpheus (ruling D-M25-05/D-M26-06)
**Why consolidated:** Switch built and measured the cache; Tank independently instrumented and confirmed the byte identity from the allocator side; Mouse then measured a further claimed halving and caught himself before publishing someone else's win as his own.

**The cache:** per-subgraph `HashMap<(cpu_ptr, byte_size), GpuBuffer>` in `VulkanSession`; tensors ≥32 KB promoted after first fence-signal; subsequent calls serve `borrowed_ref` handles — no re-upload. Per-inference upload fell from ~1997.6 MiB (full weight set every call) to steady-state **0.755–0.817 MiB** (Mouse's later, more precise byte-exact figure), a **~2646:1** ratio, flat/linear across 1/2/3/5-run sweeps on both devices (Tank's independent confirmation via `session_staging_upload_bytes`/`alloc_device_upload_bytes`, agreeing to 1.0002x with a separate allocator-side accounting). CB-caching (command-buffer reuse) was predicted and measured to buy <1% further gain once the weight cache landed, because most descriptor bindings (activations) still change every call — not implemented this session on that basis.

**Mouse's correction (the halving that wasn't):** after claiming `SimplifiedLayerNormalization` + `Gather`, Mouse measured per-inference upload at 0.38 MiB and was "one step from reporting a halving as the payoff of my work." Built the pre-change control commit and re-ran the same instrument: the halving (1997.6→0.756 MiB) was **already present in the control** — it belongs to residency (Switch's work), not op coverage. Mouse's own two claims account for exactly 6,136 bytes/inference, not the halving. "Had I skipped the control I would have published a 2x that was someone else's and mine only by accident" — R13's confirming-result-gets-more-scrutiny clause made concrete. Corrected boundary figures now pinned as constants with unit tests: upload 399,376 B/inf (46.6%), readback 457,344 B/inf (53.4%), total 0.817 MiB — asymmetric, not the old symmetric 0.756/2 model.

**Morpheus's ruling:** rank 1 on the performance ladder survives (residency), but its *content* changes from "make the weights resident" to "make residency **bounded**" (arena lifetime/eviction) — a performance mechanism that fails into silent CPU fallback (Defect 1, above) "is a correctness defect wearing a performance costume." M1's weight-residency criterion (<1% upload/inference) stands at 0.0002 today — comfortably passed while the criterion stays open on its interlocks, "the clearest evidence yet that the interlocks are the criterion."

---

### 2026-08-01T09:53:14-07:00: q_gemv column-tile kernel — the first measured kernel speedup, and the Intel portability gap (consolidated)

**By:** Switch (`gemv-column-tile`, `intel-gap-separated`, `kernel-spread-reconciliation-niobe`), Fact Checker (`gemv-hardware-clock`)

**The kernel change:** Niobe's device-clock census made `q_gemv_matmul_nbits_f16` rank 1 by Amdahl (95-98% of GPU time). Switch reproduced her 40.202 ms/inference baseline independently before touching anything ("that agreement is the only reason I trust either number"), predicted ~18 ms post-tile, then rewrote the kernel around a `QB_COLS` column tile (workgroup computes multiple adjacent output columns, reusing the activation row load), workgroup sizing that divides rather than covers `blocks_per_col`, hoisted scale multiply, paired unpacked fp16 loads, and a paired non-atomic store. Measured (NVIDIA, both `STEADY`): 40.390 → **12.294 ms** GPU busy/inference (**3.29x**), kernel mean 244.09 → 65.36 µs (**3.73x**) — beat his own prediction. Negative ablation result: forcing the atomic store back cost nothing (468.32 vs 465.18 µs) — his leading hypothesis for the bottleneck was wrong.

**The Intel gap:** raw ratio was 13.5x (Intel/NVIDIA) at baseline. Contention made single-run Intel/NVIDIA ratios unusable (2.65x disagreement between two "identical" Intel runs), so Switch normalised against an untouched control kernel (`gqa_f16`) measured in the *same* run — contention is common-mode and cancels in the ratio. Result: **2.85x of the 13.5x gap was our kernel design** (baseline), unaffected by the arithmetic/workgroup-sizing changes (2.85→2.87, device-neutral), and reduced to **1.36x by the column tile alone** — "the tile is the entire portability fix." Diagnosis: the baseline's redundant per-column activation reload and per-column barrier/reduction hit Intel's weaker shared-memory bandwidth and barrier throughput hardest; NVIDIA's bandwidth/occupancy hid it.

**Fact Checker's independent verification:** corrected the *hardware*-bandwidth-only framing — the theoretical RTX 4060/Iris Xe bandwidth ratio is 3.08x (256/83.2 GB/s), not 13.5x, so the measured 13.52x gap leaves a **4.39x residual that is genuinely our design** (matches Switch's 2.85-4.6x range depending on build). Confirmed Intel's 52.0833 ns/tick counter is reference-clock (19.2 MHz) based and trustworthy — CPU load changes GT work-per-tick, not the tick conversion itself, so `NO_STEADY_TAIL` correctly reflects real workload instability, not a broken instrument. Recommended keeping the subgroup-free shared-tree kernel mandatory (portability) while adding optional subgroup/hybrid variants without assuming width 32 — llama.cpp's structural advantage is packed/vector loads and multiple accumulators, not subgroup ops.

---

### 2026-08-01T09:53:14-07:00: The EP is GPU-bound and one kernel is 95-98% of it — old ranking void

**By:** Niobe (`gpu-bound-lever-rerank`)

The prior residency/declines/fence-wait/kernels ranking was derived from a phase decomposition taken while Phi-3.5 ran on CPU fallback and is void — it described a run in which this EP did not execute. With 353 of 363 nodes now executing in one Vulkan island, the new ranking: **#1 `q_gemv_matmul_nbits_f16`** (95.11% NVIDIA / 98.28% Intel of all GPU time — everything else sums to ≤5%, so by Amdahl perfecting everything else buys at most that); #2 command-buffer reuse (23.5% of NVIDIA in-Compute time, **NVIDIA-only, 1.0% on Intel**); #3 5.97% unattributed inside Compute (needs spans from Switch); #4 device-backed allocation (`UNOBSERVABLE` per R12 until §6.5 closes); fence-wait GPU idle **retired to dead** (0.99%/0.18%) — "the fence wait is the GPU working." This claim is robust to a contended machine because contention inflates host work and cannot touch the GPU's own timestamp counter, so it can only push GPU-busy share up, never down, on a quieter box.

---

### 2026-08-01T09:53:14-07:00: Guard D — NameError vs detection, proven by two-polarity mutation testing, R13 (consolidated)

**By:** Trinity (`guard-d-fires-two-polarity`, `broken-guard-vs-detected-fallback`, `guard-d-counts-islands-not-nodes`), Morpheus (ruling D-M27-02, R13), Tank (ruling: R13 is an axis, not a sixth census state)
**Why consolidated:** Morpheus's own specimen (he merged the broken guard and reported it working), Trinity's remedy (mutation-tested two-polarity protocol), and Tank's taxonomic ruling on where R13 sits, all describing one incident and its closure.

**The incident:** `assert_vulkan_executed_runtime` (Guard D) raised `NameError: name 'pathlib' is not defined` at its first statement — it had never read a profiling event. Merged; the suite went `8 passed → 5 failed`; Morpheus reported "Guard D works." The red matched the prediction, which is precisely why nobody looked further.

**Trinity's remedy:** `RuntimeError`(instrument failure, e.g. unreadable trace) is now a distinct exception type from `AssertionError` (fallback genuinely detected), so the distinction survives to the pytest reader rather than living only in a message a human reads with an existing hypothesis. Landed via mutation testing, not code review: `tests/ops/test_guard_d.py` runs the real guard and **four deliberately broken mutants** (NameError reconstruction, always-passes, inverted polarity, wrong-provider-key) against a paired-polarity protocol — all four must fail it, the real guard must pass it. New standing rule: any verdict-rendering harness function ships a paired-polarity self-test in the always-on lane, or is listed `unfalsified` in the census with a reason — "I read the code" is not a third option.

**Guard D's count is fused islands, not graph nodes (R11, caught before it bit):** ORT emits one profiling `Node` event per fused island, so a healthy run reports `1` even though 354/364 graph nodes ran. Remedy: tag, don't rename (`describe_vulkan_execution_count`) — the count is a presence signal only, never a volume signal.

**Morpheus's rule (R13):** *a check has at least three terminal states — PASS, FAIL(condition), ERROR(instrument) — and must report them as three distinct tokens.* Not R10 (absent), R11 (misnamed) or R12 (correct, wrong world) — here the check *ran, failed, and its failure wore the costume of its finding.* Second clause, the more dangerous half: *a result that confirms a prediction deserves more scrutiny than one that contradicts it* — quote the failure text, never the failure count.

**Tank's ruling:** R13 is not a seventh/sixth census state — it is an axis over all census states (a property of the *channel* the verdict travels down, e.g. pytest's two-token summary line), not a property of an instrument's position in the system. Consequence: `audit_instruments.py` now reports three terminal tokens (PASS/FAIL(drift)/ERROR(instrument)) via `main_guarded`, and `test_wiring_census` timing out under contention is ruled `ERROR(instrument)`, never `FAIL(condition)` — the census is deterministic and byte-based, so a timeout is evidence about the box, not the call graph.

---

### 2026-08-01T09:53:14-07:00: `model_output_equivalence` becomes a record — five-token vocabulary, `UNATTRIBUTED` added (consolidated)

**By:** Trinity (`equivalence-record-vocabulary`), Morpheus (ruling D-M27-01), Link (`unattributed-vocabulary` — imports, does not redefine)
**Why consolidated:** Morpheus's ruling, Trinity's implementation, and Link's CI-side adoption of the exact same vocabulary rather than inventing a second one.

**The specimen that motivated it:** before Switch's `alloc(size=0)` fix, ORT hit `EP_FAIL … Falling back to CPUExecutionProvider` on Phi-3.5's empty-past KV inputs and **raised nothing**; `get_providers()` still listed the Vulkan EP (fixed at session-create time); the correctness gate then compared CPU against CPU and returned `MATCH`. Every gate in the lane passed while this EP executed zero nodes.

**The five tokens** (`tests/ops/_verdict.py`, canonical; precedence `SPLIT-FRAME → UNMEASURED → UNATTRIBUTED → MATCH/DIVERGENT`): `MATCH` (agree AND ≥1 own-provider node executed — the only token permitting a quoted triple/ratio); `DIVERGENT` (ran, wrong answer — owner: kernel authors); `UNATTRIBUTED` (ran, executed **zero** own-provider nodes — a CPU-vs-CPU comparison — owner: whoever owns runtime fallback, **never** kernel authors); `SPLIT-FRAME` (two attribution witnesses disagree — report nothing); `UNMEASURED` (no comparison performed). Two distinctions that must never collapse: `UNATTRIBUTED` is not `DIVERGENT` (different owner, different fix) and not `UNMEASURED` (it ran and agreed — "the more dangerous of the two precisely because it looks like a pass — it *was* a pass, for a whole day"). Binding clause: `MATCH` is unrepresentable at zero own-provider count — a caller may not pass a literal; the attribution comes from an instrument we do not own (ORT's own profiling), never our own `dispatches_executed`, which lives inside the frame in question. **Retroactive consequence, not grandfathered:** every prior `MATCH` on this project is `UNATTRIBUTED` until re-emitted with a profile beside it.

**Link's adoption:** all CI-side tokens/constructors are **imported** from Trinity's `_verdict.py`, contributing no token of Link's own — "a second vocabulary would be R11 in its purest form: two names for one measurement, appearing to close." `epctl --check-verdict` (Link's own earlier Session-8 brief) does not exist; the canonical gate is `epctl --check-counters ... --require-dispatches 1`, extended to fail on `UNATTRIBUTED`/`SPLIT-FRAME`/missing `executed_by`.

---

### 2026-08-01T09:53:14-07:00: CI lanes now carry criterion 10's gate — `operational` vs `green`, no lane is green yet (consolidated)

**By:** Link (`criterion10-gate-wired`, `lane-classification`, `unattributed-vocabulary`)

Every CI lane now runs a right-sized gate artifact (§7.8.1's 2-node fp32 `Add→Relu`) through five checks in order: (1) `gate_chain_fp32.py` writes `UNMEASURED` before opening any session, promotes only on a completed comparison; (2) `check_verdict.py`, an independent second reader in a separate process, rejects `MATCH` with empty/zero-count `executed_by`; (3) `epctl --check-counters --require-dispatches 1` (Tank's, third reader); (4) `check_fatal_log.py` grep for `Falling back` under `always()` — a second witness with a genuinely different failure mode from (1)-(3) (a grep cannot `NameError`); (5) a negative control removing the ICD, required to go red with `FAIL(condition=UNATTRIBUTED)` text.

**`operational` vs `green` (§7.4.4):** `operational` = exists, builds, runs, reports — a prerequisite, not a satisfaction. `green` = the pass condition includes an *attributed* `MATCH`, and the lane has demonstrated it can fail. **No CI lane is `green` yet** — both build lanes now *carry* the gate but have not been *observed* to pass it on a runner; Link will not classify a lane green from reading its YAML (R10 applied to his own work). The WSL lavapipe result (196 tests, subgroup_size=8, barrier parity 58/0) is explicitly `operational, not green` and "I will not launder it" — its supportable claims are the capability diff and a third independent barrier-parity implementation, nothing about correctness. Subgroup-32 falsifier chain's previously-unsecured link (tests actually executing on GPU, not silently falling back) is now closed by the gate + teed fatal-log grep. OQ-12 Android coverage corrected to ~67.33% as of 2026-07-30 (carried from prior session, restated here with both error directions).

---

### 2026-08-01T09:53:14-07:00: Full-suite baseline, both devices — 976s under 3.2x contention, and the "68 failed" regression that did not exist

**By:** Trinity (`full-suite-baseline-both-devices`)

345 tests/device, 38 failed (dev0) / 34 failed (dev1) / 267-271 passed, both runs ~976s against a ~5min quiet reference (**3.2x slower** per Niobe's contention guard, CONTENDED before and after both runs). No timing figure from either run is quotable. 31/38 (dev0) and 31/34 (dev1) failures share one cause and are **main's pre-existing state**, not a regression from this branch (`git diff --stat origin/main` shows the relevant test files byte-identical) — `assert_vulkan_claims` correctly fails because `Min` is staged-but-not-enabled per §8.9's evidence gating, a policy/expectation mismatch, not a kernel bug; Trinity declined to loosen the assertions to manufacture green. `test_wiring_census` failed with `subprocess.TimeoutExpired` on dev0 only — ruled `UNATTRIBUTABLE` (see Tank's R13-timeout ruling above), not counted as pass or fail. Guard D itself passed in production, reporting `3 fused-island executions` on real hardware — the R10 falsifier a code reading could never supply. One open item handed to Mouse, unresolved: two sessions in one process disagreeing about whether the same node form is claimed (1/1 then 0/1).

---

### 2026-08-01T09:53:14-07:00: All wall-clock figures withdrawn; machine-quiescence gating and admissibility grading (consolidated)

**By:** Niobe (`contention-guard`, `admissibility-audit`, `niobe-endtoend-withheld-priors-withdrawn`), Switch (`measurement-contamination`), Morpheus (rulings D-M26-01 standing directive, D-M27-01/04 withdrawal)
**Why consolidated:** the gating mechanism (Niobe), its application to Switch's own session-28-30 data (Switch), and the governance ruling that makes withdrawal mandatory rather than optional (Morpheus) are one policy enacted from three directions.

**Niobe's gate:** `bench/` now requires a machine-quiescence verdict (`QUIET`/`CONTENDED`/`UNMEASURED`, default `UNMEASURED`) via three independent instruments sharing no input (system-wide idle survey; a persisted-reference tachometer; an in-band trace contention signature) before releasing any duration; a non-`QUIET` verdict withholds medians/delta/ratio rather than printing them with a warning. Motivated by the coordinator's control: `vulkan.record` went 19,460→184,356 ms (9.5x) under six-agent load, undetected by drift-only checks. `bench/admissible.py` separately grades stored artifacts `ADMISSIBLE`/`INADMISSIBLE`/`WITHDRAWN`/`NOT_A_RESULT` — "absence of a check is a refusal, not a default green." Applied to Mouse's GQA files: a naive before/after read as 5.44x speedup, but the CPU baseline moved 18.0x across a Vulkan-only change — both readings inadmissible; a would-be "first win" (1.10x) also fails four of five gates. Niobe polled ~2.5 hours for a quiet window across the session and never got one, and explicitly declined to publish the exciting (fast) sample while withholding the disappointing (slow, Intel) one — both fail the same three checks for the same reason.

**Switch's application to his own data:** sessions 28-30 timing was taken under multi-agent contention; correctness findings and qualitative *direction* survive (cache reduces upload; recording cost is fixed per-Compute), but the quantified "2.85x speedup from weight cache" and the Intel-faster-than-NVIDIA inversion are explicitly withdrawn as contaminated.

**Morpheus's governance:** the standing directive to push performance continuously (D-M26-01) does **not** add a performance gate to M0 — "slowness is loud, wrongness is silent," and a speed gate is passable by claiming nothing; it changes sequencing (performance work runs continuously and in parallel, instrumented as a *rate obligation* — a falsifiable series, not a threshold) and adds one clause on Morpheus's own authority: no timing figure is quotable from a run whose verdict is not (now: attributed) `MATCH`. Every wall-clock figure this project holds, **including the previously-quoted 3.1x/3.7x**, is withdrawn — they were measured on runs where this EP executed zero nodes (CPU-vs-CPU). Published instead: `docs/PERF.md` §13.4.3, **NVIDIA 40.201 ms/inference GPU-busy device-clock, 0.033% RSD** (first admissible number in the project's history) — contention inflates host work but cannot touch the GPU clock. Intel is `NO_STEADY_TAIL` and withheld — an iGPU shares its power budget with loaded CPU cores, so its device clock is not contention-immune.

---

### 2026-08-01T09:53:14-07:00: Instrument census gains a sixth mechanical state, `unfalsified`, and R10/R11 named as separate rules (consolidated)

**By:** Tank (`instrument-census`), Trinity (`harness-instrument-census`), Morpheus (rulings `r10-unwired-and-perf-criterion`, `r11-naming-and-decomposition` — full text of D-M-R10, D-M-702, D-M-M0-4, D-M-PERF, D-M-SEQ4, D-M25 series also recorded here for provenance)
**Why consolidated:** Tank's Rust-side static census and Trinity's Python-side harness census are explicitly one census with two domains sharing one baseline file (schema bumped 1→2), motivated by the same finding (Guard D) and governed by the same two Morpheus rulings.

**Tank's four original states** (absent/uninvoked/unreachable/misnamed) found and fixed several specimens: `SIBLING TOTAL` excluded `Phase::Prepack` from its own hand-summed total (a composite whose total excluded a term); the dead-instrument screen itself was defeated twice — once by NOT-WIRED strings naming their own instrument (textual reference-counting scored dead instruments as wired), once by a comment/string stripper leaving an unterminated quote that mis-scored six wired counters as dead — now a single-pass stripper with a 7-case self-test that must pass before the census reports anything. Rule candidate offered to Morpheus: *"name the thing this number excludes; if nobody can, it is not a measurement, it is a label."*

**Trinity's harness domain (`unfalsified`, the state that would have caught Guard D):** *called, possibly often, but no always-on test has observed it in both polarities.* R10's `uninvoked` screen would **not** have caught Guard D — it had four production callers. Decided from the test AST (non-GPU-gated test calls it inside `pytest.raises` AND another outside one); first run found `assert_matches_cpu` — the correctness oracle itself — has been watched agreeing 9 times, disagreeing 0, and "we would not know" if it were `return` — first in the queue by blast radius. Ordering: `absent → uninvoked → unfalsified (harness) → unreachable → out-of-frame → misnamed`.

**Morpheus's R10:** *a mechanism's existence is a claim about the call graph, not the source tree; the falsifier for "X is wired" is an artifact whose content varies with input — never a code reading, never an author-set flag.* Six specimens found this way, including `partition.rs`'s `retain_viable` (called only from `#[cfg(test)]`) and the EP's own validation messenger (layer loaded, output to stderr, no listener). Companion rule §7.0.2: *a claim is a scheduling decision, not a capability statement — correctness is necessary, not sufficient; net-benefit is a property of an op in a graph at a coverage level, never of the op alone* (evidence: `SkipSimplifiedLayerNormalization` and `Cast`, both flipping from harmful to helpful as coverage changed). M0 tally as of that ruling: 6 met/4 partial/2 not met of 12; no performance criterion enters M0 (four reasons, chiefly "slowness is loud, wrongness is silent"), but a disclosure obligation is added instead (the end-to-end CPU ratio, never omitted, never gated) and M1 gains a non-gameable counter-before-a-clock criterion.

---

### 2026-08-01T09:53:14-07:00: Mouse — the last ten CPU nodes: claim 2, decline 8 permanently

**By:** Mouse (`last-ten-nodes`)

Phi-3.5's remaining 10 CPU-side nodes were three structurally different things, not ten gaps: 2 claimed (`SimplifiedLayerNormalization` needing a second predicate row for the ORT GenAI builder's spelling; `Gather` with float-only caps, deliberately not widened to `ANY` to avoid silently corrupting the integer-index `seqlens_k` output), and 8 declined **permanently on technical merit**: 6-node INT64 control plane (would need `shaderInt64` + a Cast matrix + a reduction template to move scalar arithmetic the host computes for free), 1 `Shape` node (round-tripping 16 bytes the host already holds), 1 `If` control-flow node (GRAPH-typed attribute, no subgraph-execution machinery, would force a fence stall mid-island for session-invariant outputs). Result, both devices, byte-identical: 355 claimed/1 island/8 declines/0 cut-instances; 24 CPU node-executions over 3 runs (8×3), argmax 30751 matching CPU. Per R13, no wall-clock figure is quoted (`phase_containment` was RED on both devices at time of writing, predating Niobe's fix above). Standing guidance: the remaining 8 should not be reopened as coverage debt — the number to move is boundary bytes, not the claimed percentage.

---

### 2026-08-01T09:53:14-07:00: `retain_viable` gate has never run on the real model, and the byte transfer model just moved 1,750x — the highest-risk combination in the partition path (consolidated)

**By:** Mouse (`retain-viable-gap`, `transfer-recalibration`)
**Why consolidated:** Mouse explicitly links the two findings as one risk in his own text — the gate that would catch an under-claim has no real exercise, at precisely the moment its threshold moved by three orders of magnitude.

**`retain_viable` gap:** the net-benefit economics gate is WIRED per the wiring census (correctly — R10-resolved), but Phi-3.5 partitions into exactly **one** cluster, so `only_one_cluster == true` short-circuits `GetCapability` before the gate ever runs — `viable_islands_retained == 0` means the gate was **bypassed**, not that it rejected everything, and the census vocabulary cannot currently express that distinction (present-and-0 vs UNWIRED, correctly separated by type, but "bypassed" vs "ran and found nothing" is not). The gate's only exercise is a synthetic two-branch test built specifically to give it something to decline — "improving the partition result made the partition gate untestable on the thing it partitions." Escalated to Morpheus (does WIRED-synthetic-only need its own census state?) and Niobe (a second real model that partitions into ≥2 clusters, or an accepted counterfactual run, is what closes this).

**Transfer recalibration:** the brief's claimed halving (1997.6→0.756 MiB) was Switch's residency win, already present in a pre-Mouse control commit — Mouse's own two op claims account for exactly 6,136 of 399,376 bytes/inference (see weight-residency entry above for the full correction). Corrected boundary: 0.817 MiB total (upload 46.6%, readback 53.4%, asymmetric). Consequence: with transfer now ~1,750x cheaper, the net-benefit gate's remaining discriminating power sits almost entirely in `fixed_ns`, the one parameter with **no measurement behind it** (transfer's *nanoseconds*, unlike its *bytes*, still cannot be calibrated under R13 without an attributed-`MATCH` run) — for small-boundary islands the gate has degenerated to `2 × fixed_ns`, insensitive to the byte count it nominally reasons about. An under-declining gate is a **silent** failure mode (slow inference, never a wrong answer) — paired with the gate never having run on the real model, this is flagged as the highest-risk combination in the partition path.

---

### 2026-08-01T09:53:14-07:00: Criterion 3/4/5/12 controls landed; validation-messenger scope clarified (consolidated)

**By:** Trinity (`criterion-3-4-5-12-controls`), Switch (`criterion3-lane-gap`, `validation-control-trinity`)

Trinity landed paired positive/negative controls for criteria 4 (ICD-present vs absent) and 5 (shaders-compiled vs shader-less), the criterion-3 validation lane (armed check, in-lane plant, clean-after-fix), and the criterion-12 wiring-census harness, all passing on both devices. Switch's `criterion3-lane-gap` flagged that the EP's own-instance messenger plant (`ep_messenger_fires_for_planted_fence_leak`) was `#[ignore]`d and so outside the always-run lane — resolved by Trinity's subprocess wrapper (`cargo test --lib --release -- --ignored ...`), which Switch confirmed is the correct and sufficient home; the `#[ignore]` stays (the test deliberately provokes a real Vulkan error and must own the process). Switch's one open ask: the wrapper should require `Instance::validation_armed()` before treating a silent messenger as a pass, else a machine without validation layers reports a green messenger test while listening to nothing — "the exact shape of the NameError-as-green-guard mistake." (See separate entry above for the related but distinct finding that a "0 errors at shutdown" gate on the *production*, never-destroyed device is `UNOBSERVABLE`, not `0`.)

---

### 2026-08-01T09:53:14-07:00: Morpheus — M0 criterion 10 reopened on `UNATTRIBUTED`, cheap re-closure specified; ranked performance order survives

**By:** Morpheus (ruling D-M27-03, D-M27-04 — see also the R12/R13/vocabulary rulings folded into their respective entries above)

M0 criterion 10 (and, through it, criterion 2) is **reopened**, not "met with recorded scope" — the prior evidence was void (measured a different thing: CPU-vs-CPU) rather than narrow, so scope cannot repair it. Reopening is deliberately cheap and specified in advance, not punitive: **three consecutive attributed-`MATCH` runs in one session on Phi-3.5 close it the day they arrive, with no new conditions** — the multi-run requirement already existed (2026-07-30 cross-run gate), this only applies it to a row not previously assessed against it. Tally: 4 met/6 partial/2 not met of 12 (was 6/4/2). The genuine new evidence — ORT's own profiling now reports 1 fused-island Vulkan execution + 10 CPU (Mouse's exact declined set) with argmax matching CPU — "is worth more than the reading it replaces," but is one run where three are required. The ranked performance order (residency > net-benefit declines > fence-wait > kernels) survives unchanged in *position*, because it was always derived from counts and ratios, never from withdrawn wall-clock figures — "a count and a ratio do not care what the absolute number was." Rank 1's *content* moves to bounded residency (see weight-residency entry above). The project did not go backwards: the EP executed a real model on GPU for the first time today, and per-inference upload fell 1997.6→0.756 MiB — "both are real; neither is a criterion. A milestone table is not a progress report — it is a list of claims we are prepared to defend."
