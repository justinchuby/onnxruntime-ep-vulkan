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

<!-- =============================================================================== -->
<!-- ROUND 6 DECISIONS -- 2026-08-01T17:16:56-07:00 (instrument-audit round)         -->
<!-- 32 inbox records read and merged; consolidated into 26 entries below where two  -->
<!-- or more records described one event from independent angles (R9: corroboration, -->
<!-- not redundancy — both angles preserved). Round 5 (18 entries, ~46KB, written     -->
<!-- 2026-08-01T09:53:14-07:00) is NOT archived this round: it is ~7.5 hours old,     -->
<!-- under the 24-hour floor, even though it already exceeds the 40KB volume         -->
<!-- threshold — the floor overrides the threshold. Round 5 stays live because this  -->
<!-- round's device-clock retraction and min()-direction corrections correct claims  -->
<!-- made only hours ago and must stay visible beside them (supersession beats       -->
<!-- recency).                                                                       -->
<!-- =============================================================================== -->

### 2026-08-01T17:16:56-07:00: The device-clock gate was found blind to bias — discovery, ruling, and dual CI/bench implementation (consolidated)

**By:** Switch (`steady-tail-is-not-certification`), Morpheus (ruling, R9 amendment 5, §10.0 obligations 8/8b), Niobe (`device-state-companion-mandatory`), Link (`no-duration-without-device-state`)
**Why consolidated:** one finding narrated from discovery through ruling through two independent implementations in one day.

**What:** `gpu_steady_tail` (Niobe's `bench/`) is a variance test over a suffix and **cannot see a bias in level**. Switch measured it two ways: (1) a real run under 3 foreign GPU processes, truncated to end while contention was still live, reports `STEADY 126.647 ms` — **10.99× wrong**, RSD 0.79–0.91%; (2) a board pinned at its 210 MHz idle clock (verified `SOLE_TENANT`) reports `STEADY 246.72–246.735 ms` — **21.4× wrong**, RSD as low as **0.1163%**, zero discarded. **In both cases the wrong number carried the better RSD than the true measurement.** A uniformly wrong series is a perfectly steady one.

**Morpheus's ruling:** this is not R11 (all four of R11's obligations pass on this specimen — it is precisely a rule that would certify the defect it is asked to catch, disqualifying it), not R12, not R13, not R7. It is **R9, amended**: rule 5, the anti-correlated falsifier — *where a check's confidence measure is computed from the same series as the quantity it certifies, ask which way the check moves when the quantity is wrong; if it moves with the reader's confidence, tightening the threshold admits more of the failure, not less.* Remedy is a different instrument, never a tighter bound. §10.0 gains obligation 8: a device-clock figure is quotable only with a device-state record (tenancy verdict + SM-clock min/median/max vs board max) covering the *same window as the statistic*, or it is `STEADY_UNCERTIFIED` — never a plain `STEADY`. Obligation 8b: two figures are comparable only if their device-state records agree; a before/after where the "before" predates the requirement is not a pair (Switch's own 12.183 ms NVIDIA baseline is retired on this ground, his own standard applied to himself).

**Niobe's bench implementation:** `phases.gpu_steady_tail` demoted from gate to precondition; every tail is born `certification: UNCERTIFIED` and only a `bench/device_state.py` companion (importing Switch's `probe_gpustate.py`) can promote it to `QUOTABLE`. Five terminal verdicts: `QUOTABLE`, `WITHHELD` (a detection), `UNCERTIFIED` (no record/no number — not a detection), `UNOBSERVABLE` (R12, no route to certify — not a detection), `ERROR` (R13, companion failed). Discriminator is **peak** SM clock, not median: median does not discriminate (both a correct run and a 21.4×-wrong run sample median 210 MHz because `nvidia-smi` samples the whole command including idle host phases), peak separates by an order of magnitude (2280–2490 MHz correct vs never leaving 210 MHz pathological).

**Link's CI implementation:** `ci/check_device_state.py` runs in all three lanes on `always()`; a lane publishing a duration without a certified companion goes red on the same run. Absence of a producer is never a waiver — a CI runner with no GPU telemetry is the loophole obligation 8 exists to close. Instrument-dump exemption (ORT's own `dur` fields, the EP's staging counters) is a closed code-level list with a reason per entry, never an `--exclude` flag, and excused figures still print as `STEADY_UNCERTIFIED (carried, not claimed)`.

**Consequence for M1:** criteria 1/2/4 (bytes and counts) needed no correction all week; criteria 3/5/6 gain obligation 8 as an admissibility interlock. The 40.201 ms figure (Niobe, 0.033% RSD) is **re-qualified, not withdrawn** — quotable only as `≤ 40.201 ms, device state unrecorded`; Morpheus rejected the argument that it sits in a separate "boosted regime" outside the attack's reach (Switch's own record shows 210→2490 MHz *within a single run*, so a governor is continuous and there are not two regimes).

---

### 2026-08-01T17:16:56-07:00: lavapipe ruled `none_structural` — no CI lane can ever certify a device-clock figure, permanently

**By:** Link (`lavapipe-device-state-ruling`)

All three CI lanes run lavapipe, a software rasteriser with no SM clock and no GPU-sense tenancy. Ruling, written down before it is discovered the hard way: **a CPU renderer can never certify a device-clock figure — permanent, not pending.** Not "no producer yet": there is no subject to measure, and a producer that reported the CPU's frequency under a GPU field name would be obligation 8b's failure mode (two records that appear to agree because they share key names) dressed as coverage. `ci/device_state.py` registers `cpu_renderer` as `none_structural`, distinct from `unimplemented` (a producer could exist — AMD, Intel, Apple, Adreno, Mali) and `available` (NVIDIA). A *wall-clock* figure on lavapipe remains a legitimate, weaker, unstarted claim, paired with a host-state record (quiescence, CPU frequency vs package max, host tenancy) — the two records must never trade names, or lavapipe becomes the cheapest certifying platform in the matrix by having no device clock at all.

---

### 2026-08-01T17:16:56-07:00: Intel device-clock ruled `none_available` — permanent on this hardware; attack the 4.39× residual with counts

**By:** Niobe (`intel-clock-uncertifiable-permanent`)

Distinct from lavapipe's ruling: on Intel Iris Xe there **is** a real, varying clock, but **no producer exists for it** on this machine, and the producers that do exist are structurally the wrong kind of quantity. Searched and struck off: `\GPU Engine`/`\GPU Adapter Memory` counter sets (no MHz counter), `root\wmi` GPU classes (none present), `Win32_VideoController` (no core clock), `nvidia-smi` (NVIDIA-only, exits 6 on the Intel board), Vulkan core + admitted extensions (no clock query), and engine `Running Time` as a proxy — **inadmissible**, because a duration moves the same way as the figure it would certify (a second copy of the quantity, not a second quantity from outside the series). Consequence: Intel is `UNCERTIFIED(partial_companion)` at best, forever, until a producer (vendor telemetry, an elevated driver interface) exists — platform enablement, not analysis, and not on M0's path. The actionable half: the 4.39× of the 13.52× Intel/NVIDIA kernel gap that bandwidth does not explain must be attacked with counts and shapes (dispatch counts, bytes, occupancy, instruction mix, barrier counts) — quantities invariant under load, clock and tenancy — because on Intel there is now no alternative to asking that way. The Iris Xe's repeated `NO_STEADY_TAIL` refusals are *consistent with* a wandering clock and are **not evidence of one** — that is the tail gate refusing, not the clock gate reporting.

---

### 2026-08-01T17:16:56-07:00: Device-state companion splits into two independent axes — tenancy and clock; half a companion never certifies

**By:** Niobe (`windows-tenancy-half-companion`)

The §10.0 obligation-8 companion is now two axes, each with its own producer, verdict and silence set. A record with a tenancy verdict and no clock record is `TENANCY_ONLY`, certifying as `UNCERTIFIED(partial_companion)` — it never releases a number. Windows' vendor-neutral WDDM counters (`\GPU Engine(*)\Running Time`/`Utilization Percentage`) are wired in as a tenancy producer for any adapter, including Intel, via `bench/win_gpu_counters.py`. Why half must not certify: `base_b` was verified sole-tenant *and* 21.4× wrong at the project's second-best RSD, because the board never left 210 MHz — a tenancy-only companion on that run reports `SOLE_TENANT` and is *correct*, and the figure is still wrong by 21.4×; a partial record that certified would be a worse loophole than an empty one because it looks like diligence. The asymmetry that makes the half usable anyway: a tenancy-only record may **subtract** confidence and may never add it — `FOREIGN_GPU_WORK` with no clock record still resolves `WITHHELD`; `SOLE_TENANT` with no clock record resolves `UNCERTIFIED(partial_companion)`, never a pass.

---

### 2026-08-01T17:16:56-07:00: A contention companion must learn its own PID while the child is still alive — Niobe reproduces Switch's own bug at her own call site

**By:** Niobe (`companion-own-pid-live`)

The first run under the mandatory device-state companion reported `FOREIGN_GPU_WORK` in 93% of samples against a single PID holding 0.0 MiB on an otherwise-idle machine. The foreign process was our own worker: `_run_worker` used blocking `subprocess.run`, so the companion learned the PID only after the child had already exited, and ancestry checks (which must be live) resolved every sample against `own_root = None`. Fixed with `Popen` + an `on_start(pid)` callback fired before `communicate()`. **Switch's own `_is_ours` docstring already named this exact failure** from his own first run of the same instrument ("a contention detector that counts our own worker as a stranger fires on every run and is therefore not a detector — it is a constant"); Niobe imported his instrument and reproduced his bug at the call site, because the warning was about the instrument and her defect was in how she wired it. Generalisation: **importing an instrument does not import its preconditions** — a property of the instrument's contract, not its API, and nothing in the signature enforces it. `ERROR(instrument)` under R13, never a detection.

---

### 2026-08-01T17:16:56-07:00: Tenancy instruments that are wrong produce a *cleaner* record, not a noisier one; a settled tail is not automatically a quotable one (consolidated)

**By:** Niobe (`reads-clean-failure-shape`, `steady-tail-coverage-floor`)

Every bug found while building the WDDM tenancy sampler produced a clean record (`SOLE_TENANT`, no error) rather than an obviously broken one — wrong device ordering in the LUID join (watched an adapter the workload never touched), a starved sampler (three samples over a 62s window), and PDH's per-process instance-list cache (invisible to a query opened before the job started, including our own worker for a whole 60s run). All three move *with* the reader's confidence (R9 amendment 5) and cannot be repaired by tightening — the fix is that a clean record must now carry **positive evidence** it watched the right device over the right window (`UNOBSERVABLE(self_not_witnessed)`, and a blind-gap limit that fired for real on the first NVIDIA corroboration run). Separately: `gpu_steady_tail` gains a third non-quotable verdict, `MARGINAL_TAIL` — a suffix clearing the 2% RSD bar is only quotable at `n >= 8` **and** `coverage >= 50%` of usable inferences; the discriminator is coverage, not `n` (a genuine warmup is a short prefix, a device that never settled produces a short flat suffix). Its first act refused two of Niobe's own runs, including one that read as a 33% GPU improvement from the barrier fix and was false — under the floor, pre/post GPU figures agree to 0.02%.

---

### 2026-08-01T17:16:56-07:00: `bench/` reads worker stderr with Tank's `decode_both`, not a decoder of its own

**By:** Niobe (`worker-stderr-decoder`)

`bench/phi35.py` captured its worker with `text=True`, but ORT's default Windows logging sink writes UTF-16LE on the same handle Tank's new runtime WARN now uses, producing a `UnicodeDecodeError` inside the reader thread and a follow-on crash reading `None`. Rule: capture bytes, never `text=True`, on any child that can reach ORT's logging sink; use Tank's `decode_both` (`probe_broken_commitment.py`) as the one decoder for that channel rather than writing a second dialect; if the decoder is unavailable, `bench/phi35.py` returns `ERROR(instrument=stream_decoder)` rather than falling back to a private decode. Note for whoever merges `squad/tank`: `probe_broken_commitment.py` is not yet on `origin/main`, which is a prerequisite for the bench harness there.

---

### 2026-08-01T17:16:56-07:00: The barrier fix, measured as an interleaved A/B — host −4.3 to −5.8×, device unchanged

**By:** Niobe (`barrier-ab-result`)

Same harness, same machine, DLL alternated A/B/A/B in one sitting on NVIDIA RTX 4060. `record` host median fell 16.412/20.344 ms → 3.780/3.687 ms (4.3–5.5×); leaf-only over 43 recordings fell 810.0 → 139.0 ms (5.8×); GPU busy (matched pair, n=43, 100% coverage) moved 13.3463 → 13.3432 ms (0.02%). Interleaving scrambles load ordering (the second post-fix run ran under *more* foreign load than the first pre-fix run and was still 4.3× cheaper), and the diff between commits touches only `vk/barrier.rs`/`vk/session.rs` — no shader changes — so host movement without device movement is exactly what Switch's mechanism predicts. Corroboration: Switch's derived ~94 ns/barrier-struct and Niobe's measured 86–106 ns/struct agree within 13%, two different instruments on the same quantity. Explicitly not claimed: any end-to-end wall clock or ratio (gate says `CONTENDED` throughout), and not a 3.5× win versus the published 40.201 ms — between those figures lie the GEMV rewrite, partitioner fusion, residency and this fix; 40.201 ms is retired as a baseline, not beaten by one.

---

### 2026-08-01T17:16:56-07:00: The EP was emitting 147,618 buffer barriers per inference; and `min()` bounds from above, not below (consolidated correction)

**By:** Switch (`record-is-host-bound`)
**Why flagged as a correction that must stay live:** this entry contains Switch's own retraction of an inequality direction he stated earlier this session, and the retraction must not be archived ahead of the claim it corrects.

**Finding:** `vulkan.record` (building the command buffer) took more wall time (14.414 ms min) than the GPU execution it described (12.156 ms min), with 96.4% of `record` unnamed by any child span. Mechanism, measured: the per-dispatch intermediate barrier was emitted **per buffer** — 417 intermediates × 354 dispatches = **147,618** `VkBufferMemoryBarrier` structs per inference, each heap-allocated and driver-validated on the host while the GPU waited. Switch's first hypothesis (`std::env::var_os` in the per-dispatch loop) was measured and **falsified by 170×** before being acted on — a discipline the record credits explicitly, since "fixing" it would have shipped a no-op with a plausible story attached. Fix: one global `VkMemoryBarrier`/`VkMemoryBarrier2` replacing the 417 per-buffer ones — strictly more conservative (a superset of the dependency set, which was already effectively total since every kernel in the island reads what an earlier one wrote), so it cannot introduce a race and can only cost over-synchronisation, of which there was none here. Measured: `record` min 14.414 → 2.704 ms (predicted 0.6–1.6, over-predicted); GPU busy 12.183 → 11.589 ms STEADY, **falsified favourably** and not fully claimed as caused by the barrier change (a pipeline barrier is also GPU front-end work, but the series ramps and RSD is 1.23% against baseline's 0.103%, so a GPU clock effect is not excluded). Both devices: NVIDIA `MATCH`, STEADY; Intel `MATCH`, `NO_STEADY_TAIL` (expected, per Fact Checker's finding). Control-kernel ratio `q_gemv/gqa_f16` unchanged at ~1.72 on both devices — the 4.39× residual stays closed.

**The correction:** Switch originally claimed minimum-over-inferences is a **lower** bound on uncontended cost. It is an **upper** bound: `observed = true + delay`, `delay >= 0`, so `min(observed) >= true`. **Two upper bounds do not bound a difference from below** — "≤14.414 ms before" and "≤2.704 ms after" does not by itself prove an improvement, let alone its size of 5.33×. The *direction* of the change remains certain because it rests on the barrier count (147,618 → 354), which is contention-independent; the *magnitude* (5.33×) must be requoted as an estimate, not a bound. A second instrument note: `cmd_upload` is nested inside `record`, so summing named phases by name double-counts — `probe_hostphase.py` now separates `TOP` from `NESTED` and prints an explicit `unaccounted` row.

---

### 2026-08-01T17:16:56-07:00: Switch's admissibility ledger for 2026-08-01 and resume note

**By:** Switch (`admissibility-ledger-2026-08-01`)

Written at a pause, labelling every figure measured that day rather than carrying anything forward unlabelled. **Admissible (counts, not clocks):** barrier structs 147,618→354, island shape (355 kernels/417 intermediates), `var_os` cost (0.083 ms/inference, negligible), correctness (`MATCH` both devices, 426/426 tests). **Estimate, not a bound:** `vulkan.record` 14.414→2.704 ms, "5.33×" restated as an estimate under contention. **Not admissible, withheld:** every GPU-busy before/after taken that day (11.589/11.525/11.524 ms STEADY has no certified "before" to pair against — the 12.183 ms baseline predates the tenancy/clock requirement). **Admissible only as evidence about the instrument:** the 126.647 ms (10.99×) and 246.72–735 ms (21.4×) `STEADY` misreads — never to be quoted as phi-3.5 results. Rescues the project's measurement history: the two clock regimes (idle ~247 ms, boosted ~11.5–41 ms) are 21× apart and do not overlap, so a figure's regime is recoverable from its magnitude after the fact — Niobe's 40.201 ms and Switch's own 40.390→11.525 ms series are both necessarily boosted-regime and stand; what does not stand is the *reasoning* that certified them (low RSD means steady, not "good measurement"). Resume note: nothing mid-flight, tree clean, four priority items closed; first task on resume needs a quiet window to take a certified barrier before/after and to close the still-open duty-cycle mechanism (needs an elevated shell for `nvidia-smi --lock-gpu-clocks`, blocked on Justin).

---

### 2026-08-01T17:16:56-07:00: A sole-tenant GPU is not a certifiable one — the duty-cycle mechanism blocks certification even under eight minutes of verified tenancy

**By:** Switch (`sole-tenant-is-not-certifiable`)

Three attempts at a certified NVIDIA A/B, committed with their device-state companions so the refusals are legible. Two runs were verified `SOLE_TENANT` — one over 134 samples, one over 327 samples across 478 seconds — and both still resolved `UNCERTIFIED`, `MARGINAL_TAIL`. Mechanism: the EP is host-bound, so the board is never asked for sustained enough work to hold a boost clock; SM clock ranged 210–2010 MHz (median 210, i.e. idle) with utilisation median 0%, and the GPU-busy series drifted 3.8× across the run (74 ms opening to 19.6 ms closing) — a series that drifts that much cannot hold 2% RSD over half its length, so Niobe's coverage floor fires correctly. Consequence: idling the whole team would not have bought a certifiable figure that day — tenancy was not the binding constraint. `nvidia-smi --lock-gpu-clocks` with an elevated shell is now blocking two things at once (closing the duty-cycle mechanism and being the likeliest route to a certifiable figure), still blocked on Justin. Switch's own uncertified archive (11.5 ms) is now positively suspect against this run's 19.6 ms flat suffix at a lower peak clock (2010 vs 2490 MHz) — a reason to distrust the archive, not a reason to quote 19.6 ms, which is equally `UNCERTIFIED`.

---

### 2026-08-01T17:16:56-07:00: NVIDIA's device clock is contention-immune, Intel's is contention-coupled — measured; the 70% kernel-spread claim withdrawn permanently

**By:** Switch (`host-gpu-decoupling-measured`)

Paired host/device traces (ordinal attribution, never timestamp — the calibration anchor has 314 ms of uncertainty on Intel) over 15 traces, three builds, both devices, two orders of contention. Cold-start excursion (1.4–2.5 s) is entirely host-side (shader/pipeline compile) on both devices, with negligible extra device cost. Warm: on NVIDIA, host spread exceeds GPU spread by a median 1.34× (up to 2.27×); on Intel the two are equal to within 1–5% in all six runs — the iGPU inherits host contention on its device clock ~1:1, the discrete part does not. Between-run reproducibility: NVIDIA steady device time reproduces to 0.16–0.47% across separate processes while the host figure for the same runs moves up to 17%. Consequence: NVIDIA device-clock numbers do not need a quiet machine; Intel's do (confirms Fact Checker's tick-vs-work-per-tick finding); any wall-clock number needs one on both parts. Reconciliation with the withdrawn 49.4/83.8/71.0/58.5 ms figures: the hypothesis that the spread was host-side is **falsified** for these numbers (they were `gpu_ns`, not host); 49.4 is a reproducible ramp level (49.58/49.59 ms before stepping to 40.2), the ramp *length* is not reproducible (5 vs 3 inferences, same build), and the steady level (0.033% within-run RSD, 0.47% between-run) is the same claim as Niobe's — no disagreement to settle. 83.8 and 71.0 ms exceed the entire 15-trace corpus and came from a run with no `executed_by` key, `UNATTRIBUTED` under Trinity's `_verdict.py`, which now refuses that shape at construction. **"70% spread" is withdrawn permanently.** Next bound (not figure): host span exceeds GPU busy by at least 2.3× even in the quietest inference observed — the next order of magnitude is in the host path, not `q_gemv`.

---

### 2026-08-01T17:16:56-07:00: Allocator adopts the session's device by identity, not index agreement — §6.5's selector-1 gap closed and two-armed-verified

**By:** Switch (`allocator-adopts-by-identity`, `index-space-one-space`)

Tank showed §6.5's prior "closure" (both selectors reporting `SHARED`) was a coincidence: the allocator always asked for factory index 1 regardless of selector, and on selector 0 the session happened to also offer index 1. Selector 1 (session offers index 0) exposed it as `SPLIT-DEVICE`, and the earlier "fixed" report was taken with the env pin set, which forces exactly one device to be advertised and hides the defect on the one path (`ep.device_index` session option) the harness actually uses. Mechanism: the allocator's index (ORT's binding, constant across selector) and the session's index (the physical index our selector opened, varies with it) have no arithmetic relation; the fallback stood up a *second* `VkDevice` on a missed-index lookup. Fix: resolve by device identity — `Exact` (index matches), `SoleDevice` (exactly one device on offer, missed index — adopt it, frame `SHARED`), `NoOffer`, `Ambiguous` (>1 device, no match — stand one up). **Verified two-armed rather than one-armed**, per R10: a criterion requiring the allocator's index to *differ* between the two selectors and match the session's offered index in each — a pair with the same index on both arms fails even if both report `SHARED`, because that is precisely the pre-fix coincidence. Result: selector 0 → allocator index 1/session offers 1=NVIDIA; selector 1 → allocator index 0/session offers 0=Intel — **the artifact's content varies with its input**, the fourth two-index-space defect on this project and the first closed by construction rather than by making one arm agree. `alloc_device_buffer_binds = 6` on both selectors (Tank's counter has left 0); `alloc_device_authoritative_spans` stays 0 by design (all 9 spans still carry host staging — a measured zero, not an absence).

---

### 2026-08-01T17:16:56-07:00: The engine binds the EP's device buffers as inputs, and mirrors writes back to the device in return

**By:** Switch (`engine-binds-device-buffers`)

Closes relay item 3: `vk::session::dispatch_ort` gains a Step 1a that asks `host_device_memory::bind_target_for` whether an input already has a bindable `VkBuffer`, skipping allocation and re-upload when it does. Declines rather than assumes on three conditions: the span must have a buffer; the frame must be `SHARED` (binding across a second `VkDevice` is undefined and could *appear* to work on a UMA part, the worst failure mode); offset must be 0 with `offset+len <= size` (binding an interior pointer at 0 would read the neighbouring tensor and produce plausible wrong numbers). This creates an obligation in the same change: bound inputs make the old "session reads/writes only through staging" asymmetry a staleness bug, so `write_outputs_to_ort` now calls `transfer::mirror_to_device` after every output write. Measured (`probe_sec65.py`, three sessions per run): `alloc_device_buffer_binds` left 0 → 6 on both devices; `session_device_allocs` fell 21→15 (exactly the 6 no-longer-needed allocations, in a counter Switch did not touch); `alloc_device_uploads` rose 6→9 (the 3 new output mirrors). Control (device memory off): frame `OFF`, binds 0, `alloc_device_authoritative_spans` the string `"UNOBSERVABLE"` — the shipped default is provably inert. `alloc_device_authoritative_spans` still reads 0 on both devices — binding inputs is necessary for authoritative residency, not sufficient; the M1 residency criterion stays open. 426 tests pass.

---

### 2026-08-01T17:16:56-07:00: Packed 128-bit loads close the Intel residual (4.39× → 1.03×); restated in counts and Switch's own multiplier claim demoted (consolidated)

**By:** Switch (`packed-loads-residual-closed`, `packed-loads-in-counts`)

**The A/B (answering Fact Checker's `packed-loads-and-accumulators` hypothesis):** `InB` declared `uvec4[]`, one 128-bit load replacing four dependent 32-bit ones where a blob is a whole number of 16-byte units (spec constant `QB_PACKED`), feeding four independent accumulators instead of one serial chain. Predicted before building (11.3 ms NVIDIA GPU busy, 310 µs Intel kernel) and measured (11.567 ms, 297.15 µs) — NVIDIA over-predicted in the direction Switch's own roofline argument warned it would (headroom was already small there). Interleaved A/B (machine would not go quiet): NVIDIA 1.07×, Intel 1.327–1.385×, gains disproportionately Intel's as a bandwidth-bound-kernel/narrow-memory-pipe prediction requires. Against an untouched control kernel (`gqa_f16`), the design-attributable Intel/NVIDIA excess falls baseline 2.85× → column-tile 1.50× → packed-loads **1.03×** — inside the control's own run-to-run spread. **The residual is closed, not reduced.** Cumulative: RTX 4060 GPU busy 40.390→11.567 ms (3.49×); Iris Xe kernel 3804.85→297.15 µs (12.8×).

**The restatement in counts, and the demotion:** every device-clock figure for `q_gemv` is `UNCERTIFIED`/`WITHHELD` under the device-state gate, and Intel cannot be certified at all — so per §10.0.4 the structural claim is carried by SPIR-V load counts instead. Reading the emitted module (not the GLSL): the packed path issues 1 load of `v4uint` per 16-byte blob vs 4 loads of `uint`; across 161 `MatMulNBits` nodes and 116,324,352 blobs/inference, InB load instructions fall **465,297,408 → 116,324,352**. **Switch's own earlier claim** (from `538db70`) that the serial accumulator chain "pins memory-level parallelism near one outstanding load" **does not survive**: the four 32-bit loads are independent and can all be in flight; what was actually serialized is four accumulator read-modify-writes, restructured to a depth-2 tree (4→1 RMWs, 4→3 serial FP adds) — real, countable, and nowhere near a 4× effect. The byte model predicts a 3.15× total-byte reduction from column-tiling (measured 3.73× on the kernel clock, 18% agreement between independently-derived quantities) and explains why packed loads' own ceiling is small: **packed loads move zero bytes, only instructions**, so on a bandwidth-bound kernel their headroom is inherently small. An instrument self-defect caught before publishing: the first probe version reported identical load censuses for both arms because `glslc` emits the packed/scalar branch as an unfolded `OpSpecConstantOp` that optimization passes do not fold — fixed with `--fold-spec-const-op-composite`, and `arms_must_differ` now refuses to publish an artifact where the two arms agree.

---

### 2026-08-01T17:16:56-07:00: RAI-011 closed — the net-benefit gate now evaluates Phi-3.5's island, and "bypassed" is no longer a zero (consolidated)

**By:** Tank (`net-benefit-gate-observable`), Mouse (`net-benefit-gate-one-entry-point`)
**Why consolidated:** Tank made the bypass legible without changing behaviour; Mouse then removed the bypass itself. The two changes meet at one counter and neither could be read correctly without the other.

**The problem:** Phi-3.5 partitions into exactly one cluster, so `GetCapability`'s `if only_one_cluster { Claim } else { evaluate(...) }` made the net-benefit gate not merely unexercised on the project's only real model, but **unreachable** — `viable_islands_retained == 0` read identically for "the gate ran and rejected everything" and "the gate never ran."

**Tank's half (observable):** `record_net_benefit_decision(evaluated: bool)` — one call, an unconditional `clusters_seen` increment plus a conditional `evaluations`/`bypasses` increment that cannot drift apart. `viable_islands_retained` becomes token-or-integer: `"UNWIRED"` (never reached), `"UNOBSERVABLE"` (seen but every cluster bypassed — the event cannot have occurred in this frame), or a real integer. Companion token `net_benefit_gate` ∈ `UNWIRED|BYPASSED|EVALUATED|MIXED`.

**Mouse's half (the fix itself):** one entry point, no branch in front of it — `partition::gate_islands(islands, model, policy)` calls `evaluate` once per island, unconditionally; the single-island exemption is applied *after* evaluation as `GateOutcome::SoleIslandOverride(RejectReason)`, carrying the verdict it overrode, so an override that discards the verdict is a bypass wearing a different name. New counter `net_benefit_sole_island_overrides`, distinct from `viable_islands_retained`, so "the gate retained it" and "the gate rejected it and we kept it anyway" can never share a digit. `net_benefit_gate_bypasses` is now expected to be permanently 0 in shipping; non-zero means a second entry point was reintroduced. Falsifier: `probe_net_benefit_gate.py`, five env-knob configurations, verdict moves exactly where the predicted flip (3,836,739.6 ns) says it should on both devices while `claimed_nodes` stays 355 throughout — a constant proves nothing, a predicted flip landing on both sides does.

---

### 2026-08-01T17:16:56-07:00: The island lever re-derived on fresh evidence — 355/1/0, no decline creates a cut or sits INTERIOR

**By:** Mouse (`declines-reattributed`)

Re-run because `SimplifiedLayerNormalization` and `Gather` are now claimed, changing the graph shape since the §7.8.4 histogram was written — the conclusion had to be re-derived, not re-quoted. Result, both devices, byte-identical: 355 claimed, 1 island, 8 declines, **0 cut-instances**. New probe (`probe_decline_position.py`, reads the claim log without running the model) categorizes each decline: 5 `DETACHED` (INT64 control plane, never touches a claimed tensor), 3 `EDGE_ENTRY` (feed claimed nodes, fed by none — prologue), **0 `INTERIOR`**. The probe's own falsifier: `INTERIOR` together with `island_count == 1` would be a contradiction (a claimed→declined→claimed path inside one island is a cycle ORT could not have fused) — empty on both devices, as required. `If` stays declined (GRAPH-typed branch attributes, a `BOOL[]` predicate forcing a fence stall mid-island); its `EDGE_ENTRY` position gives no argument for reopening it. 355/363 is not the number left to move — boundary bytes are (see the `fixed_ns` entry below).

---

### 2026-08-01T17:16:56-07:00: `fixed_ns` cannot change any partition decision; the estimator's own boundary-byte figure is off by 104,116× — R11 applied to Mouse's own model

**By:** Mouse (`fixed-ns-sensitivity`)

No timing figure is quoted (the device-clock gate that would calibrate nanoseconds is itself blind to bias, per the consolidated entry above). With the estimator's own bytes (`GetCapability`'s 89,199,100,032 boundary bytes for Phi-3.5's island), the economics check rejects at every `fixed_ns` swept, 0 to 1e8, six runs, both devices, all identical — the byte term alone exceeds the margin requirement by ~968×. With the *measured* boundary (upload+readback = 856,720 B), the verdict is also constant across the whole plausible range (flip solves to ~3.80 ms/transfer, 63× above the current 60 µs guess). **The finding that came out of the sweep:** the two boundary figures disagree by **104,116×** (89,199,100,032 vs 856,720 bytes) — not noise. `GetCapability`'s estimator deliberately over-counts every claimed node's outputs as boundary bytes and substitutes 128 for every unknown dimension; on a 355-node fused island with symbolic `sequence_length` those choices compound five orders of magnitude off the instrumented figure. **Mouse applies R11 to his own model:** the decomposition looked closed (bytes calibrated, a gate comparing compute to transfer, a counter proving the gate ran), and the two sides of the comparison came from different sources, only one a measurement — the gate's apparent strictness on Phi-3.5 is an artifact of its own byte estimator, and the anchor exemption is the sole reason the model is claimed at all (turn it off and the EP declines the entire graph, at every `fixed_ns`, both devices). Deliberately not fixed in this change — it fails safe toward CPU and is load-bearing for the anchor exemption's design intent — and is now the top of the partition backlog, ahead of any nanosecond calibration, ranked above Mouse's own nanosecond work.

---

### 2026-08-01T17:16:56-07:00: A claimed node whose `Compute()` fails now WARNs through ORT's own sink, unconditionally (RAI-008/RAI-010)

**By:** Tank (`broken-commitment-warn`)

Implements RAI Ruling 2. When a claimed node's `Compute()` returns non-OK, the EP emits a WARNING through ORT's registered logger, naming the subgraph, every node and op_type, a condition token, and the fact that CPU re-execution follows while `get_providers()` still lists the Vulkan EP. Three load-bearing properties: the call site is in `ep.rs::compute` immediately after `guard_ffi_status`, so a panic-converted `ORT_EP_FAIL` and a normal non-OK status disclose on the same path; it bypasses the `log` crate entirely (`forward_to_ort` directly) so no env `LevelFilter` can suppress it; scope is enforced by position (declined nodes never get a `Compute` call at all) rather than by a predicate, so there is no filter to keep correct. Two-polarity mutation-tested: removing the disclosure call fails the positive polarity (no marker on ORT's sink); disabling the early-return guard fails the negative polarity (a BROKEN COMMITMENT line on an otherwise-clean run). Discovered en route: ORT's default Windows sink writes UTF-16LE, and a UTF-8 grep over it matched nothing — the first version of this probe reported FAIL for a WARN that was delivered correctly, fixed by decoding four ways (this is the same channel Niobe's `bench/` decoder entry addresses independently).

---

### 2026-08-01T17:16:56-07:00: `ALLOW_MISSING_GLSLC=1` is unreachable on any host with the Vulkan SDK installed

**By:** Trinity (`allow-missing-glslc-unreachable`), owed to Tank (`build.rs`)

`build.rs::find_glslc()` only checks `ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC` after three search steps fail, and the third step is an unconditional scan of `C:\VulkanSDK` that succeeds on any machine capable of running this project's tests — so criterion 5's negative, as DESIGN.md words it, cannot be executed on any real test host. Requested fix: check the env var as a hard override *before* the search. Trinity's interim evidence: `tests/ops/_shaderless.py` builds from an emptied shader source set, which is equivalent by the artifact's own emitted reason string (present in the shader-less polarity, absent in the compiled one) — the equivalence is why the route is legitimate, not why the result is believed. Also flagged: `epctl --probe-loader`/`--dump-capabilities` do not consult `SHADER_MODULES` at all, so a shader-less build still reports 2 devices and 50 live ops — if `epctl` is meant to be the diagnostic front door, it is currently blind to this condition.

---

### 2026-08-01T17:16:56-07:00: The instrument census could not see a pytest fixture and scored it as a fabricated detection

**By:** Trinity (`census-screen-cannot-see-fixtures`)

Adding `ops/conftest.py` to Tank's `audit_instruments.py` frame (it was omitted — the census could not see the file that decides what a failure *is*) immediately produced a fabricated detection: `require_vulkan`, with 37 dependents, reported `UNINVOKED calls=0`, because a pytest fixture is depended on by naming it as a parameter — there is no `ast.Call` node for a call-expression screen to count, so any such screen scores every fixture in the repository as dead code. Fixed additively: `_fixture_instruments()` detects `@pytest.fixture` from the decorator list and counts fixture-parameter references as callers (`require_vulkan` now reads `calls=37, unfalsified` — fixtures never supply a polarity, so `unfalsified` is the honest terminal state for them). Baseline regenerated: 8 rust uninvoked (was 9 — real drift, `device_buffer_for` got wired on main), `CENSUS VERDICT: PASS`. Recorded but not implemented: the polarity model is raise-based, so every instrument that returns a token instead of raising (`classify_validation_probe`, `_classify_failure`, `classify_clean_read_frame`) reads `unfalsified` even when fully exercised — generalisation for whoever implements it: *screened iff a non-gated test asserts two different return values for two different inputs*, the same sentence as R10's falsifier.

---

### 2026-08-01T17:16:56-07:00: Criterion 3(a) is read in the dispatch window; the shutdown window is `UNOBSERVABLE`

**By:** Trinity (`criterion3a-frame-of-record`), raised by Switch

Switch flagged that the now-leaked production `VkDevice` (per §6.5) makes any "0 validation errors at shutdown" gate `UNOBSERVABLE` by construction. Trinity's first cut grepped the whole `cargo test` transcript for `VUID-`, mixing dispatch-time and shutdown-time lines — a leak-time VUID could blame a clean dispatch, or loosening the grep to fix that would lose the dispatch-time reading too (R9 amendment 5: a check that moves with the reader's confidence cannot be repaired by tightening). Fix: the reading is taken in a named frame recorded alongside the number — dispatch window (process start to the last verified per-device dispatch) is the frame of record; shutdown window is `UNOBSERVABLE`, never `0`. Falsifier: the same VUID line is placed on either side of a movable boundary and the classification must change — pins the splitter against one that keys on the VUID text (or nothing) rather than the frame boundary. Guards: a run that skipped every device raises `InstrumentError` rather than reading as clean, and "both devices" is checked (`dispatched_devices == capable_devices`) rather than assumed from running the test twice.

---

### 2026-08-01T17:16:56-07:00: A negative control must be shown to have fired, not asserted to — Link's own constant and Trinity's independent confirmation (consolidated)

**By:** Link (`icd-suppression-was-a-constant`), Trinity (`negative-control-must-be-shown-to-fire`)
**Why consolidated:** the same defect, found by Link building the fix and independently reproduced by Trinity in her own lane — corroboration, not restatement.

**Link's finding:** the Windows ICD negative control guarded itself with an inline match on `"passed the §7.2 capability gate"`, a string `rust/src/vk/instance.rs:1072` prints on **every** run (`n=0` when suppression works). The substring is present in both polarities, so the guard matched always, short-circuited to exit 0 always, and **the negative control had never once executed** — worse, it was silent-positive, reporting "the gate cannot fail" and blaming the gate. Fixed with `NO_ICD_RE` matching the actual failure text plus a process-exit-code witness that must agree with it; a second bug found only by running the fix against the real binary (the capability-gate line is never printed at all when the ICD is genuinely suppressed — `vkCreateInstance` fails before enumeration — so Link's own first draft would have misclassified every successful suppression).

**Trinity's independent confirmation, in her own lane:** `gate_line_present` came back `true` in **both** polarities on both devices, for the identical reason — a suppressed run's own error block quotes the same phrase the control was checking for. Trinity generalizes the rule: a witness artifact for "under condition X the EP does nothing" must contain both polarities from the same lane, plus a token computed by the suppression's own classifier (not a substring the test author chose), plus proof (via `library_sha_prefix`) that both rows share a binary. She also found the same defect's mirror image in her own wiring census: the `gpu_tracer` line read `ONNXRUNTIME_EP_VULKAN_TRACE_FILE`, which nothing defines (`trace.rs::ENV_TRACE` is `ONNXRUNTIME_EP_VULKAN_TRACE`), so it reported `OPTIONAL-UNWIRED` on every run it ever made — an always-true screen and an always-false screen are the same defect, only the direction of the lie differs.

---

### 2026-08-01T17:16:56-07:00: The wiring census — one census, mechanisms imported rather than re-implemented, `ledger_lookup` the last `UNWIRED`

**By:** Trinity (`wiring-census-one-census`)

`rust/tools/audit_instruments.py` (Tank's) is the repository's one source-level instrument census; `tests/ops/test_wiring_census.py` calls it via `main_guarded` rather than building a second one — Tank dropped his own harness-census WIP unmerged for exactly this reason. Other people's classifiers are imported, not re-implemented (Link's `check_device_state.py`/`check_icd_suppression.py::classify`, Tank's `ort_sink_warns`) — a second copy of a classifier is a second thing to be wrong, silently disagreeing with the first. JSON-only observables (net-benefit fields, the broken-commitment channel, typed `viable_islands_retained`) are not in the C ABI struct, so the census spawns a fresh child per polarity with the environment set before the DLL loads (Windows UCRT env caching makes a late-set env var invisible). Census run, both devices: exactly one mechanism reports `UNWIRED` — `ledger_lookup` (criterion 11, `xfail(strict=True)`, Mouse is building it) — everything else (`partitioner`, `net_benefit_gate: EVALUATED`, `broken_commitment_warn`, `device_state_guard: FIRED`, `instrument_census: PASS`) reads a real, moved value. No duration is quoted anywhere in the artifact (obligation 8).

---

### 2026-08-01T17:16:56-07:00: Criterion 5's CI gate is not exposed to denominator inflation, but the invariance property is itself unenforced

**By:** Link (`criterion5-denominator-not-exposed`)

Morpheus's named attack — a share is gamed by inflating its total, so a lane could satisfy a threshold by making the whole run slower — does not reach `gate_chain_fp32`: its verdict is a function of counts and one exact comparison (`own_provider_execution_count`, `counters_dispatches_executed`, `profile_node_events`, `max_abs_diff`), with no total and therefore no denominator; both negative controls (op-coverage refusal, ICD suppression) are equally clock-invariant, per §10.0.4. Explicitly not claimed as foresight: the gate is count-based because CI timing thresholds are flaky, an operational convenience that happens to coincide with the correct evidentiary property — and the property is **not protected by anything**. Nothing stops a future contributor adding a ratio-over-a-total and reintroducing the exposure; the named risk in the other direction (§10.0.4) is handing a reader a bare count and letting them supply the clock. Partial mitigation in place (`ci/check_device_state.py` requires a device-state record the moment a duration is added), but a check that the gate's inputs are all clock-invariant is unstarted and named as such rather than implied to be covered.

---

### 2026-08-01T17:16:56-07:00: A second, loader-independent negative control; a vocabulary outage must name its own kind; Android citation rule

**By:** Link (`gate-negative-control-and-vocab-outage`)

`ci/gate_chain_fp32.py --artifact decline_probe` builds a single-`Det`-op graph (an op this EP does not implement) so every lane must report `FAIL(condition=UNATTRIBUTED)` even though the EP loads and the device passes §7.2 — the existing ICD-removal control only reproduces "the EP could not start," not "the EP started and executed nothing," which was the failure live on 2026-07-30 that a green suite hid. Consequence for Windows: §7.4.1's elevated-runner loader bug (LunarG silently ignores `VK_DRIVER_FILES`/`VK_ICD_FILENAMES` when elevated, and GitHub's Windows runners are elevated) means the ICD-removal control there may never have fired; it now probes the loader first and reports `ERROR(instrument=icd_suppression_ineffective)` rather than a false pass. `ci/check_vocabulary.py` runs before every gate and reports one of three distinct tokens for a missing `tests/ops/_verdict.py` (present/absent-from-checkout/broken) rather than one blanket error, so an outage is legible rather than universally on. Android citation rule: the sync2 figure is quoted only as "~32.67% as of 2026-07-30, simultaneously a ceiling and a floor," never as a bare percentage — and lavapipe is not Adreno or Mali, so no lavapipe result may be cited as Android evidence. No lane was promoted to green by any of this; the CI lanes carry the gate and have still not been observed on a runner.
