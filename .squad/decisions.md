# Squad Decisions

<!-- Round 1: 2026-07-28T17:59:54-07:00 (design session)              → ARCHIVED -->
<!-- Round 2: 2026-07-28T22:28:08-07:00 (implementation round)        → ARCHIVED -->
<!-- Round 3: 2026-07-29T09:00:39-07:00 (first-hardware round)        → ARCHIVED -->
<!-- Round 4: 2026-07-30T19:05:03-07:00 (correctness-and-measurement) → ACTIVE  -->

## Active Decisions

<!-- ═══════════════════════════════════════════════════════════════════════════════ -->
<!-- ARCHIVAL POINTER — 2026-07-30T20:58:11-07:00                                  -->
<!-- Rounds 1–3 + earlier-today session entries archived to:                        -->
<!--   .squad/decisions-archive/2026-07-30T20-58-11-07-00-rounds-1-3.md            -->
<!-- Archived: 122 entries total (57 from 07-28, 22 from 07-29, 43 from 07-30      -->
<!--   predating this Scribe session — the 43 include agent direct-to-file writes  -->
<!--   from sessions earlier today; many overlap with inbox-merged Round 4 entries) -->
<!-- Restored from archive: 1 entry (2026-07-30T05:48:29 Morpheus metric-of-record -->
<!--   — today-dated, semantically active, should remain in the live file)          -->
<!-- Trigger: decisions.md reached 164 KB (volume threshold: 40 KB).               -->
<!-- Policy: archive all entries before the current Scribe run's Round separator.   -->
<!-- Floor (refined): "before the current session's Round marker" not "calendar     -->
<!--   day" — the floor was mis-stated as calendar-day; the implementation used the -->
<!--   Round 4 separator which is correct; 43 same-calendar-day entries from        -->
<!--   earlier sessions were archived, which is acceptable since the boundary is    -->
<!--   the Scribe-run boundary, not the wall-clock date.                           -->
<!-- Supersession: corrections are always in later rounds; the live file is         -->
<!--   authoritative; archived entries may have been superseded.                   -->
<!-- Previous archival check (2026-07-30T19:05:03-07:00): 0 entries, age gate only -->
<!-- ═══════════════════════════════════════════════════════════════════════════════ -->

### 2026-07-30T05:48:29-07:00: Metric of record gated on `model_output_equivalence`; M0 sequencing — criterion 10 before CI tail [morpheus-metric-correctness-gate.md]

**By:** Morpheus (Lead / EP Architect)
**What:** The metric of record (triple: `claimed_op_coverage`, `island_count`, `largest_island_flops`) is now gated on `model_output_equivalence`: `MATCH` → triple may be reported as a result; `DIVERGENT` → triple may not be reported as progress (run reports which outputs, max-abs-diff, argmax/top-k agreement); `UNMEASURED` → triple may not be reported as progress (may be reported as a claim-path diagnostic, labelled `UNMEASURED`). Four rulings: (1) `UNMEASURED` is the default, not a soft `MATCH`. (2) Verdict is per artifact at producer-at-version; never generalises. (3) Gate not a term — correctness is not commensurable with coverage so must not appear in a row that invites arithmetic. (4) Comparison is against a CPU-only run of the same session on the same artifact, not a stored golden vector. Owners: Trinity emits, Niobe carries in `PERF.md`, Mouse carries in census. Sequencing ruling: criterion 10 goes `MATCH` before the M0 CI tail (lavapipe/Windows/Linux) is worth closing; CI lanes must carry criterion 10's gate when they run, or they measure the same silence in a new location. Link is not blocked — parallel Linux lane work is correct and prerequisite for running criterion 10 on CI.
**Why:** Coverage went 0 → 161 nodes and the model went from correct via CPU fallback to wrong via GPU. A coverage number that rises while the answer becomes wrong is worse than no number — it recruits effort in the wrong direction with the full authority of a metric of record. `UNMEASURED` as the default is R7 arriving for the fifth time on the execution path.


<!-- =============================================================================== -->
<!-- ROUND 4 DECISIONS -- 2026-07-30T19:05:03-07:00 (correctness-and-measurement)  -->
<!-- 32 inbox records merged; 3 consolidated entries; 27 effective entries          -->
<!-- =============================================================================== -->

### 2026-07-30T19:05:03-07:00: Binding-arity defect -- zero logits root cause, independent diagnosis by Switch and Mouse (consolidated)

**By:** Switch, Mouse (independent routes to same root cause)
**Why consolidated:** Switch reached the root cause through runtime diagnostics (validation layer + output-byte dump probe); Mouse reached it through static code analysis (session.rs dispatch path). Per R9: agreement raises confidence only when routes are genuinely independent -- this is evidence, not redundancy. switch-merge-mouse-fix documents the merge resolution; switch-kv-cache-explained documents the KV symptom as the same root cause. All four inbox records describe one event from different angles.

**Root cause:** push_dynamic_kernel built a descriptor-set layout with n_bindings = n_plan_inputs + n_plan_outputs (4 for 3-input MatMulNBits without zero_points). The translate handler (quant.rs) correctly emits 5 binding tokens -- scales is bound twice at slots 2 and 3 as a zero-point placeholder. The shader q_gemv.comp declares 5 bindings (0-4); slot 4 is output OutY. On the dynamic-shape path, the pipeline layout had only 4 slots; the write to slot 4 was against an undeclared descriptor. Both Intel and NVIDIA silently discard writes to undeclared descriptors. DeviceLocal output buffer stayed at zero-initialized value. compute_failures: 0 throughout -- the failure was downstream of the dispatch, invisible to all execution-status counters.

**Switch's diagnostic route:** Added DUMP_OUTPUT_BYTES probe; found 33 of 257 dispatches produced all-zero staging buffers (32 qkv_proj + 1 lm_head). Push-constant and workgroup dump confirmed dispatch geometry correct. Armed validation layer; messenger reported "vkCreateComputePipelines(): SPIR-V uses descriptor [Set 0, Binding 4] but the binding was not declared." This was the direct pointer.

**Mouse's diagnostic route:** Read compile_impl -> push_dynamic_kernel -> ShapeOnlyRecorder::dispatch. Counted binding tokens from node-desc counts (4) vs translate-handler output (5). The mismatch was readable in the code without running.

**Why static-shape tests passed:** test_matmulnbits_fp16_matrix uses concrete rows, so is_dynamic = False. Static path runs through CompileRecorder which captures the 5-element binding vector from the translate handler directly. Only the symbolic-batch form triggers push_dynamic_kernel.

**Fix:** ShapeOnlyRecorder::dispatch() now captures k.bindings (the translate-handler token sequence at Compute time). At dispatch, eff_bindings for dynamic kernels comes from the capture; both pipeline layout entry count and descriptor buffer writes iterate over eff_bindings. Mouse's structural split (captured 4-tuple + captured_bindings: Option<Vec<u64>>) adopted. Switch's diagnostic infrastructure and plant/messenger test retained per M0 criterion 3 requirement. Merged to main as 93aca04.

**KV-cache symptom (same root cause):** Tank observed two populations -- logits exactly 0.0 on runs 2+ (dirty arena, something wrote zeros) and KV cache bitwise different between runs (unwritten, shows arena residue). Post-fix probe_run2.py: BIT-IDENTICAL across all 65 outputs on both devices for 3 runs. The KV cache is produced entirely by CPU ops (SDPA); dirty arena residue on run 2+ was the consequence of CPU attention computing from zero QKV inputs.

**General invariant added:** For any dynamic-shape kernel, push_dynamic_kernel is a positional approximation. Pipeline descriptor count MUST come from the translate-time capture (ShapeOnlyRecorder::captured_bindings), not from kernel.bindings.len() (the compile-time count). Any translate handler that produces a different binding count than n_inputs + n_outputs will trigger this mismatch for dynamic-shape kernels -- the fix covers the whole class.

**Falsifier:** test_matmulnbits_fp16_dynamic_batch and test_matmulnbits_fp16_dynamic_batch_multirun -- FAIL pre-fix, PASS post-fix, both devices.

---

### 2026-07-30T19:05:03-07:00: Multi-run evidence required; cross-run consistency gate (consolidated)

**By:** Mouse (rule authorship), Trinity (gate implementation)
**Why consolidated:** Tank's finding led Mouse to formulate the rule (mouse-multirun-evidence-rule) and led Trinity to implement the corresponding gate (trinity-multi-run-cross-run-2026-07-30). Same event, different angles.

**What:** ORT's memory-pattern planner does not engage on run 1 of a session. The arena is OS-zeroed. An unwritten output buffer shows zeros on run 1 -- indistinguishable from a kernel that computes zeros. From run 2 onward the arena is dirty; an unwritten buffer shows residue, a zero-computing kernel still shows zeros. Therefore: a single-run MATCH cannot distinguish "written correctly" from "unwritten in a clean arena." Any test claiming write-completeness MUST run the session at least 3 times.

**Cross-run comparison:** Run-N vs run-N+1 (same session, identical feeds, byte-level) detects unwritten buffers in dirty arenas. This is a distinct axis from EP-vs-CPU: cross-run detects write completeness; EP-vs-CPU detects arithmetic correctness given a write occurred. Both are necessary; neither is sufficient.

**Tank's measurement:** 0 interior pointers at 1 run, 596 per run from run 2 onward, linear, zero intercept. The planner nearly stops allocating and freeing from run 2.

**Mouse's additions:** test_matmulnbits_fp16_dynamic_batch_multirun (3 runs, symbolic_batch=True) and test_phi35_vulkan_multirun_logits_stable (3 runs, Phi-3.5). Both PASS post-fix.

**Trinity's gate:** test_phi35_vulkan_matches_cpu_logits now runs 3 EP runs and uses run 2 (dirty arena) for EP-vs-CPU comparison. test_phi35_vulkan_cross_run_consistency asserts byte-equality of all 65 outputs across 3 runs. Byte comparison uses a.tobytes() == b.tobytes() -- not max|a-b| which returns NaN when either side has NaN, even when bit-identical. compare_layers() changed to np.nanmax(abs_diff) + nan_count field.

**Falsifier:** test_phi35_vulkan_cross_run_consistency was xfail(strict=True) until the binding-arity fix; xfail removed after probe_run2.py confirmed BIT-IDENTICAL across all 65 outputs.

---

### 2026-07-30T19:05:03-07:00: Device-backed allocation -- prediction pre-registered, result recorded, prediction falsified on discrete GPU (consolidated)

**By:** Tank (D-T82 through D-T89)
**Why consolidated:** Prediction (tank-device-memory-prediction.md) was written before implementation as required by the coordinator. Result (tank-device-memory-result.md) was written at session end. Both are facts about the same experiment; the prediction is load-bearing precisely because it was falsified.

**Prediction (D-T82, written before implementation):** Device-backed allocation on its own will make Phi-3.5 SLOWER on both devices. RTX 4060: 1.1x-1.6x slower (PCIe crossing); Iris Xe: 1.0x-1.2x slower (UMA). Mechanism: session.rs resolves every kernel input through host_backing_for and allocates its own GpuBuffer per Compute call; device-backing changes where CopyTensors writes, not the re-upload loop.

**Result (D-T83 through D-T89):** Falsifier fired on the discrete card. RTX 4060: 0.94x (faster), not 1.1x-1.6x slower. Iris Xe: 1.01x (within noise). Root cause of falsification: alloc_device_uploads = 386 for 19 inferences -- upload fires once per allocation (at weight deserialization, during warm-up, outside the measured timing loop).

**Mirror architecture (D-T83):** Making a device-backed handle refuse to yield a host address caused CopyTensors failures and ORT CPU fallback. Fix: every span gets a real VkBuffer in DEVICE_LOCAL memory in addition to its host staging block (mirror). Host receives writes from ORT; device receives vkCmdCopyBuffer from host; device never read back. Correctness by construction (one writer, one direction). alloc_device_backed_spans > 0; alloc_device_authoritative_spans = 0 (never incremented). staging_verdict() adds MIRRORED state.

**epctl still exits 1 (correct):** Every span is both staged and device-backed; the flag's question ("tensors resident in device memory where kernels read them") is still answered No.

**Seam handed to Switch:** transfer::device_buffer_for(ptr, len) -> Option<(BufferView, usize)>. The usize interior offset is not optional -- the planner sub-divides one span across several tensors; binding at offset 0 for an interior pointer overwrites a neighbour.

---

### 2026-07-30T19:05:03-07:00: Fact Checker -- OQ-12 figure currency; opset-26 verified; Q/DQ oracle risk; no-bump class growing

**By:** Fact Checker
**Date:** 2026-07-30T07:05:09-07:00

**OQ-12:** The 68.57% sync2 Android coverage figure (31.43% gap) is a live database snapshot from 2026-07-28. As of 2026-07-30 the figure is ~67.33% (gap ~32.67%). The figure is simultaneously a ceiling on the legacy-path benefit (sync2-lacking devices may also fail the S7.2 device gate) and a floor on the real gap (gpuinfo.org over-represents newer hardware). Drop conditions for the legacy barrier path: both Android AND Windows database coverage >= 99% simultaneously, AND OQ-12 confirms gap devices fail S7.2. Neither condition is close today. Legacy path justified.

**Opset 26 verified:** ONNX 1.21.0 (2026-04-27) introduced opset 26. ONNX 1.22.0 (2026-06-15) introduced opset 27. ORT 1.28 supports through opset 27. The onnxruntime.ai compatibility table is stale. "Support opset 26" is fully exercisable against ORT 1.28, with headroom to opset 27.

**onnx#8182 Q/DQ oracle failure (AFFECTS OUR TIER PLAN):** Q/DQ-23 and Q/DQ-25 reference implementations not registered in onnx <= 1.22.0. ReferenceEvaluator silently falls back to opset-21 implementations. Result: TypeError on output_dtype (detectable) or silent wrong reference (if basic Q/DQ form tested without those attributes -- not detectable by C2). Action: pin onnx >= 1.23 for Q/DQ-23/-25 oracle; avoid output_dtype/precision in conformance tests until 1.23 ships. Trinity owns the guard (assert_qdq_reference_oracle_safe).

**No-bump class:** 9 known instances as of 2026-07-30 (6 existing + #8182 + #8099 ScatterND + #8194 TopK). All three new instances post-date onnx 1.22.0. The table needs a monthly sweep owner (Mouse or Trinity).

---

### 2026-07-30T19:05:03-07:00: Link -- lavapipe first claimed-node execution; gate artifact; subgroup-32 red instrument; is_uma verified

**By:** Link
**Dates:** 2026-07-30T05:48:29-07:00, 2026-07-30T08:21:19-07:00

**Lavapipe first claimed-node execution:** Linux lavapipe (Mesa 25.2.8, Ubuntu 24.04, WSL2, Vulkan 1.4.318) executed claimed nodes end-to-end. Subgroup_size=8 confirmed. Sync2 promoted to 1.3 core -> Barriers::select -> Sync2Backend::Core. Barrier parity: 58 passed / 0 failed / 28 skipped. Zero lavapipe-specific failures. Zero numerical correctness failures. Lane is operational, NOT green (green requires criterion 10 gate artifact with MATCH verdict; UNMEASURED is the default by construction).

**WSL build notes:** CARGO_TARGET_DIR must be set outside /tmp (systemd private-tmp recycling); WSL root bash subshells have empty PATH; use wsl -d Ubuntu -u root for elevated operations. VK_ICD_FILENAMES works for non-elevated processes in WSL.

**Gate artifact for lavapipe:** gate_chain_fp32 -- Add(X,Y) -> Relu(Z) on fp32 [256]. Claims 2 nodes; 1 island of 2 nodes; fp32 ew_binary + ew_unary proof keys. Feed: linspace(-1,1,256) + ones(256) (spans zero, exercises Relu clamp). Oracle: ORT CPU EP, FP32_ELEMENTWISE tolerance. fp16 deferred until storageBuffer16BitAccess and shaderFloat16 confirmed on lavapipe. Trinity implements the verdict file mechanism; Link wires it into the lavapipe lane spec (PLATFORMS.md S7.8).

**Subgroup-32 red instrument (closed):** A shader baking gl_SubgroupSize == 32 produces wrong reduction outputs on lavapipe (subgroup_size=8). Wrong outputs diverge from CPU reference. test_elementwise.py on lavapipe goes RED. New shader templates must have a lavapipe numerical test before op moves Staged -> Ready. This is a meaningful threat because the test checks numerical output, not just dispatch.

**is_uma verified correct:** "Every heap is DEVICE_LOCAL" predicate correctly returns false for RTX 4060 with ReBAR (two heaps, DEVICE_LOCAL|HV + empty) and true for lavapipe (one DEVICE_LOCAL|HV heap), for the correct reason. Unit test at caps.rs:671-679 specifically tests the ReBAR case.

**OQ-12 figure update adopted from Fact Checker:** Figure now ~67.33% (gap ~32.67%) as of 2026-07-30; pull date updated in PLATFORMS.md with floor-and-ceiling analysis preserved.

---

### 2026-07-30T19:05:03-07:00: Morpheus -- claiming gated on proven correctness (S7.0.1 / S8.9)

**By:** Morpheus
**Date:** 2026-07-30T06:32:18-07:00

**Ruling:** Claiming is gated on proven correctness. An op we have not proven correct on a form is, for claiming purposes, an op we cannot run on that form. S7.0 was silent on ops that can dispatch but have not been proven; S7.0.1 closes that gap.

**Mechanism:** Claimability derived per-form from a proof ledger. claim_decision claims only if status == Ready AND the ledger holds an entry under the node's proof key. No entry -> decline with [unproven]. Proof key 7-tuple: (domain, op_type, opset_bucket, element dtypes, kernel_variant_key, shape_class, populated_optional_input_set). Ledger generated by differential harness, never hand-edited, baked into cdylib. Tier 1 (per-form proof) gates claiming; Tier 2 (per-producer-at-version model proof) gates reporting and can retract Tier 1.

**Escape hatch:** ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN takes a comma-separated list of full proof keys only -- no boolean, no wildcard. A parser that can express "everything" must not exist (enforced by 6 planted rejection tests). Default is safe: unset -> unproven forms decline. Three disclosures: session WARN naming every enabled key; unproven_forms_enabled in counters artifact; epctl --check-counters fails on non-empty list without --allow-unproven. Available in release builds (Trinity and Link's lanes build release).

**Cost acknowledged:** Phi-3.5 goes 161 -> 0 claimed nodes when gate activates. The honest number was already 0 (DIVERGENT verdict voids the metric triple; the reporting was wrong, not the code).

**OpStatus::Live retired:** Deprecated as a hand-written duplicate of a machine-known fact (R7: derive, do not declare). Add-i32 Live on an f32-only predicate was the first specimen; MatMulNBits Live on a FLOAT mask with f32-only proof was the second, and it shipped.

---

### 2026-07-30T19:05:03-07:00: Morpheus -- correctness gate is precondition for *green* lanes, not for bring-up

**By:** Morpheus
**Date:** 2026-07-30T06:32:18-07:00

**Ruling:** Operational (lane exists, executes, reports) and Green (lane result is admissible as evidence) are different. Link may declare Operational without criterion 10; Green requires the gate (criterion 10 MATCH verdict). Made unrepresentable: a lane's pass condition includes the verdict field; UNMEASURED reports UNMEASURED (not PASS). A lane cannot accidentally go green.

**Gate artifact sizing:** Not Phi-3.5 on CI (2.2 GB on a software rasterizer in CI is infeasible). Each lane carries the smallest real producer-at-version model that (a) claims non-zero nodes, (b) has an island of >= 2 nodes, (c) exercises >= 1 proof key per claimed dtype. Trinity chooses and pins; Link wires.

**Why not gate-as-follow-on:** Three ungated lanes agreeing are three more corroborating instruments added to a set with no falsifier -- the precise mechanism by which the wrong conclusion became persuasive on 2026-07-30.

---

### 2026-07-30T19:05:03-07:00: Mouse -- accuracy_level=0 vs =1 oracle pinning is correct

**By:** Mouse
**Date:** 2026-07-30T09:14:00-07:00

ORT CPU kernel maps levels 0-3 to SQNBIT_CompFp32 (fp32 accumulation). Only level 4 selects int8 VNNI. GPU shader accumulates in float acc = 0.0 unconditionally; attribute not passed as push constant or spec constant. Levels 0 and 1 are identical computations on x86. Oracle pinning (level 1) is correct for a model declaring level 0. No change needed. If ORT 1.28+ changes the level-0/1 mapping: re-run test_matmulnbits_accuracy_level_pinning() to verify.

---

### 2026-07-30T19:05:03-07:00: Mouse -- multi-node island dispatch: intermediate buffer aliasing root cause and fix

**By:** Mouse
**Date:** 2026-07-30T09:14:00-07:00

**The defect:** After partition.rs wiring, dispatch_accounting RED: compute_calls 1 != expected 1023. 33 islands compiled live; only 1 Compute call dispatched successfully. model_output_equivalence = MATCH appeared to hold (CPU vs CPU). compute_failures: 0 throughout.

**Root cause:** Positional token scheme resetting next_bind = 0 per kernel in ShapeOnlyRecorder. For 2-node island {A,B}: A's output -> token n_plan_inputs+0 at Compile time; B's output -> token n_plan_inputs+1. At Compute pre-pass: ShapeOnlyRecorder restarts for B -> B's output -> n_plan_inputs+0. Dispatch: eff_bindings for B has token n_plan_inputs+1 -> j=1 >= n_ort=1 -> gpu_temps[0] (empty) -> panic. guard_ffi_status caught panic; ORT abandoned EP after first failure; all subsequent inferences on CPU.

**Fix:** Name-based token assignment from the island's plan at Compile time. Token ranges non-overlapping by construction: 0..n_plan_inputs (ORT inputs), n_plan_inputs..n_plan_inputs+n_plan_outputs (ORT outputs), intermediates, alloc_temp scratch. compile_impl builds name_to_token: HashMap<String, u64>. Inter-kernel barriers (SHADER_WRITE -> SHADER_READ) on intermediates between dispatches.

**Results:** Intel: 2954.6 ms -> 807.2 ms (3.7x). NVIDIA: 7.9x -> 4.1x slowdown (still slow, but all 1023 Compute calls executing). dispatch_accounting GREEN: compute_calls 1023 == 33 x 31 on both devices.

**The general rule:** A mechanism that exists in a file and not in a call graph is indistinguishable from a mechanism that does not exist. partition.rs existed in full; the wiring was absent; the improvement was invisible to every counter except dispatch_accounting.

---

### 2026-07-30T19:05:03-07:00: Mouse -- proof ledger scaffolding (S8.9 groundwork)

**By:** Mouse
**Date:** 2026-07-30T08:29:00-07:00

Built: OpStatus::Ready (new variant; Live deprecated); DeclineCode::Unproven (tag "unproven"); ProofKey struct (7-tuple per S8.9.2, string form domain::op_type/opset_bucket/dtypes/variant/shape_class/inputs); claim_unproven_keys() (parses env var, comma-separated full keys only; invalid key -> WARN + empty list); ledger_contains() stub (always false pending Trinity's harness); 6 planted rejection tests (reject *, 1, all, bare op-type, empty).

Gate NOT yet activated in claim_audit -- requires Trinity's harness, build.rs bake, ledger read, and CI gate.

The populated_optional_input_set component was motivated by the binding-arity defect: MatMulNBits without zero_points has 4 bindings; with zero_points has 5. A proof of the 4-binding form cannot cover the 5-binding form.

The 6 planted rejection tests must never be deleted or weakened without a S8.9.4 amendment.

---

### 2026-07-30T19:05:03-07:00: Mouse -- SkipSimplifiedLayerNorm island measurement; alloc_temp infrastructure fix

**By:** Mouse
**Date:** 2026-07-30T09:14:00-07:00

**Finding:** Promoting SkipSimplifiedLayerNorm (128 nodes) increased island count 257 -> 321 (delta: +64 islands). No islands merged. Coordinator's prediction of 128-200 fewer islands was wrong (falsifier fired).

**Why:** Claiming a new op type adds islands before it removes them. Islands removed only when the newly claimed op is the *last* unclaimed gap between two existing islands. On Phi-3.5, other unclaimed ops (Mul, Sigmoid) sit in the same regions; ORT cannot merge the new SkipNorm islands with MatMulNBits islands while those remain.

**New rule (in OP_COVERAGE.md S7.1.4):** Use the declined_nodes histogram to compute island-removal potential before implementing an op. The question is: "how many of X's nodes are the last unclaimed gap between two existing islands?"

**alloc_temp fix:** skip_norm is the first translate handler to call alloc_temp in production (slot 3 absent -> scratch buffer for residual write). dispatch_ort had never exercised this code path; buf_bindings indexed only gpu_outputs, panicking on temp tokens. Fixed: CompileRecorder::alloc_temp records pending_temp_sizes; dispatch_ort allocates gpu_temps pool; routes temp tokens via temp_starts[ki] + (j - n_ort). Falsifier: test_skip_norm_f32_slot0_matches_cpu and test_skip_norm_f16_slot0_matches_cpu were RED before fix (panic), GREEN after.

---

### 2026-07-30T19:05:03-07:00: Niobe -- first real measurement; benchmark discipline; tracer wiring gap

**By:** Niobe
**Date:** 2026-07-30T08:21:19-07:00

**First measurement (D-N41), 257 islands, staging-bound, model_output_equivalence = MATCH:**
| device | vulkan | cpu-only | delta | per-island lower bound |
|---|---|---|---|---|
| Intel Iris Xe (UMA) | 2790.7 ms | 229.8 ms | +2561 ms (12.1x) | >= 9.96 ms |
| RTX 4060 (discrete) | 1465.9 ms | 185.9 ms | +1280 ms (7.9x) | >= 4.98 ms |

dispatch_accounting: 7967 == 257 x 31 GREEN on both devices.

**Benchmark disciplines (D-N29 through D-N38):** Benchmark computes its own S10.0 verdict in-run. Correctness gate before timing; timing unreachable on non-MATCH. island_count from subgraphs_live counter only. dispatch_accounting = hard check, no tolerance. stats.drift() reports trending runs (a trending run is invalidated by more samples, not improved by them -- measured on Iris Xe: 724 -> 695 -> 903 -> 1447 -> 2080 -> 2669 -> flat near 2790 ms). --repeats=3 whole-process repeats; within-run rsd != cross-run stability. baseline_disagreement fires above 2x (CPU baseline 218 ms vs 665 ms on same hardware minutes apart due to page-cache pressure after loading a 2.2 GB model).

**Timestamp audit (D-N37 through D-N39):** Intel: 52.0833 ns/tick, 36 valid bits (wrap period 3579 s). NVIDIA/lavapipe: 1.0 ns/tick, 64 bits. timestamp_audit.py exits non-zero when period_mistake_detectable_on or mask_exercisable_on lists are empty. Intel Iris Xe is the ONLY local instrument for period or mask bugs; CI has none.

**Tracer wiring (D-N40) -- four separate facts, fourth is NOT DONE:** dependency pinned OK; module written and tested OK; env wiring OK; called from execution path: NO. Verified empirically: 257 islands, 4 inferences, no trace file. Switch addressed in session 25.

**Key implications:**
- Cheapest large win is fewer islands, not faster MatMulNBits. At >= 5 ms/crossing on NVIDIA, every GQA (32) and SkipNorm that lands removes two crossings.
- Intel costs ~2x more per island than discrete while having no bus to cross -- argues for fixed per-submission cost, not per-boundary PCIe transfer. Hypothesis, pending S3 timestamps.
- 85.9% of inference runtime (recording + fence-wait idle + submit) involves no GPU kernel work. Kernel optimisation is not the best move right now.

---

### 2026-07-30T19:05:03-07:00: Rai -- silent inference: two verdicts, the line, and the mechanism

**By:** Rai (RAI Reviewer)
**Date:** 2026-07-30T07:12:15-07:00
**Status:** Red on shipping architecture; Reviewer Rejection Protocol applies. Green on gated architecture.

**Verdict 1 (the architecture without gate):** An EP that claims nodes and writes zeros without notification crosses the line at RAI framework S6.1 -- "silent model degradation without user awareness." The user cannot distinguish a session where the EP ran correctly from one where it silently produced zeros. Rai rules Red on the shipping-without-gate architecture.

**Verdict 2 (Morpheus's gate):** The S8.9 proof-ledger gate closes the loop. With the gate active: unproven forms decline; CPU EP runs; user gets a correct answer. Rai rules Green on the gated architecture, conditional on the three disclosures being present and non-bypassable.

**The line:** An EP may decline, fall back, and log. It must not silently produce wrong answers. The difference between "no GPU" and "GPU running incorrectly" must be observable without reading source code. compute_failures: 0 while producing zeros fails this standard.

**Independence note:** Rai's ruling does not depend on the engineering rationale and would not change if the engineering rationale were withdrawn -- stated explicitly so a reader cannot mistake parallel reasoning for corroboration (R9's failure mode applied to agents rather than counters).

---

### 2026-07-30T19:05:03-07:00: Switch -- ENGINE_ACCEPTS_RUNTIME_EXTENTS flipped; 97 nodes unlocked (D-S18-01)

**By:** Switch
**Date:** 2026-07-30T01:00:00-07:00

ENGINE_ACCEPTS_RUNTIME_EXTENTS = true. 97 nodes on Phi-3.5 (Mul x64, Sigmoid x32, Sub x1) were declined [dynamic-shape] solely because the engine baked extents at Compile. No new kernels needed -- only the dispatch path changed. Three preconditions met: (1) CompiledKernel stops baking at Compile for dynamic nodes via DynKernelRecipe; (2) dispatch_ort reads real shapes at Compute via GetTensorSizeInBytes + read_tensor_desc_from_ort; (3) translate handlers re-run against real shapes via ShapeOnlyRecorder. OQ-15 resolved for M1: re-record per shape (M2+ bucketing is an optimization for persistent-buffer mode). Verified: 161 claimed on both devices; variable seqlen (seq=1 and seq=5 in same session) correct on both devices; cargo ci green.

---

### 2026-07-30T19:05:03-07:00: Switch -- EP validation messenger; fence-leak plant; counter scoping fixes

**By:** Switch
**Date:** 2026-07-30T03:52:28-07:00

**EP-side messenger:** EP_VALIDATION_ERROR_COUNT AtomicU32 static; validation_log_callback installed as VkDebugUtilsMessengerEXT callback routing ERROR to log::error! AND incrementing the counter; Instance gains debug_messenger; Instance::create requests VK_EXT_debug_utils when enable_validation=true.

**Fence-leak plant (M0 criterion 3):** env-gated (ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION). vkDestroyDevice fires VUID-vkDestroyDevice-device-05137. EP_VALIDATION_ERROR_COUNT = 1 confirmed on RTX 4060.

**Counter scoping fixes:** Removed FIRST_DISPATCH_DUMPED one-shot; record_dispatches() calls dump_if_requested() on every dispatch (conftest.py Add-probe no longer overwrites Phi-3.5 counters). Islands regex fixed: count distinct args["op_name"] values among VulkanExecutionProvider events (actual ORT plugin-EP event names have a large hash). Verified: compile_calls=1, subgraphs_live=161, compute_calls=161, compute_failures=0, dispatches_executed=161, islands=161 on both devices.

Multi-run interior-pointer safety test: test_phi35_multi_run_same_session_interior_pointer_safety -- 5 consecutive inferences, asserts all 5 outputs bit-identical and dispatches_executed == 5 x subgraphs_live.

---

### 2026-07-30T19:05:03-07:00: Switch -- VkQueryPool GPU timestamps; tracer end-to-end; phase split measured

**By:** Switch
**Dates:** 2026-07-30T11:27:08-07:00 (implementation), 2026-07-30T15:41:27-07:00 (session 27 results)

**Implementation:** timestamp.rs + barrier.rs helper (cmd_write_compute_timestamp confines PipelineStageFlags to barrier.rs per layering rule 7.5). Bracketing calibration: host_anchor_us = (before_submit_us + after_fence_us) / 2; no VK_EXT_calibrated_timestamps (not universal on MoltenVK). tracer().export() wired in VulkanEp::drop() -- was the missing link (tracer accumulated spans; file never written). record_partition() cross-owner edit to ep.rs.

**Empirical confirmation (D-S27-01):** 322 GPU kernel spans with real durations on Intel Iris Xe (52.0833 ns/tick). 33 islands in vulkan.getcapability event args. 1 span per claimed node.

**Phase split (D-S27-04), 34 submissions, 322 kernel dispatches, 1 inference:**
| Metric | Intel Iris Xe | NVIDIA RTX 4060 |
|---|---|---|
| GPU kernel total | 784.6 ms | 48.3 ms |
| Fence-wait total | 893.8 ms | 82.7 ms |
| Record total (CPU) | 1340.3 ms | 1316.6 ms |
| Per-submission record time | 39.4 ms | 38.7 ms |

**85.9% of runtime involves no GPU kernel work:** recording 68.3% + fence-wait GPU idle 16.3% + submit 0.3%. Command-buffer re-recording is the dominant mechanism (not driver dispatch overhead; not per-boundary transfer). NVIDIA recording (1317 ms) is 27x GPU kernel time (48 ms). Niobe's fixed-per-submission hypothesis: CONFIRMED that there is a fixed cost, but the dominant term is re-recording, not submit overhead (submit is 0.3%).

**Expected speedup from record-once/replay:** 94% NVIDIA, 60% Intel. This is the highest-leverage optimization available.

**Declined-op histogram (D-S27-05):** GQA x32 staged (primary island splitter), Gather x2, Cast x2, SimplifiedLayerNorm x1, Shape x1, ReduceSum x1, Sub i64 x1, Greater x1, If x1. GQA is Mouse's highest-priority kernel (potentially reduces 33 -> 1 island if all are bridge positions).

---

### 2026-07-30T19:05:03-07:00: Switch -- session 26: multi-node island merge resolution + clippy fixes

**By:** Switch
**Date:** 2026-07-30T15:41:27-07:00

Mouse's dc36166 wired partition.rs into GetCapability (321 -> 33 islands, Intel 2954 ms -> 807 ms). git merge origin/main conflicted in session.rs. Resolution: take BOTH sides for all three conflict hunks. alloc_temp named/positional mode: Mouse's named mode adopted for multi-node first-kernel case (gpu_intermediates token names must be stable across kernels within an island). dispatch_ort opening: tracer Phase guards + upload record_transfer() alongside n_plan_inputs/n_plan_outputs declarations. Submission section: split-fence wait + GPU timestamp read alongside gpu_intermediates in both free_all calls.

dispatch_ort now 11 arguments: #[allow(clippy::too_many_arguments)] added (deferred to M2+ refactor). Mouse's new ep.rs unsafe blocks documented with SAFETY comments per -D clippy::undocumented_unsafe_blocks. Verified: 366 tests, 0 failed; Mouse's fp16 tests PASS on both devices; trace file EXISTS; messenger EP_VALIDATION_ERROR_COUNT=1.

---

### 2026-07-30T19:05:03-07:00: Tank -- allocator exercised through claimed path; single-run structural blindness; two-population finding

**By:** Tank
**Dates:** 2026-07-30T05:01:09-07:00, 2026-07-30T05:48:29-07:00

**D-T67 -- Allocator in ORT's path:** 648 pointers came back at the base of device handles in one op-table run (255 allocations, 255 matched frees, both devices). All 255 spans: host memory. alloc_device_backed_spans: 0. "Device memory is on" != "tensors are on the device."

**D-T68 -- Single-run structural blindness:** Every tests/ops/ helper is _session(...).run(...) -- one run, session dropped. 184 single-run sessions: pointers_interior = 0. One five-run session: 52 interior pointers. The suite that runs most often is structurally incapable of producing a single interior pointer. probe_planner.py --require-interior fails when zero interior pointers observed.

**D-T69 -- Staging verdict as assertion:** allocator::tally::staging_verdict() computes whole-run claim at teardown with MIRRORED state. epctl --check-counters --require-device-memory asserts it. Key absent -> exits 3 (absent key is not zero, is not a pass).

**D-T75/D-T76 -- Two-population finding:** With allocator IN or OUT of path: logits 0.0 in all positions on all runs (something wrote zeros); KV cache outputs 1..64 bitwise different between run 1 and runs 2+ (unwritten, dirty arena). Two distinct failure modes. Handed to Switch and Mouse (binding-arity bug resolved both).

**D-T77 -- Interior-pointer path at model scale:** 3-run Phi-3.5 probe -> 1192 interior pointers. probe_run2.py and probe_planner.py --require-interior are currently the only lanes reaching this path.

---

### 2026-07-30T19:05:03-07:00: Tank -- coverage falsifier confirmed; 2.09 GB scope error resolved; quarantine status

**By:** Tank
**Date:** 2026-07-30T07:51:58-07:00

**D-T78 -- Run-count confirmed:** probe at PROBE_RUNS=1 -> 0 interior pointers (matches coordinator's zero). At 2: 596, at 3: 1192, at 4: 1788 -- linear, zero intercept. Coordinator's reading CONFIRMED. Reserved-VA design met a real planner at 2 GB, zero guard-band hits in 21,460 opportunities (two vendors).

**D-T79 -- Scope error resolved:** The "322 device handles still live" warning was a scope error. HandleRegistry is process-global; reading live_spans at one allocator's release counted spans from other still-running sessions. alloc_allocations: 2511, alloc_frees: 2511 on Phi-3.5 under pytest -- ORT hands back every span. Warning corrected: scoped to warn! only for the last allocator on a device.

**D-T80 -- Counter reproduced the bug it was built to fix:** First version of alloc_frees_after_release tested allocators_released > 0 (monotone), reporting 2508 late frees on a healthy run. Same scope error as the warning it replaced. Fixed to require allocators_live == 0. General rule: a diagnostic that names an owner is making a scope claim, and the scope claim needs the same falsification as the value.

**D-T81 -- Quarantine gap worse under multi-run:** ORT's free traffic grows by only 2 per additional run (planner stops allocating from run 2). Multi-run is the regime LEAST likely to present a stale handle. The gap will not close with more runs. Status unchanged: "ORT has not presented us with a freed handle under any allocation pattern we have run." Not "quarantine is verified."

---

### 2026-07-30T19:05:03-07:00: Tank -- model-scale allocator verification (D-T72 through D-T77)

**By:** Tank
**Date:** 2026-07-30T05:48:29-07:00

**D-T72 -- NaN-comparison instrument fixed:** probe_run2.py first used np.max(np.abs(a-b)) which returns NaN when either side has NaN, even if bit-identical. Replaced with raw-byte equality (a.tobytes() == b.tobytes()) + np.nanmax over finite subset. The verdict survived (outputs really differ) -- the instrument was right by luck, and a lucky instrument is indistinguishable from a reliable one.

**D-T73 -- Verification 1 at model scale:** Phi-3.5 (2.2 GB), DEVICE_MEMORY=1, 3 runs, both devices: pointers_interior: 1192, pointers_in_guard_band: 0, alloc_allocations: 427, alloc_bytes: 2,095,251,328. Reserved-VA design met 1192 interior pointers from a real planner, zero guard-band hits in 21,460 opportunities. All 427 spans: host staging. epctl --require-device-memory exits 1 (correct).

**D-T74 -- Quarantine:** pointers_use_after_free: 0 across 18,460 pointer observations. Honest statement: "ORT has not presented us with a freed handle under any pattern we have run." Not "quarantine is verified."

---

### 2026-07-30T19:05:03-07:00: Tank -- remeasure prediction for 33-island configuration (D-T90, sealed)

**By:** Tank
**Date:** 2026-07-30T16:11:31-07:00
**Status:** Prediction, sealed before measurement. Coordinator asked for a prediction falsifiable in advance.

**Claim:** alloc_device_backed_spans substantially unchanged (within +/-10% of 427); DEVICE_MEMORY on/off wall-clock ratio ~1.00 on BOTH devices. Mirror upload fires once per allocation (at weight deserialization during warm-up, outside the measured loop). Island count changes boundary crossings per inference, not tensor allocation count.

**Weaker claim (stated separately):** The 0.94x speedup on RTX 4060 (321-island measurement) does not reproduce under 33 islands; ratio returns within noise of 1.00.

**Falsifiers:** backed spans outside 380-470; ratio outside 0.97-1.03 in a direction that repeats across 3 process repeats; alloc_device_uploads scaling with inference count.

**NOT claimed:** device backing is useful. alloc_device_authoritative_spans = 0 and expected to stay 0.

---

### 2026-07-30T19:05:03-07:00: Trinity -- model_output_equivalence verdict implementation

**By:** Trinity
**Date:** 2026-07-30

**Verdict field:** JSON-only addition. C struct VulkanEpCounters unchanged; abi_version stays 1. Correct design: the EP cannot compute the verdict (no oracle access); it is computed externally and written into JSON. to_json_with_equiv() on VulkanEpCounters; to_json() calls it with UNMEASURED default; dump_observations_if_requested() reads existing verdict and rebuilds (preserves MATCH/DIVERGENT written by Python gate before teardown overwrites).

**epctl enforcement:** EquivalenceDivergent (DIVERGENT with dispatches >= required -> exit 1); EquivalenceUnmeasured (UNMEASURED with dispatches >= required -> exit 3). DIVERGENT always fails. UNMEASURED always exits 3. No bypass flag. Check order: ABI -> dispatches -> guard-band -> dispatch count -> equivalence.

**One declared gap:** write_equivalence_verdict() Python write path has no Rust unit test -- correctness verified by inspection only. If this write path has a bug, epctl will report the wrong verdict and no automated check will catch it before a human runs epctl manually.

---

### 2026-07-30T19:05:03-07:00: Trinity -- Q/DQ opset-23+ oracle guard; ORT 1.28 ceiling corrected; no-bump audit ownership

**By:** Trinity
**Date:** 2026-07-30T08:46:10-07:00

**Q/DQ guard:** assert_qdq_reference_oracle_safe(opset, attributes) raises RuntimeError for any ReferenceEvaluator path on Q/DQ >= 23 with output_dtype or block_size. _probe_qdq_reference_oracle() runs at pytest_configure; prints status to stderr. No current test uses ReferenceEvaluator for Q/DQ at any opset.

**ORT 1.28 ceiling corrected to opset 27:** ORT 1.28 registers ONNX opsets through 27 (upgraded to ONNX 1.22.0). The onnxruntime.ai table is stale. "Support opset 26" is fully exercisable, with headroom to opset 27.

**No-bump ownership split:** Trinity owns detection (session-start probe + guard function per affected oracle path; extensible pattern). Semantic audit of whether a new no-bump correction affects ops we claim is Fact Checker's standing task (monthly sweep of onnx PRs merged since previous release). Findings from Fact Checker go into OP_COVERAGE.md per-row notes and are relayed to Trinity for probe/guard additions.

---

### 2026-07-30T19:05:03-07:00: Trinity -- shape-delta test refound; three-class taxonomy; Max exception

**By:** Trinity
**Date:** 2026-07-30T09:14:00-07:00

**What was wrong:** test_uninferred_shape_ep_declines failed on 12 CLASS-1 ops after Switch's runtime-extents fix made them claimable without output-shape annotation. The test asserted DECLINE; the code had gotten better. The false premise was in the function name (invisible to docstring-grep audit).

**Three-class taxonomy (in module docstring):** CLASS 1 (rank-known, extents-symbolic) -> claimable with ENGINE_ACCEPTS_RUNTIME_EXTENTS=true; CLASS 2 (rank-unknown) -> not claimable; CLASS 3 (data-dependent) -> permanently unclaimable.

**Fix:** test renamed test_ep_claims_without_output_annotation with inverted assertion (CLAIMS). test_inferred_shape_ep_claims[Max-fp32-dyn] -> xfail(strict=True) (Max declines even after annotation; root cause unknown).

**Audit method extension:** Must also grep function names containing _ep_declines or _ep_claims, not only docstrings. This is the transferable finding: a test whose only statement of its premise is its function name is invisible to docstring-grep audits.

---

### 2026-07-30T19:05:03-07:00: Trinity -- xfail markers flipped; binding-arity coverage dimension named

**By:** Trinity
**Date:** 2026-07-30T09:16:27-07:00

**xfail removed:** test_phi35_vulkan_matches_cpu_logits and test_phi35_vulkan_cross_run_consistency both active correctness gates. Evidence: argmax vk=cpu=30751, top-10 overlap 10/10 on both devices, BIT-IDENTICAL across all 65 outputs on both devices via probe_run2.py. max|vk-cpu| ~0.03 (fp16 accumulation-order divergence, expected).

**Binding-arity as coverage dimension:** Bug was at (3-input) x (symbolic/dynamic batch). Static-batch forms used push_static_kernel/push_normal_kernel, which captured bindings correctly. Dtype (fp16) was not the axis; input arity x batch-path was. Proof-ledger populated_optional_input_set component would have caught it: 3-input form and 4-input form get different keys.

**Coverage axis added to test_matmulnbits.py:** axes tracked: bits | block_size | input_arity | batch_path | dtype | rows | runs. Covered: 3-input x static, 4-input x static, 3-input x symbolic. Declared gap: 4-input x symbolic (lower priority; on the debt list).

**epctl FreeAfterRelease test isolation:** Tank's new test hit Trinity's equivalence check (base snapshot was UNMEASURED). Fixed by using snapshot_match() as base for FreeAfterRelease tests. FreeAfterRelease is checked BEFORE equivalence in the check order (unconditional, like OutOfBounds).

---
