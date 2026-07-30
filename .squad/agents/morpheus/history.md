# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Lead / EP Architect — architecture, design docs, scope, review
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- SUMMARIZED by Scribe 2026-07-28T22:28:08-07:00 — original entries compressed; decisions.md is the canonical record for all rulings. -->

### [SUMMARY] Sessions 1–6: architecture, baseline, contrib, OQ rulings (2026-07-28)

**DESIGN.md authored (session 1):**
- MLX pipeline shape transfers (factory, vtable, registry, convex clustering, repo layout). Memory ownership does NOT transfer — Vulkan requires OrtAllocator + OrtDataTransferImpl + staging + barriers.
- ORT allocator is pointer-based; VkBuffer is not — opaque tagged-handle registry resolving to `(VkBuffer, offset)` chosen.
- llama.cpp base shaders target vulkan1.2, ExecuTorch targets VK_API_VERSION_1_1. "Requires 1.3" claim was wrong; verified by Fact Checker.
- Claim rate is a bad metric; **fused-region compute volume** (`largest_island_flops`) is the metric of record. Island count + largest fused region must appear in every benchmark.
- Record-once / replay-many (Compile→Compute). ExecuTorch model, not llama.cpp's per-eval re-record.
- Device test must assert `VulkanExecutionProvider` node placement — CPU fallback vacuously passes.

**Baseline frozen — OQ-1 (session 2, after Link's measurements):**
- Reversed provisional `synchronization2`+`subgroup_size_control` hard requirements. Link: 31.43% Android gap on sync2; MoltenVK reports the extension but `subgroupSizeControl=VK_FALSE`.
- **Frozen gate:** Vulkan ≥1.1, compute queue, `maxComputeWorkGroupInvocations ≥ 256`, `maxComputeSharedMemorySize ≥ 16384`, subgroup BASIC, one DEVICE_LOCAL + one HOST_VISIBLE memory type. No required extensions.
- Rule: a requirement that excludes the machines you test on has not been tested. Capability shortfalls degrade op coverage, not device availability.
- Khronos layer shim (Link's Option B) rejected: AOSP loader searches only APK owner's nativeLibraryDir; we are a plugin. Precedent was false.
- Dual-backend barrier seam: backend selected once at device init. `ep.force_legacy_barriers` session option forces legacy path in CI.

**OQ-11 ratification + contrib domain reversal (session 2):**
- `ai.onnx`-only was wrong: ORT GenAI emits `com.microsoft` ops (GQA, RotaryEmbedding, MatMulNBits, LinearAttention) directly for Qwen graphs. For scope questions: read the exporter, not the standard.
- Admitted as **named ops** (not a domain predicate). `if domain == "com.microsoft"` is forbidden. Registry key is the allowlist; graph census in CI is the drift alarm.
- OQ-12 experiment defined: §11.1 fixes devices, pass bar (≥1.5× vs phone's own CPU EP, zero numerical failures), and all four outcome consequences in advance.

**OQ-3 ruling (session 3 — Tank's proposal adopted):**
- BDA is a second shader architecture (requires `GL_EXT_buffer_reference`), not an optimization. Does not remove the side table. MoltenVK support Apple-Silicon-only. **No BDA at all.**
- **Reserved VA registry:** `VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` on Windows, `mmap(PROT_NONE)` on POSIX. Real unique spans. Stray dereference = MMU fault, not silent corruption.
- Rule: prefer designs that make a hazard impossible by construction.

**OQ-4 ruling (session 4 — code was right):**
- **Hard Vulkan SDK build dependency.** No checked-in SPIR-V fallback.
- Checked-in `.spv` that drifts from `.comp` is silent wrong-numbers in the build system. Freshness-hash defeats the purpose. Same shape as layer shim and BDA — under-exercised second path.
- `ALLOW_MISSING_GLSLC=1` escape hatch must produce an inert artifact (zero devices, zero claims), not subtly broken. No release artifact from escape-hatch builds.

**Oracle validation + accuracy_level pinning (session 4):**
- CPU EP works as oracle for quantized path (MatMulNBits fp32). `accuracy_level` pinned at 1 — level 4 diverges ~3.6e-3; fp16 NaN/Inf on ORT 1.27 (null-allocator PrePack bug).
- Bit-layout correctness (dequantize) goes to NumPy (independent spec), not CPU EP — shared misreading passes both sides.
- Oracles that change with the machine are not oracles. Pin all CPU-sniffed knobs.

**llama.cpp accelerant + OQ-M6:**
- Rai 🟢 Green. No obligation for reading/learning. Obligation attaches only on substantial source adaptation.
- Block format mismatch = no code copying. Tiling strategy, subgroup reduction shape, dequant-in-register patterns **do transfer** (Switch confirmed). Budget algorithm study time.
- Mouse's "useless" claim was too strong — he answered "can this be copied?", Switch answered "does reading save time?". Both right. Adjudicating on Mouse alone would have been wrong.

**Key process lessons:**
- Mark unverified claims as unverified *in the document* — a lead's wrong entry propagates into everyone's assumptions.
- When two owners appear to disagree, check whether they are answering the same question.
- Pre-commit the conditions under which you will widen before the data arrives. Write the reversal conditions in the document.
- Every time you rely on the team remembering something, write a test instead.
- "Performed a fusion" and "implemented a fused node" differ: GQA arrives as one node; decomposing it materializes a [B,H,S,S] score in VRAM. Implementing is conservative; decomposing is reckless.
- A positive result on a named risk: bank the conditions (pinned `accuracy_level`), not just the headline.

**Milestone status (as of 2026-07-28):**
- M0: ORT loads plugin, enumerates Vulkan device, runs Add node, matches CPU EP, on lavapipe CI.
- M1 gate: template infrastructure before op #1, reported ops-per-kernel ratio ≥ 8.
- M2: device allocator, reserved-VA registry. Gated on M2: LLM path (KV cache cross-subgraph boundary).
- M3: Android tuning (budget only if A+B devices pass all three OQ-12 stages).

**Open questions (as of 2026-07-28):**
- OQ-12: hardware experiment (Adreno 5xx + Mali Bifrost devices).
- OQ-14: fp16 device share on Android (product-scope question).
- OQ-15: shape-agnostic dispatch / `vkCmdDispatchIndirect` (Switch).
- OQ-16: LinearAttention/CausalConvWithState schema stabilization (T5a gated on upstream).
- OQ-13: zero-copy IO binding (Tank, post-M2).

---

## 2026-07-29T08:13:58-07:00 — coverage is producer-relative, and T3's first kernel

**The most dangerous kind of wrong is a correct answer to a narrower question than you asked.** Our
op inventory was derived from *emitted graphs* — the thing I praised when ratifying it, and I still
would. But it was derived from **one exporter's** emitted graphs and then reasoned about as "what a
Qwen3 graph looks like". Justin's own `mobius` builder emits `ai.onnx::Attention` @ 23,
`RMSNormalization` and `RotaryEmbedding` with no fused skip-norm; `MatMulNBits` is the only op the
two toolchains share. A Qwen3 from our own user's own toolchain would have declined five nodes per
layer across 28 layers for **want of a table row, not a kernel** — every op already implemented or
planned. Nothing was misread. Nothing was missing from the list. The list answered a different
question.

**Standing rule: a coverage number is meaningless without naming the producer it was measured
against.** "We support Qwen3" is not well-formed. Model architectures are not expressed in ONNX —
*exporters* are, and two exporters over identical weights and identical mathematics disagree on
domain, on fusion boundaries, and on which optional inputs exist. This is now §8.5 and it is
enforced structurally: the census is indexed by producer, and every tier exit criterion that names a
model names the producer that built it.

**Verify against the artefacts of the people you are building for, first.** The finding came from
Justin saying 这都是我们的项目 and Mouse actually reading the repo. The reference architecture, the
target models, and the *exporter* are all his. I had treated the exporter as an environmental
constant.

**Shared kernel, separate gate.** `RMSNormalization` shares `simplified_layer_norm`'s handler
(asserted by function-pointer identity — the right way to say "same kernel" so it cannot drift), but
`ai.onnx::Attention` gets its own predicate, because attribute names, illegal combinations and
optional-input indices differ, so one predicate over both is wrong about one of them **in the
permissive direction**. Same asymmetry as C2 item 7. The kernel is where duplication is expensive;
the gate is where duplication is *cheap and protective*. Do not let "we already have that kernel"
argue for "we already have that predicate".

**Prefer the standard domain where a producer offers one** — opset-versioned means the monotonic
range check is back, which is the exact thing C2 exists to compensate for the absence of. Every op
served from `ai.onnx` is an op outside the contrib risk surface.

**Sequencing decisions must state whether they are also scope decisions, or they will be read as
scope decisions.** I ruled T3 starts with `ai.onnx::Attention` rather than GQA. Two real reasons:
it decouples T3 from Switch's unfinished aliasing seam, and it gives us a model family we can build
and iterate on at the desk on the first milestone that actually dispatches anything. But the
honest objection is that this optimises for *our* tooling while ORT GenAI is what external users
hit — so I attached constraints rather than waving it away: T3 exits only per-producer on both
columns, `largest_island_flops` reports per producer, and the KV-cache contract is designed for
GQA's requirements even though the first kernel does not use them. **Designing the memory contract
around the easier consumer is how the second consumer becomes a rewrite.**

**Test your own ruling against the counterfactual and say the result.** The strongest form of the
T3 argument is not "local iteration is faster" — it is that the standard-domain form is *also* the
lower-risk claim surface, so both considerations point the same way. I wrote down that had the
standard form been riskier I would have ruled the other way and eaten the CI latency. A ruling that
cannot name the condition that would have flipped it is a preference wearing a ruling's clothes.

**A "no" without a trigger is a question that gets re-asked every quarter.** The crate evaluations
came back mostly negative and the good one is `onnx-runtime-ir`'s deferral: it names a *structural*
fact — ORT hands us `OrtGraph`/`OrtNode` across a C ABI and we never see a protobuf, so an external
IR means copying the graph into a second representation inside someone else's process — rather than
a maturity judgement that expires, and it names what would reverse it. Deferrals here should look
like that. Separately: `onnx-shape-inference` is the **cheapest coverage in the plan** and I nearly
filed it as harness polish — it turns `[dynamic-shape]` declines into claims with zero Rust
changes. Coverage that costs no kernel should be sequenced as coverage.

**Refresh disclosure sections in the favourable direction too.** §9.1.2 said the machine had no ICD
and no `glslc`; it now has both, two conforming GPUs, and 168 compiling variants — and *still* has
never dispatched a shader. Had I left it, the next reader would have found one false detail and
discounted the whole section, including the part that is still true and still the point. Also added:
the local GPUs are a development loop, not coverage — nothing they run is recorded, gated or
reproducible by anyone else, and **a result obtained only on this desk is not a result this project
has.**

**Mechanical note to self: the `edit` tool applies `old_str` → `new_str` literally.** Three times
now I have anchored on a heading and omitted it from the replacement, silently deleting `### 8.2`
and `### M0`. Both were caught by re-listing headings afterwards. Always re-grep the heading
structure after any edit anchored on a heading.

---

## Cross-agent context appended (2026-07-29T09:00:39-07:00) — first-hardware round

📌 **CI is the only place shaders are verified; red CI blocks all merges (2026-07-29, Trinity + Morpheus):** `README.md` now carries a CI badge and explicit callout. A shader running locally is a development loop, not coverage. A result obtained only on this desk is not a result this project has.

📌 **`onnx-genai-models` (`mobius`) finding (2026-07-29, Mouse D-M6-04):** Standard-domain ops now registered. Coverage is relative to a named producer. Morpheus D21: a target model is "covered" only when a named producer is specified. The M1 census must be per-producer; "producer emits no GQA" must be an explicit row.

📌 **T3 sequencing fixed: `ai.onnx::Attention` first (2026-07-29, Morpheus D23):** GQA sequenced after. The `bind_aliased_output` seam must be designed for both consumers' requirements (Switch + Mouse must coordinate).

📌 **`rustfmt --edition 2021` silently no-ops on edition-2024 crate (2026-07-29, Tank D-T12):** Always use `cargo fmt --all`.

---

## 2026-07-29T09:47:45-07:00 — the first dispatch, and a disclosure whose failure mode inverted

**A disclosure section has to be rewritten when the good news arrives, and that is the hardest time
to write it well.** §9.1.2 existed to stop us overclaiming execution. On 2026-07-29 a kernel
actually ran — one elementwise shader, 1024 elements, on Intel Iris Xe and NVIDIA RTX 4060, zero
validation errors on both. The temptation is to relax the section. The correct move was to notice
that **the failure mode inverted**: yesterday's risk was claiming execution we had not done;
today's is letting "we dispatch on two GPUs" stand in for "the EP works", and the gap between those
two sentences is the entire project. So the section now carries three qualifiers that must travel
with any citation of the result — **one kernel, no ORT, one OS** — and the line about a result
obtained only on this desk is more load-bearing than it was, not less. A disclosure that relaxes on
good news is one nobody should trust on bad news.

**Do not declare a milestone on evidence that bypasses what the milestone is about.** M0 says *a
stock ORT loads the plugin, enumerates a device, runs a graph containing an `Add` on that device,
and matches the CPU EP*. The dispatch came from a Rust integration test — no ORT, no graph, no
claim path. Every clause after "enumerates" is still open. Six criteria met, one partial, two unmet,
and **the two unmet are the two that define M0**. Assessing criterion by criterion rather than in
aggregate is what made that visible; an aggregate report would have read "mostly there". The
dispatch retired the risk that the innermost step was wrong. That was a real risk and it is not the
milestone.

**A failed capability probe is indistinguishable from a device with no capabilities.** This is the
sharpest thing I learned this turn. `let _ = props2.push_next(..)` silently discarded the whole
`pNext` chain — `ash`'s builders are `#[must_use]` and return rather than mutate — so every chained
capability read zero and subgroup size looked like 0. Nothing was wrong with the device, the driver
or our reading of the spec: **we never asked the question**, and "no answer" and "answer is zero"
were the same bytes. Worse, the ambiguity had already bitten: lavapipe's `supportedStages = 0` was
recorded as a device fact and we can no longer tell from the record which class of error it was.
Hence §7.9: a probe reports **three** states, the third being *not determined*, never silently
coerced to "not supported"; an all-zero chain on a ≥1.1 device is treated as probe failure until
proven otherwise; and `--dump-capabilities` prints the raw values, because a derived boolean cannot
be audited but the number behind it can.

**When two people independently reach the same wrong answer, it is the natural mistake and needs
structural prevention, not review.** `detect_uma` returned true for a *discrete* RTX 4060 because
ReBAR maps VRAM `HOST_VISIBLE`; the correct predicate is that **every** heap is `DEVICE_LOCAL`.
Niobe hit the identical trap in the benchmark harness at the same time. Two people, one intuition,
one wrong answer — that is a design defect in the predicate's phrasing, not two lapses. The general
rule I extracted: **where a predicate's two plausible readings differ in which direction they fail,
choose the one that fails toward the extra copy.** Staging is the safe default; skipping it is the
optimisation; the burden of proof sits on the optimisation.

**Neither bug was visible on one device or on lavapipe.** So capability-derived behaviour is not
trusted until it has run on one integrated *and* one discrete device. The Intel half is the more
valuable one — it is the stricter implementation, which makes it a conformance oracle rather than a
second sample. Justin said this before we had evidence for it and he was right.

**Amend a rule the moment the evidence arrives, rather than waiting to be asked.** Mouse was going
to raise whether §8.5's producer rule needs a version. It does — he is re-deriving against
`onnxruntime/mobius` at default **opset 24**, and the same builder at a different opset changes the
op set we must serve, which is §8.5's own failure one level finer. I amended it to *producer at
version* immediately. Waiting to be formally asked, when the evidence is already in hand, is
ceremony.

**A deferral recorded as a structural fact survives the collapse of the reason people thought it
had.** Justin withdrew the trust objection to `onnx-runtime-ir`. The deferral is untouched, because
it never rested on trust: it rests on our being a guest in ORT's address space, handed
`OrtGraph`/`OrtNode` across a C ABI and never seeing a protobuf. A reversal must defeat *that* or
meet the named trigger. "The original objection has weakened" is a third thing and is not a reason.
D24's insistence on triggered deferrals paid off on its first test.

---

## 2026-07-29T15:02:55-07:00 — reading the file instead of the builder, and errors that are not randomly signed

**When a claim can be checked against an artifact, checking it against a description of the artifact
is not checking it.** This is now the project's characteristic failure and I can name four
instances: the CPU-EP oracle (§9.1), the fingerprint audit (C2 item 7), the capability probe (§7.9),
and now the producer census. Each time we had a plausible derived answer and no check against the
thing itself.

**The recurrence is the finding, not the individual corrections.** The producer lesson arrived three
times at successively finer grain — wrong producer, then right producer at the wrong revision, then
right producer at a pinned revision *whose output had never been read*. Each pass I correctly
narrowed the claim and each pass I landed one level short of the evidence. Three iterations with the
same shape is one mistake, not three, and the tell was that every correction moved along the same
axis. **If a fix makes the claim more precise without moving it closer to an artifact, expect to be
back.** Mouse's formulation is now §8.5's rule of record: *builder source is intent; the model file
is the fact.*

**Coverage percentage is not merely a weak metric, it can be inverted.** I had `largest_island_flops`
as the guard and thought it sufficient. Mouse's simulation showed claiming `Cast` on gpt-oss moving
coverage 28%→54% while island count went 52→125 — more ops claimed, strictly worse partitioning.
And Phi-3.5 sits at 34–35 islands from T1 through T3 until `MatMulNBits` collapses it to **one island
of 364 nodes**: the useful transition is a **cliff, not a slope**, so partial coverage of that graph
is worth exactly nothing. The metric of record is now the triple `(coverage, island_count,
largest_island_flops)` and no member may appear alone. **A single-number guard is a number that can
be gamed by accident** — I did not need a bad actor, only a plausible next step.

**The corollary I did not expect: we may implement an op and decline it.** If claiming an op raises
island count without raising largest-island FLOPs, declining it is the better answer for that graph.
That is not a contradiction of "claimed never outruns translatable" — it is the reverse direction,
and the reverse direction is a legitimate optimisation.

**"Which kernel do we write first" and "which model proves it" are different questions, and I
answered them with one decision.** §10.0.2 chose `ai.onnx::Attention` and also implied the
demonstration target. Mouse separated them and was right. Phi-3.5 is a better *demonstration* — MHA,
softcap 0, no SWA, no sinks, no Q/K norm, uniform RTN int4, five op types over 353 of 366 nodes —
while `ai.onnx::Attention` remains the right *first kernel* for reasons untouched by the census.

**Follow your own argument in both directions or drop it.** I justified §10.0.2 partly on "a path
whose models we can run on the desk is worth materially more". Phi-3.5 is on the desk today and no
Qwen3 graph is. An argument good enough to sequence a kernel is good enough to re-target a
demonstration, and quietly letting it stop applying when it points somewhere inconvenient is how a
rationale degrades into a preference.

**Adopt a criterion that can only be met by the thing working.** `MatMulNBits` claimed ⇒ Phi-3.5
partitions into one island of ≥360 nodes. One falsifiable number, unreachable gradually because of
the 34→1 cliff, worth more than every coverage percentage in the document.

**Our errors are not randomly signed — four of five census contradictions were permissive.** A
random distribution does not land four-to-one. The worst: the GQA predicate never read inputs 1 and
2, so it would have **claimed** a packed-QKV node and handed the kernel a fused tensor where it
expected a query — and both real models pack on every layer, so this was the normal path, producing
wrong numbers rather than a decline. Third independent observation of the same asymmetry (C2 item 7,
§7.9, this). **Too-strict fails loudly; too-permissive fails silently and produces numbers.** So the
audit question is not "is this right?" but **"in which direction is this wrong?"**, and an audit that
finds nothing must say which permissive failures it looked for.

**A predicate that does not read an input cannot reject on it.** Silence about an optional input is
acceptance of it. Every optional input a schema defines must be enumerated and explicitly accepted
or declined. This is the mechanical form of the lesson above and it is the one that would actually
have stopped packed QKV.

**Report progress on the ABI seam as progress on the ABI seam.** ORT now enumerates our EP on both
GPUs with full metadata — M0's "loads and enumerates" clauses are satisfied. Criterion 2 moved from
not-met to partial and the milestone did not move at all, because everything from "runs a graph"
onward is still open. Good news that lands on a criterion you already listed is easy to report
honestly precisely *because* you listed the criteria first.

---

## 2026-07-29T16:00:55-07:00 — evidence our own tooling manufactured, and a criterion a skip could satisfy

**The most dangerous wrong input is a number our own code printed.** A frozen architectural decision
— removing gate criterion R5 — was justified in writing by lavapipe's `supportedStages = 0`. Mesa
26.1 lavapipe supports subgroup BASIC in compute; that reading was our own `push_next` probe bug.
Every step to the decision was reasonable: read a capability, notice it excludes a platform we care
about, weigh the requirement, change it, record why. **This failure is invisible to diligence** —
more care at any step would not have caught it, because the flawed input entered at the only point
nobody thinks to question. Written up as §10.0.1 R6, and it is the cleanest specimen I have.

**The decision survived only because it had two reasons and the load-bearing one was not the number.**
§7.0 argues R5 out of the gate from first principles and never mentions lavapipe. Had the observation
been load-bearing rather than merely presented, this would have been a reversal. The rule I take
from it and have written down: **when a decision rests on both a principle and a measurement, record
which one would have to fail for the decision to change.** If the answer is the measurement, the
decision is provisional and must say so. I did not do that for R5, and I could not tell from my own
document which reason was doing the work.

**Publish the correction even when the conclusion stands.** Nothing operational changed — R5 stays
out of the gate, the code is right, the test pinning it is right. The temptation was to fix a comment
quietly. But `PLATFORMS.md` LVP2 states the same false premise as an *observed device quirk* and as
the cause of the §7.2 change, so anything downstream reasoning about lavapipe inherits it. **A right
answer reached through false evidence is an unaudited answer.**

**A number taken with a broken instrument is not evidence merely because it was written down.** Every
capability reading recorded before the probe fix is now provisional and must be re-observed with
`epctl --probe-loader` before it is cited again. Corroboration means a *second instrument*, never a
second reading from the same code.

**A criterion that a skip can satisfy is not a criterion.** The barrier-parity test reported "`Add`
is not yet Ready (did not claim this node form)" on the same machine and day that
`test_add_is_claimed` passed — two of our tests disagreeing about whether `Add` is claimed, because
they build different node forms, and the disagreement surfaced as a **skip with a plausible reason**
rather than a failure. I amended M0 criterion 8 to require a non-zero executed-dispatch count per
lane. **A skip is a permissive outcome dressed as a neutral one** — same sign as R5's four-of-five
finding, which is now the third context in which that asymmetry has bitten.

**Exit criteria written in advance make good news easy to report honestly.** `Add` now executes
through ORT on both devices — the first result traversing the whole stack. Criterion 2 moves to met,
criterion 9 moves *back* to partial because this same revision found the sibling docs carry a false
premise, and M0 remains not met because its sentence ends "on both Windows and Linux, on a software
rasterizer, in CI" and that tail is not decoration. Nothing about that assessment required judgement,
which is the whole value of having listed the criteria before there was anything to report.

**The best number in the test suite is a failure count.** 125 failures on real hardware, every one
the harness refusing to score a CPU-fallback run as a pass. A weaker harness would have reported
~240 green. **A differential test that does not verify the EP ran is a test of ORT, not of us** —
Trinity implemented in the harness what §9.1.2 had only asserted in prose, and the implementation is
worth more than the prose.

---

## 2026-07-29T19:42:07-07:00 — a standard that yields the first time it costs something was never a standard

**I did not declare M0 on the best day the project has had.** 45 op rows live through ORT, both
barrier backends executed and bit-exact on two vendors, the legacy path that carries ~31% of Android
exercised for the first time in the project's history. Every clause of M0's sentence up to the dash
is satisfied. Everything after the dash — *on both Windows and Linux, on a software rasteriser, in
CI* — is untouched, and all of today's results are on one desk. **My own §9.1.2 line was written when
the news was thin, and its whole value is that it applies unchanged when the news is good.** The
CI and software-rasteriser clauses went into M0's sentence on day one precisely because local
hardware success is the easiest result here to obtain and the least transferable to claim.

**Writing exit criteria in advance is what makes good news cheap to report honestly.** Two criteria
moved this revision, one to met and one *down* to not met, and neither required a judgement call. I
could downgrade criterion 9 on a defect this document introduced without it feeling like a
concession, because the criterion was already written.

**"No errors surfaced" is what a disabled instrument reports.** I declined to close criterion 3 on a
validation run that raised nothing, and required a **positive control** — a planted violation the
lane catches, or the layer's startup banner asserted. The general rule: **a clean result from an
instrument whose liveness was not established is not a result.** Converting "we saw nothing" into
"we would have seen something" is nearly free and is the only version that carries information. I
had written two rules of exactly this shape this week (§7.9's failed probe, R7's dead claim log);
accepting a silent validation layer days later would have made both of them decorative.

**Our instruments fabricate negatives, and a negative asks for no explanation.** R7. A `OnceLock`
cached the claim-log path, a pytest process set the variable per-call, and a missing file read as
"not claimed" — a dead instrument returning a plausible answer.

**The lesson was not layer 1, it was layer 2.** Layer 1 was a bug. Layers 2 and 3 were **competent
engineering applied to a false premise**, and each made things worse: the workaround hit an access
violation, and the workaround's workaround (a hand-declared per-op `live` flag against per-form
claims) would have bought exactly the vacuous pass it was introduced to prevent. **When a fix
produces a second, unrelated-looking failure, suspect the diagnosis before extending the fix.** The
cost of a fabricated negative is never the negative; it is everything built on top of it.

**Derive, do not declare.** Any fact the code already knows must be computed from the code, never
restated in a second language — and at the same granularity the code uses. The `live` flag was
per-op while our claims are per-form, which is how `Add-i32` carried `live=True` against an f32-only
predicate. A hand-written duplicate of a machine-known fact is a fork, and it drifts permissively:
stale `True` buys a pass, stale `False` merely skips. R5's asymmetry again.

**Similarity is not a measurement.** §8.7, from Mouse declining to promote the attribute-carrying
activations despite their satisfying every stated condition: a push-constant tail is **a new code
path, not a new expression inside an exercised one**, and a wrong `params[0]` offset is invisible to
every live op because they all push zeros there and read none of them. The operational test I wrote
down: *what could be wrong with the new op that the exercised ops are structurally incapable of
detecting?* Template leverage is the basis of the M1 estimate, which makes it the place where this
project is most tempted to convert structural similarity into evidence.

**Corrections must be chased to every file that inherited them, or they are not corrections.** §7.2's
false lavapipe premise reached `PLATFORMS.md` as an *observed device fact*. `ENGINE.md` and the Rust
comments were fixed; LVP2 was not, so criterion 9 is not met. Stopping a correction at the file where
it was noticed is the exact failure R6 exists to name.

---

## 2026-07-29T21:14:03-07:00 — the roadmap was ordered by our assumptions, and one real model reordered it

**A decline histogram is a different measurement from an op census, and I had never asked for one.**
The op inventory was accurate — Phi-3.5 really does contain those ops in those counts. It answered
*which ops does this graph contain*, and the roadmap needed *why do this graph's nodes get declined*.
Same artifact, different question, different answer: 258 declined on symbolic shapes, 100 on missing
kernels. **Fourth landing of the same failure** — wrong producer, wrong revision, unread model file,
now unasked question. It has survived three explicit corrections of itself, which is the strongest
available evidence that it is structural. Rule of record: **coverage planning is driven by the
decline histogram of a real graph, not by its op histogram.** An op census says what to build; only
a decline census says what to build first.

**Read a histogram of first-match codes as floors and ceilings, never as a partition.** This is the
part I nearly missed and it inverts the naive reading. `claim_decision` checks key → opset → schema
→ **status** → predicate and records the *first* failure, so the 100 `staged` nodes never reached
the shape check — their shape viability is unknown, and three kernels unlock *at most* 100. The 258
had already passed registration, opset, schema and status, so shape is their *sole* remaining
blocker. **The asymmetry is larger than the raw ratio, not smaller.** Early codes are ceilings, late
codes are floors, and two decline counts are not comparable without the check order.

**"Ahead of" versus "alongside" is a dependency question, not a priority question.** Every
interesting extent on a decode path is symbolic. A kernel written against static extents is **not a
partial version of the kernel we need — it is a different kernel**, so sequencing kernels first
manufactures rework. When asked to prioritise between two workstreams, check first whether one is
upstream of the other; if it is, priority is not the question being asked.

**Write criteria that a workaround cannot satisfy.** M1's new criterion is *two different concrete
values of a symbolic dimension in one session* — deliberately unsatisfiable by resolving shapes
ahead of time, which is exactly the substitution available here. That is the third hole of this shape
I have closed this month: a skip could satisfy criterion 8, a switched-off validation layer could
satisfy criterion 3, a static preprocessing step could have satisfied a naive shape criterion. **It
is now a drafting rule: for every criterion, ask what the cheapest thing is that satisfies the words
without satisfying the intent.**

**A tool that improves our numbers without improving the product is the §9.1.2 hazard in its purest
form.** `onnx-shape-inference` resolves symbolic dims statically. That is right for a fixed-shape
test artifact and useless for inference, where sequence length varies per call. It stays adopted as
an oracle and is now explicitly barred from being read as a solution to §8.8.

**A ruling made on one model names its reversal condition.** gpt-oss showing `dynamic-shape` below
`staged` reverses this. I do not expect it — the mechanism is architecture-independent — but "I do
not expect it" is exactly what produced R8.

**Record the machinery that worked, not only the plan that did not.** A 2.2 GB fp16 model with
external data and an `If` prologue loaded, ran, declined all 363 nodes with machine-readable reasons,
fell back to CPU, and was bit-identical across sessions. Conservative claiming did precisely what it
was designed to do at a scale nothing else has tested. **The roadmap was wrong; the safety net was
not**, and a design record that only logs the errors will mislead about where the risk actually sits.
