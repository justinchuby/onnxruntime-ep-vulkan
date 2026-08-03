# RAI Audit Trail

> Append-only evidence log. Entries are redacted — never contains raw secrets or harmful content.

<!-- Rai appends findings below -->

---

## Audit Entry — 2026-07-28T19:16:08-07:00

**Review type:** On-demand — IP/licence compliance (OQ-M6) + first RAI pass
**Files reviewed:** `LICENSE`, `README.md`, `docs/DESIGN.md`, `docs/ENGINE.md`, `docs/PLATFORMS.md`, `docs/OP_COVERAGE.md`, `.squad/decisions/inbox/mouse-op-coverage-plan.md`
**External licences fetched:** llama.cpp `LICENSE`, ExecuTorch `LICENSE`, ONNX Runtime `LICENSE`

### Findings

| ID | Category | Severity | File | Finding | Status |
|----|----------|----------|------|---------|--------|
| RAI-001 | Credentials/secrets | ✅ Clear | All source files | No hardcoded credentials, API keys, tokens, or secrets found in any `.rs`, `.toml`, `.json`, `.md` file. | Resolved — no action |
| RAI-002 | IP Compliance — OQ-M6 | 🟢 Green | N/A | Reading llama.cpp MIT shaders as reference: fully permitted. Conditions for adaptation documented in `docs/THIRD_PARTY.md`. | Resolved — decision filed |
| RAI-003 | Platform honesty | 🟡 Advisory | `README.md` | README platform table lists Android and macOS without CI-coverage caveat. All physical hardware rows in `PLATFORMS.md` are marked "untested." User-facing README does not reflect this. Link is aware. | Open — Link to address |
| RAI-004 | Terminology | 🟡 Advisory | `docs/OP_COVERAGE.md` | "fusion whitelist" (2 instances). Policy prefers "allowlist". | Open — Mouse to address at next revision |
| RAI-005 | Performance claims | ✅ Clear | All docs | No published performance claims or speedup numbers found. Tier exit criteria are goals (design doc), not user-facing claims. README clearly states "Status: pre-implementation." | Resolved — no action |
| RAI-006 | Deceptive/ungrounded claims | ✅ Clear | All docs | All uncertain claims are marked ⚠️ Unverified. vulkan.gpuinfo.org data is correctly attributed with date. No hallucinated citations found. | Resolved — no action |

### Verdict Summary
- 🔴 Critical: 0
- 🟡 Advisory: 2 (RAI-003, RAI-004) — work proceeds, non-blocking
- 🟢 Green: 4 (RAI-001, RAI-002, RAI-005, RAI-006)

---

## Audit Entry — 2026-07-30T07:12:15-07:00

**Review type:** On-demand — silent-inference RAI verdict (R9 event, §10.0.1)
**Files reviewed:** `docs/DESIGN.md` (§8.9, §9.1.2, §9.1.3, §10.0, §10.0.1), `.squad/rai/policy.md`, `.squad/agents/Rai/charter.md`, `.squad/decisions/inbox/switch-ep-messenger-and-plant.md`, `.squad/decisions/inbox/tank-allocator-claimed-path.md`
**External state reviewed:** `DESIGN.md` Morpheus rulings at `557bf24` (§8.9.1 claiming gate, §9.1.3 compute_failures, criterion 10 DIVERGENT, criterion 11 NOT MET)
**Requested by:** Justin Chu

### Context

Phi-3.5-mini (2.2 GB, int4-quantised) run against `main` at `557bf24`:
161 `com.microsoft::MatMulNBits` nodes claimed and accepted by ORT, `compute_failures: 0`,
`dispatches_executed: 161`, test suite green. Vulkan output: logits `[0.0000, 0.0000]`, argmax 0.
CPU output: logits `[-13.0859, 13.0312]`, argmax 30751. Top-10 token overlap: 0/10.
Deterministic on Intel Iris Xe and RTX 4060. No error surfaced at any layer. Mouse has a kernel
fix in flight. Morpheus has ruled on the engineering gate (§8.9.1, §9.1.3, criteria 10 and 11).

### Findings

| ID | Category | Severity | Attaches to | Finding | Status |
|----|----------|----------|-------------|---------|--------|
| RAI-007 | Engineering defect — defective `MatMulNBits` kernel | 🟡 Advisory | The instance: one wrong fp16 kernel | Kernel produces zeroed output. Mouse fixing; criterion 10 tracks it. Instance is an engineering matter. Does not independently require RAI lockout — Morpheus's criterion 10 (NOT MET, DIVERGENT) already blocks M0 on this. | Tracked — Mouse and Morpheus own the fix |
| RAI-008 | Deceptive system claim — silent wrong inference output | 🔴 Critical | The class: architecture permits silently-wrong inference output at any layer with no disclosure | The EP reports success (`compute_failures: 0`, populated output tensors, no exception, no log line) while producing output entirely disconnected from the input. A user cannot distinguish this from "model performs badly." In an autoregressive decode loop, a single zeroed-logit dispatch produces an unbounded stream of wrong tokens, each fluent in form. No mechanism existed at R9 time to prevent this; no user-visible disclosure exists at the session layer. This property survives Mouse's fix — the next unproven kernel will be equally silent. | 🔴 OPEN — structural; fix is the proof ledger (criterion 11, NOT MET) plus a session-layer WARN |
| RAI-009 | Missing user-visible disclosure at runtime | 🔴 Critical | Architecture — absence of session-layer disclosure | `compute_failures: 0` is the only compute-path signal a user observes. §9.1.3 constrains its reading by prose and a `model_output_equivalence` verdict on the counters file. Neither is visible to a user who receives a session object and runs inference. There is no runtime WARNING when claimed forms carry UNMEASURED proof status. | 🔴 OPEN — requires runtime WARN mechanism at session creation |

### Verdict Summary
- 🔴 Critical: 2 (RAI-008, RAI-009) — **Reviewer Rejection Protocol applies to the architecture**
- 🟡 Advisory: 1 (RAI-007) — instance tracked by engineering criteria, non-blocking as a separate RAI matter
- New 🟢: 0

**Overall architectural verdict: 🔴** — the EP cannot ship op claims until the proof ledger (criterion 11) is implemented and a session-layer disclosure mechanism exists. The 🔴 does not block Mouse's kernel fix or Mouse's test work; it attaches to the shipping decision, not to the investigation.

**Falsifier for this verdict:** The 🔴 is discharged when: (a) criterion 11 is MET — no form is claimable without a ledger entry, verified by a planted `[unproven]` decline and a CI check; AND (b) a runtime WARN is emitted at session creation naming every claimed form whose proof status is UNMEASURED or whose last model-level verdict is DIVERGENT. An EP that satisfies both conditions has an instrument that would go red on the next silent-zeroing event — specifically, the ledger check would decline the unproven form before dispatch, and any future DIVERGENT result would demote the form automatically (§8.9.2 rule 4). The RAI concern evaporates when the instrument exists.

---

## Audit Entry — 2026-08-01T09:53:14-07:00

**Review type:** Recall — re-verdict RAI-008/009 against a materially changed system; fallback disclosure; performance-directive RAI dimension; llama.cpp licence boundary
**Files reviewed:** `docs/DESIGN.md` (current, last revised 2026-07-31T07:45:10-07:00; §8.9.4–8.9.7, §9.1, §10.0 all four amendments, §10.0.1 R9–R13, M0 table rows 2–3, 10–12, line 371, line 1108–1122 single-cluster bypass hazard), `.squad/rai/audit-trail.md` (my own 2026-07-30 entry), `.squad/agents/Rai/history.md`
**Requested by:** Justin Chu, explicitly against my own stated falsifier from RAI-008

### Context since the 2026-07-30 verdict

1. **Correctness evidence improved genuinely.** After Switch fixed `Allocator::alloc(size=0)` returning `None` on Phi-3.5's `[1,32,0,96]` KV-cache inputs, ORT's own profiling trace — an instrument this project does not own — recorded an attributed execution: one fused `VulkanExecutionProvider` island (~354 of 364 nodes) plus ~10 `CPUExecutionProvider` events matching Mouse's declines exactly, `argmax 30751` matching CPU, repeated across runs. This is real and I am not discounting it.
2. **The same window produced two more silent-fallback events**, both with the EP as proximate cause: a weight-cache OOM (`gpu-allocator failed to allocate 14155776 bytes`) causing a silent mid-session fallback, and the KV-cache `alloc(size=0)` case itself before the fix — `ORT printed EP_FAIL … Falling back` and raised nothing, `get_providers()` still listing `VulkanExecutionProvider` because the provider list is fixed at session creation. Fifth documented sighting of that exact line on this project. 50 KV-cache outputs were never written in the dirty-arena case, detectable only by cross-run divergence.
3. **Guard D now catches the fallback line itself in the test lane**, with a two-polarity control (it must state what it observed even on failure — `NameError` is not a detection, `0 Vulkan node events, providers seen: [CPUExecutionProvider]` is). This is a CI-lane instrument, not a production disclosure a user calling `session.run()` would ever see.
4. **Criterion 11 (the proof ledger) is confirmed scaffolding-only** — `DeclineCode::Unproven` and a list-only parser exist in `registry.rs`; there is no ledger, so nothing is looked up, and the three planted controls promised for it do not exist. Table status: "Not met — scaffolding only."
5. **New fact not yet in `DESIGN.md`: the single-cluster bypass now plausibly bites on our only real model.** §7's recorded hazard — `GetCapability` skips the net-benefit check (`partition::evaluate` / `retain_viable`) whenever there is exactly one surviving cluster, on the premise "there is no competing partition," which Morpheus already recorded as wrong (the competing partition is always CPU fallback) — was written when Phi-3.5 had 33 clusters. Phi-3.5 now converges to **one fused island**. `viable_islands_retained == 0` on this model is therefore structurally ambiguous between "the gate ran and rejected everything" and "the gate was bypassed," and today's evidence indicates the latter. This is a second gate, not the proof ledger, but it is the same failure shape RAI-008 names: a mechanism that would matter is not exercised on the one artifact we have.

### Findings

| ID | Category | Severity | Finding | Status |
|----|----------|----------|---------|--------|
| RAI-008 (re-verdict) | Deceptive system claim — silent wrong inference output | 🔴 **Stands, unsoftened** | My own falsifier required (a) criterion 11 MET AND (b) a session-creation WARN for UNMEASURED/DIVERGENT forms, both instrument-verified. (a) is false on the record: "Not met — scaffolding only," no ledger, no planted controls. (b) is also false: §8.9.7's session-creation INFO/WARN disclosure is *designed* but I have seen no artifact it produced — R10 applies to my own review exactly as it applies to Mouse's or Trinity's code, and "review of a mechanism is not complete until the reviewer has seen an artifact it produced" binds me too. Good correctness news does not discharge a disclosure gate; that is precisely the substitution R9 forbids — a verdict is not evidence for a *different* claim. **I am not softening this because the news is good.** | 🔴 OPEN |
| RAI-010 | Falsifier gap in my own RAI-008 ruling | 🟡 Advisory, self-correction | My 2026-07-30 falsifier (a)+(b) covered *claim-time* disclosure (proving a form before claiming it, warning at session creation). It did not anticipate *mid-session runtime* Compute() failure — the actual mechanism of both new incidents. A form can be correctly proven and claimed, pass every claim-time check, and still fail at runtime under a condition the proof did not cover (empty tensor shape, OOM), triggering ORT's silent internal fallback with no signal at any layer available today. My falsifier is necessary but was not sufficient; extending it below. | Extending RAI-008's discharge test — see below |
| RAI-011 | Second unwired gate on the only real artifact | 🟡 Advisory, flagged for Mouse/Morpheus, not independently blocking | The single-cluster bypass (§7, line ~1114) plausibly now excludes the net-benefit check on Phi-3.5's single fused island. Same shape as R10 (a mechanism true of one entry point, silent on another) applied to a different gate. Does not raise the severity of RAI-008 — it is evidence *for* the same class of finding, not a new class — but should be named so it is fixed alongside, not after, criterion 11. | Flagged — Mouse owns `GetCapability`/`retain_viable` wiring |

### Extended falsifier for RAI-008 (supersedes the 2026-07-30 version)

The 🔴 discharges only when **all three** hold, each verified by an instrument with a planted control, none by prose:
**(a)** Criterion 11 MET — no form claimable without a ledger entry, a planted `[unproven]` decline demonstrated declining in CI.
**(b)** A session-creation disclosure (§8.9.7) is observed to run — INFO naming every claimed form's proof key and ledger entry, WARN on any UNMEASURED/DIVERGENT form, INFO when zero nodes are claimed — verified by an artifact it produced, not by reading the code that would produce one.
**(c) [new]** A runtime disclosure fires when a *previously claimed* node's `Compute()` returns a non-OK status and ORT is about to fall back — a WARN through ORT's own logging sink, naming the node, the error condition, and that CPU re-execution follows — verified by a planted Compute-failure control (positive) and a normal successful run asserting the WARN does *not* fire (negative), the same two-polarity discipline Guard D now uses for the fallback line itself.

Any one of the three, alone, leaves a silent path a user can hit: (a) alone permits claim-time-proven forms to fail silently at runtime (the actual 2026-07-30→08-01 incidents); (b) alone tells a user what was claimed at start but not what failed mid-session; (c) alone catches runtime failures of already-claimed forms but does nothing to stop unproven forms from being claimed and dispatched in the first place. **The class-level hazard does not close until a user calling `session.run()` cannot receive a silently wrong or silently degraded answer without a signal reaching a channel they can observe**, at claim time and at run time both.

### Ruling 2 — fallback disclosure through ORT's logging sink

**Question:** should the EP itself WARN through ORT's logging sink when its `Compute()` fails and a fallback will follow?

**For:** a user who selected `VulkanExecutionProvider` and silently receives CPU-only inference has been given a different product without being told — this is §1.3's compatibility promise and §8.9's disclosure principle applied to a later point in the pipeline than either currently covers. It has now happened five times, twice since my last verdict, both times with the EP as proximate cause and zero signal at any layer a user could observe. Silence here is the load-bearing mechanism of RAI-008's hazard, not a side effect of it.

**Against:** an EP that logs loudly on every recoverable condition trains users to filter it out; if declined ops (the large majority of any real graph today — 258 dynamic-shape declines alone on Phi-3.5) each produced a WARN, the signal would drown in noise and the one WARN that matters would be read exactly as often as the 258 that don't.

**Where the line sits, mechanically, not in prose:** the two cases are mechanically distinguishable and must be treated differently.
- A node that was **never claimed** (declined at partition time) falls back to CPU *by design*; that is the plan, not a failure, and is already covered once, in aggregate, by §8.9.7's session-creation disclosure ("the EP claimed nothing" / claimed-count summary). No per-node WARN belongs here — this is the noise case above.
- A node that **was claimed** — the EP told ORT it would compute it — and whose `Compute()` then returns a non-OK status is a broken commitment at runtime. This is rare by construction (a claimed node already passed the claim-time proof gate) and is exactly the category that produced both new incidents. **This case must WARN, every time, with no opt-out**, because an EP reneging on a claim it already made is the one condition where "it happened before and was fine" is never a safe inference for the next occurrence.

**The mechanism, not the prose:** the panic/error guard around `Compute()` (line 371, already converting a Rust panic into `ORT_EP_FAIL`) is extended so that on any non-OK return it calls ORT's own logger (the session's registered `OrtLoggingFunction`/`Logger_LogMessage`, not this project's own log crate) at WARNING, before returning the status — naming the node name, op_type, the specific error condition (allocator/shape/etc.), and that CPU re-execution will follow. It must go through **ORT's** sink specifically, because that is the channel already carrying the "Falling back" line itself and the one channel a user with ORT logging configured is already watching; a WARN in our own private log is invisible to exactly the audience that matters. A planted control (force an `alloc(size=0)` or synthetic Compute error) must assert the WARN fires, and a normal successful run must assert it does not — the same two-polarity discipline Guard D was built under, because a WARN that cannot be shown *not* to fire on a good run is not a detector, it is a printed opinion.

### Ruling 3 — is there an RAI dimension to the standing performance directive?

**Not yet — and here is the trigger, named in advance rather than discovered after the fact.**

Morpheus's ruling (the directive "changes the calendar and not one gate") already closes the crudest version of this question: it forecloses the cheapest cheat (do less GPU work to win the ratio) by requiring a non-zero claimed count and an attributed `MATCH` before any timing figure is quotable at all. That protects *coverage* from the performance directive. It does not by itself say anything about *accuracy*, and the prompt's own framing is correct that accuracy is a second lever with the same shape: the tolerance budget (fp16 accumulation, max diff 0.031–0.035 against CPU on a ±13 range) is an accuracy decision, currently justified and gated by Trinity's sign-off (§9.1, "never widened to make a red test green," requires a note in the test).

That gate is real today and I am not finding a violation of it. The RAI dimension is *latent*, not present, and it becomes active at a specific, nameable point:

- **Trigger 1 — undisclosed widening.** Any change to a documented `rtol`/`atol` or accumulation-order tolerance that lands with Trinity's engineering sign-off in the commit but with no corresponding update to a **user-facing** accuracy statement (a precision/accuracy section in `PERF.md` or the README, analogous to the wall-clock-ratio disclosure Morpheus already mandates). Engineering sign-off answers "is this justified"; it does not answer "does the user know." Those are different obligations, exactly as a claim gate and a session disclosure are different obligations (§8.9.7).
- **Trigger 2 — device-conditional precision.** If a kernel variant or `accuracy_level` is ever selected *because* a device is slow (trading fidelity for speed differentially by hardware) rather than fixed per producer-at-version, a user's answer would depend on which GPU they own, silently. This is the accuracy analogue of `UNATTRIBUTED` — a number that is correct about the wrong world and looks identical to one that isn't.
- **The mechanism that would make the boundary visible, ready to propose the day either trigger fires:** a precision-disclosure obligation paired with the existing performance-disclosure obligation — every milestone report/benchmark carries the accuracy budget (tolerance, max diff, the device(s) it was measured on) beside the wall-clock ratio, not only in a test file. A report with the speed number and no accuracy number would be exactly as incomplete, under this rule, as one with neither — which is the same logic Morpheus already applied to the ratio itself.

**Until one of these two fires, this is Trinity's engineering gate, not my finding.** I am naming the trigger now, per R9, precisely so that if it fires, the observation is already specified and cannot be argued down to a one-off after the fact.

### Ruling 4 — llama.cpp licence boundary, for Switch (one paragraph, actionable)

llama.cpp is MIT-licensed, so its GLSL/SPIR-V shader source is legally readable and even legally copyable with attribution — the licence itself does not require Switch to avoid the code, and "study the structure, not copy code" is a project engineering-quality discipline, not a legal constraint. The boundary that matters is the one recorded in `docs/THIRD_PARTY.md` from the OQ-M6 ruling: **algorithms, tiling strategies and subgroup techniques cannot be owned (idea/expression dichotomy) — reading llama.cpp's packed-load and multi-accumulator *approach* and reimplementing the technique in Switch's own idiom for `q_gemv_matmul_nbits_f16` produces independent work with no attribution obligation.** Only if Switch adapts substantial *expression* — copies recognisable GLSL structure, control flow, variable names or comments from llama.cpp's own shaders — does the result become a derivative work; MIT permits this freely, and the only consequence is an entry in `docs/THIRD_PARTY_NOTICES.md` (already templated at `docs/THIRD_PARTY.md` §10) carrying llama.cpp's copyright notice and licence text. The practical test to apply while working: *could you write this shader without the original open next to you?* If yes, it's independent, no notice needed. If the structure requires the original to reproduce, it's derivative, MIT-permitted, and needs the notice. Nothing here blocks Switch's optimisation work; crossing from "study" to "copy" costs a documentation entry, never a legal risk.

### Verdict Summary
- 🔴 Critical: 1 (RAI-008, re-verdict — **stands, unsoftened, extended falsifier**)
- 🟡 Advisory: 2 (RAI-010 self-correction of my own falsifier gap; RAI-011 second unwired gate flagged for Mouse)
- Rulings issued, non-severity: fallback disclosure mechanism (Ruling 2); performance-directive trigger, not yet fired (Ruling 3); licence boundary restated (Ruling 4)

**Overall verdict: 🔴 unchanged.** Correct output on three consecutive runs is real progress and does not discharge a disclosure obligation — those are different claims, and RAI-008's falsifier was written to be about disclosure, not about correctness, specifically so that this day would not be mistaken for a discharge.

**Falsifier for RAI-008 (extended):** see the three-part test above — (a) criterion 11 MET with planted CI control, (b) session-creation disclosure §8.9.7 observed to run via a produced artifact, (c) a two-polarity-verified runtime WARN on claimed-node `Compute()` failure. **Falsifier for Ruling 2:** the mechanism is falsified if a planted Compute-failure control fails to produce the WARN, or if the WARN fires on a normal successful run (false positive) — either observation means the mechanism is not what this ruling requires. **Falsifier for Ruling 3:** either named trigger firing — an undisclosed tolerance widening, or a device-conditional precision choice — converts this from "not yet" to an active finding; their absence is what keeps it advisory rather than critical.

---

## Audit Entry — 2026-08-02T03:21:20-07:00

**Review type:** Fresh pass at Justin's request via coordinator — re-status RAI-008 against material progress since 2026-08-01; four new RAI-surface items; sanity check of the coordinator's own conduct this session
**Files reviewed:** `docs/DESIGN.md` (current, last revised 2026-08-02T02:02:23-07:00 — criterion 10 closing ruling, criterion 11 table row with the four discharge conditions and their status, criterion 12's four-conjunct enumeration, §10.0 obligation 8/8b the device-state companion, wiring census with `broken_commitment_warn` and `net_benefit_gate`), `.squad/decisions/inbox/morpheus-criterion-10-closes.md`, `.squad/decisions/inbox/trinity-criterion11c-and-equivalence-authority.md`, `.squad/decisions/inbox/niobe-per-dispatch-is-not-a-tenancy-signature.md`, `git log` (main at `57d7018`), branch `evidence/criterion-rerun-null-witness`
**Requested by:** Justin Chu, via coordinator

### Context

Criterion 10 (model-level correctness, attributed) closed tonight on the condition written in
advance: real Phi-3.5, three consecutive attributed `MATCH` runs in one session, both devices,
`executed_by` from ORT's own profiler (3 `VulkanExecutionProvider` island executions vs 24 CPU),
`cross_run_identical_to_run1 = true` on all three. Main: Rust 459/0 across ten targets, Python
union 479/0, clippy green, 80 lane checks, both device wiring censuses `unwired: []`. Criterion 11
(the proof ledger) is **closer, not closed** — Morpheus refused the cheapest satisfaction ("a ledger
generated from the claim table would make the check unable to fail") and wrote four discharge
conditions before the tally could move: (a) provenance and (d) the three-token miss path are DONE
(Mouse); (b)(iii) the digest-refusal control is DONE (Mouse); (c), Trinity's — `ledger_hits` shown
to move with its input — is still open. Criterion 12 (wiring census) was over-claimed as closed by
the coordinator and corrected by Morpheus the same night, on the coordinator's own report.

### RAI-008 — re-status against the extended (2026-08-01) falsifier

My falsifier from the last pass has three parts. Checking each against tonight's artifact, not
against the prose describing it:

| Part | Requirement | Status tonight | Evidence |
|---|---|---|---|
| (a) | Criterion 11 MET — no form claimable without a ledger entry, planted CI control | **Still NOT MET** | Table row 11: "Not met — scaffolding only," open specifically on Trinity's (c). Three of four discharge conditions landed; the row is correctly still open, and correctly not closed by the agent who supplied the artifacts (Morpheus's own ruling on this exact temptation, restored this session after a merge nearly dropped it) |
| (b) | Session-creation disclosure (§8.9.7) — INFO per claimed form + proof key/ledger entry, WARN on UNMEASURED/DIVERGENT | **Still NOT MET, in flight** | §10 criterion-list text still reads "Owner: Tank at session creation" with no artifact produced yet; the escape-hatch WARN (criterion 11 item (c) in the older enumeration) is a narrower, already-built cousin — it discloses only when the hatch is manually enabled, which is not this obligation |
| (c) **[new since last pass]** | Runtime WARN on a *claimed* node's `Compute()` failure, two-polarity control | **MET** | Wiring census now carries `broken_commitment_warn` (Tank): planted `ONNXRUNTIME_EP_VULKAN_FORCE_COMPUTE_FAILURE` reads `channel='ORT_SINK' broken_commitments=1 fault_injection='ACTIVE' ort_sink_warn_lines=1` against a clean run reading `channel='UNOBSERVABLE' broken_commitments=0` — exactly the two-polarity discipline my falsifier asked for, and it is in the census, not behind a flag |

**Verdict: RAI-008 remains 🔴 OPEN. It is not discharged — two of three parts are unmet — and I am
recording, without softening either direction, that one part I added last session has actually
landed, verified, in the shape I specified.** This is the correct way for a 🔴 to move: not by the
news being good, but by an instrument I named actually going green. I am glad to write that
sentence for once instead of a caution against writing it.

**RAI-011 (single-cluster bypass) — update.** The census now reports `net_benefit_gate: EVALUATED
clusters_seen=1 evaluations=1 bypasses=0 sole_island_overrides=1 viable_islands_retained=0` — bypass
and override are now separate, typed fields instead of sharing one `0` (R12; this is the exact fix
RAI-011 asked for). **Downgrading RAI-011 to resolved-as-instrumented**: the ambiguity I flagged
(bypassed vs. all-rejected reading the same) is gone; whether the gate's *economics* are right on a
single-island graph is Mouse's and Morpheus's question now, not a disclosure gap.

### New RAI surface, traffic-lighted

**1. The `CLAIM_UNPROVEN` escape hatch — 🟡 Advisory, not 🔴.** The hatch cannot silently claim
unproven forms: it takes a list of exact proof keys only (a planted `*`, `1`, and bare op-type are
all rejected — C1's enforcement shape), logs at WARN naming every enabled key, and
`unproven_forms_enabled` fails `epctl --check-counters` without an explicit `--allow-unproven`. That
is real disclosure infrastructure, not a name on a settings screen. **The residual risk is real and
worth naming precisely: a user who is annoyed by the WARN can pass `--allow-unproven` once and
silence it for good**, at which point the escape hatch behaves exactly like the pre-ledger world for
every key on that list. This is not a defect in the mechanism — an escape hatch that could not be
disabled would not be an escape hatch — but it means the ledger's guarantee is only as strong as the
friction around `--allow-unproven`. **WHAT would raise this to 🔴:** if `--allow-unproven` ever
becomes settable via a config file, environment default, or anything other than an explicit
per-invocation flag a human types, the friction is gone and the hatch becomes the silent path again.
**Recommendation, not a blocker:** `--allow-unproven` should require a reason string logged beside
the enabled-keys line (mirrors this project's own `DeclineCode` discipline — every decline already
carries a reason, an override should carry one too), and CI should refuse to green a lane that uses
it, exactly as `epctl` already refuses without the flag. Owner: Tank, non-blocking.

**2. Is there an RAI obligation on performance claims for an inference accelerator, beyond
Morpheus's engineering gate? Yes, and I recommend writing it into policy rather than leaving it as
one project's discipline.** Morpheus's obligation (no timing figure quotable outside `MATCH`, the
device-state companion, `STEADY_UNCERTIFIED` as a fourth state, the two-polarity Guard-D pattern
applied to `gpu_steady_tail`) is the engineering version of a principle that is squarely RAI's: **an
accelerator's performance claim is a claim a user acts on** — it informs purchase decisions, capacity
planning, and trust in every other number the project publishes. `baseline_certified` being the
cleanest run by every dispersion measure the project owns and still 21.4× wrong, caught only by the
device-state companion, is the sharpest specimen available anywhere in this project of *engineering
rigor discharging what would otherwise be an RAI failure* — a number that looked unimpeachable by
every test available and was wrong by more than an order of magnitude. **My recommendation: add a
policy line under Deceptive Patterns in `.squad/rai/policy.md` — "A performance figure for an
accelerator or ML system is an ungrounded factual claim (🔴) unless it carries the record of the
instrument that could have contradicted it."** This does not change today's practice, which already
meets that bar; it makes durable what is currently one engineer's vigilance, so that the standard
survives a personnel change or a rushed milestone in a way a discipline that lives only in one
person's rulings does not. **Trigger for treating this project's own practice as falling short:** any
future benchmark quoted without its device-state companion or its `MATCH`/attribution frame.

**3. Intel iGPU performance is permanently uncertifiable on the hardware in this project's matrix,
and that is a disclosure gap nobody has named as one yet — 🟡 Advisory, escalating to a hard
pre-publication gate.** §10.0's obligation 8 already establishes that a platform with no clock
telemetry reports `STEADY_UNCERTIFIED` forever, and names Intel's iGPU as *more*, not less, exposed
(shared power budget with loaded CPU cores). If that is durable — no counter, no WMI class, no tool,
no Vulkan query surfaces a usable device clock on this hardware — then **every performance figure
this project will ever publish is NVIDIA-only, permanently, and a user on Intel silicon receives an
EP whose speed characteristics on their own hardware can never be stated.** This is not currently a
violation because no performance figure has shipped publicly yet (Morpheus's own withdrawal
discipline has kept it that way). **It becomes a blocking disclosure requirement, not merely
advisory, the day any performance figure ships publicly (README, release notes, PERF.md marked
final rather than in-progress):** that artifact must state, explicitly and not in a footnote, that
timing figures are certified on NVIDIA only and that Intel iGPU performance is and will remain
uncertified absent new instrumentation — the same prominence Morpheus already requires for the
wall-clock ratio itself (§10.0 disclosure obligation, "never omitted, including when it is worse
than 1.0"). Owner: Link to confirm the "permanently" claim holds after a genuine cross-platform
search (WMI, `NVML`-equivalent, `VK_EXT_calibrated_timestamps` extensions, driver-specific counters)
before this is written as a permanent fact rather than a current one; Niobe/PERF.md to carry the
disclosure text once a figure ships.

**4. `rust/src/trace.rs` ownership — 🟢 Green, closed by the coordinator's own action.** Assigning
ownership tonight to the file holding the project's only sanctioned tick conversion, after Link's
static screen proved it is the single entry point for raw ticks in the tree, is exactly the right
remedy and needs no further finding from me. One process recommendation, non-blocking: this
project's ownership assignments currently live in prose (rulings, this session's message) rather
than in a single artifact a new contributor or a script could read. A `CODEOWNERS`-shaped file
(or a table in `DESIGN.md` itself) would make "who owns this file" a lookup rather than a memory,
which is the same class of improvement R10 asks for applied to governance instead of mechanisms.

### Sanity check of the coordinator's conduct this session

- **Resolving merge conflicts inside other agents' authored documents, then disclosing after rather
  than asking before — 🟡 Advisory, not a violation, with a process refinement.** The one concrete
  instance reviewed (the withheld-tally sentence, restored 2026-08-02T01:42:02) shows the right
  outcome: the coordinator's own ruling notes "neither side was a superset," he "correctly declined
  to splice" prose, told Morpheus, and the sentence was restored with the underlying finding
  unchanged throughout (Mouse's evidence was never disputed — only whether the tally could move).
  **Telling-after was sufficient in this instance because nothing load-bearing was lost for longer
  than one review cycle and the correction was public.** I would not generalize "tell after" as the
  standing rule, though: the specific risk is a conflict resolution that *silently* drops a
  dissenting verdict rather than a supporting sentence — had this been a rejected 🔴 finding rather
  than a restored sentence, telling-after would be too late by definition, because the finding would
  already have shipped without it. **Recommendation:** for merge conflicts inside a named agent's
  *verdict* text specifically (not supporting prose), stop and flag before completing the merge,
  because a verdict is the one artifact class where "ask forgiveness" and "the thing already shipped
  wrong" are the same event.
- **Preserving unclaimed artifacts on `evidence/criterion-rerun-null-witness` rather than committing
  to main or discarding — 🟢 Green.** Textbook handling of a finding whose owner had not ruled:
  neither buried nor prematurely promoted to a claim. No finding.
- **Pushing with three known, precisely-attributed, in-flight red items rather than holding
  twenty-one verified commits off origin — 🟢 Green.** This matches the project's own established
  convention throughout `DESIGN.md`'s M0 table — partial/open criteria are the normal state of a
  transparently tracked milestone, not a defect to hide behind a hold. The distinguishing test is
  the one this project already applies everywhere else: is each red item named, owned, and does its
  status say so, rather than being silent or claimed green. It is. No finding.
- **Over-claiming criterion 12 as closed, then correcting publicly — 🟡 Advisory, self-corrected,
  and worth naming as a pattern rather than an incident.** This is a real instance of exactly the
  witness/discharge conflation Morpheus's own ruling names ("he held a witness and read it as a
  discharge"), and it is at least the second time this conflation has occurred on this project by a
  different name (RAI-008/009's own history has a structurally identical shape: a correct run being
  read as a discharge of a disclosure obligation). **The self-correction, same session, is the
  standard this project has set for itself and I am not treating it as a violation** — but I record
  it because a pattern repeating across two different people (Mouse's union-defect specimen,
  the coordinator's criterion-12 specimen) in one week is worth a durable check rather than a
  standing vigilance requirement on any one person. **Recommendation:** a small doc-consistency
  script asserting that any prose claiming a criterion "closed"/"met" agrees with that criterion's
  own table cell before a status update is posted to the team — mechanical, cheap, and it converts
  a recurring human error into an `ERROR(instrument)` the register already has a name for.

### Verdict Summary
- 🔴 Critical: 1 (RAI-008 — **remains open, unchanged severity, one of three falsifier parts now
  genuinely discharged**)
- 🟡 Advisory: 4 new (escape-hatch friction; performance-claim policy recommendation; Intel
  permanent-uncertifiability disclosure, escalating to a hard gate on first public figure;
  witness/discharge conflation as a recurring pattern) + RAI-011 downgraded to resolved-as-instrumented
- 🟢 Green: 3 (branch preservation; pushing with attributed red items; trace.rs ownership assignment)
- Overall: **no 🔴 escalation this pass.** Nothing reviewed tonight — the coordinator's conduct
  included — rises to a genuine 🔴 beyond the one already open and already tracked.

**Falsifiers, named per new item:** escape hatch (2) — falsified into 🔴 if `--allow-unproven`
becomes settable other than as an explicit per-invocation flag; performance-claim policy (3) —
falsified (i.e., shown unnecessary) if this project ever quotes a figure without its companion and
suffers no consequence, which would mean the discipline was never load-bearing; Intel disclosure
(4) — falsified if Link's search finds a usable device-clock surface on Intel after all, in which
case "permanent" becomes "not yet solved" and the obligation softens accordingly; witness/discharge
pattern — falsified if the doc-consistency script, once built, never fires, meaning the human
discipline alone was already sufficient.

---

## Audit Entry — 2026-08-03T07:40:00-07:00

**Review type:** Recall via coordinator — rule on RAI-008(a) (Tank's CI-plant repair, `137e40f`);
judge the honesty of the proof-ledger decline message together with Link's counters finding; spot
check `docs/DESIGN.md` §0 for overstatement
**Files reviewed:** `rust/src/registry.rs` (`Ledger::state_for`, `parse_ledger`, `claim_decision_audited`),
`rust/src/counters.rs`, `rust/src/disclosure.rs`, `tests/ops/test_proof_ledger.py`,
`bench/results/link-linux-downstream/{counters-linux.json,pytest-linux.log}` (Link's worktree, live),
`bench/results/_ledger_counters.json` (main, post-`137e40f`), `docs/DESIGN.md` §0/§10 (criterion 11 row),
`.squad/decisions.md` Round 10, `.squad/agents/tank/history.md`
**Requested by:** Justin Chu, via coordinator

### Findings

| ID | Category | Severity | Finding | Status |
|----|----------|----------|---------|--------|
| RAI-008(a) re-check | Falsifier discharge | 🔴 stays OPEN | Tank's `137e40f` repairs the CI plant (rot fixed, non-vacuous membership assertion added, 14/0) — real, verified, correctly-scoped maintenance. It does not discharge RAI-008(a), whose falsifier requires Criterion 11 MET as a whole. Criterion 11's DESIGN.md row is still NOT MET, open on an unrelated defect the plant fix does not touch: no predicate reads `LedgerEntry.device`, so `wiring_census-dev1.json` reads 6 forms proven on a device (`device0`, a selector ordinal) nothing has ever proven anything on. | Not credited — criterion 11 must close on the device-attribution defect, not on the plant |
| RAI-012 | Deceptive/misleading diagnostic — ledger-fault decline message names the wrong subject | 🟡 Advisory, named 🔴 trigger | Confirmed live in the tree, not hypothetical: `counters-linux.json` reads `ledger_faults: 97, ledger_entries: 0`; `pytest-linux.log` has 0× the true-cause line (`proof ledger fault: ...`, from the `OnceLock` in `registry::ledger()`) against 42× the generic decline text (`no proof ledger entry ... nothing has proven it correct on this form`), which is false in both clauses — 97 entries exist, some proving these exact forms elsewhere, and the true cause is a whole-ledger parse/digest fault. `Ledger::state_for` blankets every key to `Unproven` when `self.faults` is non-empty, bypassing the `SubjectChanged` branch built to name a frame mismatch accurately. Fail-safe output is preserved (decline is still correct), so this is a diagnostic-honesty defect, not RAI-008's silent-wrong-output class. Not Linux-specific: the blanket applies on the certified Windows lane too, on any whole-ledger fault (corrupted install, hand-edited file, `ONNXRUNTIME_EP_VULKAN_LEDGER_FILE` digest mismatch) — it just hasn't fired there yet because the Windows ledger currently parses cleanly. | 🟡 OPEN — owner Mouse (message construction + `OnceLock`/`attach_default_ort_logger` ordering). Escalates to 🔴 if it ships unfixed on Windows and fires against a real user. |
| RAI-013 | Disclosure emitted but below default visibility — does not discharge §8.9.7 | 🟡 Advisory | `_ledger_counters.json` (main, post-`137e40f`): `session_disclosure_infos: 1`, `session_disclosure_infos_to_ort_sink: 0`, `session_disclosure_info_channel: "BELOW_ORT_THRESHOLD"`, `ort_sink_severity_threshold: "WARNING"`. The self-report is honest and itself good instrumentation (R9/R13 discipline) — credited. But a normal user, ORT's default logging severity unchanged, never sees this INFO line: "we emitted it and can prove it didn't land" is not "the user was told," the same substitution my 2026-08-02 ruling named on a different artifact. | 🟡 OPEN — owner Tank (raise this disclosure to WARN, or Morpheus documents the gap in DESIGN.md §0.2 explicitly) |
| — | `docs/DESIGN.md` §0 accuracy | 🟢 Green, one omission | Spot-checked `oracle_outputs_within_tolerance: 62` and `argmax_cpu`/`argmax_vk: 30751` against `criterion10-dev0.json` — matches exactly. No overstatement found in the sampled claims; the section's existing self-correction history (withdrawn "bit-identical," withdrawn `x/x` weight-amplification identity) stands. §0.2 omits both RAI-012 and RAI-013's current state, which belong there by its own stated purpose. | 🟡 minor — Morpheus to add one line each once Mouse's/Tank's fixes land or are scoped |

### Verdict Summary
- 🔴 Critical: 1 (RAI-008 — **remains open; (a) explicitly NOT credited by this repair, (b) and (c)
  stand as previously credited**)
- 🟡 Advisory: 2 new (RAI-012 ledger-fault decline message; RAI-013 unseeable INFO disclosure) + one
  minor doc-omission note on DESIGN.md §0.2
- 🟢 Green: Tank's CI-plant repair credited as real maintenance, not as criterion closure; DESIGN.md
  §0's sampled numeric claims check out against cited artifacts
- Falsifiers: RAI-012 is falsified-upward (🔴) by an unfixed occurrence on the Windows lane against a
  real user; RAI-013 is falsified-downward (discharged) by either the severity change or the doc
  addition landing.
