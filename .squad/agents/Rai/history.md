# Project Context

- **Project:** onnxruntime-ep-vulkan
- **Created:** 2026-07-29

## Core Context

Agent Rai initialized and ready for work.

## Recent Updates

📌 Team initialized on 2026-07-29

## Learnings

Initial setup complete.

---

## Session — 2026-08-01T09:53:14-07:00 (Recall — re-verdict against a materially changed system)

### Task
Justin recalled me to re-examine RAI-008/009 against my own stated falsifier, now that the EP
genuinely executes correctly on Phi-3.5 (attributed `MATCH`, ORT profiling as an instrument we do
not own) but has also produced two more silent-fallback events since my last ruling. Also asked:
should the EP WARN through ORT's logging sink on Compute failure; is there an RAI dimension to
Justin's standing performance directive; and a llama.cpp licence boundary paragraph for Switch.

### Key Facts Established

- **RAI-008 stands, unsoftened.** My 2026-07-30 falsifier required (a) criterion 11 MET and (b) a
  session-creation WARN, both instrument-verified. Neither holds: criterion 11 is confirmed
  "scaffolding only" (no ledger, no planted controls), and §8.9.7's disclosure design has no
  artifact I have observed it produce (R10 applies to my own review, not only to others' code).
  **Correct output today is not evidence about disclosure — those are different claims**, and
  conflating them would be the exact substitution R9 forbids.

- **My own falsifier had a gap, and I am recording it as a self-correction rather than pretending
  it was always complete.** RAI-008's (a)+(b) test covered claim-time disclosure. It did not
  anticipate *mid-session runtime* `Compute()` failure on an already-proven, already-claimed form
  — which is exactly what caused both new incidents (weight-cache OOM; `alloc(size=0)` on
  zero-length KV tensors). A form can pass every claim-time gate and still fail silently at
  runtime under a condition the proof never exercised. Extended the falsifier to a three-part
  test: ledger (a), claim-time disclosure (b), and a new runtime WARN on claimed-node Compute
  failure (c), each requiring an instrument with a planted, two-polarity control — none by prose.

- **Fallback disclosure ruling: WARN through ORT's own sink, but only on broken commitments, not
  on ordinary declines.** The mechanical line: a node never claimed falling back to CPU is the
  plan, already disclosed once in aggregate at session creation — no per-node signal needed. A
  node that *was* claimed and then fails its `Compute()` call is the EP reneging on a commitment
  it already made to ORT, and that must WARN every time, through ORT's logging sink specifically
  (the channel already carrying "Falling back", not our private log), verified by a two-polarity
  planted control — same discipline Guard D now uses for the fallback line itself.

- **The performance directive has a latent RAI dimension, not yet an active one, and I named the
  trigger rather than declaring "not yet" without one.** Morpheus's ruling already forecloses the
  coverage-side cheat (do less GPU work to win a ratio). It says nothing about the accuracy side:
  tolerance widening under performance pressure, or device-conditional precision selection, are
  the accuracy-lever equivalent of that same cheat and are currently gated only by Trinity's
  engineering sign-off — an "is this justified" gate, not a "does the user know" one. Two named
  triggers: undisclosed tolerance widening (engineering sign-off lands, user-facing accuracy
  section doesn't), or precision keyed on device speed rather than fixed per producer-at-version.
  Either firing converts this from advisory-latent to an active finding.

- **llama.cpp licence boundary restated for Switch in one paragraph**: MIT permits reading and
  even copying with attribution; the "study, don't copy" instruction is an engineering-quality
  choice, not a legal one. The idea/expression line still governs: technique reproduced in
  Switch's own idiom is independent work, no notice; substantial expression adapted is a
  derivative work, MIT-permitted, costs one `THIRD_PARTY_NOTICES.md` entry, never a legal risk.

### Learnings

1. **A falsifier can be wrong by being incomplete rather than by being false.** My 2026-07-30
   falsifier was never falsified in the sense of being shown untrue — it was shown *insufficient*
   by an incident it didn't cover. R9 applies recursively: naming what would falsify a verdict
   includes staying alert to whether the named test actually spans the space of ways the
   underlying hazard can recur. It didn't, and the fix is to extend the test, not to defend it.

2. **Good news and a disclosure gate answer different questions, and the discipline is refusing to
   let one stand in for the other.** "The output is correct on three runs" and "a user would learn
   if it weren't" are independent claims. Softening a disclosure verdict because the correctness
   news improved is exactly the failure mode RAI-008's falsifier was written to prevent, and
   Justin's recall prompt tested precisely whether I would hold that line under good news.

3. **A latent RAI concern deserves a named trigger, not a permanent "not applicable."** Declining
   to raise a verdict is only defensible if paired with the specific observation that would raise
   it — otherwise "not yet" quietly becomes "never," which is the same failure as an unfalsifiable
   green.

4. **Mechanism-not-prose applies to my own rulings, not only to the ones I review.** Both the
   fallback-disclosure ruling and the extended falsifier specify a planted, two-polarity control
   as part of the ruling itself, not as a follow-up someone else designs later.

### Deliverables
- Appended to `.squad/rai/audit-trail.md` (RAI-008 re-verdict, RAI-010, RAI-011, four rulings)
- Appended to `.squad/agents/Rai/history.md` (this entry)
- Created `.squad/decisions/inbox/rai-recall-reverdict.md`

---

## Session — 2026-08-02T03:21:20-07:00 (Fresh pass — criterion 10 closes, criterion 11 nearly, coordinator conduct)

### Task
Justin asked for a fresh RAI pass via the coordinator: re-status RAI-008 now that criterion 10
closed on the condition set in advance and criterion 11's proof ledger landed most of its four
discharge conditions; assess four new possible RAI-surface items (the escape hatch, a
performance-claim policy question, Intel's permanent performance-uncertifiability, and
`trace.rs`'s prior lack of an owner); and sanity-check the coordinator's own conduct this session
(merge-conflict resolution inside others' documents, preserving an unclaimed branch, pushing with
known red items, and self-correcting an over-claimed criterion).

### Key Facts Established

- **RAI-008 remains 🔴 OPEN, and I checked it against my own three-part falsifier rather than
  against the good news surrounding it.** (a) criterion 11's ledger — still not met, correctly:
  Morpheus refused to let the row close on the artifact-supplier's own tally, and it is open
  specifically on Trinity's one remaining discharge condition. (b) the §8.9.7 session-creation
  disclosure — still not built, owned by Tank, in flight. (c) **the part I added last session — a
  runtime WARN on a claimed node's `Compute()` failure — has actually landed**, as
  `broken_commitment_warn` in the wiring census, with the exact two-polarity control I asked for
  (`fault_injection='ACTIVE'` vs a clean run reading `UNOBSERVABLE`). **A 🔴 moving because a named
  instrument actually went green, one part at a time, without the whole verdict softening, is the
  falsifier discipline working as designed** — I recorded the progress precisely and did not let it
  bleed into the other two unmet parts.

- **RAI-011 (single-cluster bypass) downgraded to resolved-as-instrumented.** The wiring census now
  reports `bypasses` and `sole_island_overrides` as separate typed fields instead of one collapsing
  `0` — the exact ambiguity I flagged is gone. Whether the *economics* are right on a single-island
  graph is now an engineering question for Mouse/Morpheus, not a disclosure gap for me.

- **The escape hatch (`CLAIM_UNPROVEN`) is real disclosure infrastructure, not a facade — but its
  guarantee is only as strong as the friction around disabling the WARN.** Recommended a reason
  string requirement on `--allow-unproven`, mirroring this project's own `DeclineCode` discipline
  (every decline already carries a reason; an override should too). Named the exact trigger that
  would raise this from 🟡 to 🔴: `--allow-unproven` becoming settable other than as an explicit
  per-invocation flag.

- **Performance claims for an accelerator are an RAI matter, and I recommended writing that into
  policy rather than leaving it as one engineer's discipline.** `baseline_certified` — the cleanest
  run by every dispersion measure this project owns, and 21.4× wrong, caught only by the
  device-state companion — is the sharpest specimen on this project of engineering rigor
  discharging what would otherwise have been a deceptive, ungrounded claim shipped with full
  confidence. Proposed a `.squad/rai/policy.md` addition: an accelerator performance figure is a
  🔴 deceptive claim unless it carries the record of the instrument that could have contradicted it.

- **Intel iGPU performance is on track to be permanently uncertifiable on this project's hardware,
  and nobody had named that as a disclosure question until asked.** Not a violation today — no
  performance figure has shipped publicly — but I named it as a **hard gate** that activates the
  day any figure does ship: the artifact must say, at the same prominence as the wall-clock ratio
  itself, that timing is NVIDIA-only and Intel is and will remain uncertified absent new
  instrumentation. Asked Link to confirm "permanently" holds after an exhaustive platform search
  before it is written as durable fact.

- **Sanity-checked the coordinator's own conduct and did not manufacture a finding to seem
  balanced.** Branch preservation and pushing with attributed red items are both 🟢 — they match
  conventions this project already applies to itself throughout the M0 table. The merge-conflict
  handling (tell-after) was sufficient in the one instance reviewed because nothing load-bearing
  was lost for longer than one cycle and the fix was public — but I distinguished that from the
  higher-risk case (a dissenting *verdict*, not supporting prose, silently dropped in a conflict),
  where tell-after would be structurally too late, and recommended stopping before completing a
  merge specifically for verdict text. The criterion-12 over-claim was a real instance of the
  witness/discharge conflation Morpheus's own ruling names, correctly self-corrected same session —
  I logged it as a recurring pattern (a second person, same conflation, in one week) rather than
  either dismissing it or treating a self-corrected, disclosed error as a violation.

### Learnings

1. **A falsifier is doing its job when it can move a verdict one part at a time without moving the
   whole verdict.** RAI-008 stayed 🔴 tonight while I explicitly credited one of its three parts as
   discharged — that asymmetry (partial credit, no change in severity) is the correct behaviour of
   a conjunctive test and is worth modeling explicitly rather than reporting only the aggregate.

2. **Sanity-checking a requester's own conduct requires the same discipline as reviewing anyone
   else's work: name what's fine as fine.** Three of four coordinator conduct items were 🟢 on
   inspection; manufacturing advisory findings to appear balanced would be its own violation of
   R9 (a finding with no falsifier, issued to satisfy an expectation rather than an observation).

3. **A recurring human error pattern (witness read as discharge) across two different people is
   worth a mechanical check, not a renewed vigilance instruction to either person.** This mirrors
   R10/R11's own lesson applied to process rather than code: a rule that depends on someone
   remembering to apply it decays the same way an unwired mechanism does.

4. **A latent RAI concern (the performance-claim policy question) is worth writing into durable
   policy the moment a project's own practice already meets the bar it implies** — codifying what
   is currently good discipline is cheaper before a personnel change or a rushed milestone tests it
   than after.

### Deliverables
- Appended to `.squad/rai/audit-trail.md` (RAI-008 re-status, RAI-011 downgrade, four new advisory
  items, coordinator-conduct sanity check)
- Appended to `.squad/agents/Rai/history.md` (this entry)
- Created `.squad/decisions/inbox/rai-fresh-pass-2026-08-02.md`
- Working branch: `squad/rai` (not pushed)

---

## Session — 2026-08-03T07:40:00-07:00 (RAI-008(a) recall; the ledger-fault message; INFO-below-threshold)

Recalled via coordinator for three rulings: Tank's RAI-008(a) CI-plant repair (`137e40f`), Link's
finding that the proof-ledger decline message misnames its subject under a whole-ledger fault, and a
spot check of `docs/DESIGN.md` §0 for overstatement.

**RAI-008(a): NOT credited.** Tank's plant repair (imported `PLANTED_CONTROL_CASE`/`_KEY`, fixed a
stale docstring prediction, added the missing non-vacuous membership assertion, 14/0) is real,
verified, correctly-scoped maintenance — credited as that. It does not discharge RAI-008(a), whose
falsifier requires Criterion 11 MET *as a whole*; Criterion 11's own DESIGN.md row is still NOT MET,
open on the unrelated device-attribution defect (`LedgerEntry.device` unread by any predicate — six
forms read `ALL-PROVEN` on a device nothing has proven anything on). A repair to the control is not a
repair to the criterion. RAI-008 stays 🔴 OPEN.

**New finding, verified live in the tree (not hypothetical):** `bench/results/link-linux-downstream/
counters-linux.json` reads `ledger_faults: 97, ledger_entries: 0`; the matching `pytest-linux.log`
has **0×** the true-cause line (`proof ledger fault: ...`, emitted once from a `OnceLock` in
`registry::ledger()`) against **42×** the generic decline text (`no proof ledger entry ... nothing
has proven it correct on this form`) — false in both clauses, since 97 entries exist and some prove
these exact forms elsewhere; the true cause is a whole-ledger parse/digest fault, not key absence.
Traced to source: `Ledger::state_for` blankets every key to `Unproven` whenever `self.faults` is
non-empty, bypassing the `SubjectChanged` branch built for exactly this kind of accurate naming.
Fail-safe output is preserved (still declines correctly to CPU), so this is a diagnostic-honesty
defect, not RAI-008's silent-wrong-output class — named 🟡 (RAI-012), with an explicit 🔴 trigger:
unfixed and firing on the certified Windows lane against a real user. Owner: Mouse.

**INFO disclosure below ORT's default threshold (RAI-013), 🟡.** `_ledger_counters.json` post-`137e40f`
honestly reports `session_disclosure_infos_to_ort_sink: 0` / `BELOW_ORT_THRESHOLD` / threshold
`WARNING`. Credited: the self-report is good instrumentation. Ruled: an honestly-labeled emission
that a user will never see by default does not discharge §8.9.7's obligation — same substitution my
2026-08-02 ruling already named ("a verdict is not evidence for a different claim"). Owner: Tank
(raise severity) or Morpheus (document the gap explicitly in §0.2) — either closes it.

**DESIGN.md §0 spot check: no overstatement found**, two numeric claims verified against
`criterion10-dev0.json` exactly. One omission noted: §0.2 should name both open items above, by its
own stated standard.

### Learnings
- **A repair to a control is not a repair to what the control was gating.** RAI-008(a)'s falsifier is
  a conjunction (criterion MET *and* planted control correct); crediting the conjunction on one
  conjunct's repair would be the same witness/discharge conflation this project has already named
  twice this week by other people's hands. Worth re-checking every time a "the CI check now works"
  commit arrives framed as closing a criterion it only partially touches.
- **A message can be safe and still be false.** RAI-008's hazard class is silently wrong *output*;
  this session's ledger-fault message is a case where the output stays correct (fail-safe decline)
  but the stated *reason* is false and actively misdirects remediation. These need separate severity
  tracks — conflating them either over-blocks safe-but-mislabeled behavior at 🔴 or, worse, lets a
  real diagnostic-honesty defect hide inside a "no RAI-008 recurrence" verdict.
- **"We can prove it didn't reach the user" is not the same claim as "the user was told."** This is
  the second time this exact substitution has needed naming (first: correctness vs. disclosure,
  2026-08-02). It generalizes: any obligation phrased as an *action* (disclose, warn, inform) is not
  satisfied by evidence about the *attempt*, however honestly instrumented the attempt's failure is.

### Deliverables
- Appended to `.squad/rai/audit-trail.md` (RAI-008(a) re-check, RAI-012, RAI-013, DESIGN.md spot check)
- Appended to `.squad/agents/Rai/history.md` (this entry)
- Created `.squad/decisions/inbox/rai-008a-not-credited-and-the-ledger-fault-message.md`
- Working branch: `squad/rai` (not pushed to `main`)

---

📌 Team update (2026-07-30T19:05:03-07:00) — Scribe

Two findings apply to every agent on the team:

**(a) A mechanism that exists in a file but not in a call graph is indistinguishable from
one that does not exist.**  Verification by reading is insufficient.  Verify by running.
Five such mechanisms surfaced in this single batch: partition.rs, the GPU tracer,
model_output_equivalence, compute_failures, and should_claim_island.  In every
case the code was correct; the wiring was absent; the absence was invisible to review.

**(b) 85.9% of inference wall-time involves no GPU work** (recording 68.3%, fence-wait
idle 16.3%, submit 0.3%; GPU kernels 14.1%).  Optimising GPU kernels before the
command-buffer recording bottleneck is resolved is low-leverage.  Align work priorities
accordingly.


📌 Team update (2026-08-01T17:16:56-07:00): Intel device-clock figures are permanently uncertifiable on this hardware (`none_available`, no producer exists and none of the available proxies are the right kind of quantity) — attack the Intel/NVIDIA residual with counts and shapes, not clocks — decided by Niobe


📌 Team update (2026-08-01T17:16:56-07:00): All wall-clock figures remain withdrawn; only counts, bytes and certified-companion device-clock figures are quotable — decided by Switch, Morpheus, Niobe, Link


📌 Team update (2026-08-01T17:16:56-07:00): `ledger_lookup` is the last `UNWIRED` mechanism in the instrument census (criterion 11); Mouse is building it — decided by Trinity, Mouse


## [SUMMARY] Compressed entries

<!-- SUMMARIZED by Scribe 2026-08-02T15:26:11.334894 -->

- **Session — 2026-07-28T19:16:08-07:00 (OQ-M6 / First RAI Pass)** — Resolved OQ-M6: licence compliance ruling for reading/adapting llama.cpp Vulkan shaders. Conducted first RAI pass of all project docs.
- **Session — 2026-07-30T07:12:15-07:00 (R9 event — silent inference RAI verdict)** — On-demand RAI ruling on the R9 event (§10.0.1): the EP claimed and dispatched 161 `MatMulNBits`

📌 Team update (2026-08-04T12:25:00-07:00): Morpheus's ruling — the screening question for a
criterion motion is not "is this true?", it is "what does it admit?" (the criterion-10
unsatisfiability finding was refuted without a run; the false premise was harmless, but relaxing the
criterion on its strength would have admitted two of three real failures with no element moving).
Applies to any motion you rule on against a criterion or a prior finding. — decided by Morpheus

📌 

📌 Team update (2026-08-04T20-25-00-07-00): Trinity's model-scale float64 oracle finding -- at model scale both EPs are ~6x further from true than from each other (common error, not opposed) -- this bounds what an AGREE between the two EPs can mean: consistency-with-each-other is not evidence of correctness against the reals. -- decided by Trinity

📌 Team update (2026-08-04T20-25-00-07-00): Tank's "loudly logged, silently returned" finding -- `Compute()` is entered, the intermediate `ep_inter_76` allocation is refused (67108864 bytes), three disclosures fire at ORT's default severity, but exit 0 / finite logits / `get_providers()` all read as success. Also: `disable_cpu_ep_fallback=1` screens the partition, not the fallback -- arm D at ctx 1 with no island produces the identical refusal sentence. -- decided by Tank
