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

## Session — 2026-07-28T19:16:08-07:00 (OQ-M6 / First RAI Pass)

### Task
Resolved OQ-M6: licence compliance ruling for reading/adapting llama.cpp Vulkan shaders. Conducted first RAI pass of all project docs.

### Key Facts Established
- **All four external sources** (llama.cpp, ExecuTorch, ORT, gpuinfo.org) are compatible with this project's MIT licence. No copyleft, no conflicts.
- **Reading is always free.** The idea/expression dichotomy means algorithms, tiling strategies, and subgroup techniques cannot be owned. Attribution only triggers on substantial code adaptation.
- **ExecuTorch BSD-3** adds a no-endorsement clause: cannot use "Meta", "Arm", "Qualcomm" etc. names to promote this project without permission.
- **SPIR-V derived from adapted GLSL is a derivative work.** Attribution via THIRD_PARTY_NOTICES.md in the distribution package is sufficient — no need to embed text in the binary.
- `docs/THIRD_PARTY_NOTICES.md` does **not** need to exist pre-implementation. Template provided in `docs/THIRD_PARTY.md` §10.

### Learnings
1. **Fetch licences rather than assuming.** Both llama.cpp and ORT are MIT; ExecuTorch is BSD-3 not MIT — the distinction matters for the no-endorsement clause.
2. **CC-BY 4.0 data (gpuinfo.org) is a documentation concern, not a code licence concern.** Already correctly attributed.
3. **Pre-implementation projects often have platform coverage claims without CI evidence.** Flag this early — it is easier to caveat in a README now than to correct after users have relied on the claim.
4. **"whitelist" in technical ML docs** (fusion allowlist) is a common legacy term. Flag it as advisory; do not require a blocking fix.
5. **The grey-zone rule** that proved most useful: "Could you write this code without looking at the original?" If yes, it's independent work. If the structure requires the original to reproduce, it's derivative.

### Deliverables
- Created `docs/THIRD_PARTY.md`
- Created `.squad/decisions/inbox/rai-oq-m6-license-ruling.md`
- Appended to `.squad/rai/audit-trail.md`

---

## Session — 2026-07-30T07:12:15-07:00 (R9 event — silent inference RAI verdict)

### Task
On-demand RAI ruling on the R9 event (§10.0.1): the EP claimed and dispatched 161 `MatMulNBits`
nodes on Phi-3.5-mini, `compute_failures: 0`, test suite green, model output all zeros.
Required: (1) two traffic-light verdicts separated by object, (2) ruling on whether silent wrong
numerical output is RAI or engineering with a test for the line, (3) mechanical obligation, (4)
explicit agreement or disagreement with Morpheus.

### Key Facts Established

- **Two objects, two verdicts, deliberately separated.**
  - **The instance** (wrong fp16 kernel): 🟡 — engineering defect, Mouse fixing, criterion 10 tracks
    it, does not independently require RAI lockout because Morpheus's gate already blocks M0.
  - **The class** (architecture permits silent wrong inference at any layer): 🔴 — the proof ledger
    (criterion 11, NOT MET) and a session-layer runtime WARN must exist before the EP ships any
    op claims. This property does not go away when Mouse's fix lands.

- **The line between RAI and engineering quality is real and I located it.** The test: *does the
  defect produce a confident, well-formed, contextually disconnected output in a system whose
  output will be consumed and acted upon by a human, with no signal at any available layer?* If
  yes: RAI. If the error degrades output proportionally, introduces measurable noise, or fails
  hard: engineering. The zeroed-logit scenario passes the RAI test; a BLAS epsilon error does not.
  Justin's candidate line holds with one refinement: "systematically disconnected from the input"
  is the essential distinguishing property, not just magnitude-invisibility.

- **The counter-argument steelmanned and rejected.** Every numerical library has bugs; an EP is
  infrastructure far below the application layer; if wrong `MatMulNBits` is RAI, all BLAS bugs
  are too. The response: the BLAS comparison fails the test — BLAS errors are proportional, visible
  by comparison, or cause hard failures. Zeroed logits produce fluent, plausible, contextually
  meaningless output at full speed, with success signals at every layer. The autoregressive decode
  loop amplifies one wrong dispatch into an unbounded stream. The inflation objection would be
  compelling if the failure mode were similar to BLAS; it is not.

- **Morpheus agreement and disagreement explicitly recorded.**
  - Agree: claiming gated on proven correctness (§8.9.1) — this is correct and I concur fully.
  - Agree: `compute_failures: 0` may never be read as a correctness signal (§9.1.3) — correct.
  - Agree: `model_output_equivalence` ∈ {MATCH, DIVERGENT, UNMEASURED} as metric of record — correct.
  - **Disagree (scope):** Morpheus's mechanism is a *claim gate* — it prevents future claiming of
    unproven forms. It does not address *user disclosure at the session layer*. A developer who
    runs inference against a session that claims proven forms gets no runtime signal about the
    proof state of what they're running. I require a session-creation WARN when claimed forms have
    UNMEASURED status. This is an additional obligation Morpheus's current ruling does not impose.

- **R9 applied to my own verdict.** Falsifier stated: the 🔴 is discharged when criterion 11 is
  MET (proof ledger prevents any unproven dispatch) AND a runtime WARN is emitted at session
  creation for every claimed form with UNMEASURED or DIVERGENT proof status. An instrument that
  would go red on the next silent-zeroing event must exist and be verified by a positive control.

- **Pre-release framing honoured.** The 🔴 attaches to the shipping decision, not to investigation
  or fix work. Pre-M0, the obligation is to design and implement the mechanism before claiming,
  not to halt development. The hole must not be designed in; the hole has not yet been shipped.

### Learnings

1. **The two-verdict discipline matters.** Conflating "there is a bug" with "the architecture is
   unsafe" would have produced either a verdict too harsh (block Mouse's fix work) or too soft
   (treat a structural property as a one-time issue). Separate the instance from the class.

2. **The RAI/engineering line is a test, not a category.** The category is useless without the
   test. The test is: does the failure produce confident, contextually disconnected output in a
   human-facing system with no available signal? Write the test first; apply it second.

3. **A claim gate is not a disclosure mechanism.** Morpheus's proof ledger prevents future wrong
   claims; it does not tell a current user what the proof state of their session is. These are
   different obligations and require separate mechanisms.

4. **R9 applied to a verdict is the same operation as R9 applied to a claim.** Name the
   observation that would make the verdict wrong before issuing it. A finding without a falsifier
   is an opinion. The falsifier for my 🔴 is operational and stated in the audit entry.

5. **The autoregressive loop is a force multiplier on this failure class.** One wrong dispatch is
   not one wrong output — it is an unbounded wrong sequence. This is not true of BLAS errors or
   one-shot inference systems. Document it as a distinguishing property of LLM inference contexts.

### Deliverables
- Appended to `.squad/rai/audit-trail.md` (findings RAI-007, RAI-008, RAI-009)
- Created `.squad/decisions/inbox/rai-silent-inference-verdict.md`
- Appended to `.squad/agents/Rai/history.md` (this entry)

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

