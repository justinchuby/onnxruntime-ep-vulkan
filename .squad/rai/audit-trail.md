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
