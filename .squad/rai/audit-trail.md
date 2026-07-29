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
