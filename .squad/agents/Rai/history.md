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
