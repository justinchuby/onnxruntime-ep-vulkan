# Project Context

- **Project:** onnxruntime-ep-vulkan
- **Created:** 2026-07-29

## Core Context

Agent Scribe initialized and ready for work.

## Recent Updates

📌 Team initialized on 2026-07-29

## Learnings

Initial setup complete.

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

