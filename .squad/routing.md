# Work Routing

How to decide who handles what.

## Routing Table

| Work Type | Route To | Examples |
|-----------|----------|----------|
| Architecture, scope, design docs | Morpheus | Crate layout, `docs/DESIGN.md`, EP design, opset scope, milestone calls |
| ORT plugin EP ABI, FFI, build | Tank | `sys.rs`/`ep.rs`/`factory.rs`, C ABI vtables, `build.rs`, cdylib export, EP registration |
| Vulkan device, memory, shaders | Switch | `engine.rs`, SPIR-V/GLSL kernels, descriptor sets, barriers, command buffers, allocators |
| ONNX operators, registry, partitioning | Mouse | `ops/*`, `registry.rs`, capability reporting, shape/dtype validation, CPU fallback rules |
| Correctness testing, conformance | Trinity | Differential tests vs ORT CPU EP, tolerances, `tests/conformance/`, regression suites |
| Benchmarks, profiling, perf regressions | Niobe | `bench/`, timestamp queries, roofline analysis, speedup claims, perf CI |
| Platform/hardware support, toolchains, CI matrix | Link | Capability detection, driver quirks, MoltenVK/Android/NDK, Vulkan SDK, runner matrix, packaging |
| Code review | Morpheus | Review PRs, enforce layering, quality gates |
| Scope & priorities | Morpheus | What to build next, trade-offs, decisions |
| Session logging | Scribe | Automatic — never needs routing |
| RAI review | Rai | Content safety, credential detection, ethical review |
| Claim verification / devil's advocate | Fact Checker | Verify API/extension/version claims, challenge design assumptions |

## Domain Overlaps — Tie-Breakers

| Situation | Primary | Why |
|-----------|---------|-----|
| A new op needs a new shader | Mouse specs the semantics, Switch writes the shader | Semantics vs GPU implementation |
| A kernel is slow | Niobe diagnoses, Switch or Mouse changes the code | Measure ≠ optimize |
| An op fails on one vendor only | Link triages the driver quirk, then hands the fix to Switch or Mouse | Quirk ownership |
| Test fails only in CI | Trinity owns the test, Link owns the runner/environment | Test vs platform |
| Should we require a Vulkan extension? | Link proposes, Morpheus decides | Baseline is an architecture call |

## Issue Routing

| Label | Action | Who |
|-------|--------|-----|
| `squad` | Triage: analyze issue, assign `squad:{member}` label | Lead |
| `squad:{name}` | Pick up issue and complete the work | Named member |

### How Issue Assignment Works

1. When a GitHub issue gets the `squad` label, the **Lead** triages it — analyzing content, assigning the right `squad:{member}` label, and commenting with triage notes.
2. When a `squad:{member}` label is applied, that member picks up the issue in their next session.
3. Members can reassign by removing their label and adding another member's label.
4. The `squad` label is the "inbox" — untriaged issues waiting for Lead review.

## Rules

1. **Eager by default** — spawn all agents who could usefully start work, including anticipatory downstream work.
2. **Scribe always runs** after substantial work, always as `mode: "background"`. Never blocks.
3. **Quick facts → coordinator answers directly.** Don't spawn an agent for "what port does the server run on?"
4. **When two agents could handle it**, pick the one whose domain is the primary concern.
5. **"Team, ..." → fan-out.** Spawn all relevant agents in parallel as `mode: "background"`.
6. **Anticipate downstream work.** If a feature is being built, spawn the tester to write test cases from requirements simultaneously.
7. **Issue-labeled work** — when a `squad:{member}` label is applied to an issue, route to that member. The Lead handles all `squad` (base label) triage.
