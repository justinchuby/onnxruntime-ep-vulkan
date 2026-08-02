# Squad Team

> onnxruntime-ep-vulkan

## Coordinator

| Name | Role | Notes |
|------|------|-------|
| Squad | Coordinator | Routes work, enforces handoffs and reviewer gates. |

## Members

| Name | Role | Charter | Status |
|------|------|---------|--------|
| Morpheus | Lead / EP Architect | .squad/agents/morpheus/charter.md | 🏗️ Active |
| Tank | Runtime & FFI Engineer | .squad/agents/tank/charter.md | 🔧 Active |
| Switch | Vulkan Compute Engineer | .squad/agents/switch/charter.md | ⚡ Active |
| Mouse | Op Coverage Engineer | .squad/agents/mouse/charter.md | 🧩 Active |
| Trinity | Test & Conformance Engineer | .squad/agents/trinity/charter.md | 🧪 Active |
| Niobe | Performance Engineer | .squad/agents/niobe/charter.md | 📊 Active |
| Link | Platform & Hardware Support Engineer | .squad/agents/link/charter.md | 🌐 Active |
| Scribe | Session Logger | .squad/agents/scribe/charter.md | 📋 Built-in |
| Ralph | Work Monitor | .squad/agents/ralph/charter.md | 🔄 Built-in |
| Rai | RAI Reviewer | .squad/agents/Rai/charter.md | 🛡️ RAI |
| Fact Checker | Fact Checker | .squad/agents/fact-checker/charter.md | 🔍 Verifier |

## Coding Agent

<!-- copilot-auto-assign: false -->

| Name | Role | Charter | Status |
|------|------|---------|--------|
| @copilot | Coding Agent | — | 🤖 Coding Agent |

### Capabilities

**🟢 Good fit — auto-route when enabled:**
- Bug fixes with clear reproduction steps
- Test coverage (adding missing tests, fixing flaky tests)
- Lint/format fixes and code style cleanup
- Dependency updates and version bumps
- Small isolated features with clear specs
- Boilerplate/scaffolding generation
- Documentation fixes and README updates

**🟡 Needs review — route to @copilot but flag for squad member PR review:**
- Medium features with clear specs and acceptance criteria
- Refactoring with existing test coverage
- API endpoint additions following established patterns
- Migration scripts with well-defined schemas

**🔴 Not suitable — route to squad member instead:**
- Architecture decisions and system design
- Multi-system integration requiring coordination
- Ambiguous requirements needing clarification
- Security-critical changes (auth, encryption, access control)
- Performance-critical paths requiring benchmarking
- Changes requiring cross-team discussion

## File Ownership Notes

- **`rust/src/trace.rs`** — assigned to **Niobe** (2026-08-02T02:03:46-07:00, coordinator decision,
  recorded by Scribe). Reason: it holds the project's only sanctioned tick-to-nanosecond conversion
  and timestamp calibration arithmetic — measurement, not counters/FFI plumbing — and Niobe already
  owns the instruments that consume it. The file had no roster owner (flagged by Link,
  `link-conversion-call-sites-static-screen`, 2026-08-01). Tank may have the stronger claim on
  counters/FFI grounds and has been notified in his history.md; reassignment is a one-line change to
  this note and to `ci/tick_conversion_allowlist.json` if he objects.

## Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — cross-platform Vulkan plugin execution provider for ONNX Runtime, in Rust
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` (mirror its `rust/`, `tests/conformance/`, `bench/`, `python/`, `docs/` layout)
- **Stack:** Rust cdylib plugin EP · Vulkan 1.1+ compute · SPIR-V/GLSL · ONNX Runtime C API · Python bindings · GitHub Actions
- **Target platforms:** Windows, Linux, Android, macOS (MoltenVK); NVIDIA / AMD / Intel / Adreno / Mali; lavapipe & SwiftShader for GPU-less CI
- **Created:** 2026-07-29
- **Cast universe:** assigned 2026-07-28
