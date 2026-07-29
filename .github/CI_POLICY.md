# CI Policy

## Why CI is critical here

CI is the **only place in this project where a shader ever executes** (Morpheus DESIGN.md §9.1.2).
On a developer workstation, `cargo test` runs unit and integration tests in Rust but never touches
the GPU. The lavapipe software-rasterizer lanes in CI are the sole source of empirical evidence
that GPU dispatch paths are correct.

A red CI lane is therefore not a routine nuisance. It means:
- Zero confidence that any shader executes correctly.
- Any subsequent green run is potentially testing a broken baseline.
- The differential test oracle (Vulkan EP vs ORT CPU EP) is not running.

## Pre-finish checklist (run before ending a turn that touches CI config)

Before handing off any change to `.github/workflows/*.yml`:

```bash
# 1. Parse both workflow files — catches YAML syntax errors before the push.
python -c "
import yaml, sys
for f in ['.github/workflows/ci.yml', '.github/workflows/conformance.yml']:
    try:
        yaml.safe_load(open(f))
        print(f'OK: {f}')
    except yaml.YAMLError as e:
        print(f'FAIL: {f}'); print(e); sys.exit(1)
"

# 2. Spot-check any shell continuation (\) in run: blocks — a backslash
#    that outdents below the run: block's indent level breaks YAML parsing.
#    Visual inspection is sufficient; the yaml.safe_load above catches it.

# 3. After the coordinator commits and pushes: gh run list --limit 1
#    Confirm the run starts (not "This run likely failed because of a workflow file issue").
```

**Rationale:** The YAML parse step was added after three consecutive CI failures caused by
unverified CI config changes: wrong package name, package not in Ubuntu repos, and a YAML
syntax error from an outdented shell continuation. A CI change is not verified until something
parses or executes it — the parse step is that verification for syntax, and the `glslc --version`
precondition step is that verification for tool availability.

## The rule

**Before landing a commit, check `gh run list --limit 5` and confirm the badge is green.**

```
gh run list --limit 5
```

If CI is red:
1. Do not land unrelated work on top of a red lane.
2. Find the red commit and fix or revert it before continuing.
3. Report the failure in the squad decision record if the root cause is non-obvious.

## Lane structure

| Lane | Runner | Vulkan device | Purpose |
|---|---|---|---|
| `rustfmt check` | ubuntu-latest | none | Format-only; fast |
| `Build + op tests (Linux, lavapipe)` | ubuntu-22.04 | Mesa lavapipe (CPU) | **Primary correctness lane** |
| `Build + op tests (Windows, lavapipe via mesa-dist-win)` | windows-latest | Mesa lavapipe (CPU) | Windows correctness + no-ICD fallback |

Both test lanes enforce:
- `glslc --version` passes (precondition check) before the build starts
- `ONNXRUNTIME_EP_VULKAN_VALIDATE=1` (Khronos validation layers, zero errors)
- `-D warnings` in RUSTFLAGS (warning-clean Rust)
- `pytest tests/ops` with `--strict-markers` (no unknown markers silently ignored)

## Required status checks

**Not yet configured as branch-protection rules.** Adding them requires a GitHub organization-level
setting that Justin must enable once the initial green run lands. This document and the README badge
serve as the interim enforcement mechanism.

TODO(coordinator): configure `main` branch protection to require:
- `Build + op tests (Linux, lavapipe)` ✓
- `Build + op tests (Windows, lavapipe via mesa-dist-win)` ✓
- `rustfmt check` ✓

## Investigating a failure

```bash
# List recent runs
gh run list --limit 10

# Get failure log for a specific run
gh run view <run-id> --log-failed

# Find the root cause quickly
gh run view <run-id> --log-failed | grep -E "error|Error|::error"
```

For the Linux lavapipe lane specifically:
- If the build step fails with `glslc not found`: the `shaderc` package was not installed or
  changed its binary location. The CI step "Verify GLSL compiler (glslc)" should have caught
  this with a clear error before the build.
- If the Vulkan EP panics during tests: check validation layer output (`VK_INSTANCE_LAYERS` is
  set to `VK_LAYER_KHRONOS_validation`). The panic message includes the validation error.

---
*Owned by Trinity (Test & Conformance Engineer). Created 2026-07-29T02:56:28-07:00.*

## Why CI is critical here

CI is the **only place in this project where a shader ever executes** (Morpheus DESIGN.md §9.1.2).
On a developer workstation, `cargo test` runs unit and integration tests in Rust but never touches
the GPU. The lavapipe software-rasterizer lanes in CI are the sole source of empirical evidence
that GPU dispatch paths are correct.

A red CI lane is therefore not a routine nuisance. It means:
- Zero confidence that any shader executes correctly.
- Any subsequent green run is potentially testing a broken baseline.
- The differential test oracle (Vulkan EP vs ORT CPU EP) is not running.

## The rule

**Before landing a commit, check `gh run list --limit 5` and confirm the badge is green.**

```
gh run list --limit 5
```

If CI is red:
1. Do not land unrelated work on top of a red lane.
2. Find the red commit and fix or revert it before continuing.
3. Report the failure in the squad decision record if the root cause is non-obvious.

## Lane structure

| Lane | Runner | Vulkan device | Purpose |
|---|---|---|---|
| `rustfmt check` | ubuntu-latest | none | Format-only; fast |
| `Build + op tests (Linux, lavapipe)` | ubuntu-22.04 | Mesa lavapipe (CPU) | **Primary correctness lane** |
| `Build + op tests (Windows, lavapipe via mesa-dist-win)` | windows-latest | Mesa lavapipe (CPU) | Windows correctness + no-ICD fallback |

Both test lanes enforce:
- `glslc --version` passes (precondition check) before the build starts
- `ONNXRUNTIME_EP_VULKAN_VALIDATE=1` (Khronos validation layers, zero errors)
- `-D warnings` in RUSTFLAGS (warning-clean Rust)
- `pytest tests/ops` with `--strict-markers` (no unknown markers silently ignored)

## Required status checks

**Not yet configured as branch-protection rules.** Adding them requires a GitHub organization-level
setting that Justin must enable once the initial green run lands. This document and the README badge
serve as the interim enforcement mechanism.

TODO(coordinator): configure `main` branch protection to require:
- `Build + op tests (Linux, lavapipe)` ✓
- `Build + op tests (Windows, lavapipe via mesa-dist-win)` ✓
- `rustfmt check` ✓

## Investigating a failure

```bash
# List recent runs
gh run list --limit 10

# Get failure log for a specific run
gh run view <run-id> --log-failed

# Find the root cause quickly
gh run view <run-id> --log-failed | grep -E "error|Error|::error"
```

For the Linux lavapipe lane specifically:
- If the build step fails with `glslc not found`: the `shaderc` package was not installed or
  changed its binary location. The CI step "Verify GLSL compiler (glslc)" should have caught
  this with a clear error before the build.
- If the Vulkan EP panics during tests: check validation layer output (`VK_INSTANCE_LAYERS` is
  set to `VK_LAYER_KHRONOS_validation`). The panic message includes the validation error.

---
*Owned by Trinity (Test & Conformance Engineer). Created 2026-07-29T02:56:28-07:00.*
