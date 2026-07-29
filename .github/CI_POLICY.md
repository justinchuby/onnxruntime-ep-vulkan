# CI Policy

**Owned by Trinity (Test & Conformance Engineer). Updated 2026-07-29T09:39:59-07:00.**

---

## Why CI is non-negotiable

CI is the **only place in this project that proves portability** (Morpheus §9.1.2).
On a developer workstation, local hardware is Windows + discrete NVIDIA + Intel integrated.
A test that passes locally but silently relies on NVIDIA behavior is not a test — it is a
coincidence. **The lavapipe lanes prove cross-platform correctness; local hardware is a
development loop.**

A red CI lane means:
- Zero confidence that any shader executes portably.
- The differential oracle (Vulkan EP vs ORT CPU EP) is not running.
- Any subsequent green run is testing an unverified baseline.

---

## Portability rules (standing directive, 2026-07-29T09:39:59-07:00)

These rules are **structural**, not a checklist item. They apply to every test Trinity writes
and every tolerance Trinity sets, continuously.

### 1. Intel is the spec-conformance oracle

Intel Iris Xe (Vulkan 1.4.309, UMA) on Justin's dev machine is the strictest Vulkan
implementation available locally. **If a test passes on NVIDIA and fails on Intel, we
relied on something unspecified — the test or the EP code is wrong, not Intel.**

Consequence for the harness:
- An Intel-only failure MUST be investigated as a portability bug, not worked around.
- Never add a vendor-conditional tolerance or a vendor-conditional skip.
- An `@pytest.mark.portability` failure on Intel is evidence for Switch, not a harness bug.

### 2. No vendor special-casing

A fix that takes the form "skip this on Intel" or "widen tolerance for NVIDIA" is forbidden.
If the EP produces different numerical results on different vendors, either:
  a. The tolerance is legitimately different (document it with a measurement, not a guess), or
  b. The shader relies on undefined behavior (fix it in the shader).

The one permitted exception: driver bugs that are filed, tracked, and have a timeline.
These must be marked with `pytest.mark.xfail(reason="vendor bug: <URL>", strict=True)` so
they become visible fails when the driver is fixed.

### 3. UMA memory model

Iris Xe (and Adreno, Mali) expose memory types that are both `DEVICE_LOCAL` and `HOST_VISIBLE`.
A staging path that assumes a separate upload heap never runs on these devices. **Iris Xe on
Justin's desk is the closest available proxy for the Adreno/Mali memory model.**

When `--vulkan-devices 0` is Iris Xe and device 1 is NVIDIA, running with `--vulkan-devices 0,1`
exercises both memory models simultaneously. Any test that passes on device 1 and fails on
device 0 is a UMA portability bug.

### 4. Local results are development loops only

A number obtained only on Justin's desk is not a project result (Morpheus §9.1.2). Quote from
coordinator directive:
  *"Local hardware is a development loop; CI proves portability; physical Android and macOS
   coverage remains absent (OQ-12)."*

When reporting a measured result (coverage delta, tolerance, timing), always state the source:
  GOOD: "15/15 cases resolve after shape inference — local measurement, CI pending"
  BAD:  "15/15 cases resolve after shape inference"

### 5. Portability failures are routed, not silenced

When a test fails on one device but not another:
- Do NOT change the tolerance.
- Do NOT add a vendor skip.
- DO record the failure in `.squad/decisions/inbox/trinity-portability-<slug>.md`.
- DO route to Switch (shader issue) or Mouse (claim-predicate issue).

---

## Pre-finish checklist (run before ending a turn that touches CI config)

Before handing off any change to `.github/workflows/*.yml`:

```bash
# 1. Parse both workflow files (catches YAML syntax errors before the push).
python -c "
import yaml, sys
for f in ['.github/workflows/ci.yml', '.github/workflows/conformance.yml']:
    try:
        yaml.safe_load(open(f))
        print(f'OK: {f}')
    except yaml.YAMLError as e:
        print(f'FAIL: {f}'); print(e); sys.exit(1)
"

# 2. Spot-check shell continuations (\) in run: blocks.
#    A backslash that outdents below the run: block's indent level breaks YAML parsing.
#    yaml.safe_load above catches it; visual inspection confirms.

# 3. After coordinator commits and pushes:
#    gh run list --limit 1
#    Confirm the run starts (not "This run likely failed because of a workflow file issue").
```

**Rationale:** This step was added after three consecutive CI failures from unverified config
changes: wrong package name, package not in Ubuntu repos, and a YAML syntax error from an
outdented shell continuation. A CI change is not verified until something parses or executes it.

---

## The rule

**Before landing a commit, check `gh run list --limit 5` and confirm the badge is green.**

If CI is red:
1. Do not land unrelated work on top of a red lane.
2. Find the root cause and fix or revert before continuing.
3. Record the failure in the squad decision record if the root cause is non-obvious.

---

## Lane structure

| Lane | Runner | Vulkan device | Purpose |
|---|---|---|---|
| `rustfmt check` | ubuntu-latest | none | Format-only; fast |
| `Build + op tests (Linux, lavapipe)` | ubuntu-22.04 | Mesa llvmpipe (CPU, Vulkan 1.3.255) | **Primary portability lane** |
| `Build + op tests (Windows, lavapipe)` | windows-latest | Mesa llvmpipe via mesa-dist-win | Windows portability + no-ICD fallback |

*Local extended device matrix (not CI — development loop only):*

| Device | Vulkan | Memory | Notes |
|---|---|---|---|
| Intel Iris Xe Graphics | 1.4.309 | UMA (DEVICE_LOCAL + HOST_VISIBLE) | Strictest spec conformance; proxy for Adreno/Mali |
| NVIDIA GeForce RTX 4060 Laptop | 1.4.325 | Discrete (separate upload heap) | High throughput; permissive on UB |

Both CI test lanes enforce:
- `epctl --probe-loader` exits 0 before pytest starts (hard failure if no capable device)
- `glslc --version` passes before the build (precondition check)
- `ONNXRUNTIME_EP_VULKAN_VALIDATE=1` (Khronos validation layers, zero errors)
- `-D warnings` in RUSTFLAGS (warning-clean Rust)
- `pytest tests/ops` with `--strict-markers`

---

## Required status checks

**Not yet configured as branch-protection rules.** Requires GitHub organization-level setting.

TODO(coordinator): configure `main` branch protection to require:
- `Build + op tests (Linux, lavapipe)` ✓
- `Build + op tests (Windows, lavapipe via mesa-dist-win)` ✓
- `rustfmt check` ✓

---

## Investigating a failure

```bash
gh run list --limit 10
gh run view <run-id> --log-failed
gh run view <run-id> --log-failed | Select-String -Pattern "error|Error"  # PowerShell
```

**Linux lavapipe specific:**
- `glslc not found` → `shaderc` package not installed or changed binary location. The "Verify
  GLSL compiler" step should have caught this; check why it did not.
- EP panics during tests → check VK_LAYER_KHRONOS_validation output. The panic message includes
  the validation error.
- `epctl --probe-loader` exits non-zero → lavapipe not enumerated. Check Mesa installation and
  VK_ICD_FILENAMES.

**Windows lavapipe specific:**
- `llvmpipe` not in vulkaninfo output → registry ICD not found; check
  `HKLM:\SOFTWARE\Khronos\Vulkan\Drivers` and mesa3d\x64\ DLL path.
- `Select-String -Quiet` used for boolean checks (not `-match`/`-notmatch`, which are array
  filters on multi-line output).
