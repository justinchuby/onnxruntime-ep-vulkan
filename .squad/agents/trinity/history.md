# Trinity (Test-Conformance) — history.md

## Learnings

### [SUMMARY] Rounds 1–17: CI, barriers, oracle, claim log, Phi-3.5, env var finding (2026-07-28–2026-07-30)

**Rounds 1–5 (archived):** Differential harness (EP vs CPU). Profiling-JSON claim assertion. Barrier parity layer. Two lavapipe CI lanes (Linux and Windows). ORT CPU EP quantised oracle with `accuracy_level=1`; fp16 gated on ORT ≥1.28. Windows ICD registration: `HKLM\SOFTWARE\Khronos\Vulkan\Drivers` (loader ignores `VK_ICD_FILENAMES` under elevation). PowerShell array-operator bug fixed. `glslc` via LunarG apt repo (`shaderc` not in Ubuntu's own).

**Round 11 (2026-07-29T09:19:35-07:00) — shape inference and device parametrization:**
`apply_shape_inference(model_bytes)` in all harness paths. Device parametrization across indices wired (blocked on Switch `ep.device_index`; now available as `--vulkan-devices 0,1` or `VULKAN_DEVICE_INDEX=0,1`).

**Round 12 (2026-07-29T09:39:59-07:00) — portability:**
Cross-platform portability is structural, not a review step. Test assertions derived from reported device limits, not constants.

**Round 13 (2026-07-29T10:34:41-07:00) — onnx oracle pin:**
`onnx>=1.22.0` explicit in requirements.txt + CI (not transitive). `_assert_oracle_versions()` refuses at `pytest_configure` if ORT<1.28 or onnx<1.22.0. Attention-24 tests marked `xfail(strict=True)` — onnx 1.22 has wrong reference implementation; `_ONNX_ATTENTION24_FIXED_VERSION=None` until onnx 1.23 ships.

**Round 14 (2026-07-29T15:12:13-07:00) — M0 exit criteria and env var finding:**
`is_vulkan_claimed` reverted to profiling-JSON — post-load `os.environ` changes are unreliable for the loaded DLL on Windows (UCRT cache vs `GetEnvironmentVariableW` timing). CLAIM_LOG still correct for subprocess/shell use. M0 exit criterion 5 (shader-less binary claims nothing) implemented with `epctl` check.

**Round 15 (2026-07-29T17:02:47-07:00) — Intel AV and `live` flag:**
Intel `pytest` AV crashes wrapped in `try/except Exception → return False` in `is_vulkan_claimed`. Per-op `live` flag introduced; vacuous pass on `Add-i32` vs f32-only predicate resolved.

**Round 16 (2026-07-29T20:26:56-07:00) — Phi-3.5 end-to-end:**
Phi-3.5-mini-instruct (2.2 GB) through EP: loads, runs, 65 outputs, bit-identical across sessions, variable sequence length correct.

**Round 17 (2026-07-29T21:24-07:00) — barrier parity and gpt-oss:**
Barrier parity: 46 passed / 28 skipped on both devices, both backends. Non-zero dispatch counts — first execution of legacy barrier backend. gpt-oss attempt: 8-bit nodes only (ORT CPU EP lacks INT8 oracle for this config). Oracle pin: ORT CPU EP can be used as quantised oracle with `accuracy_level=1` for fp16/int4 ops; fp32 nodes use numeric tolerance.

**Current state:**
- `pytest` green. Phi-3.5 end-to-end verified. Legacy backend verified.
- M0 open: validation positive control (criterion 3) — "no errors" = no validation layer loaded. Needs planted violation in Switch's `vk/**`.
- CI lanes stable: Windows ICD registry, Linux `glslc` via LunarG, PowerShell array bug fixed.
- Multi-device parametrization wired; active on Switch landing `ep.device_index`.
- Attention-24: `xfail(strict=True)` until onnx>=1.23 ships. `_ONNX_ATTENTION24_FIXED_VERSION` is the knob.
---

## 📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)

### Worktree layout and inbox portability constraint
The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree. `.squad/decisions/inbox/` is **gitignored** — records written in a worktree do NOT travel with the branch. The inbox in `main` is authoritative.

### Vulkan SDK path
`C:\VulkanSDK\1.4.350.0` — installed but **not on the default PATH**. `glslc` discovery must search this path; `VULKAN_SDK` env var is the canonical pointer.

### Local hardware — both GPUs pass the §7.2 gate
- Intel Iris Xe: Vulkan 1.4.309, UMA, `subgroup_size=32`, 32 KiB shared. Spec-conformance oracle. Do not special-case Intel.
- RTX 4060 Laptop: Vulkan 1.4.325, discrete, `subgroup_size=32`, 48 KiB shared.
- Lavapipe (CI): `subgroup_size=8`, 32 KiB shared, `is_uma=true`. CI exercises the mobile-warp path. LVP2 retracted.

### ORT's planner hands back interior pointers from run 2 onward
Memory-pattern planner does not engage on run 1. From run 2 onward hands back interior pointers. 52 observed on two GPUs, `pointers_in_guard_band=0`. Gate: `epctl --check-counters <file> --require-dispatches 1`.

### Execution counters file is the instrument for "did anything execute"
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` — always-on JSON. `dispatches_executed > 0` is the only reliable indicator.

### `push_next` must rebind, never discard
`let _ = props2.push_next(..)` silently discards pNext chain. Rebind, never discard. Root cause of LVP2, `subgroup_size=0`, ReBAR UMA misclassification.

### First real execution: 45 ops Live, 161 nodes claimed on Phi-3.5
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. M0 not declared — open: validation positive control, CI lanes green.

### Performance metric is a TRIPLE (Niobe — critical)
`(claimed_op_coverage, island_count, largest_island_flops)` per producer at version. Portability floor = §7.2. `SUBGROUP_SIZE_IS_GUARANTEED=False`.

---

## Round 18 (2026-07-30T09:14:00-07:00) — False-premise test audit + correctness gate

**Trigger:** Switch's runtime-extents merge moved 258 dynamic-shape declines to 0.
161 MatMulNBits nodes are now claimed, compiled, dispatched (`compute_failures: 0`)
on both Intel Iris Xe and RTX 4060.  The entire test suite was green.
Coordinator's `phi35_vk_vs_cpu.py` script revealed: `vk range [0.0000, 0.0000]` vs
`cpu range [-13.0859, 13.0312]`.  argmax 0 vs 30751.  top-10 overlap 0/10.

**Root cause:** `test_phi35_cpu_output_matches_between_sessions` compared two VulkanEP
sessions against each other.  Its docstring premise ("0 claimed nodes, both fall back to
CPU") became false.  Two all-zero GPU sessions are bit-identical — vacuous pass.
Classic R7: absence of instrument reads as negative.

**What I did (squad/trinity branch):**

1. **New correctness gate:** `test_phi35_vulkan_matches_cpu_logits`
   - One VulkanEP session vs one CPU-only session.
   - Guard A: `EP_NAME in sess.get_providers()` before comparison.
   - Guard B: `max|logit_vk| > 0.1` before token comparison.
   - Asserts: argmax match + top-10 overlap ≥ 5/10.
   - `xfail(strict=True)` — fails on current main (all-zero logits).
   - When Mouse's fix lands: XPASS → suite errors → forced xfail removal.

2. **Renamed `test_phi35_cpu_output_matches_between_sessions`** →
   `test_phi35_vulkan_session_determinism`. New premise: "same inputs, two VulkanEP
   sessions → bit-identical outputs." Valid property regardless of claimed count.
   Docstring explains the history.

3. **Renamed `test_phi35_variable_seqlen_fallback`** → `test_phi35_variable_seqlen`.
   "fallback" label removed; false "0 claimed nodes" docstring corrected.

4. **`assert_ep_in_providers(sess)`** added to `_models.py` — fast provider-list
   guard for tests that own their sessions.  Documents when to use it vs
   `assert_vulkan_claims`.

5. **Docstrings updated:** module docstring, `test_phi35_session_loads_and_declines_cleanly`,
   `test_gptoss_session_loads_and_declines_cleanly` — all false "0 claims expected" text
   removed or corrected.

**Full audit result — same defect class across the suite:**

| Test | Was false premise present? | Fix |
|---|---|---|
| `test_phi35_cpu_output_matches_between_sessions` | Yes — "0 claimed / CPU fallback" | Renamed; new gate (xfail) |
| `test_phi35_session_loads_and_declines_cleanly` docstring | Yes — "all 366 declined" | Corrected |
| `test_phi35_variable_seqlen_fallback` docstring | Yes — "0 nodes claimed" | Renamed + corrected |
| `test_gptoss_session_loads_and_declines_cleanly` docstring | Yes — "0 claims (all fp16)" | Corrected |
| `test_matmulnbits_*` | No — `assert_vulkan_claims` present | OK |
| `test_barrier_parity` | No — `live` flag guard correct | OK |
| `test_permanent_cpu_fallback_ops` | No — `assert_vulkan_does_not_claim` | OK |

**Unguarded paths that remain (by design):**
- `test_phi35_vulkan_session_determinism`: compares two VulkanEP sessions. Determinism is
  a valid property; correctness is a companion test's job.
- `test_phi35_variable_seqlen`: crash-absence test. Shape-driven "outputs differ" is
  acceptable; documented explicitly.
- `test_gptoss_session_loads_and_declines_cleanly`: no CPU oracle for gpt-oss-20b
  (QMoE issue). Deferred until Mouse confirms gpt-oss MatMulNBits numerics.

**New standing rule (supplements R7):**
Any test comparing two `EP_PROVIDERS` sessions is a DETERMINISM gate, not a CORRECTNESS
gate. A correctness gate requires one `EP_PROVIDERS` session + one `["CPUExecutionProvider"]`
session + Guard A (provider list check) before comparing.

**Mouse dependency still open:** `accuracy_level` — model declares 0, oracle is pinned
at 1. Mouse has been asked twice for confirmation.


📌 Team update (2026-07-30T05:48:29-07:00): A green suite has been shown not to imply a correct model. Phi-3.5: 161 MatMulNBits dispatched, compute_failures:0, entire suite green — vk logits all-zero (argmax 0 vs CPU argmax 30751). R9 (Morpheus): for every claim, name the instrument that would go red if the claim were false; if none, the claim is UNMEASURED. model_output_equivalence verdict required alongside all counter summaries; default UNMEASURED. Any comparison must first assert EP_NAME in session.get_providers() before calling sess.run() — failure to do so compares CPU to CPU and reports agreement. Coordinator's own first comparison reported bit-identical on both devices due to this exact error. Trinity has landed xfail(strict=True) correctness gate. M0 criterion 10 added (NOT MET: DIVERGENT). Criteria 2, 4, 5 reopened. — decided by Morpheus, Trinity, Switch, Mouse; coordinator-verified.
