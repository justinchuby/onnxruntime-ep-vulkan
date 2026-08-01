# Trinity (Test-Conformance) — history.md

## Learnings

### [SUMMARY] Rounds 1–17: CI, barriers, oracle, claim log, Phi-3.5, env var finding (2026-07-28–2026-07-30)

**Rounds 1–5 (archived):** Differential harness (EP vs CPU). Profiling-JSON claim assertion. Barrier parity layer. Two lavapipe CI lanes (Linux and Windows). ORT CPU EP quantised oracle with `accuracy_level=1`; fp16 gated on ORT ≥1.28. Windows ICD registration: `HKLM\SOFTWARE\Khronos\Vulkan\Drivers` (loader ignores `VK_ICD_FILENAMES` under elevation). PowerShell array-operator bug fixed. `glslc` via LunarG apt repo (`shaderc` not in Ubuntu's own).

**Round 11 (2026-07-29T09:19:35-07:00) — shape inference and device parametrization:**
`apply_shape_inference(model_bytes)` in all harness paths. Device parametrization across indices wired (blocked on Switch `ep.device_index`; now available as `--vulkan-devices 0,1` or `VULKAN_DEVICE_INDEX=0,1`).

---

## Round 19 (2026-07-30T20:33:50-07:00) — paired controls, validation lane, wiring census

**Trigger:** Morpheus's ruling: criteria 4 and 5 "partially met — unchanged, and deliberately
untouched by today," on five consecutive days. Criterion 3 moved backward (messenger was
wired to stderr with no in-process listener). Criterion 12 added (wiring census).

### Criterion 4 & 5 — paired positive controls

Added to `test_claim_diagnostics.py`:
- `test_icd_present_ep_advertises_nonzero_devices` — criterion 4 positive. ICD present → 2
  devices advertised → Add claimed. PASS on both Intel Iris Xe and RTX 4060.
- `test_shaders_compiled_ep_claims` — criterion 5 positive. 46 live ops → Add claimed.

Polarity pair confirmed in same lane: negative (no ICD / shader-less) + positive (ICD +
shaders) are now siblings in the same `pytest tests/ops` invocation.

### Criterion 3 — test_validation.py

Three tests, each covering a distinct property:

**(a) Armed:** `test_validation_messenger_armed` — `epctl --probe-validation` exits 0.
VALIDATION ARMED: VK_LAYER_KHRONOS_validation installed and messenger live.

**(b) Plant in lane:** `test_ep_messenger_plant_fires_in_lane` — invokes the Rust
`#[ignore]`'d `ep_messenger_fires_for_planted_fence_leak` as subprocess. EP_VALIDATION_
ERROR_COUNT = 1 after planted fence leak. Plant is now in the pytest lane (criterion:
"a control that must be opted into is not in the lane"). PASS.

**(c) Clean after fix:** `test_validation_clean_after_binding_arity_fix` — `add_f32_
dispatches_end_to_end` under validation shows zero VUID errors on both devices after
Switch's binding-arity fix. Prior "no errors" reading was void (messenger wired to stderr
with no in-process listener). This is the fresh, valid reading.

**What Switch needs:** Remove `#[ignore]` from `ep_messenger_fires_for_planted_fence_leak`
in `rust/src/vk/dispatch_integration.rs` so the plant also runs in the cargo-test lane.
Python lane is covered; Switch's change makes it doubly covered.

### Criterion 12 — test_wiring_census.py

`test_wiring_census` emits `[WIRING CENSUS] mechanism: value` lines. Uses
`OrtEpVulkanGetExecutionCounters` (ctypes C ABI) instead of COUNTERS_FILE — avoids the
Windows UCRT env-var cache issue (post-load `os.environ` changes are invisible to the loaded
DLL; established in round 14).

Current census on this machine (commit 6f28af8):
```
partitioner: dispatches=1, subgraphs_live=1, claimed_from_profiling=1
partition_identity_check: VACUOUS (single-node graph — expected)
gpu_tracer: OPTIONAL-UNWIRED (opt-in by design)
model_output_equivalence: UNMEASURED (phi35 not in cache for this run)
retain_viable: UNWIRED -> xfail(strict=True), owner Mouse
ledger_lookup: UNWIRED -> xfail(strict=True), criterion 11 not met
validation_messenger: ARMED
layering_lint: PASS
```

**Known-unwired items:** `retain_viable` and `ledger_lookup` are `xfail(strict=True)`.
They XPASS when Mouse wires `retain_viable` into GetCapability and criterion 11 is met.

**Design note (R10 sub-rule, §10.0.1):** For model_output_equivalence UNMEASURED means
Phi-3.5 is not in the dev machine cache for this specific run; the canonical MATCH is
in `test_phi35.py`. For the census this is informational — `test_phi35.py` owns the
assertion. UNMEASURED does not fail the census.

**layering lint in census:** Runs `cargo test --test layering` (debug, not --release) to
avoid relinking the release DLL while it is loaded. CI uses the same (no `--release` flag
in `.github/workflows/ci.yml`).


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


---

## Round 20 (2026-07-31T00:26:22-07:00) -- Guard D: runtime-fallback vacuous-pass closed

Trigger: Allocator::alloc(size=0) returns None for KV-cache [1,32,0,96] inputs. ORT prints EP_FAIL, falls back to CPU silently. get_providers() still shows EP. test_phi35_vulkan_matches_cpu_logits PASSED (CPU-vs-CPU). Fourth fallback trap instance.

Fixes (commit 5b3518b):
- _models.py: count_vulkan_executions_from_profile, assert_vulkan_executed_runtime (Guard D)
- test_phi35.py: Guard D in 4 tests (matches_cpu, cross_run, f16_logits, multirun_stable)
- test_wiring_census.py: model_output_equivalence validity condition documented
- Device labels fixed (commit 00c576c): selector 0=NVIDIA, selector 1=Intel

Needed from Switch:
1. alloc.rs: zero-size alloc must return valid zero-length buffer, not None
2. Remove #[ignore] from ep_messenger_fires_for_planted_fence_leak

---

## Round 21 (2026-07-31T07:20:03-07:00) — Guard D proven to fire; harness joins the census

**Trigger:** Guard D (`assert_vulkan_executed_runtime`) had raised `NameError: name 'pathlib'
is not defined` at its first statement since the day it landed and had **never read a single
profiling event**. Coordinator fixed the call site on main (`3ea42fd`, `Path(profile_path)` —
`from pathlib import Path` was already present, so not adding a redundant import was the
right call and I would have done the same). Coordinator also merged it, saw the suite go
`8 passed` -> `5 failed`, and told the team "Guard D works". It had crashed.

**The lesson I am writing down for myself, not for them:** the red matched the prediction,
and that is exactly why it should have been read harder. A guard that raises `NameError` and
a guard that correctly detects a fallback are indistinguishable to every reader we have.

### 1. Two-polarity proof that Guard D fires — `tests/ops/test_guard_d.py`, 12 tests, 0.3 s

No GPU, no model, no EP. Synthesised ORT profiling JSON, so it is immune to the 9.5×
contention swing and always in the lane.

| Polarity | Input | Outcome |
|---|---|---|
| NEGATIVE | 0 VulkanEP `Node` events | `AssertionError`, names providers that did run |
| POSITIVE | 1 event | returns `1` |
| POSITIVE | 5 events | returns `5` |
| INSTRUMENT | missing file | `RuntimeError` |
| INSTRUMENT | corrupt JSON | `RuntimeError` |

**The part that is evidence rather than agreement (R9):** the protocol is factored into one
function and pointed at four deliberately broken guards, each of which must FAIL it —
`_mutant_nameerror` (byte-accurate reconstruction of the shipped defect), `_mutant_always_passes`,
`_mutant_inverted`, `_mutant_wrong_provider_key` (reads `args['ep']`; passes the negative
polarity for the WRONG reason, which is why both polarities are mandatory). Paired control:
the real guard must PASS the same protocol, or a protocol that rejected everything would also
score green.

**Generalisable:** the falsifier for an instrument does not need the world the instrument
observes; it needs a document the instrument reads. Synthesise the input.

### 2. Broken guard vs detected fallback — now different exception types

`AssertionError` = fallback detected, a finding about the EP, route to Switch/Mouse.
`RuntimeError` = `[Guard D instrument failure]`, a finding about the guard, route to me.
Anything else = an unanticipated harness break. Call sites in `test_phi35.py` catch the two
separately so the distinction survives to the reader of pytest output.
Recorded limit: two sites use `pytest.fail()`, which flattens the type; there the distinction
lives in the message. Weaker, and stated rather than glossed.
Companion rule: on `RuntimeError` **nothing** is written to `model_output_equivalence` — not
even `UNMEASURED`, which is a statement about a run that happened.

### 3. R11 caught before it bit — the count is fused islands, not graph nodes

One VulkanEP `Node` event = one **fused island**. Phi-3.5 runs 354 of 364 graph nodes in ONE
island, so a healthy run reports `1`, and `1 node event(s)` reads as a catastrophe.
Not renamed (Tank's `Phase::Record` precedent — compatibility outranks elegance); tagged.
`describe_vulkan_execution_count()` is the single wording, all three print sites go through
it, and the failure message states what was counted.
Presence signal, never volume: `0` conclusive, `>=1` participation, coverage from the
counters JSON. A partition change that merges islands makes this number FALL while the EP
does more work.

### 4. Harness domain added to Tank's census — schema 2, one census, two domains

`rust/tools/audit_instruments.py` extended (additively; Rust path untouched) +
`instrument_census.json` -> schema 2. Told Tank I edited his file and offered to move the
collector if he prefers — but the baseline does not split.

**Why `uninvoked` would not have caught Guard D:** it had four production callers from day
one. Tank's question ("has it got a caller?") answers WIRED and is correct. New state, third
mechanically decidable one, after `uninvoked` in his ordering:

> **unfalsified** — called, but no always-on test has watched it in BOTH polarities. Decided
> from the test AST: screened iff a non-GPU-gated test calls it inside `pytest.raises(...)`
> AND another non-gated test calls it outside one.

Ordering is now `absent -> uninvoked -> unfalsified -> unreachable -> out-of-frame -> misnamed`.

**First run, 9 harness instruments, ONE screened — the one that just cost us a day:**
- `assert_qdq_reference_oracle_safe` is genuinely **UNINVOKED**. conftest.py:341 prints a
  banner telling readers to call it; test_matmulnbits.py:121 cites it in a comment. Both are
  prose. There is no call. **My file, my defect, found by the screen on its first run.**
- `assert_matches_cpu`: 9 accept-polarity observations, **0 reject**. It is the correctness
  oracle and the suite cannot distinguish it from `def assert_matches_cpu(*a, **k): return`.
  Not a claim it is broken — a record that we would not know. First in the queue.

The screen is itself screened: `tests/ops/test_harness_census.py`, 7 always-on tests, two
synthetic trees differing only in the presence of a `pytest.raises` block must produce
DIFFERENT verdicts. Includes the case that matters: a paired self-test behind `require_vulkan`
does NOT earn `screened`, because that was exactly Guard D's situation.

### 5. Full-suite baseline, both devices

Binary verified fresh against `main@2ebd786` first — the first attempt ran against a DLL that
predated the merge and I threw those counts away rather than quote them.

| | Device 0 — RTX 4060 | Device 1 — Intel Iris Xe |
|---|---|---|
| failed | 38 | 34 |
| passed | 267 | 271 |
| skipped | 30 | 30 |
| xfailed | 10 | 10 |
| wall | 976.05 s | 976.62 s |

Niobe's guard read `[RED] CONTENDED` before and after both runs (3.82 then 1.40 foreign cores
mean; occupancy probe VACUOUS — no quiet reference for this host, and untested is not passed).
976 s against a ~5 min quiet reference is **3.2× slow**, so **no timing figure from either run
is quotable**, and `test_wiring_census` (dev0) died of `subprocess.TimeoutExpired`, which I am
recording as **UNATTRIBUTABLE** rather than as a defect or a pass.

**Attribution:** `test_op_table.py` and `test_elementwise.py` are byte-identical to
`origin/main`, there is no `rust/src/` change, and every `_models.py` hunk is at line 844+
inside the Guard D block. `assert_vulkan_claims` (line 564), which raises 31 of the failures,
is untouched. This is main's state measured, not a regression from this branch.

**31 failures, one cause:** §8.9 evidence gating declines `Min`/`Max`/`Cast`/comparisons/
bitwise as `[staged]` — "compiles but has never executed on a device, so claiming it would be
a bet" — against a suite that still asserts they are claimed. Policy/expectation mismatch,
decision not patch. **I am not loosening 31 assertions to make a suite green.**
Open question for Mouse: the same run logs two sessions, the first `claimed 1/1 nodes -> 1
island`, the second `claimed 0/1`. Two sessions in one process disagreeing about one node
form is a ledger effect or an R12 frame problem. Answer before anyone edits tests.

**Device delta is 4 and it is a discriminator:** dev0 alone fails `session_determinism`,
`f16_matmulnbits_logits_nonzero`, `multirun_logits_stable` (plus the timeout). The discrete
GPU fails three determinism tests the integrated GPU passes; Intel remains the oracle and it
is the one that agrees with itself. `interior_pointer_safety` fails on BOTH (6e-08 vs 0.0,
fp16, run 2) — same defect, two vendors, so it is ours.

**Guard D in production:** it executed and PASSED inside `test_phi35_vulkan_cross_run_consistency`
on device 0, reporting `3 fused-island executions`. Not 0, not 1 — **its output varies with
its input on real hardware.** That is the R10 falsifier for "Guard D is wired" that no code
reading can supply.

### Owed to others
- **Mouse:** the two-session claim disagreement (above). Also still open from Round 18:
  `accuracy_level` — model declares 0, oracle pinned at 1, asked three times now.
- **Tank:** I edited `rust/tools/audit_instruments.py`. Additive only. Say the word and I
  move the collector; the baseline does not split.
- **Switch:** zero-size alloc must return a valid zero-length buffer, not `None`; and the
  `#[ignore]` on `ep_messenger_fires_for_planted_fence_leak` (carried from Round 20).
- **Me, next:** falsify `assert_matches_cpu` first, then wire `assert_qdq_reference_oracle_safe`.
