# Tank (Runtime-FFI) — history.md

## Learnings

### [SUMMARY] Sessions 1–13b: crate foundation, ORT bindings, logging crash, allocator, execution verification (2026-07-28–2026-07-30)

**Sessions 1–7 (archived):** Crate structure (`ort-ep-vulkan`, cdylib). ORT C API bindings via bindgen. Three-number version negotiation: EXPECTED 28 / MIN 24 / negotiated 28. `logging::forward_to_ort` null-pointer crash fixed (ORT annotates `file_path` as `_In_z_` and dereferences unconditionally). `tests/cdylib_load.rs` dlopens shipped cdylib and resolves exports. `tests/portability.rs` added after `ort::wchar_t` broke Linux lane. `cargo ci` command added with edition preflight.

**Session 9 (2026-07-29T10:50:02-07:00) — Compile/Compute seam:**
`Compile`→`OrtNodeComputeInfo`→`dispatch_ort` wire complete. Inputs/outputs from fused node, not subgraph body. `ep.rs` imports via `engine.rs` re-export (layering rule 4.3). `Compute` must return real `OrtStatus` on failure — null means success to ORT.

**Session 10 (2026-07-29T20:26:56-07:00) — CI and counters:**
`cargo ci` edition preflight: refuses rustfmt that doesn't know the crate's edition. Execution counters (`rust/src/counters.rs`): six relaxed atomics, always on, `ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` env var, written on first dispatch and at teardown. `epctl --check-counters`: exit 0/1/3 (1≠3 distinction is load-bearing for CI). `glslc` discovery now searches `C:\VulkanSDK\<version>\Bin\` as fallback.

**Session 11 (2026-07-29) — allocator and lavapipe crash:**
Real allocator: 64 GiB VA reservation per device (`MEM_RESERVE`/`PROT_NONE`), `BTreeMap<usize, Span>`. Lavapipe crash diagnosed: OOB storage buffer access = real host fault; `robustBufferAccess` not enabled. Lavapipe `subgroup_size=8`, `maxComputeSharedMemorySize=32 KiB`.

**Session 12 (2026-07-29) — device memory and probe failure:**
Device memory behind `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` (default off). `transfer.rs` (724 lines): `CanCopy`, `CopyTensors`, `Release`. `EpDevice_AddAllocatorInfo`: do NOT release `MemoryInfo` after success (bounded intentional leak; ORT retains the pointer). `probe_allocator.py` was a false-green machine — must check counters file for `dispatches_executed > 0`.

**Session 13 (2026-07-30) — interior pointer verification:**
ORT's planner does NOT engage on run 1 (records the pattern, hands back sub-ranges from run 2 onward). 52 interior pointers observed across 5 runs, identical on Intel and NVIDIA, all within span, `pointers_in_guard_band=0`. Every earlier "0 interior" probe was pointed at the wrong moment — instrument failure, not negative result. `allocator::ledger` classifies every pointer by `LookupError` taxonomy.

**Session 13b — positive controls and honest scope:**
Quarantine detector positive-control present (`the_quarantine_detector_fires_when_a_stale_handle_is_presented`). `pointers_use_after_free=0` in real sessions is worth nothing alone — detector proven able to fire, not exercised by ORT. `probe_planner.py` runs session in child process. Phi-3.5 still claims 0 nodes as of this session (blocked on Switch's runtime extents — now landed). CI contract: `pointers_in_guard_band > 0` is a hard assertion.

**Current state:**
- `cargo ci` — green, 300 tests.
- Allocator ready, interior-pointer observation complete.
- Device memory blocked on at least one claimed node (now unblocked — Switch's extents landed 161 claims).
- D-T51: quarantine detector not yet exercised by a real ORT allocation pattern.
- Standing: `ort::wchar_t` Windows-only bindgen type; `tests/portability.rs` guards Linux lane.
---


<!-- SUMMARIZED by Scribe 2026-08-01T20:39:12-07:00 -- older entries condensed below; full text lives in git history -->

### [SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)

- **📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)** — ### Worktree layout and inbox portability constraint The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree.
- **Session 10 — 2026-07-29T20:26:56-07:00 — CI has never executed a claimed node** — **The task was "make CI prove it", and the first thing I found was that CI cannot currently prove anything: both lanes crash.** Run `30510593046`, eight consecutive failures.
- **Session 11 — 2026-07-29 — the allocator, and a crash that was mine** — **What I built.** `src/allocator.rs` — a real `OrtAllocator` over a per-device reserved virtual-address arena.
- **Session 12 — 2026-07-29 — device memory becomes real, and a probe that lied to me** — **Worktree.** Moved to `C:\Users\justinchu\dev\ep-vulkan-tank` on `squad/tank`.
- **Session 13 — 2026-07-30 — the verification I could not get was an instrument problem** — **Worktree** `C:\Users\justinchu\dev\ep-vulkan-tank`, branch `squad/tank`, rebased on `main`.
- **Session 13b — 2026-07-30 — every mechanism casts a shadow** — The coordinator ruled that decision records must be written into the integration tree's inbox, because `.squad/decisions/inbox/` is gitignored and a record written in a worktree is invisible to everyone.
- **2026-08-01 — Tank — the broken-commitment WARN, through ORT's own sink, with a control that bites** — Ruling 2 specified the mechanism; my job was to build it and then to make it *falsifiable*.
- **2026-08-01 addendum — the load was misattributed, and my evidence is unaffected by the correction** — The coordinator withdrew his attribution of the machine load: it is a second development project of Justin's running CPU **and GPU** tests, not squad orchestration.
- **STOP POINT 2026-08-01T11:39 — read this first if you are resuming as Tank with no memory** — **Everything is committed.** Worktree `C:\Users\justinchu\dev\ep-vulkan-tank`, branch `squad/tank`, commit `bce87cd`, on top of `main` at `17c2fab`.

📌 Team update (2026-08-02T02:03:46-07:00): ust/src/trace.rs had no roster owner (flagged by Link). The coordinator assigned it to Niobe — timestamp calibration and trace-event arithmetic are measurement, and she already owns the instruments that consume it. Recorded in 	eam.md's new File Ownership Notes section. Tank may have the stronger claim on counters/FFI grounds and may object; reassignment is a one-line change if so. — decided by Scribe

---

## 2026-08-02 — Tank — the session-creation disclosure (§8.9.7 / RAI-009), with the arm that matters

**Worktree** `C:\Users\justinchu\dev\ep-vulkan-tank`, branch `squad/tank`, merged on `main`
(`ca283a9` and later). Rebuilt before verifying — Mouse's R12 fourth generalisation says the frame
for a test result is *the binary that ran it*, and `Copy-Item` preserving `LastWriteTime` is enough
to make cargo re-run a stale one.

**What the requirement was.** A user creating a session against a build that would claim ops whose
correctness is `UNMEASURED` or known `DIVERGENT` must be told **at session creation**, not left to
discover it from a wrong answer later. RAI-008's second discharge condition.

### What I built

- `rust/src/disclosure.rs` (new, mine). `FormEvidence` = `Proven` / `Unmeasured` / `Divergent` /
  `LedgerFaulted`; `evidence_for(key)`; `disclose_claimed_forms(forms)`; `disclose_zero_claims(...)`.
  Four states, not three: `LedgerFaulted` is an **instrument** state (R13) and is not a finding
  about the form, and `Unmeasured` vs `Divergent` is the distinction RAI-008's falsifier names.
- `rust/src/logging.rs` — `info_through_ort_sink`, sibling of `warn_through_ort_sink`. The INFO and
  the WARN are a *pair* and must travel down the same channel, or a user who sees the WARN looks
  for its context in a log we never wrote to.
- `rust/src/registry.rs` (Mouse's, two additive changes, decision record filed):
  - `Ledger::demoted` + `demotion_for(key)`. A non-`MATCH` verdict already faults the ledger; now
    it is also **remembered**. Without this, "the evidence measured this and it was wrong" and
    "nothing has ever measured this" both arrive at the disclosure layer as "no proof".
  - `claim_decision_audited(view)` returning the whole `ClaimAudit`; `claim_decision` delegates.
    `ep.rs` needs each claimed node's proof key, and calling `claim_audit` a second time would
    **double-count `proven_key_lookups`** — which is criterion 11's evidence. One audit, one
    lookup, two readers.
- `rust/src/counters.rs` — `session_disclosures`, `claimed_forms_{proven,unmeasured,divergent,
  ledger_faulted}`, `session_disclosure_warns[_to_ort_sink]`, and two tokens:
  `claimed_form_evidence` (`UNOBSERVABLE` / `NO-CLAIMS` / `ALL-PROVEN` / `UNMEASURED-PRESENT` /
  `DIVERGENT-PRESENT` / `LEDGER-FAULTED`) and `session_disclosure_channel` (`UNOBSERVABLE` /
  `ORT_SINK` / `PRIVATE_LOG_ONLY`). The C ABI struct is unchanged; JSON only.
- `rust/src/ep.rs::GetCapability` — collects distinct claimed forms during the existing claim loop
  and discloses once, before any fusion decision. The zero-claim branch gets the aggregate INFO.

### Is the new observable in-frame at the moment it must be read?

Yes, and deliberately. `record_session_disclosure` **writes the counters artifact at the instant of
the disclosure**, the same argument as `record_broken_commitment`. The moment this observable must
be read is session creation; a session that claims unproven forms is by construction one that may
end abnormally, and an observable readable only at a shutdown that no longer occurs is out-of-frame
by construction — the fifth state in my own vocabulary, and the exact hazard I created for myself
when I leaked the production device to close §6.5.

And `claimed_form_evidence` reads **`UNOBSERVABLE`, never `ALL-PROVEN`**, when no disclosure ran.
A claim set that was never assembled is not a claim set that came back clean. That substitution is
precisely the §6.5 coincidence: agreement produced by the absence of the event rather than by its
outcome.

### The control — both arms, in the lane, PLANTED

**In-lane Rust (`ep::tests::session_disclosure`, not `#[ignore]`)**, driving the real
`disclose_claimed_forms` through a fake `OrtApi` whose only live slot is `Logger_LogMessage`:
- red arm: a planted `UNMEASURED` form → exactly one WARN at `ORT_LOGGING_LEVEL_WARNING` on ORT's
  sink, naming the form, the kernel, `UNMEASURED`, and the env var that caused it;
- green arm: an all-proven claim set → **no** WARN, with `d.proven >= 1` asserted **first** so the
  silence is not the silence of a session that claimed nothing;
- mixed arm: the WARN names the unproven form and **not** the proven one.

The proven key is read out of the shipped ledger, never pasted. A hardcoded key that drifts turns
the *green* arm green for the wrong reason.

**Mutation-tested — all three bite:**

| mutation | caught by |
| --- | --- |
| A — suppress the WARN entirely | 5 tests, incl. both ORT-sink red arms |
| B — warn unconditionally | **only** the two green arms (2 tests) |
| C — count `UNMEASURED` as proven | 3 tests |

Mutation B is the one that matters: nothing except the good-run polarity notices a WARN that always
fires. That is the whole of Rai's point, demonstrated rather than asserted.

**Out-of-process (`rust/tools/probe_session_disclosure.py`), PASS on device 0 and device 1**,
artifact `bench/results/session-disclosure-control.json`:
- arm A: `mul_f16_unproven` (Mouse's planted case) enabled through the escape hatch. The proof key
  is **learned from the claim log in a first pass**, never hardcoded — otherwise a drifted key
  makes the plant silently inert. Result both devices: `claimed_form_evidence=UNMEASURED-PRESENT`,
  `session_disclosure_channel=ORT_SINK`, WARN present on ORT's sink naming the key.
- arm B: `add_f32` (in the ledger), no hatch. Both devices: `claimed_forms_proven=1`,
  `claimed_nodes=1`, `claimed_form_evidence=ALL-PROVEN`, **zero** WARNs on ORT's sink. Non-vacuity
  is checked before the silence, and a FAIL is raised if the silence is vacuous.
- `decode_both` is **imported** from `probe_broken_commitment.py`, not re-implemented. ORT's sink
  is UTF-16LE on Windows and our narrow line shares the handle; a second decoder is a second thing
  that can drift, and the defect it reintroduces is a delivered WARN reported as absent.
- R13 blindness control: if neither arm sees anything on ORT's sink the probe returns
  **ERROR(instrument)**, exit 4, never FAIL. Verified by disarming arm A's plant: the probe
  returned ERROR, not PASS.

**Evidence class: `PLANTED`**, recorded as such in the artifact itself. Both arms' `UNMEASURED`
condition is one I constructed; it is reachable only by naming a key in
`ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN`, and no production build of this repo has ever produced one.
What is demonstrated is the *warning path*, not the frequency of the fault. The `DIVERGENT` arm is
more planted still — it needs a ledger line no generator writes — and is covered by a unit test on
`parse_ledger` (`a_divergent_ledger_line_is_remembered_as_a_demotion`) rather than end to end.

### Verification

`cargo build --lib` and `--release` rebuilt from the merged state. `cargo test --lib`:
**469 passed / 0 failed** (was 459 before this work; +10). `cargo clippy --all-targets`: clean, no
warnings at all. `cargo fmt --check`: my files clean; the pre-existing drift in `partition.rs`,
`allocator.rs`, `epctl.rs` and two spots in `counters.rs` is not mine and I left it alone.

### No inadmissible measurement

Nothing here has a time term. Every figure above is a count or a token from a log or a JSON
artifact, so `machine_quiescence: CONTENDED` does not bear on any of it.

### Next step if I am a fresh session

1. `offer_shared_device` — with Switch, by M2 entry. Morpheus ruled it intended and opt-in (it has
   a production caller in `vk/session.rs`), but its **recorded reason has expired**: the source says
   the transfer cannot be written until the handle→VkBuffer seam is filled, and that seam is filled.
   He explicitly did not rule that the flag should flip. Find the live reason for `OFF` (host memory
   wearing a device handle: risk, no measured benefit) or re-decide it. *A default defended by a
   reason its own documentation does not give is a default nobody has re-decided.*
2. RAI-011, handed to Mouse (below) — check he has what he needs.
3. `transfer.rs::device_buffer_for` is still uninvoked; `alloc_device_buffer_binds` still 0. That
   is Switch's half to bind. Coordinate, do not build it.

### `union_check.py --run` — RED, and it is not mine

`python tests/union_check.py --run` returns `FAIL(condition=union_red)`: **154 failed / 276 passed
/ 30 skipped / 9 xfailed** in one process (54 min). The dominant failure text is

> `VulkanExecutionProvider did not execute any node of this model — the CPU-match check would be a
> vacuous pass. Providers seen: ['CPUExecutionProvider'].`

with the EP's own decline reason immediately above it:

> `[unproven] no proof ledger entry for com.microsoft::SkipSimplifiedLayerNormalization/1+/...`

**That is the ledger gate working.** The `ops` lane exercises forms the 9-entry
`evidence/proof_ledger.jsonl` does not prove, nothing in `tests/` arms
`ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN`, so the EP correctly declines and the op tests correctly
refuse a vacuous CPU-only pass.

**Attributed, not assumed.** I did not stop at the nearest available cause. I stashed
`rust/src` + `rust/tools`, rebuilt release from the merge base, and ran
`tests/ops/test_skipnorm.py tests/ops/test_shape_inference_delta.py`:

| build | result |
| --- | --- |
| merge base, my changes stashed | 19 failed / 18 passed |
| same commit, my changes applied | 19 failed / 18 passed |

Identical. My work does not move this number in either direction. The remedy is either regenerating
the ledger over the op-lane forms (`rust/tools/gen_proof_ledger.py`) or arming the hatch in
`tests/ops/conftest.py` — **Mouse's gate and Trinity's harness, not mine**, and I have not touched
either. Flagging it rather than fixing it in someone else's file.

**Rebuilt before every verdict above.** Cargo's fingerprint does not notice a restore that
preserves `LastWriteTime`, so each of the three states above was compiled, not assumed.

---

## Session 22 — 2026-08-02 — the census frame is declared, not implied (`bench/` IN FRAME)

**Input:** Niobe found that `audit_instruments.py` scanned `rust/src` and `tests/` and never
`bench/`, while printing `CENSUS VERDICT: PASS`. My own `misnamed` state, applied to the census.

**Ruling: the defect was the silence, not the scope.**

- `bench/` is IN FRAME as a third domain (`BENCH_INSTRUMENT_FILES`, 10 modules, 90 fns).
- `FRAME_DIRS`: every top-level source directory has an IN/OUT decision **with a reason**.
  `ci/` is the deliberate OUT — its `check_*` files are invoked by name from workflow YAML and
  are censused by `ci/lane_inventory.py`; two censuses over one tree is the failure to avoid.
- `BENCH_HELD_OUT`: every `bench/*.py` is screened or held out with a reason.
- Undeclared directory or bench module => `FAIL(drift)`. Falsifier both ways in
  `bench/results/census_frame_falsifier.txt`; `undeclared()` is pure with a 4-case self-test
  that runs before `--check`.
- `frame_report()` prints scanned vs held-out on EVERY path. A disclosure on one path only is
  one the diagnosing reader never sees.

**The finding inside the fix.** First cut reused the harness name vocabulary. `bench/phases.py`:
**37 top-level functions, 0 selected** — and that file holds `gpu_steady_tail`,
`decomposition_identity`, `trace_matches_counters`, `valid_bits_applied`, `red_flags`. A lexical
extension would have printed "bench/ scanned" over a module it saw nothing in. Selection is now
structural (module-public top-level defs), the same rule the Rust screen uses for `pub fn`.

**85 `unfalsified` is a property of the screen, not a verdict on Niobe's tests.** Polarity comes
from `pytest.raises`; most bench instruments are total functions returning a token. Said out loud
in the report and in `rulings`. Did not invent a value-polarity heuristic — crediting a polarity
the screen did not observe is Guard D with the sign flipped.

**New findings:** `bench/devices.py::by_index` uninvoked (Niobe's). 4 rows screened, 85 unfalsified.

**Trinity's two guards:** `--write-baseline` absorbed them on the first run — an open red turned
green. Added `not_baselined_on_purpose`, which `--write-baseline` now subtracts mechanically.
`--check` is `FAIL(drift)` on exactly those two and nothing from my three new arms; that is the
evidence the new arms are at baseline. `pytest tests/ops/test_harness_census.py` 9 passed,
1 FAIL(condition), 0 ERROR(instrument).

**Kept the box quiet:** static analysis only, no GPU run, no Rust rebuild — the change is Python
and the verdict it supports is about the census, not about the build.

**Decision record:** `tank-census-frame-is-declared.md` (D-T87).

## Session 23 — 2026-08-02 — offer_shared_device re-decision: the KV hypothesis, refuted and re-aimed

**Asked:** does arming the device-memory provider remove the host round-trip for
`past_key_values`/`present`? Counters only, no clock, and do not let a ctx-0 result decide it.

**Answer: no, and the mechanism says why.** Readback is byte-identical in both lanes —
393,216.0 B per past token on the admissible 0->128 segment, `readback_slope_delta = 0.0`.
`bind_target_for` is called on inputs only (vk/session.rs:1143-1148); the output readback is an
unconditional sum over `actual_output_byte_sizes` (vk/session.rs:1957-1974). **No output-side
bind exists, so no configuration can decline the readback.** Switch's arena is not an alternative
route — it is the only one, and the output-side bind is the seam it needs.

**The half I did not expect:** arming collapses upload from 2.29 GB to 1.57 MB over five inferences
(399,376 B -> 8 B per inference), `alloc_device_buffer_binds` off 0 for the first time. That is
weight residency, and it makes M1's weight-residency criterion measurable. Replaced the expired
default-OFF rationale in factory.rs with that measurement. Did **not** flip the default: the benefit
is on the upload axis and the VRAM cost at long context is unmeasured, and this is Switch's and
Niobe's to hold with me.

**Ran Niobe's probe unchanged, once per lane.** Re-deriving her slope would have made my falsifier
depend on her arithmetic in the other direction, which is Mouse's rule applied to myself.

**Two instrument findings, both of which presented as data.**

1. Both 512 points were truncated. My first diagnosis — a failed best-effort snapshot write leaving
   a well-formed prefix — was **wrong**, and I am recording that it was wrong. The real cause,
   quoted: `vkWaitForFences failed: The logical device has been lost` ->
   `Falling back to ['CPUExecutionProvider'] and retrying.` -> **EXIT=0**. The control matters:
   the identical text appears in the **default** lane at the same point, so device loss at ctx 512
   is not a cost of arming. Differencing the truncated pair before I had a screen produced an
   apparent **6.7% KV saving that was an observation ending early** — the shape of a real result.
   The only signal that refused it was `compute_calls < iters` and `uploads == readbacks + 1`.
   Sixth mode again: a mechanism failing indistinguishably from what it measures. R13 classifies
   it; the exit code cannot.
2. Added `counters_snapshot_writes` / `counters_snapshot_write_failures` so a stale snapshot is
   self-announcing. Verified live: `{'counters_snapshot_writes': 2, ...write_failures: 0}`. Not
   the ctx-512 cause, but the mode it closes was unobservable.

**Census:** main's new bench modules arrived undeclared and the frame check caught them — exactly
what it was built for. `ceiling.py` and `clock_log.py` in frame (both render a verdict;
`clock_log.window` returns UNOBSERVABLE for an unrecorded window, which is the R12 distinction),
`test_ceiling.py` held out as a test module. Baseline updated, Trinity's two guards still held out
on purpose, `CENSUS VERDICT: PASS`.

**Decision:** `tank-device-memory-kv-re-decision.md` (D-T88).
