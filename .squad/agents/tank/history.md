# Tank (Runtime-FFI) — history.md

## Learnings

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

📌 Team update (2026-08-02T14-42-30-07-00): Niobe measured that past context length 2048 we are link-bound, not memory-bound (KV-cache readback exact at 393,216 B per past token, ratio 1.000000). This changes what the present-copy KV-cache fix is worth once contexts grow past that point, and connects to `offer_shared_device`, whose default is `OFF` and whose recorded reason for staying off Morpheus has separately ruled expired. Worth revisiting whether the default should change now that the bottleneck downstream of it has moved. — decided by Niobe

📌 Team update (2026-08-02T14-42-30-07-00): Link's independent whole (enumerated from production Rust the census does not write: 14 counter fields, 10 Phase variants, 26 env switches = 50 surfaces) found 12 instrumented surfaces observed by no census mechanism. You have now put `bench/` in frame for the census, but these 12 are Rust surfaces (not bench/), still uncovered, and remain Link's subject to close, not yours — noted so the two efforts aren't conflated as one gap closing when the other lands. — decided by Link

## [SUMMARY] Compressed entries

<!-- SUMMARIZED by Scribe 2026-08-02T15:27:07.854635 -->

- **[SUMMARY] Sessions 1–13b: crate foundation, ORT bindings, logging crash, allocator, execution verification (2026-07-28–2026-07-30)** — **Sessions 1–7 (archived):** Crate structure (`ort-ep-vulkan`, cdylib). ORT C API bindings via bindgen. Three-number version negotiation: EXPECTED 28 / MIN 24 / negotiated 28. `logging::forward_to_ort` null-pointer crash
- **[SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)** — - **📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)** — ### Worktree layout and inbox portability constraint The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squa
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

📌 Team update (2026-08-02T22:37:04-07:00): Mouse's `mouse-counters-abi-mirror-equality` finding — `device_losses` was inserted mid-struct into `VulkanEpCounters` without a `COUNTERS_ABI_VERSION` bump; three ctypes mirrors kept the old layout, so `dispatches_executed` silently read `device_losses` and `unproven_forms_claimed` silently read `ledger_entries` — nothing went red because the wrong number was stable and plausible. Every counter reading you took through a ctypes mirror between `a52024f` and `4d47362` is suspect. Mirrors must now assert exact `struct_size` equality, not `>=`, and declare the ABI version they were written against. — decided by Mouse

📌 Team update (2026-08-02T22:37:04-07:00): Switch's `KV_CAN_STAY_DEVICE_RESIDENT` ruling settles the question your device-memory re-decision (D-T88) opened: ORT does permit binding an `OrtValue` in this EP's device memory as a graph output, bit-identical to unbound — the project was never in the "ORT forbids it" world. The obstacle was `transfer.rs`'s own host-staging-authoritative invariant, now fixed as a per-span `device_authoritative` flag. Your finding that arming the allocator does not touch the KV round trip (input-only bind, unconditional output readback) stands unchanged — the per-span fix is the output-side seam your writeup named as owed to Switch, and it is now landed. — decided by Switch

## Session 24 — 2026-08-02 — the last six ops, and the disclosure half nobody had ever seen

**The six were seven.** The baseline was taken before anything was touched:
`tests/ops/test_op_table.py` + `test_elementwise.py` = **7 failed / 120 passed**. Mouse had
counted two suites separately and `test_clip_no_bounds` lived in the other one. Final:
**0 failed / 127 passed**. `cargo test --lib` 492 -> **505 / 0**. Clippy clean. Ledger 97 -> **106**.

**IsInf and Clip: a selector is a specialisation constant, not a shader variant.** The record's
premise was right (a selector is not a float and cannot ride `pc.params`) and its conclusion was
one step too far. `PipelineKey` is already keyed on `(shader_stem, spec_constants)` and
`build_spec_info_data` already maps index -> `constant_id`; id 2 was free. Declaring it in the
shared `indexing.glsl` faulted all 97 ledger entries at once (11 lib tests -> ERROR(instrument)) —
the digest gate catching a one-line header edit made to serve two modules. Scoped into the two
templates instead. Commit `164f9bf`.

**I re-derived a repair the record had already argued against, and the record was right.** I
invented a `SelectorError::NoBoundPresent` refusing a bounds-free `Clip`. It contradicts this
project's own `Identity` precedent. Reversed to `Ok(0)`.

**Cast is a pair-keyed template** (`Template::EwCast`, 6x6 `pair_stems!`, 36 modules, 11 refused
for `shaderInt64`). Promotion made **two latent defects reachable**: `claim::cast` declined every
opset-19+ `Cast` because ORT reports a defaulted `saturate` as present, and `variant_key` rendered
every `Cast` proof key as `metadata`. A `Staged` row's predicate is never really run — **staging
hides bugs in the code staging says is ready.** Three test feeds were also vacuous
(`Cast-fp32-to-i32` truncates ~68% of a standard normal to zero). Commit `26fd93f`.

**Flatten/Reshape were not a defect.** A census of the whole Phi-3.5 graph (363 node records) found
**zero** of either. The rows asserted an *intention* no version of this EP has ever satisfied.
`claim=False`, argued, with a named reachable falsifier — not deleted.

**RAI-008(b): the INFO half had never been witnessed.** The probe that certifies the §8.9.7
disclosure ran ORT at WARNING in *both* arms, where an INFO is invisible by construction. Adding
arm C found that the EP emits the record, ORT's own `Logger_GetLoggingSeverityLevel` reports a
threshold that admits it, and the line never appears — at any host severity — while the WARN from
the identical call site always does. Blindness control rules out the boring explanation. Cause is
inside ORT 1.28 and is **not** established. The channel counter's tokens are therefore
`OFFERED_TO_ORT` / `BELOW_ORT_THRESHOLD`, never `ORT_SINK`: **a counter that overstates delivery is
the defect the counter exists to catch, wearing the counter's badge.**

**The planted control had moved out from under two probes.** `mul_f16_unproven` became
`sub_f16_dyn_unproven` because populating the op-suite ledger *proved the old plant's form* — the
control fired on its own author. `probe_session_disclosure.py` said ERROR(instrument) and exited 4;
`tests/ops/probe_ledger_arms.py` had the same dead name and **exited 0 with the arm errored**. Both
now import `PLANTED_CONTROL_CASE`; the ledger-arms probe exits 4 on any errored arm.
**A planted control names a condition the team is working to eliminate — it will rot by design.
Import the plant; never spell it.**

**An OBSERVED arm for criterion 11(c), supplied not tallied.** `Cast` f32->i32 HITs and f32->u8 is
KEY-ABSENT on models identical but for the `to` attribute. No plant, no environment variable —
strictly stronger than the arranged control. Trinity's (c) artifacts refreshed from
`ledger_entries: 9` to 106. The tally is not the artifact-supplier's.

**ABI caution:** no conclusion in this session rests on a ctypes-mirror counter read from the
`a52024f`..`4d47362` window. The Phi-3.5 census is a claim log, not a counter struct.

**Decisions:** `tank-selectors-are-spec-constants-not-variants.md`,
`tank-shape-ops-stay-unregistered.md`, `tank-cast-is-a-pair-keyed-template.md`,
`tank-the-info-half-was-never-witnessed.md`. Commits `164f9bf`, `26fd93f`, `1578fcd` on `squad/tank`.
