# Link (Platform-Support) — history.md

<!-- CONDENSED-AT: 4725a94cb8466162e891630736efc36d7bbf0de1 -->

## Learnings

### [SUMMARY] Sessions 1–6: extension availability, capability set, CI root causes, hardware matrix, LVP2 retraction (2026-07-28–2026-07-30)

OQ-1 measured: `VK_KHR_synchronization2` at 68.57% on Android, `VK_EXT_subgroup_size_control` at 85.88%. Option B (Khronos layer shim) rejected. Wgpu/Dawn/Godot's claimed Vulkan 1.2 requirement found false (extensions, not 1.2 core). Windows CI: `VK_ICD_FILENAMES` ignored under elevation — must register ICD in `HKLM\SOFTWARE\Khronos\Vulkan\Drivers`. Linux CI failure was a compile error, not Vulkan (`glslc` via LunarG apt repo). Hardware matrix: both CI lanes CI-VERIFIED, both local GPUs local-dev-verified; Intel Iris Xe is the spec-conformance oracle (UMA proxy for mobile memory model) — never special-case its failures. Cross-platform generality is structural, not a review step; every `cfg` is a portability hazard. **LVP2 RETRACTED:** Mesa 23.2.1 lavapipe supports subgroup BASIC+ARITHMETIC+BALLOT+SHUFFLE+SHUFFLE_RELATIVE+QUAD in compute (`subgroup_size=8`); the original `supportedStages=0` reading was our own discarded `push_next` chain, not a device fact.

**Current state (end of session 6):** PLATFORMS.md current, both CI lanes verified, M0 criterion 9 met.
---


<!-- SUMMARIZED by Scribe 2026-08-01T20:39:12-07:00 -- older entries condensed below; full text lives in git history -->

### [SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)

- Round 4 worktree layout note (squad/switch, squad/mouse, squad/tank worktrees, `main` integration tree).
- Session 8 — gate artifact design, `is_uma` verification, subgroup red instrument, lane classification (`operational` vs `green` per DESIGN.md §8.9).
- Session 9 — criterion 10's gate wired into CI lanes, new fourth verdict state `UNATTRIBUTED`, lanes reclassified, subgroup-32 falsifier chain restated end to end.
- Session 10 — gate closes the loop, `ERROR(instrument)` made readable: ran both polarities on both devices, positive `GATE: PASS`/`MATCH` identical digest on both GPUs; found the Windows negative control (`VK_ICD_FILENAMES` removal) likely never effective under elevation — third instance of an instrument outage wearing a detection's costume; added `ci/check_vocabulary.py` preflight with distinct `ERROR(instrument=...)` tokens. Learning: a negative control depending on a documented-unreliable mechanism isn't a control.

📌 Team update (2026-08-01T09:53:14-07:00): The EP genuinely executes now — 3 VulkanExecutionProvider fused-node events (~355 graph nodes in one fused node) + 24 CPU per run, 65/65 outputs bit-identical, argmax 30751 matching CPU; coverage figures are execution, not offer. All wall-clock figures including 3.1x/3.7x are withdrawn under R13 pending device-clock measurement. Switch holds exclusive claim on device-clock measurement while agents run in parallel. — decided by Scribe

---

### [SUMMARY] Sessions 12-18b: lane composition, criterion 12 extent, device-loss reporting, cross-platform Linux, ORT log level width, open-reds register (2026-08-02 -- 2026-08-03)

- **[SUMMARY] Session 12 -- composed workflow, PLANTED vs OBSERVED (2026-08-02)** -- Registered `hostfree.tautological_assertions` as `UNDEMONSTRATED` (Switch's own screen never actually exercised either defect that occurred), demoting `lane-checks` green->operational -- then went further and added an orthogonal `PLANTED`/`OBSERVED` axis to every green check via `validate()`, since conflating "somebody planted a defect and it caught it" with "it caught something nobody planted" applied a double standard (his own tick screen is PLANTED too). Built `ci/check_lane_inventory.py --union-with <ref>` (union of step-name lists, not a merge -- avoids merge-conflict semantics) after a tautological-assertion defect from Switch's branch became unclassified only in the union of two correct merges. Assigned `rust/src/trace.rs` to Niobe.
- **[SUMMARY] Session 13 -- criterion 12 extent (2026-08-02)** -- Criterion 12 NOT closed; built `ci/check_census_completeness.py` enumerating the true whole from production Rust the census doesn't write (14 counter fields + 10 trace `Phase` variants + 26 env switches = 50 surfaces), so numerator and denominator have independent authors.
- **[SUMMARY] Session 15 -- device-loss reporting defect (2026-08-02)** -- Built `ci/check_device_loss.py` + 14-arm negative control; live catch found a SECOND, earlier unreported device loss (2026-07-31, Intel, different call site, two days before Tank's) -- proving the reporting defect outlives whatever Switch fixes. Ruled three device-loss mechanisms have three extents, never one guarantee; `disable_cpu_ep_fallback` is structurally blind to a loss on an already-accepted session. Found (routed, not patched) that `_verdict.FATAL_LOG_MARKERS` never matches ORT's real fallback text, so `check_fatal_log` was reading Tank's log as clean while it announced the fallback twice.
- **[SUMMARY] Session 16 -- cross-platform Linux answer (2026-08-02)** -- Built WSL Ubuntu access; found `cargo test --lib` doesn't compile on Linux (11 `i32`/`u32` errors on `OrtLoggingLevel`, routed not patched); a red step skips the rest of its job so 7 Linux steps had never run (new status `GATED_NEVER_RUN`); the real defect is that the proof ledger keys per-toolchain (`shader_digest` hashes the compiled SPIR-V, so a different glslc faults all 74 entries with no kernel change) while `device` sits unread on 74/75 entries -- digest-agreeing-on-an-unproven-device fails open and nothing watches it. Built `ci/check_ledger_portability.py` (3 LIVE arms).
- **[SUMMARY] Session 13-reprise -- review of Trinity's marker fix (2026-08-02)** -- Upheld Trinity's decline to widen `FATAL_LOG_MARKERS` (would have reddened his own negative control, deleting the evidence for the reach it demonstrates). Found two defects of his own the same shape: `marker_cross_check()` compared physical lines by string equality instead of matched spans (broke on trailing punctuation); `check_device_loss.py` scanned un-normalised text, blind to UTF-16LE including its own primary condition. New rule: an arm must assert the property wanted, never the defect currently present, or fixing the defect reddens the control meant to catch it.
- **[SUMMARY] Session 14 -- ORT log-level width, seven steps run (2026-08-02)** -- `OrtLoggingLevel` is `c_int` on MSVC / `c_uint` on GCC (representational, not a signedness bug); rejected an eleven-site `as i32` cast fix on principle (preserves the exact assumption that caused the bug) in favour of typing the three real declarations, and made the rejection executable as portability rule P3. Found his own prior "DLL SHA-256 identical = no Rust changed" method was unsound: Windows gave six distinct hashes across six rebuilds of an unchanged tree (cargo just didn't relink) while Linux's `.so` was byte-identical -- same shape as an empty patch reading like a clean file. `cargo check --all-targets` now runs before clippy on both lanes (clippy had nothing to say about a crate that couldn't compile).
- **[SUMMARY] Session 17 -- closing the sweep (2026-08-03)** -- "Windows-only" for 12 failures was a category error: Windows is the only lane with a device that executes phi-3.5 (lavapipe declines to CPU), so it meant *only observable on*, not *broken on*; survivor is a real fp16 accumulation finding (ULP 0.0->4.0 monotonic, only layer 31 K/V), routed to Mouse/Switch. His own dead-guard screen convicted his own prior decision (38 dormant `BUILD_SKIPPED` guards left in for a tidy diff, re-armed by one new line). Built `ci/check_build_precondition.py` and `ci/check_flake_witness.py`.
- **[SUMMARY] Session 18 -- open-reds register (2026-08-03)** -- Three reds (`audit_instruments`, lane-checks, union_check) were failing into a void -- nothing read their exit codes. Found his own "132 passed, 3 failed (known census reds)" had silently become 4 after a merge reintroduced deleted `BUILD_SKIPPED` guards -- an accepted red and a new red are indistinguishable when the only record is a number in someone's head. Built `ci/check_open_reds.py` + `ci/open_reds.json`: every guard declares its expected colour (both falsifiable), with `unaccounted_red`/`stale_acceptance`/`signature_changed`/`lease_expired` arms; five accepted reds catalogued with named owners for the first time.
- **[SUMMARY] Session 18b -- register's first real user found its defect (2026-08-03)** -- Mouse closed three accepted reds and DELETED their entries per the tool's own instructions -- but `audit_instruments --check` went from ruling on 8 subjects to 5 and printed PASS over a red it had stopped watching (an eighth uninvoked accessor was missed). Repaired: `subjects` is append-only, closing a red flips `expect` to green (never deletes), `retired` requires owner+date+reason and is not a suppression list -- the frame line now prints `N ever declared = M ruled on now + K retired`.
## Session 19 (2026-08-04) -- a cited artifact has a frame, and a merge reverted a field inside 115 surviving entries

Assigned: regenerate the three Linux artifacts Rai found were cited as evidence
for a fix committed three hours AFTER them; then ask whether a cheap invariant
would make a stale citation detectable; then close or register two limitations
of my own that were living in prose.

Did all three. The regeneration is the smaller half of what came back.

THE FINDING. First regeneration came back 50 failed / 372 passed against 8
failed / 633 passed the round before. Ledger --check on the fresh .so read
6 PROVEN-ELSEWHERE + 115 SUBJECT-CHANGED -- the exact inverse of session 18.
Cause: eee65aa (Mouse, Conv) regenerated evidence/proof_ledger.jsonl from a base
predating my merge aea0147 and restored the withdrawn pre-normalize_shader_text
source_digest on 115 of 121 entries. No key lost, no entry deleted, file GREW
116 -> 122. Key census 0 VANISHED. Loss invariant 0 missing. Shrinking-write
guard silent. Windows --check PASS, because stale source + identical SPIR-V is
SOURCE-COSMETIC, which forgives. Linux declines the same staleness as
SUBJECT-CHANGED. Second firing of my own session-18c sentence, with a count:
only the declining platform could see it, and a field inside a surviving entry
is invisible to every instrument that counts entries.

Repaired with --backfill-frame --rewitness-source (115 re-witnessed) plus a
rebuild -- the ledger is include_str!'d, so an unrebuilt --check reads the baked
copy. After: Windows 121 identical / 0 SOURCE-COSMETIC; Linux 9 failed / 676
passed, gate MATCH, counters PASS, portability PASS.

BUILT. ci/check_artifact_frame.py + evidence/proof_rewitness.json + a
frame-witness arm and a shallow-clone guard in ci/check_ledger_census.py + an
xtent requirement on accepted reds + a new known_limits section in
ci/open_reds.json that makes each screen ADMIT its own bounded gap by token
every run.

THREE ERRORS I MADE, each of a class the tools exist to prevent. (1) My first
frame-witness arm was SYMMETRIC -- it convicted my own repair as loudly as the
regression, because from values alone the two are one event seen from opposite
ends. The answerable question is whether the writer DECLARED the move, not what
the value did. (2) A hand-typed screened_since sha resolved to nothing, so
merge-base --is-ancestor failed for every revision, everything fell out of
frame, and the screen printed PASS having ruled on nothing. (3) Ancestry is not
the frame boundary: eee65aa was authored on squad/mouse, which forked BEFORE the
merge, so an ancestry test excused the one event the arm exists for. Fixed with
position in the --full-history --topo-order walk. All three are planted arms now.

Also learned: a known limit is NOT an accepted red. test_lane_checks enforces
owner != link on accepted reds, and it is right to; a bounded gap in a screen is
held by the screen's own author, which is exactly the case that rule forbids.
Filing one as the other means either weakening the rule or writing an owner you
do not mean. New category, and a test that no id appears in both.

Evidence: Windows cargo test --lib 553 passed / 0 failed / 4 ignored, clippy
clean with -D warnings. gen_proof_ledger.py --check PASS, 121 entries = 121
identical, loss invariant 0 missing (ONNXRUNTIME_VULKAN_EP_LIB set -- without it
the checks pass having read nothing). check_ledger_census.py PASS, 0 UNDECLARED.
negative_control_ledger_census.py 21/21 (2 LIVE, 6 REPLAYED, 13 PLANTED).
negative_control_artifact_frame.py 11/11 (1 LIVE, 2 REPLAYED, 8 PLANTED) -- the
REPLAYED arms convict the real ac4bd0b/7688356 citation gap. test_lane_checks.py
164 passed, 3 failed and the 3 are the declared census-extent reds.
check_open_reds.py: all 12 entries the declared colour, 2 known limits admitted.
audit_instruments --check PASS (declared Niobe's bench/test_paired_ratio.py).
verification_subjects PASS, 24 classified, 0 SELF. Lane inventory PASS.
Linux: WSL Ubuntu / llvmpipe LLVM 20.1.2, .so sha256 727d0f41a6a31291.
No timing claims. DEVICE_MEMORY and KV_ARENA not enabled in any lane.

NOT established: the regenerated pytest-linux.log does NOT witness the corrected
decline message, because on the repaired tree there are zero declines left to
carry it. The reading that did witness it (42 declines, all corrected, 0
old-message) was taken mid-session and overwritten by the second run; it is
recorded in numbers in the decision, not as a file.

POSTSCRIPT, same session. Ran check_ledger_census.py a second time hours after
the first and it went from PASS to 28 VANISHED / 102 undeclared moves on an
unchanged tree -- because Mouse had pushed 18ddece to squad/mouse in between and
the walk was scoped --all. It was convicting my branch for not containing
somebody else's unmerged draft, and the sentence it printed ("committed to this
ledger and no longer in it") was false. DEFAULT_SCOPE is now HEAD; both merge
parents are reachable from HEAD so a proof dropped inside a merge is still
convicted, and there are now two planted arms holding both halves.
negative_control_ledger_census.py 23/23 (2 LIVE, 6 REPLAYED, 15 PLANTED).

Four framing errors in one session and not one of them was a wrong value test.
Symmetric comparison; unresolvable boundary; ancestry mistaken for frame
boundary; scope too wide. A screen with a right question and a wrong scope
reports confidently about the wrong population, and the only reason I caught the
fourth is that I happened to run it twice while a teammate was working. That is
not a method.

📌 Team update (2026-08-04T12:25:00-07:00): the field-level reversion class has now happened twice
(your own 115-of-121 finding, then a second reversion inside a `squad/mouse` merge) and every
count-based screen read clean both times, for the same reason both times — the loss is a field
inside a surviving entry, invisible to any instrument that counts entries. Only reading the subject
arithmetic line, not the PASS line, catches it. — decided by Rai, Link, Morpheus

📌 Team update (2026-08-04T12:25:00-07:00): Rai's "the claim is true; the citation is not" (RAI-012's
cited artifacts were three hours stale) is the event your `check_artifact_frame.py` was built to
catch. Morpheus's ruling: a citation is a proof key with no subject digest — cite a state, not a
path. — decided by Rai, Morpheus

📌 

📌 Team update (2026-08-04T20-25-00-07-00): Mouse's `claimed_nodes` != `dispatches_executed` -- BERT claims 481 of 1274 rows at `GetCapability` but the partitioner's net-benefit gate retains only 4; every coverage figure quoted against `claimed_nodes` alone (including prior island/counterfactual rankings) is affected. `dispatches_executed` is the honest metric going forward. -- decided by Mouse

📌 Team update (2026-08-04T20-25-00-07-00): Tank's "loudly logged, silently returned" finding -- `Compute()` is entered, the intermediate `ep_inter_76` allocation is refused (67108864 bytes), three disclosures fire at ORT's default severity, but exit 0 / finite logits / `get_providers()` all read as success. Also: `disable_cpu_ep_fallback=1` screens the partition, not the fallback -- arm D at ctx 1 with no island produces the identical refusal sentence. -- decided by Tank

📌 Team update (2026-08-04T20-25-00-07-00): `cargo fmt --check` has been re-broken by three consecutive merges since the coordinator's fix landed and was confirmed green on CI. Recording as a CI-discipline item (a gate that is not re-verified after every merge is not a gate), not a formatting nit. -- decided by Coordinator
