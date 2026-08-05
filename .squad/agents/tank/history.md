# Tank (Runtime-FFI) — history.md

## Learnings
<!-- CONDENSED-AT: 5a544f37d3d3c2f6d7b4c9ea8e5940ce463f4c9d -->

### [SUMMARY] 2026-08-02 sessions: session-creation disclosure (RAI-009), census frame declared, offer_shared_device re-decision, last-six-ops closeout

- **RAI-009, session-creation disclosure:** a user creating a session claiming `UNMEASURED`/`DIVERGENT` ops must be told at session creation, not left to discover it from a wrong answer. Built the disclosure with a PLANTED control in both arms and verified it is in-frame at the moment it must be read (not logged too early/late). `union_check.py --run` found RED not attributable to Tank's own change.
- **Session 22, census frame declared not implied:** Niobe found `audit_instruments.py` never scanned `bench/` while printing PASS. Fixed via explicit IN/OUT decisions per top-level directory (`FRAME_DIRS`), `ci/` deliberately OUT (censused separately). Selection switched from a first lexical cut (missed all 37 fns in `bench/phases.py`) to structural (module-public top-level defs). 85 unfalsified is a property of the screen, not a verdict on Niobe's tests.
- Team update (Niobe, 2026-08-02T14-42-30): past ctx 2048 the KV path is link-bound not memory-bound (readback exact at 393,216 B/past-token) — revisits whether `offer_shared_device`'s OFF default should change.
- Team update (Link, 2026-08-02T14-42-30): 12 instrumented Rust surfaces (of 50: 14 counters + 10 Phase variants + 26 env switches) remain uncensused by any mechanism — Link's to close, distinct from Tank's bench-frame fix.
- **Session 23, offer_shared_device re-decision:** asked whether arming device-memory removes the KV host round-trip. **Answer: no** — readback is byte-identical in both lanes (`bind_target_for` is input-only; output readback is an unconditional sum with no bind seam) — Switch's arena is the only route, not an alternative. Unexpected upside found instead: arming collapses upload 2.29 GB -> 1.57 MB over 5 inferences (weight residency), used to replace the expired default-OFF rationale in `factory.rs` without flipping the default (VRAM cost at long context still unmeasured). Two instrument findings: a truncated-snapshot artifact was mis-diagnosed as a failed write, the real cause was `vkWaitForFences` device loss at ctx 512 present identically in **both** lanes (not a cost of arming) — differencing the truncated pair before a screen existed produced a false apparent 6.7% KV saving; added `counters_snapshot_writes`/`_failures` so a stale snapshot self-announces.
- Team update (Mouse, 2026-08-02T22-37-04): `device_losses` inserted mid-struct without an ABI bump made three ctypes mirrors silently misread `dispatches_executed`/`unproven_forms_claimed` as plausible-but-wrong values between `a52024f`..`4d47362` — any counter reading in that window is suspect.
- Team update (Switch, 2026-08-02T22-37-04 / 2026-08-03T04-55): `KV_CAN_STAY_DEVICE_RESIDENT` settles D-T88 — ORT does permit device-resident output binding; the obstacle was `transfer.rs`'s host-staging-authoritative invariant, now fixed. Separately: on the real graph the shipping KV lane fails to allocate at ctx 4096 (resident lane fine); ctx 8192 fails on both — the VRAM-cost tradeoff Tank deferred jointly with Niobe/Switch never becomes relevant past 4096.
- **Session 24, last six ops (were seven):** baseline 7 failed/120 passed (Mouse had split one suite in two); final 0 failed/127 passed, ledger 97->106. `IsInf`/`Clip`: a selector is a specialisation constant, not a shader variant — scoped into the two templates rather than the shared header (avoided faulting all 97 ledger entries at once). `Cast` promoted as a pair-keyed template (36 modules, 11 refused for `shaderInt64`) surfaced two latent defects a `Staged` row had been hiding: `claim::cast` declined every opset-19+ `Cast`, and `variant_key` rendered every `Cast` proof key as `metadata`. `Flatten`/`Reshape` confirmed not a defect — zero of either in the whole Phi-3.5 graph. **RAI-008(b): the INFO-severity disclosure half had never been witnessed** — the certifying probe ran ORT at WARNING in both arms where INFO is invisible by construction; a third arm found the EP emits the record and ORT's own threshold admits it, but the line still never appears — cause is inside ORT 1.28, not established; tokens renamed to `OFFERED_TO_ORT`/`BELOW_ORT_THRESHOLD`, never `ORT_SINK`. A planted control (`mul_f16_unproven`) rotted mid-round when populating the ledger proved its own form — consumers now import `PLANTED_CONTROL_CASE` rather than spelling it, and the ledger-arms probe now exits 4 on any errored arm instead of masking it as exit 0.

## Session 25 — the merge that found the seam

**Merged `origin/main` (`607056a`) into `squad/tank` as `5cd5507`.** The coordinator refused to
guess the union of my `SessionDisclosure` struct and Mouse's positional `device_unattributed`, and
was right to. Resolution: struct form, `device_unattributed` as a **field**; Mouse's
`ledger_faulted` leg and his `DEVICE-UNATTRIBUTED-PRESENT` rung kept verbatim.

**Two right changes; the join was wrong.** `disclose_claimed_forms` has two INFO branches. Mine set
`informed`; Mouse's DEVICE-UNATTRIBUTED branch discarded the return of its own
`info_through_ort_sink` call, because when he wrote it there was no INFO counter to feed. **Every
baked ledger entry is DEVICE-UNATTRIBUTED**, so the unjoined branch is the *only* INFO a real
session emits — and `session_disclosure_info_channel` read `UNOBSERVABLE` on runs that had just
emitted one. A channel counter reporting no traffic while traffic moves is worse than no counter,
because it is cited as evidence. Both branches now feed the pair; `info_reached_ort_sink` is ANDed,
because it is a claim about the INFO half as a whole.

**My own probe was asserting the defect.** `probe_session_disclosure.py` went FAIL on both devices
after the merge: it demanded `claimed_forms_proven >= 1` and `claimed_form_evidence == "ALL-PROVEN"`
— the token Mouse deleted *precisely because* it asserted a device frame nothing had checked.
Repaired to assert the property (proof-backed = proven + device_unattributed; the device is
reachable from the disclosure, in either branch's wording) rather than one author's phrasing.
PASS both devices.

**Both fixes proved by mutation, not by assertion.** Dropping `device_unattributed` from the struct:
caught by a new `device_unattributed: 1` rung in the `claimed_form_evidence` ladder test. Un-joining
the INFO branches: caught by `a_proof_backed_disclosure_informs_whichever_branch_carried_it`. Before
those arms, the obvious merge resolution would have compiled, passed, cleared clippy, and silently
re-promoted every claim in the repository to `ALL-PROVEN`.

**The ABI guard did not fire, and that is correct.** `counters_abi.py --check`: v7, 20 fields,
152 bytes, `0x16eacc53e6e18d97`, PASS. The guard covers the mirrored C struct; `SessionDisclosure`
is a Rust-side call shape and `session_disclosure_*` are JSON-only. Mouse's rename fired it because
it renamed a *mirrored* field and the hash covers names. **The guard fires on exactly what a
name-keyed ctypes reader would misread — no more, no less.**

**Census artifacts regenerated, not hand-merged.** Producer recorded:
`onnxruntime_vulkan_ep.dll` SHA-256
`5D457FBBB5B68EC7B75FDB84476C1B8EF0C8FC606D7119DB2979B516C18D2305`, release build of the `5cd5507`
tree, abi_version 7. A hand-merged reading is a reading of nothing, and a reading whose binary is
not named cannot be re-taken.

**Verified:** `cargo test --lib` 515 passed / 0 failed / 4 ignored (main was 510); clippy
`--all-targets -D warnings` clean; `tests/ops` 127 passed / 0 failed; `test_wiring_census.py`
7 passed / 1 xfailed; `probe_session_disclosure.py --devices 0,1` PASS/PASS exit 0;
`probe_ledger_arms.py` exit 0; `probe_ledger_mutations.py` 3/3 CAUGHT. No clock.

**Decision:** `tank-the-union-of-two-right-changes.md`.

**RAI-008(a), same session.** The CI check the falsifier asks for already exists
(`tests/ops/test_proof_ledger.py`, in the default `pytest tests/ops` on both lanes). Its plant was
**spelled by literal** and its docstring still recorded the *previous* plant's key as the
prediction — third sighting of plant-rot in one day. Imported `PLANTED_CONTROL_CASE` /
`PLANTED_CONTROL_KEY` and added an assertion that the plant has not acquired a ledger entry; the
membership test is non-vacuous by measurement (`False` for the plant, `True` for the sibling).
14 passed / 0 failed. **The tally is not the artifact-supplier's** — criterion 11 is Morpheus's,
RAI-008's status is Rai's.

📌 Team update (2026-08-03T10-35-00-07-00): Link generalised from his own accepted-red incident: "an accepted red and a new red are indistinguishable when the only record of the acceptance is a number in someone's head." You maintain instruments with accepted-failure states — apply the same discipline Link built (`ci/check_open_reds.py`'s `stale_acceptance` and `signature_changed` arms) rather than carrying an acceptance count from memory. — decided by Link

📌 Team update (2026-08-03T10-35-00-07-00): Rai opened RAI-013 🟡 — an honestly-labelled emission a user never sees by default is not "the user was told." You are the named owner. — decided by Rai

📌 Team update (2026-08-03T19:55:00-07:00): Switch's refutation — "Phi-3.5 has never been a valid proof subject." Re-proving GQA against a single-form graph with no flag disabled showed withholding one form and withholding nine produce the identical refusal, because Shape, ReduceSum, If have no Vulkan handler at all on this graph — the model was never exercising the claim it was cited for. This changes what a re-proof run can be asked to do: a re-proof on Phi-3.5 alone cannot distinguish "this form is broken" from "this model never reached this form." — decided by Switch

📌 Team update (2026-08-04T12:25:00-07:00): Switch's "the blocker has no control lane" finding —
ctx-512 blocks nothing (both lanes `NO_LOSS`, identical dispatch counts, memory maximally separated);
the actual blocker is option 1 (`DEVICE_MEMORY=1`, `KV_ARENA=0`), established structurally with no
rate needed, and at ctx 4096 the shipping lane cannot run at all (zero EP dispatches, deterministic
3/3). This changes what your `closes_when` can require — a closing condition resting on a rate
measured through a lane that cannot dispatch at all is not a rate, it is a null manipulation.
— decided by Switch## 2026-08-04 — mintability is an ABI question; the ctx-4096 fallback is loudly logged and silently returned
**Task 1 — `--check` could not tell whether a ledger key is mintable.** Repaired as an ABI
addition, not a Python inference: `form_mintability_report()` in `registry.rs`, exported as
`OrtEpVulkanGetFormMintability` (two-call size-then-fill, newline-separated because proof keys
contain commas in their dtype signature — this is where the shader-subject export's comma format
would have silently mis-split). It is pure over the baked SPIR-V and `ENGINE_ENABLED_CAPABILITIES`
and answers identically with no device in scope; `--check` calls it through `ctypes.CDLL` with no
ORT session at all. A missing export is `ERROR(instrument)`, never PASS. Unmintable **ledger** key
is FAIL; unmintable **retired** key is a NOTE — retirement is deliberate, and the note is what
supplies the split the 43 retired keys did not have.
**The red state is shown, not reasoned.** `probe_ledger_mintability.py`, 6/6 arms. Arm 5 is the
load-bearing one: a genuinely shaderless build reports **129/129 declared-stem keys
`mintable=no`**. Getting there cost a real finding — `build.rs::installed_sdk_glslc()` scans
`C:\VulkanSDK` after `VULKAN_SDK` and `PATH` both miss, so
`ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1` is **unreachable on this box**; the first attempt at
arm 5 reported 129/129 *mintable* because the "shaderless" build had quietly compiled shaders
anyway. The reliable route is deleting `shaders/glsl/*.comp` and `shader_variants.txt`
(worktree `ep-vulkan-tank-mint-shaderless`, kept as a reusable control). A synthetic ledger also
trips `check_baked_vs_disk`, so probe arms pass `expect_rebuild=True` or they measure staleness
instead of mintability.
**Task 2 — the brief's premise did not survive the measurement.** The ctx-4096 fault is not a
session-creation rebuild and not `alloc_device failed for input buffer`. The session creates, the
EP claims all 355 nodes, `Compute()` is entered, and the gpu-allocator refuses 67108864 bytes for
the **intermediate** `ep_inter_76`. And it is **not silent**: three disclosures fire at ORT's
default severity, not just at VERBOSE — arm E's log is byte-identical to arm B's after timestamp
normalisation. What lies is exit 0, the correct finite logits, and `get_providers()`. Honest name:
**loudly logged, silently returned**.
**`disable_cpu_ep_fallback=1` screens the partition, not the fallback.** Arm D — ctx 1, no island
retained, no allocator involvement anywhere — produces the **identical refusal sentence** as arm C
at ctx 4096. Switch's own discriminator applied to my instrument: a screen that cannot separate the
lanes is not reading the fault. On Phi-3.5 the flag is unusable, because five `[unproven]` declines
sit on the CPU EP on every run. `probe_runtime_fallback_guard.py` (arms F–I, `add_f32`, fully
claimed, failure injected) reaches the question C could not: the guard **does** block the run-time
path, but by side effect of ORT re-initialising on `['CPUExecutionProvider']`, and its message
names the user's provider list rather than the EP's broken commitment. RAI-012's shape one layer
out, in ORT rather than in us.
**The null manipulation I nearly shipped.** My first C and D named `CPUExecutionProvider` in the
provider list *while* setting the flag, so ORT refused for a *configuration* reason before
partitioning. It looks exactly like the guard firing. Both probes now pass
`providers=["VulkanExecutionProvider"]` alone and the trap is written down in the probe.
**Verified:** `cargo test --lib` 576 passed / 0 failed / 4 ignored (main was 574); clippy
`--all-targets -D warnings` clean; `gen_proof_ledger.py --check` PASS with the new subject
arithmetic line `129 mintable / 0 not`, `43 retired — 43 mintable / 0 not`;
`tests/ops/test_proof_ledger.py` 18 passed (was 14); `pytest tests/ops` 829 passed / 2 failed,
**both reproduced identically on unmodified `main` b8679a4 with main's own DLL**. Every ctx-4096
arm pinned `DEVICE_MEMORY=1` / `KV_ARENA=0` and screened on `dispatches_executed`. No clock.
**Instrument note carried forward:** ORT's Windows log sink writes wide characters into a captured
pipe, so a UTF-8 read renders every ORT line NUL-separated and every grep answers "absent". Both
probes strip NULs before matching. A log-absence claim taken without that strip is a null reading.
**Decisions:** `tank-mintability-is-an-abi-question.md`, `tank-loudly-logged-silently-returned.md`.
