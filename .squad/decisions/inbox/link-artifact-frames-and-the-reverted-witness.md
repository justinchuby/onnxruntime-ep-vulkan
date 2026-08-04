# A cited artifact has a frame, and nothing was checking it

**link**, 2026-08-04, branch `squad/link`. Requested by justinchu via coordinator, from Rai's RAI-012 finding.

## What Rai found, and what I did about it

Rai verified RAI-012 as genuinely fixed — she rebuilt the crate under WSL and ran the real op
suite: 42 declines, all carrying the corrected message, zero old-message instances. Then she
found that the three artifact files the round cited as the evidence were committed at `ac4bd0b`
(07:12) and the fix landed at `7688356` (10:33). **The claim was true; the citation was not.**

I regenerated them. Which files, which build, which platform:

| file | produced by |
|---|---|
| `bench/results/link-linux-downstream/pytest-linux.log` | `ci/link-linux-repro/link_linux_device_steps2.sh` |
| `bench/results/link-linux-downstream/counters-linux.json` | same run, `epctl --check-counters` |
| `bench/results/link-linux-downstream/verdict-linux.json` | same run, Criterion-10 gate |

Platform: WSL Ubuntu, `llvmpipe (LLVM 20.1.2, 256 bits)`, `shaderc 2023.8-1build1`.
Build: `libonnxruntime_vulkan_ep.so`, release, sha256 `727d0f41a6a31291…`, built inside WSL at
`CARGO_TARGET_DIR=/home/justinchu/link-linux-target` so it cannot clobber the Windows artifact.

## The regeneration found something much larger than the stale citation

The first regeneration came back **50 failed / 372 passed**, against 8 failed / 633 passed the
previous round. Criterion-10 `UNATTRIBUTED`, `--check-counters` exit 1, portability
`FAIL(claimed_nothing)`.

`gen_proof_ledger.py --check` against the fresh `.so` read **6 PROVEN-ELSEWHERE{toolchain} + 115
SUBJECT-CHANGED** — the exact inverse of the previous session's 115 PROVEN-ELSEWHERE.

`eee65aa` (Mouse, the `Conv` commit) regenerated `evidence/proof_ledger.jsonl` from a base
predating my merge `aea0147`, and in doing so **restored the withdrawn pre-`normalize_shader_text`
`source_digest` on 115 of the 121 entries**. The 6 survivors are exactly the entries that commit
newly proved. No key was lost, no entry was deleted, the file *grew* 116 → 122.

**Every screen in this repository read clean over it:**

| screen | reading | why it was right and useless |
|---|---|---|
| ledger key census | `0 VANISHED` | no key was lost |
| loss invariant (new) | `0 missing` | same |
| shrinking-write guard | silent | the file grew |
| `--check` on **Windows** | **PASS** | stale source + identical SPIR-V is `SOURCE-COSMETIC`, which forgives |
| `--check` on **Linux** | **115 SUBJECT-CHANGED** | the same staleness collides into a state that declines |

This is the second firing of a sentence I wrote in session 18c, now with a count attached:
**Windows masked it; only the declining platform could see it.** A field inside a surviving entry
is invisible to every instrument that counts entries.

Repaired with `--backfill-frame --rewitness-source` (re-witnessed 115, 6 already current) plus a
rebuild — the ledger is `include_str!`'d, so an unrebuilt `--check` reads the baked copy and fails
`LEDGER_DOES_NOT_DESCRIBE_THE_BUILD`. After repair: Windows `--check` PASS, 121 identical, 0
SOURCE-COSMETIC; Linux lane **9 failed / 676 passed**, gate PASS (`verdict=MATCH`,
`VulkanExecutionProvider=1`), counters PASS, portability PASS.

### For Mouse

Two things, neither of them a request to stop what you are doing.

1. **Regenerating the ledger from a stale base silently withdraws other people's frames.** It is
   now screened: `ci/check_ledger_census.py` grew a **frame-witness arm** and
   `evidence/proof_rewitness.json` is its register. If you move `source_digest` on entries you did
   not prove, declare it there — `{revision, who, keys, deliberate, why}`. An undeclared move is
   `FAIL(condition=undeclared_witness_move)`; a declaration matching no move is
   `stale_rewitness_declaration`, so the register cannot rot either. The `eee65aa` record is in
   there permanently with `deliberate: false` — deleting it would remove the check, not the event.
2. **`SOURCE-COSMETIC` is a PASS state and I think that is arguably wrong.** It forgives on the
   only platform CI runs green, and it forgave this. I have not changed it: it is your file and
   there is a real argument for it (a cosmetic source change with identical SPIR-V genuinely is
   the same subject). But the cost of the current setting is now measured rather than assumed.

## The general question, and what it cost

> Nothing in this repository forces a cited artifact to be newer than the fix it is cited for.

**Verdict: the cheap half is cheap and is built. The strong form is not cheap and I have not built it.**

`ci/check_artifact_frame.py` gives an artifact directory the frame the proof ledger already has:
a sidecar `artifact-frame.json` recording `produced_at_commit`, `platform`, per-file sha256,
optional `subject_sha256`, and `subject_paths` — the source the reading is *about*. Arms:

- `artifact_predates_subject` — has anything touched `subject_paths` since `produced_at_commit`?
  Pure `git log`. **No build, no device, no second platform.** This is the arm that catches RAI-012.
- `artifact_content_moved` — do the bytes still hash to what the frame says?
- `frame_pins_an_uncommitted_tree` — was the reading taken while the subject was dirty? Then
  `produced_at_commit` names a tree nobody has and every other arm compares against the wrong thing.
- `subject_moved` — opt-in via `--subject`, **off by default**, because the Windows `.dll` is not
  byte-reproducible across forced rebuilds (PLATFORMS.md §7.21.3) and a check that cries wolf on
  every rebuild is a check somebody switches off. Linux `.so` *is* reproducible, so the Linux lane
  can turn it on.
- Absent, incomplete or unresolvable frame → `ERROR(instrument=…)`, exit 2. Never PASS.

`ci/negative_control_artifact_frame.py`: **11 arms, 1 LIVE / 2 REPLAYED / 8 PLANTED**, ratio
printed. The REPLAYED arms reconstruct the real `ac4bd0b` frame and convict the actual 2026-08-03
gap, and separately assert that `7688356` is among the 13 commits it names — otherwise a match
would be coincidence.

**What it does not do, stated plainly:** `subject_paths` is a list the writer typed, not a
derivation from the claim. An artifact can be stale w.r.t. a fix in a path nobody named. Closing
that needs every claim to name its evidence — a claims register nobody has asked for — and I am
not building it on speculation. That residual is now a **declared known limit**, below.

**Cost, honestly:** ~250 lines of screen + ~250 of control, no build, no device, ~1s per run. The
strong form (derive `subject_paths` from a claims register) is a new register plus an editing
discipline for every artifact-citing document in the repo, and I decline it until there is
evidence the cheap form is insufficient.

### It has an immediate second customer

`kv_caller_bind_reading` is an accepted red whose whole difficulty is this problem:
*"the artifact was re-taken at 872d739 … either the obstacle went away and the assertion is stale,
or the re-take was made under conditions that do not answer the same question, and I cannot tell
which from the artifact alone."* **Switch: a frame on `bench/results/kv_device_residency-*.json`
would settle it from the file.** Stamp it when you re-take.

## The two limitations of mine that were living in prose

Both are closed, and their residuals are now **checked**, not listed.

1. **Switch's finding — a substring cannot notice a second file joining an accepted red.**
   Closed: `expect: red` entries in `ci/open_reds.json` now *require* an `extent`
   `{pattern (exactly one capture group), members}` and the observed set must match exactly.
   Growing is `FAIL(condition=extent_widened)`, shrinking is `extent_narrowed` — good news, and
   still a failure, because an acceptance that has stopped covering something must be re-read.
   Both shipped red entries carry extents. 5 new tests in `ci/test_lane_checks.py`.
   *An empty `members` list is legal and is the strictest possible declaration* — it says the
   acceptance covers nothing, so any match is widening. My first cut rejected it as "missing",
   which made the safest thing unspellable.
2. **The census is blind in a shallow clone — declared by me in session 18 and not guarded.**
   Closed: `history_is_complete()` → `ERROR(instrument=truncated_history)`, exit 2. Verified
   against a real `--depth 1` clone.

### `known_limits`: a new section in `ci/open_reds.json`, and why it is not an accepted red

`ci/test_lane_checks.py` already enforces that an accepted red's `owner != link` — *"an accepted
red owned by nobody but the person who accepted it is not owned."* That rule is right, and my
first attempt violated it: I filed the artifact-frame residual as a red I owned. The correct
reading is that **a known limit is a different kind of thing from an accepted red.** An accepted
red is a failing check somebody *else* closes. A known limit is a bounded gap in a screen, held by
the screen's own author — precisely the case that rule forbids. So it gets its own category
rather than a category it does not fit, and a test asserts no id appears in both.

Each entry names a command that makes **the screen itself admit the limit by token** and exit 1
(`--assert-known-limit <name>`). It is executed on every `check_open_reds.py` run. If a limit is
quietly closed the admission stops and the arm goes red — a stale declaration is as much a defect
as an undeclared gap. Two entries today, both mine:

| id | screen | closes when |
|---|---|---|
| `artifact_frame_subject_paths_is_a_declaration` | `ci/check_artifact_frame.py` | a claims register lands, or a real stale citation gets through a declared list |
| `ledger_census_is_unobservable_in_a_shallow_clone` | `ci/check_ledger_census.py` | the workflow declares fetch-depth 0 **and** a lane test asserts exit 2 fails the lane |

The second one is deliberate: a guard is not a fix. In a shallow CI checkout the census still
rules on nothing; it now says so loudly instead of printing PASS, and that is all.

## Three errors I made building the frame-witness arm, each of a class this file exists to prevent

Make that four — the fourth was found by running the screen twice, hours apart.

1. **Symmetry.** My first arm asked *"did the value go back to something a later revision
   replaced?"* — which convicts a repair exactly as loudly as a regression, because from values
   alone the two are one event seen from opposite ends. A screen cannot rank two alternating
   values without a frame, and the ledger records no frame for its own hashing rule. The
   answerable question is *"did the writer say they were moving it."*
2. **An unresolvable boundary printed PASS having read nothing.** A hand-typed `screened_since`
   sha made `merge-base --is-ancestor` fail for every revision, so everything fell out of frame.
   Same defect class the file exists to prevent, arriving through a typo. Now guarded by
   `rev-parse --verify`, and planted as an arm in both negative controls.
3. **Ancestry is not the frame boundary.** `eee65aa` is *not* a descendant of `aea0147` — it was
   authored on `squad/mouse`, which forked before the merge. An ancestry test excused the one
   event the arm exists for. Fixed by using position in the `--full-history --topo-order` walk.
4. **The revision walk was scoped `--all`, and `--all` is not history.** The census was green in
   the morning. Hours later, minutes after Mouse pushed `18ddece` to `squad/mouse`, the same
   screen on the same tree reported **28 VANISHED proofs and 102 undeclared witness moves** — all
   of them from a branch that is not an ancestor of anything I have. It was convicting my branch
   for not containing somebody else's unmerged draft, and the sentence it printed — *"committed to
   this ledger and no longer in it"* — was false. `DEFAULT_SCOPE` is now `HEAD`, which costs the
   screen nothing it exists for: the failure it was built to catch is a proof dropped inside a
   merge conflict resolution, and **both merge parents are reachable from HEAD**, so
   `--full-history` still sees the side the deletion came from. Two new planted arms hold both
   halves — a sibling branch's proofs are not the denominator, and a proof dropped inside a merge
   is still convicted.

Note the shape of all four: **not one of them was a wrong value test.** Every one was a wrong
*frame* — which is the same thing this session's whole assignment turned out to be about. A screen
with a correct question and a wrong scope is a screen that reports confidently about the wrong
population, and the only reason I caught the fourth is that I ran the screen a second time while
somebody else was working. **That is not a method.** It is the strongest argument I have for why
`known_limits` entries are executed rather than listed.

## Also

`bench/test_paired_ratio.py` (Niobe, main `98d5bf3`) was undeclared in the `audit_instruments.py`
bench frame, so that census read `FAIL(drift)` on main. Declared as a test module. **Niobe:** if
you mean it as an instrument rather than a caller, move it to `BENCH_INSTRUMENT_FILES` — one line.
The comment beside it is the record that nobody decided it silently.

`ci/check_artifact_frame.py` is classified in `ci/check_verification_subjects.py` as ARTIFACT:
subject = a committed reading, oracle = git. It verifies nothing against itself.

## What my verification established and what it did not

**Established.** On the repaired tree, the Linux device lane claims nodes and the Criterion-10
gate reads `MATCH` with `VulkanExecutionProvider=1` and 2 dispatches; `--check-counters` and the
ledger-portability screen both pass; Windows `--check` reads 121 entries = 121 identical with the
loss invariant at 0 missing; `cargo test --lib` is 553 passed / 0 failed and clippy is clean with
`-D warnings`. The regenerated artifacts were produced by the build named above, on the platform
named above, and now carry a frame that says so.

**Not established — and this matters for the citation Rai was asking about.** The regenerated
`pytest-linux.log` does **not** witness the corrected decline message, because on the repaired
tree there are **zero** unproven declines, so there is no decline left to carry any message. The
reading that *did* witness it was taken mid-session against the post-fix, pre-repair tree — 42
declines, all corrected, 0 old-message, `no proof ledger entry for` = 0 — and its log was
overwritten by the second run. I have not reconstructed it. So: the claim "RAI-012 is fixed"
remains **true and verified by Rai's own run**, the new artifacts witness a *different and
stronger* fact (there is nothing left to decline), and the intermediate reading is recorded here
in numbers rather than as a file. Anyone wanting the message-level evidence must re-run against
`eee65aa`'s tree, and `check_artifact_frame.py` would now stop them citing today's log for it.

**Also not established:** that `check_artifact_frame.py` catches every stale citation — it catches
staleness w.r.t. declared paths only, which is the known limit above. That the 9 remaining Linux
op-suite failures (harness_census, kv_device_residency ×2, op_table Asin/Acos, validation ×3,
wiring_census) are unrelated to any of this — I did not triage them. That no *other* field-level
reversion is hiding in the ledger: the new arm screens `source_digest`, `shader_digest` and
`toolchain`, and nothing else.

No timing claims anywhere in this work. `DEVICE_MEMORY` and `KV_ARENA` were not enabled in any
lane I ran.
