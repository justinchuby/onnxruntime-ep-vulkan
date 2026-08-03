# Trinity (Test-Conformance) — history.md

## Learnings

## Round 27 — the union defect: a required parameter and a caller that was not on my branch (2026-08-02)

**Branch:** `squad/trinity` @ `1ccbfef`, then merged `main` @ `79674d6`. Not pushed.

### What broke
`TypeError: _run_counters_child() missing 1 required keyword-only argument: 'guard'` in
`tests/ops/test_wiring_census.py::test_ledger_lookup_wired`, on both devices. I added the
required keyword-only `guard` on my branch; Mouse added the caller on his. Both green alone.

### The fix, and the decision inside it
`guard=None`, where **None means build one, not run unguarded**. `None -> unguarded`
would satisfy "callable without a guard" while silently undoing round 25: every future caller
opts out of stall detection with nothing to notice. An ambient guard loses only *bookkeeping*
(its costs never reach the census-wide ledger, so they are absent from `observed_units`),
never protection. Both halves carry falsifiers because either alone is satisfiable by the
wrong fix. `test_stall_guard.py`: **16 passed**, deterministic, no GPU.

Same shape one line down: `_BUDGET_UNITS[label]` -> `_budget_for(label)` with a generous
default. Safe because the unit is *work*: a loose work budget detects a hang later in work,
never not at all, and contention cannot stretch it. That is not true of a wall clock.

### tests/union_check.py — and only tier 3 is a gate
Tiers 1 (file intersection), 2 (`sys.path.insert(0,)` in a colliding directory) and 2b
(module-basename collisions, computed from the working tree) are **preconditions**, never
gates — R9 A5: a union can break with no intersecting file and no scanned side effect, so
their silence is a statement about the tool. It prints
`PRECONDITION(tiers=1,2; tier 3 not run)`, never a bare PASS.

Tier 2 was 21 findings and useless at first. The signal was never the insert — it was the
**population of colliding basenames**. Against pre-fix history tier 2b reports
`COLLISION device_state.py <- bench, ci` and names `bench/device_state.py` and
`ci/check_lane_inventory.py`: it retrodicts the Niobe x Link incident. Against current main
(after Niobe's rename) it reports `none`. Same code, different tree, different report —
the R10 falsifier for the tool itself.

Tier 3, both arms, real history, same target, same device:
- broken union `2fee5ef..c55a389` -> `FAIL(condition=union_red)`, exit 1, quoting the TypeError
- repaired union `main..squad/trinity` -> `PASS`, exit 0, `ERROR(instrument): 0`

Conflicts confined to `bench/results/` artefacts resolve to HEAD under
`--resolve-artifacts`; anything else stays `ERROR(instrument=merge_conflict)` — an
instrument that cannot construct its subject has not observed it. Scratch worktree lives
under the repo's parent (not Niobe's `bench/results/`) and is removed in a `finally`;
`git worktree list` shows no leak.

### The near-miss I want on the record
My first post-merge census read `proven_key_lookups == 0` / `ledger_entries=0` and I
almost filed "the S8.9 ledger is UNWIRED". The cause was a **stale
`rust/target/release/onnxruntime_vulkan_ep.dll`** predating Mouse's ledger code. After
`cargo build --release` it passed. This is Mouse's exact failure mode from earlier today,
and in a shared worktree (`trinity` and `trinity-1` map to the same directory) it is the
default state after any merge. **Rebuild after merging** is now part of my standing
obligation, not an optional step.

### Verification on the fully merged tree
- `test_stall_guard.py` — 16 passed.
- `test_wiring_census.py` dev1 (Intel) — 3 passed, `ERROR(instrument): 0`.
- `test_wiring_census.py` dev0 (NVIDIA) — 3 passed, `ERROR(instrument): 0`.

No timing figure is quoted as a claim of record (S10.0 obligation 8).

### Recommendation to the coordinator
Standing obligation: **merge `main`, rebuild, then `python tests/union_check.py --run`**
before reporting a branch done. Targeted rather than full-suite, deliberately: the full suite
is stronger but the targeted form is the one people will actually run, and it caught this.

### Routed, not edited
`docs/DESIGN.md` S10.0 (~line 2781) shows an `attribution_witnesses` example with two keys
where the record now emits six. Morpheus's file.

Decision record: `.squad/decisions/inbox/trinity-union-check-and-guard-default.md`.

📌 Team update (2026-08-02T02:03:46-07:00): Morpheus named R12's fourth generalisation — for a test result, the frame is the binary that ran it — from two of Mouse's self-caught near-misses this session: a build in a shared worktree that linked a sibling's in-flight file (nearly reported as a false ALL-DECLINED finding), and Copy-Item preserving LastWriteTime, which let cargo silently keep running a mutated binary after a restore-from-backup. Your own union-check work independently reproduced the same failure shape (a stale DLL in a shared worktree nearly read as an UNWIRED §8.9 ledger) — worth checking any mutation harness you build touches or hashes a restored file and asserts the rebuild happened before reading a result as a control. — decided by Scribe

---

## Round 28 — criterion 11(c), and which `model_output_equivalence` is of record (2026-08-02T03:10-07:00)

### Standing obligation, executed first
Fetched, merged `main`, then **rebuilt and hashed either side** — R12 generalisation 4, which
Morpheus minted from my own round-27 stale-DLL scare. `3B0C0ACD7AA7...` -> `8D07173F8AE5...`,
`rebuilt=True`. No reading was treated as a control before that assertion passed.
The merge aborted on regenerated `bench/results/wiring_census-dev{0,1}.json` and an untracked
`bench/results/census/`; both are outputs, discarded, merge clean.

### Criterion 11(c): `ledger_hits` moves with its input
Three tests in the census lane, not behind `#[ignore]`:
`test_ledger_hits_moves_with_its_input`, `test_ledger_key_discriminates_optional_inputs`,
`test_ledger_digest_refusal_is_in_the_lane`.

I **probed nine arms before writing a single assertion**. An assertion written ahead of the
reading is a test of my guess, not of the mechanism — and it mattered here, because my brief
was wrong about which control moves the counter.

The **shape-class arm is load-bearing**: static `mul_f32` vs dynamic-extent `mul_f32` is the
same op, dtype, optional-input set and one-node enumeration; only the key's `shape_class`
changes, and `ledger_hits` goes `1 -> 0`. The enumeration is identical work in both arms, so a
counter derived from the enumeration could not have moved. That is the falsifier.

The **MatMulNBits `scales` / `scales+zero_points` pair does NOT move `ledger_hits`** — both
proven, both `HIT`. Reporting it as a mover would have been R11 exactly. Its real content is
that they must be two keys, asserted on the key string: `differing == [2, 5]` over the six
components; index 5 is the populated-optional-input set (the 2026-07-30 all-zero-logits defect).

The **digest-identical arm is the control**; without it every other assertion would pass
against a check that rejects everything. Its failure classifies `ERROR(instrument)`.

**Both arms demonstrated** — `probe_ledger_mutations.py`, three mutations, all CAUGHT on
**both** devices. Quoting failure text, not counts (R13).

`ledger_lookup` promoted into `_MANDATORY_WIRED`; `_KNOWN_UNWIRED_M0` emptied (kept as a seam).
A ledger that stops being consulted now fails the lane.

### `model_output_equivalence`: `MATCH` beside `UNMEASURED`
Not two sources disagreeing. The EP has no CPU oracle; `to_json()` defaults the field. The
nested value is **a field nobody set**. The mechanical tell needs no judgement:
`write_equivalence_record` writes the token and the record in one call, so a token with no
record beside it was never written by a comparison — and in the phi35 artifacts the record key
is absent. The rule keys off **record presence, not token value**: keying off
`token == "UNMEASURED"` would read the field under suspicion, and would mislabel a genuine
comparison that legitimately concluded `UNMEASURED`. Both polarities asserted.

It did not "go" null — the phi35 bench path never calls `write_equivalence_verdict`, so that
copy has always been the default. Per R12 the reconciliation is `UNOBSERVABLE` **with a reason**;
`agreement = (outer == inner)` would have read `DISAGREE` and sent a reader after a
contradiction that does not exist.

**R9 A5 turned on my own check.** My first run flagged ~20 historical artifacts. Frozen,
unowned, non-regenerable — a permanently red gate gets loosened, so I demoted them to a printed
`PRECONDITION` and scoped the gate to the 4 certified files. Two *genuine* contradicting
readings stay a finding everywhere.

### Verified
Intel: census + equivalence + stall-guard 26 passed, `ERROR(instrument): 0`.
NVIDIA: census 6 passed, `ERROR(instrument): 0`. Mutation probe PASS both devices.
Stale-stamp falsifier fired with the right text; artifact restored.
`audit_instruments.py --check`: `CENSUS VERDICT: PASS`.

### Routed, not edited
`bench/phi35.py` should stamp `model_output_equivalence_authority` at write time so it survives
regeneration — **Niobe's file**. I stamped the four existing certified artifacts by hand.
`docs/DESIGN.md` S10.0 attribution_witnesses example still shows two keys where six are emitted
— Morpheus's file, still open.

Decision record: `.squad/decisions/inbox/trinity-criterion11c-and-equivalence-authority.md`.

### Addendum — a defect the 11(c) arms introduced, caught by their own artifacts
The tracer witness path is shared across census arms. A faulted-ledger arm dispatches
nothing by construction, so it unlinked the clean arm's tracer file and wrote none — a
**false UNWIRED available purely by running later**, invisible in a green lane and visible
only as a tracked artifact missing from `git status` after six passing tests. Fixed with
`trace=None` defaulting to the historical behaviour (no required kwarg — round 27's `guard`
lesson). Both arms shown: absent before, present on both devices after.

### union_check: FAIL(condition=union_red), and the condition is not mine
153 failed / 277 passed on the merged tree. Attribution before alarm: `git diff main --
rust/ evidence/ ops/` is empty, and the same subset scores **21 failed / 18 passed
identically** in main's own checkout with main's own DLL. The decline text is
`[unproven] no proof ledger entry for ...`. So main is red on its own — the ledger gate is
enforcing against a 9-entry ledger — and that is Mouse's ground, reported not touched.
R13: quote the failure text, and an inherited condition is not a detection about my branch.

Commit `e39bdd6` on `squad/trinity`, not pushed.

---


## [SUMMARY] Compressed entries

<!-- SUMMARIZED by Scribe 2026-08-02T15:26:03.394019 -->

- **[SUMMARY] Rounds 1–17: CI, barriers, oracle, claim log, Phi-3.5, env var finding (2026-07-28–2026-07-30)** — **Rounds 1–5 (archived):** Differential harness (EP vs CPU). Profiling-JSON claim assertion. Barrier parity layer. Two lavapipe CI lanes (Linux and Windows). ORT CPU EP quantised oracle with `accuracy_level=1`; fp16 gate
- **[SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)** — - **Round 19 (2026-07-30T20:33:50-07:00) — paired controls, validation lane, wiring census** — **Trigger:** Morpheus's ruling: criteria 4 and 5 "partially met — unchanged, and deliberately untouched by today," on five co
- **Round 29 — criterion 12: the twelve surfaces no census mechanism watched (2026-08-02)** — `unwired: []` was measured against a denominator the census supplied itself. Link's
- **Round 30 — per-output attribution: the fifth costume (2026-08-02)** — The coordinator's concern, and it was right. `attributed` was a property of the SESSION;
- **Round 31 — ORT refuses the session we spent the session learning to detect (2026-08-02)** — **Handed to me:** the user asked whether ORT has a flag preventing EP fallback. It does:
- **Round 32 — the specimen stopped being a specimen (2026-08-02, merged `a0bd22d`)** — Merged `origin/main` (18 artifact conflicts, all regenerable — took main's), **rebuilt**:
- **Round 29 — the two unfalsified guards, screened in the always-on lane (2026-08-02T11:40-07:00)** — Merged `main` (`4e70601`), rebuilt, hashed either side: `D45B3A8C8C2B...` -> `918E8FF56B2E...`,
