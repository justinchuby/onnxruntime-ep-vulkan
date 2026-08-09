# Niobe (Performance) — history.md

<!-- CONDENSED-AT: 50b55bcbf77e38e55b773fadc7c7d60b1d7dda05 -->

## Learnings

### [SUMMARY] Sessions 1–4: tracing, harness, producer provenance, portability, two fabricated speedups caught (2026-07-28–2026-07-30)

**Sessions 1–3 (archived):** `onnx-runtime-tracer` integrated; Vulkan span phases defined; GPU timestamp-query requirements routed to Switch; `bench/`/`docs/PERF.md` built with no real numbers yet; OQ-12 anchor `matmulnbits_q4_b32_K4096_N4096` ≥1.5x bar; `bench/transfer_calibration.py`/`bench/environment.py` built.

**Session 4 — portability envelope (2026-07-30):** Portability floor is §7.2 (16KiB shared/256 invocations), not the local GPU — Iris Xe is a UMA memory-model proxy, not a shared-memory-budget proxy. `bench/portability.py` `evaluate()→Verdict` {`portable`,`needs-fallback`,`unknown`}; every row `unknown` until the engine reports tile shape/workgroup size. UMA and discrete transfer models may not be blended. Two fabricated speedups caught and named ("the 1.70x that wasn't", "the 1.45x that wasn't" — both were CPU-vs-CPU or all-declined runs).

**Current state (end of session 4):** 50 tests passed, no real Vulkan bench row yet, all timing CPU-only and labelled. Standing rule: metric of record is the triple `(claimed_coverage, island_count, largest_island_flops)`, never one component alone.
---


<!-- SUMMARIZED by Scribe 2026-08-01T20:39:12-07:00 -- older entries condensed below; full text lives in git history -->

### [SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)

- Round 4 worktree layout note (squad/switch, squad/mouse, squad/tank worktrees, `main` integration tree).
- Turn 5 (2026-07-30) — first honest Phi-3.5 measurement, both devices; coverage figures go stale fast, read the counter not the briefing.
- 2026-07-30 evening — per-submission hypothesis FALSIFIED (`vulkan.submit` 0.6% on discrete part); two defects found by instrument.
- 2026-07-30 evening — the machine itself was the biggest uninstrumented variable; coordinator ran a quiet-vs-loaded control.
- 2026-07-30 late — two coordinator corrections checked before applying; one (device-label inversion) did NOT apply and would have broken correct rows.
- 2026-07-30 night — admissibility discipline established: whether a stored number may be quoted at all (R10/R11/R6-amendment-4).
- 2026-07-31 — falsifier was her own; only admissible number is device-clock; went after `phase_containment` RED.
- 2026-08-01 — Intel has no clock producer; obligation 8's only producer is `nvidia-smi`, so every Intel figure is permanently `UNCERTIFIED`; the 4.39x GEMV residual is a claim about Intel.
- 2026-08-01 — harness died on Tank's WARN (UTF-16LE/UTF-8 decode collision), fixed by borrowing Tank's `decode_both`. Ran interleaved A/B/A/B: host `record` 5.8x faster (load couldn't explain the shape), device unchanged (0.02%) — confirmed Switch's prediction. Set `MARGINAL_TAIL` floor (n≥8, coverage≥50%) which then refused her own "33% GPU improvement" headline as false. Numbers of record: NVIDIA 13.3432ms/inference GPU busy (STEADY, RSD 1.72%); Intel `NO_STEADY_TAIL` (53.4–91.3ms).
- 2026-08-01 — built `bench/device_state.py` (tenancy+clock companion) and `bench/win_gpu_counters.py`; ruled Intel device clock permanently uncertifiable (`none_available`); caught 3 of her own instrument bugs pre-publication, all of which would have made the reading falsely clean (R9 amendment 5).
- 2026-08-01 evening — R11 in a *selection*: a real zero (`alloc_device_authoritative_spans=0`) was correctly ruled, but `probe_indexspace.py` omitted companion keys needed to tell a measured zero from a pinned one — new rule: a counter interpretable only against a companion key isn't admissible without that companion on the same output face. Fixed 3 probes.
- 2026-08-02 — union pytest failure was a module-name collision (`device_state.py` existed in both `bench/` and `ci/`), not the prescribed `sys.path` fix (proved insufficient with an artifact); fixed by renaming to `bench/device_companion.py`. Built `bench/test_import_isolation.py` (base-name-collision screen + executed sys.path-mutation screen), caught 3 pre-existing leaks on first run.

📌 Team update (2026-08-01T17:16:56-07:00): Intel device-clock figures are permanently uncertifiable on this hardware — attack the Intel/NVIDIA residual with counts and shapes, not clocks — decided by Niobe

📌 Team update (2026-08-01T17:16:56-07:00): All wall-clock figures remain withdrawn; only counts, bytes and certified-companion device-clock figures are quotable — decided by Switch, Morpheus, Niobe, Link

📌 Team update (2026-08-02T02:03:46-07:00): `rust/src/trace.rs` had no roster owner; assigned to Niobe (timestamp calibration/trace-event arithmetic is measurement, she owns the consuming instruments) — decided by Scribe

### [SUMMARY] Three 2026-08-02 sessions: tenancy-signature refutation, first real-node run, byte-model promotion

- **[SUMMARY] PER_DISPATCH is not a tenancy signature (2026-08-02)** — Swept every cut -0.50..+1.00 for a foreign-tenancy discriminator; best J=+0.333 (7/9) — classes interleave, so PER_DISPATCH doesn't carry the distinction. `gpu_steady_tail` detects foreign work only via non-stationarity, never magnitude: a steady foreign load reads STEADY at 10.99x wrong — worse than the earlier idle-clock bias hole. `device_companion.certify` now load-bearing for two holes. Caught two of her own instrument bugs pre-publication: a 1000x unit error that changed no verdict (scale-invariant, so it survived a first reading), and a reference partly built from withheld medians (a withheld number used as a denominator is that number published). Accepted `rust/src/trace.rs`. 485 passed/336 skipped/0 failed.
- **[SUMMARY] First real-node run: refused figure, fragmentation cost (2026-08-02)** — Task 1 refused (CONTENDED); `12.1847 ms` stands, now always quoted with context length. Best specimen found by accident: a refused run's device-clock series is STEADY (tighter than all 28 census traces) under SOLE_TENANT with zero foreign GPU work, and is 20.18x wrong — host contention showing up as an idle device clock, undetectable by tenancy. Self-caught a backwards-fragmentation R11 error: divided a cumulative byte counter by iteration count across runs of different length and read a false 1.78x "improvement" that was really the iteration ratio — fixed with a slope instead of a quotient. Answer: 33 islands cost 33x host round-trips and 1.92x staging bytes, NOT allocations/dispatches/high-water. Unpriced dominant cost: 32 declined GQA nodes running on CPU — execution-location, not boundary, next thing to measure.
- **[SUMMARY] Contention is the baseline; counts become primary instrument (2026-08-02, later)** — Box is shared indefinitely; wall-clock is STEADY_UNCERTIFIED by default as policy (PERF.md §20) — no plan may wait for the box to settle. Promoted the byte model to `bench/ceiling.py`: GroupQueryAttention (the op that reads the KV cache) was DECLINED and ran on CPU, so `island_bytes_phi35.json` was charging KV bytes to a GPU roofline the GPU never incurred (48 MiB→3072 MiB across past_len 128→8192, up to 60.5% of modelled stream) — a bound on a machine not being run. Extent correctly reports UNOBSERVABLE (never 0) but discloses the declined figure. At past_len 0 the KV term is exactly zero, so the bound is admissible exactly where the one quotable figure (12.1847ms) lives — not a coincidence, the reason the 67% comparison survives. Noted harmless 0.5% divergence between her ceiling (67.4%) and Switch's (67.1%) — both correct about different quantities, flagged so nobody reads it as movement. Built `bench/clock_log.py` (continuous tenancy+clock recorder, retrospective certification, refuses rather than raises on truncated log lines) and a screen for the cumulative-counter-quotient defect class across all of `bench/`.

- **[SUMMARY] GQA claimed: re-derived the ceiling rather than flipping a flag (2026-08-02)** — Found a stale-record defect in her own module before touching it (`ceiling.py` printed a new DLL hash next to a claim status sourced from the previous binary) — fixed by hashing the DLL and raising on an out-of-frame record. Verified GQA on her own build: 355/363 claimed, MATCH. Measured the KV byte term instead of assuming it: readback = **393,216 B per past token** (ratio 1.000000), upload flat → `UNOBSERVABLE`, never 0. Her first probe version averaged the two staging axes and printed `0.0`, misread as refuting an exact term — caught before commit. New result: **at past_len ≥ 2048 the inference is transfer-bound, not DRAM-bound**. Teeth: an in-frame *declined* record still collapses the extent to `[0]` — discharging a refusal once must not wire it open. 586 passed, 3 failed.
  pre-existing by removing every file of mine and re-running (2 x Link's `ci/test_lane_checks.py`,
  1 x `tests/ops/test_harness_census.py::test_census_baseline_has_no_drift`). `ERROR(instrument): 0`.

📌 Team update (2026-08-02T22:37:04-07:00): Mouse's `mouse-counters-abi-mirror-equality` finding — `device_losses` was inserted mid-struct into `VulkanEpCounters` without a `COUNTERS_ABI_VERSION` bump; three ctypes mirrors kept the old layout and two other counters silently swapped meanings. Every counter reading you took through a ctypes mirror between `a52024f` and `4d47362` is suspect, including any of your link-bound/readback figures read that way. Mirrors must now assert exact `struct_size` equality. — decided by Mouse

📌 Team update (2026-08-02T22:37:04-07:00): Switch's `KV_CAN_STAY_DEVICE_RESIDENT` ruling changes what your link-bound-past-2048 measurement is a bound *on*: ORT permits binding an `OrtValue` in the EP's device memory as a graph output, bit-identical to unbound, so the round trip your 393,216 B/past-token figure describes is not a runtime limitation — it is `transfer.rs`'s own host-staging-authoritative invariant, now fixed as a per-span flag. The round trip is moved (paid whenever a caller asks for host bytes), not yet removed; your figure remains the correct measurement of the shipped (unbound) path, but it is no longer evidence of an ORT-side ceiling. — decided by Switch

📌 Team update (2026-08-03T04-55-00-07-00): Switch measured on the real Phi-3.5 graph (64 KV outputs, 6-step chain) that the shipping (host) KV lane OOMs at past context 4096 on the 8 GB discrete GPU while the device-resident lane completes — the round trip is a VRAM cost, not only the bandwidth cost your link-bound figure names. Also: ctx 8192 fails on both lanes, so the operating point the 82.2%-of-traffic KV figure is quoted at has never been reached on this hardware — bounds what your link-bound-past-2048 measurement can be extrapolated to at longer context. — decided by Switch
📌 Team update (2026-08-03T19:55:00-07:00): Switch's GLSL.std.450 table correction — the SIMT interpreter's extended-instruction table was wrong at slots 30/37/40/43. The silent-miscompute set is {37, 43} → celu, hardsigmoid, hardswish; a 
elu would NOT have been silently miscomputed (Switch corrected his own earlier headline on this point) — the actual discriminator is operand count, not op identity. Your weight-amplification measurement (re-derived at 1.000000 with three positive controls) is untouched by this and needs no re-run — it does not exercise the affected instruction slots. — decided by Switch
## 2026-08-03 night — the paired ratio, attacked and refused (`squad/niobe`)

**Task:** make "is this a high-performance EP" answerable, by testing whether a paired,
interleaved A/B ratio escapes §20's refusal. **Result: it does not, and the refusal now has an
instrument.** `docs/PERF.md` §24, `bench/results/probe_paired_ratio.py`, three records,
`bench/test_paired_ratio.py` (19 tests).

- **I built the instrument to attack the proposal, not to use it.** The six phases exist so the
  *apparatus* is priced (`solo` -> `blocked` -> `paired`) before any injection runs. That ordering
  is what found the killer: the apparatus perturbs the two arms unequally (1.64x vs the CPU EP,
  2.13x vs our own resident lane) **before any foreign load exists**. A ratio published from the
  paired phase would have carried that factor as if it were the EP.
- **The surprise: foreign GPU work made our arm faster.** `vk_lift_x = 0.771`. Interleaving with a
  ~300 ms CPU step idles the board to 825 MHz; a co-tenant holds it at 2475 MHz. §20.2's mechanism
  running backwards. **The interleaving granularity that makes a foreign episode symmetric is the
  same one that manufactures a device-axis asymmetry**, and a decode step is atomic, so there is no
  finer granularity to retreat to. This is not tunable; it is a property of pairing a GPU arm
  against a slower arm on a boost-clocked board.
- **Host contention flatters us.** Against the CPU EP the ratio improves ~1.8x purely because the
  box got busy. A number taken in a loud hour would look better than the same number taken quiet.
- **Pairing bought 1.30-1.44x variance reduction**, not the 5x it is adopted for. §10.3's 2.65x was
  the right number to size with: the cross-device form needs ~350 pairs for +-5%, and I took 72.
- **Side finding worth more than the ratio:** under 19 spinners the host-KV lane inflated **8.67x**
  while the device-resident lane inflated **2.78x**. The 393,216 B/past-token round trip is
  *host-contention* sensitive, not only link sensitive. New axis, found by accident.
- **The thing I nearly published:** the same-device ratio is 1.185 (+-4.4%) as found — and 1.081 /
  1.480 / 3.701 under three other box states. **A 3.4x swing driven entirely by what else was
  running.** Quoting 1.185 as "the KV round trip costs 19%" would have been exactly the Fact
  Checker's diagnosis of my long-lived errors: a pre-formed number re-quoted rather than
  re-derived. §24.7 is a table for that reason, and the table is the quotable unit.
- **Confirmed Switch myself rather than on trust**, and against the shipped SPIR-V rather than the
  GLSL: `spirv-dis q_gemv_matmul_nbits_f16.spv` -> eleven ext-inst, all `PackHalf2x16` (58) /
  `UnpackHalf2x16` (62); the f32 variant issues none. Neither is in `{30,37,40,43,50}`. The
  1.000000 amplification needs no re-run.
- **Trap paid for twice:** `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` must be set **before**
  `register_execution_provider_library`. Set after, the EP registers no DEFAULT allocator and the
  resident arm returns `ERROR(instrument)`.
- Intel: `clock_producer: NO_PRODUCER` in every phase, so the confound that turned out to *be* the
  story is unobservable there in principle. Permanently `UNCERTIFIED(partial_companion)`.
- **The original question is still unanswered, on purpose.** §24.12 says what it would take.

📌 Team update (2026-08-04T12:25:00-07:00): Trinity found `np.spacing` returns `inf` at fp16's
largest finite value, so a 504-unit error read back as `0.0` ULP — strictly worse than the previous
two instrument defects, because those made a sound residual look wrong and this makes a wrong
residual look sound. Relevant to any ULP figure you compute near the fp16 max-finite boundary.
— decided by Trinity

📌 

📌 Team update (2026-08-04T20-25-00-07-00): Switch -- "the 6144/8192 figures were never the shipping lane's" -- the boundary for the ctx-4096 KV-arena blocker was 2560, not 4096; the 6144/8192 figures came from a non-shipping measurement path and do not belong to the shipping-lane record. Flagging against your roofline and context-length record. -- decided by Switch

## 2026-08-07 — PR #57 revision 3 (issue #55), sole ownership

Morpheus rejected revision 2 at `15e20a8`. Link (author) and Trinity (revision 1) were both
locked out; I owned revision 3 alone, in the dedicated worktree, `origin/main` merged normally.

What the measurements said, before any code was written:

- **R1** was not a redaction defect at all. `REJECTED_REF=d5bab5d` is unreachable the moment the
  PR squash-lands, so the control's only non-planted evidence dies with the branch it guards.
- **R2** reproduced: `_echo_cmd` printed `mirror.example/pypi/simple?token=<sentinel>` raw
  while the record seam redacted the identical input.
- **R3** reproduced: `C:\Users\justin.chu@contoso.com\...` → `REDACTED@contoso.com\...`. A fuzz
  over the same regex found 13 non-idempotence counterexamples, all from that alternative.

The lesson worth keeping: **R2 and R3 are the same defect seen from two sides.** A shape-based
scanner tight enough not to corrupt `first.last@corp` paths cannot see a schemeless credential
URL, and one loose enough to see it corrupts them. No regex resolves that. What resolves it is
knowing the value: redact **by value**, then **by literal**, then — only as a backstop — **by
syntax**, with syntax requiring an explicit `//`. Every echo, error and record surface routes
through one function, and an AST test enforces that the general scanner has exactly one caller.

The second lesson: **the negative control found five holes in my own suite that 350 green tests
did not.** Three PLANTED arms came back GREEN (the `://` echo gate, the echo losing the run's own
URL, truncate-before-scrub) because defence-in-depth was masking them, one because the literal pass
was covered by the value pass, and one REPLAYED arm did not reproduce — the spelling I picked
was one revision 2 actually handled. Each was a real assertion I had not written. Write the control
before believing the suite.

Evidence is now **content-addressed**: `ci/fixtures/cleanroom-redaction/*.pysrc` + a sha256
manifest, refused on mismatch, with a LANDING arm that runs in a tree with no `.git` and
`ci/simulate_squash_cleanroom_redaction.py` that proves it against a real squash with the reflog
expired and the objects pruned. 12/12 replay arms fire with the rejected refs gone.

---

## Issue #69, revision 4 — clean-room rebuild of the Phi-3.5 evidence (rejected #102)

Morpheus rejected `c39bdbb` on PR #102. Link and Morpheus were both locked out, so I owned this
revision alone, in a fresh sibling worktree branched from exact `origin/main` (`8701812`). I never
read #102, #95, or any of their commits, branches or worktrees — everything below was measured or
written from `origin/main` and issue #69.

Three blockers had to be answered independently, and answering them honestly changed the design:

- **Decode is INCONCLUSIVE, and stays that way.** Two independent observations exist — `0.859x`
  and `0.9651x` (95% CI `[0.820, 1.136]`, power `0.346`). Neither supersedes the other. My own
  fresh reproduction landed a third, `0.999x` with a floor of `0.903x`, and the temptation was to
  let the friendlier number stand in for the older ones. The gate now refuses that by name:
  `REQUIRED_DECODE_OBSERVATIONS`. I found this because the negative-control arm that *deletes* one
  named observation came back GREEN — my third observation was quietly covering for it.
- **An RTX A1000 is not lavapipe.** The record carries driver name, driver version, API version,
  device type, UUID, LUID, PCI address and device count, and the gate fails
  `vulkan_implementation_mislabelled` if a discrete adapter is described in software terms.
- **`load_frozen()` validates nothing.** It reads and parses; identity checking lives in
  `verify_frozen_identity()`. The docstring says so and a behavioural test proves it by feeding
  `load_frozen()` a tampered artifact and asserting it returns happily.

What the instruments caught that 47 green tests did not:

1. **Calibration was not disjoint by content.** `prefill/M1/past0` and `decode/M1/past0` feed
   byte-identical tensors — same M, same empty past. The band was counting one measurement twice.
   Subject *labels* were disjoint; the *feeds* were not. The gate now compares
   `feeds_digest_by_subject`, and I re-measured the whole sweep rather than keep the first sample.
2. **The bench instrument census keys rows by bare function name.** My `gate` collided with
   `bench/contention.py::gate` and flipped that row to `screened` without a single test touching
   it. I renamed mine to `evidence_gate`. The collision class is pre-existing (`load`, `describe`,
   `audit` already collide) and belongs to the census owner, not to me.
3. **A verdict-returning instrument had no reject polarity.** `refuses()` wants a raise; my gate is
   total and returns `FAIL`. Rather than baseline twelve `unfalsified` rows I added
   `bench/_polarity.py::convicts`, registered it in `VALUE_REJECT_FN`, and gave it synthetic
   two-polarity tests of its own.

The lesson worth keeping: **the strongest result in the sweep is the one you must be most careful
not to widen.** M=128 came back at `2.103x` and M=64 at `1.597x` — better than expected — and the
headline is still one model, one prefill family, one adapter, one box. No compatible CUDA result
exists, so #69 stays open and the PR says `Refs #69`. The gate has a `headline_scope_widened`
condition precisely because the pressure to generalise arrives *after* a good measurement, not
before it.

## Issue #69, revision 5 — second-round review findings, same clean room

Eight further exact-head findings arrived against the rejected #102. I never opened it; I treated
each as a falsifier to satisfy independently on my own branch. What that cost, and what it bought:

- **Per-record provenance now fails closed, and it caught a real one.** Every raw run carries the
  loaded DLL's hash, the model resolver, external-weight metadata, device name, shader hashes and
  count, dispatch count, the providers that actually registered, and typed agreement pairs the gate
  *recomputes* from their own recorded sides. On the first re-sweep the gate REFUSED my own fresh
  artifact: `vulkan_pre72/prefill/M128/past0/r1` had registered only `CPUExecutionProvider` and
  executed zero dispatches, and had timed at 1473.8 ms against 3036.3 and 3020.6 ms either side.
  Averaged in, that silent CPU fallback would have understated the baseline by about a quarter and
  inflated every prefill ratio. The instrument caught it on live data, not in a mutation test.
- **The fix had to be structural, not statistical.** `_vulkan_actually_ran()` reads which providers
  registered and whether a dispatch happened; it never reads a timing, so it cannot select for a
  result. `_run_worker_insisting()` retries and discloses every refused attempt *with its samples*,
  and "nothing was discarded" is written as an empty list, not an omission.
- **A content digest cannot bind bytes.** A CRLF-translated artifact re-serialises to the same
  content digest, because the parse already threw the difference away. The sidecar seal binds exact
  bytes *and* exact length, hashed as read, checked before the parse. Length is compared first so a
  truncation reports as a truncation.
- **A red lane can leak a green number.** Refused rows are now sanitised — no timing, ratio,
  separation, speedup, band edge or lower bound survives — and the sanitation check had to move
  *earlier* in the gate than every check that recomputes a number out of a row, or a stripped row
  convicts on the wrong token. That ordering bug was found by the negative control, not by me.
- **Say what the code does, not what you wish it did.** The loader strips superseded blocks; it does
  not refuse them, so the docs say strips and a behavioural test pins the layer that actually
  refuses. The proof ledger has production registry, disclosure and pipeline-audit consumers, so it
  is not diagnostic-only and is no longer described that way.
- **A verdict is a reading against a band, not a fact about the world.** M64 is only indeterminate
  against the band that was committed; under a 3% band it classifies FASTER. The gate recomputes the
  readings under the alternative bands, which is what makes "indeterminate at any band" unassertable
  rather than merely discouraged.

This sitting's box was noisier than the last: calibration came back `[0.739547, 1.363869]` against
`[0.9045, 1.0467]` before. I disclosed the wide band rather than re-rolling until it narrowed. A
wide band can only suppress a verdict, never manufacture one — M32 fell to INDETERMINATE because of
it — and repairing it by selection is precisely how a band stops being evidence.

