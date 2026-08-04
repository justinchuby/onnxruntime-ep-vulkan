# Niobe (Performance) — history.md

## Learnings

### [SUMMARY] Sessions 1–4: tracing, harness, producer provenance, portability, two fabricated speedups caught (2026-07-28–2026-07-30)

**Sessions 1–3 (archived):** `onnx-runtime-tracer` integrated. Vulkan-specific span phases defined. GPU timestamp-query requirements routed to Switch (`DeviceInfo` must carry `timestamp_period_ns`, `timestamp_valid_bits`). `bench/` and `docs/PERF.md` built with no performance numbers (no kernel executed at that point). OQ-12 anchor: `matmulnbits_q4_b32_K4096_N4096`, ≥1.5× bar measured at that case only. `bench/transfer_calibration.py` sweeps doubling byte staircase, fits fixed+bandwidth model, prints paste-ready Rust literal. MVS constants replaced per device via review. `bench/environment.py` stamps OS/CPU/ORT/EP/Vulkan/env-vars at run start.

**Session 4 — portability envelope (2026-07-30):**
- **D-N24** — Portability floor is §7.2 (16 KiB shared, 256 invocations), not the smaller local GPU. Iris Xe is UMA proxy for mobile memory model, not for mobile shared-memory budget. A 32 KiB tile passing on Iris Xe is not portability evidence.
- **D-N25** — `bench/portability.py`: `evaluate(Configuration) → Verdict` in {`portable`, `needs-fallback`, `unknown`}; `quotable_as_ep_behaviour` true only for `portable`. Every row is `unknown` today (engine does not report tile shape or workgroup size). `fits_device(config, shared, invocations)` uses reported limits, not constants.
- **D-N26** — UMA and discrete transfer models may not be blended. `portability.transfer_model_merge_refusal()` closes the obvious path.
- **D-N27** — Routing to Switch: engine must report `tile_config`, workgroup size, shared-memory bytes, and memory path (UMA mapped write vs staging copy). Until then, portability column is honest but empty.
- **D-N28** — Two fabricated speedups caught: 1.70× (ORT 1.27 prints failure without raising; result was CPU vs CPU); 1.45× (EP loads under ORT 1.28, declines everything — all "vulkan" columns are CPU EP). Neither claimed. `bench/README.md` records both as "the 1.70× that wasn't" and "the 1.45× that wasn't."

**Current state:**
- `pytest bench/` — 50 passed (11 new portability tests).
- No real Vulkan bench row yet — no kernel has executed through the bench harness.
- All timing rows are CPU-only, clearly labelled.
- First quotable Vulkan row: after Switch reports tile_config + workgroup size in `DeviceInfo`, and `dispatches_executed > 0` in counters file.
- `SUBGROUP_SIZE_IS_GUARANTEED=False` constant present — both local GPUs happen to report 32, not a guarantee.
- Standing rule: metric of record is the triple `(claimed_coverage, island_count, largest_island_flops)`, never any component alone.
---


<!-- SUMMARIZED by Scribe 2026-08-01T20:39:12-07:00 -- older entries condensed below; full text lives in git history -->

### [SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)

- **📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)** — ### Worktree layout and inbox portability constraint The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree.
- **Turn 5 — 2026-07-30 — the first honest measurement (Phi-3.5, both devices)** — ### Coverage figures go stale fast — read the counter, never the briefing I was told 161 claimed nodes.
- **2026-07-30 (evening) — the hypothesis died, and two defects found by instrument** — ### My per-submission hypothesis is FALSIFIED `vulkan.submit` is **0.6%** on the discrete part.
- **2026-07-30 evening — the machine is an uninstrumented variable, and it was the biggest one** — ### What happened The coordinator captured a trace while six agents were compiling, then **ran a control**: same device, same build, same test, quiet vs loaded.
- **2026-07-30 late — two corrections from the coordinator; I checked both before applying them** — ### The one that did NOT apply to me, and applying it would have broken correct rows Coordinator: "device labels are inverted, re-label any stored result." True of the world (`select_device` indexes the **best-first sorted** list, so selector 0 = NVIDIA, 1 = Intel; `instance.rs:5
- **2026-07-30 night — admissibility: whether a stored number may be quoted at all** — **Context.** Standing directive from Justin that performance is now continuous and first-class, plus new rules R10/R11/R6-amendment-4.
- **2026-07-31 (evening) — the falsifier was mine, and the only admissible number is a device-clock one** — Merged `origin/main` (77d5d2a), rebuilt, and went after the RED `phase_containment` Morpheus's run hit.
- **2026-08-01 (later) — Intel has no clock producer, and half a companion is not one** — The task: obligation 8's only producer is `nvidia-smi`, so **every Intel figure is permanently `UNCERTIFIED`** — and Intel is where the open question lives, because the 4.39× residual of the 13.52× kernel gap that memory bandwidth does not explain is a claim about Intel.

- **2026-08-01 — the harness died on someone else's WARN, and the A/B ruling that cost me my own headline:** Merged local `main` (not `origin/main`, which lacked Switch's barrier fix and Tank's WARN); a UTF-16LE/UTF-8 decode collision on Tank's WARN line crashed the harness on its own error path (fixed by borrowing Tank's `decode_both` rather than writing a third decoder — bytes captured, decode moved out of the reader thread). Machine never quiet (4.7–11.4 foreign cores); ran interleaved A/B/A/B instead of before/after: host `record` 16.4/20.3ms→3.78/3.69ms (5.8×, load could not explain the shape), device 13.3463 vs 13.3432 ns (0.02%, no barrier-driven device movement) — confirmed both halves of Switch's prediction. Set the `MARGINAL_TAIL` floor (`n>=8` AND `coverage>=50%`) after finding two of my own bad tails (5-sample, 12% coverage reading +42% high) — its first act refused my own "33% GPU improvement" headline, which the floor proved false (0.02% agreement). Retired 40.201 ms as a baseline (survives only as a regime, not a pairable measurement — two upper bounds don't bound a difference from below). Numbers of record: NVIDIA 13.3432 ms/inference GPU busy (STEADY, RSD 1.72%); Intel `NO_STEADY_TAIL` (wanders 53.4–91.3ms, iGPU shares power budget with loaded CPU); wall-clock and Vulkan/CPU ratio withheld both devices (`CONTENDED`).
- **2026-08-01 (reconstructed) — device-state companion, and Intel's clock is permanently uncertifiable:** Built `bench/device_state.py` (tenancy verdict + clock record, only `QUOTABLE` releases a number) atop Switch's GPU sampler. Self-caught two near-misses: published a `MARGINAL_TAIL` refusal against my own lowest-RSD reading (0.086%) to prove refusal isn't a claim the number was wrong; reproduced Switch's own PID-timing bug at my call site (`Popen`+`on_start` callback fix). Then built `bench/win_gpu_counters.py` (WDDM `\GPU Engine` PDH sampler, tenancy-only, clock always `UNOBSERVABLE`) with a negative-control design (samples every other live adapter too). Ruled Intel device clock **permanently uncertifiable** (`none_available` — no MHz-carrying counter set, no WMI class, no `nvidia-smi` equivalent, engine-duration is same-source-tainted); told Switch to attack the 4.39× residual with counts/shapes, not clocks. Caught 3 of my own instrument bugs before publication (device-index swap, ancestry-cost O(n), PDH's cached-instance-list blind spot for a just-started process) — all three would have made the reading falsely *clean*, per R9 amendment 5 (ask which way a check moves when its subject is wrong).
- **2026-08-01 (evening) — R11 in a *selection*:** `alloc_device_authoritative_spans = 0` in an artifact was a real zero (Morpheus ruled correctly — type discipline, companion counters, and ceiling arithmetic all agreed), but `probe_indexspace.py`'s `KEYS` list omitted the companion keys (`alloc_staged_spans`, `alloc_device_authoritative_ceiling`) needed to tell a measured zero from a pinned one — the defect was the probe's selection, not the counter. New rule: a counter interpretable only against a companion key isn't admissible without that companion on the face of the same output. Fixed 3 probes (2 more instances found in `probe_sec65.py`, including a key that has never existed in source and printed `'<absent>'` on every run). Re-ran: verdict unchanged (confirms this was a reporting fix, not a measurement change).

- **2026-08-02 — the union failure was a name collision, and the prescribed fix would not have fixed it:** `bench/test_marginal_tail_withholds.py` broke Link's `ci/test_lane_checks.py` in one pytest process (3 failures, each file green alone). Rejected the given diagnosis (unrestored `sys.path.insert`, fix `monkeypatch.syspath_prepend`) — the errors were `AttributeError`, not `ImportError`: two files were both named `device_state.py` (mine in `bench/`, Link's in `ci/`); `sys.modules` resolves before `sys.path`, so whichever imported first bound the name process-wide, and proved the prescribed fix insufficient with an artifact (`sys.path` restored byte-identically, wrong module still resolved). Fixed by renaming mine to `bench/device_companion.py` (8 sites repointed; Link's file untouched). Built `bench/test_import_isolation.py`: a base-name-collision screen (found this was the repo's only duplicate) + an executed (not grepped) `sys.path`-mutation screen, which caught 3 real pre-existing leaks in unrelated files on its first run. Flagged (not fixed, his call) that `ci/device_state.py` and `bench/device_companion.py` cover overlapping subject matter. 402 passed / 321 skipped / 0 failed across `bench/ ci/ tests/ops/` in one process.

📌 Team update (2026-08-01T17:16:56-07:00): Intel device-clock figures are permanently uncertifiable on this hardware (`none_available`, no producer exists and none of the available proxies are the right kind of quantity) — attack the Intel/NVIDIA residual with counts and shapes, not clocks — decided by Niobe

📌 Team update (2026-08-01T17:16:56-07:00): All wall-clock figures remain withdrawn; only counts, bytes and certified-companion device-clock figures are quotable — decided by Switch, Morpheus, Niobe, Link

📌 Team update (2026-08-01T17:16:56-07:00): `ledger_lookup` is the last `UNWIRED` mechanism in the instrument census (criterion 11); Mouse is building it — decided by Trinity, Mouse

---

📌 Team update (2026-08-02T02:03:46-07:00): ust/src/trace.rs had no roster owner (flagged by Link). The coordinator assigned it to Niobe — timestamp calibration and trace-event arithmetic are measurement, and she already owns the instruments that consume it. Recorded in 	eam.md's new File Ownership Notes section. Tank may have the stronger claim on counters/FFI grounds and may object; reassignment is a one-line change if so. — decided by Scribe

---

## 2026-08-02 — PER_DISPATCH is not a tenancy signature; the tail's second hole is bigger than the first

**Merged** `origin/main` (`57d7018`). No measurement taken and no binary exercised — everything
recomputed from committed artifacts, so no figure here has a binary frame and I said so rather than
rebuild to imply one. Six agents running; my own gate would refuse a fresh run.

**Q1 — does `PER_DISPATCH` track witnessed foreign tenancy? No.** The two traces I was handed both
hold (`contended3` FOREIGN/PER_DISPATCH, `ab_p1_long` SOLE/SUBMISSION_LEVEL). Nine device-0
traces carry both a localisation and a committed `gpustate_*.json`. Sorted by
`explained_by_level` the three witnessed-foreign traces sit at the extreme bottom (-0.2265) **and**
near the extreme top (+0.8408, +0.8638). `contended` — `foreign_sample_fraction 1.0`, the most
disturbed trace in the census — is more submission-level than four of six sole-tenant traces. The
two false positives are `base_b` and `baseline_certified`, i.e. the two idle-clock 21.4x
specimens: a guard on this signal fires on the runs whose defect was clock, not tenancy.

**Swept every cut from -0.50 to +1.00 rather than report failure at one.** Best achievable J =
+0.333 (7/9). The classes interleave, so it is a signal that does not carry the distinction, not a
boundary in the wrong place. Published the whole table, chose nothing — threshold-episode discipline
a second time. Recorded the qualification against my own conclusion: n=9 is small.

**Q2 — the level moves, the verdict does not follow.** Against a sole-tenant boosted reference of
**11.5248 ms**: `contended` 11.7697 (1.021x), `ab_p0_r1` 20.6159 (1.789x), `contended3`
truncated **126.6465 (10.99x)**. So not level-blind to contention the way it is to bias. But the
verdict is dispersion, and a sustained steady foreign load is steady: contended3 truncated to
20/28/34 publishes STEADY at 126.6 ms with RSD 0.79–0.91%; full length refuses. **The refusal is a
property of run length, not sensitivity.** Ruling: *`gpu_steady_tail` detects foreign work only
through its non-stationarity, never through its magnitude; a steady foreign load is
indistinguishable from a slower GPU.* Strictly larger than the bias hole — bias needed a board at
idle clock (rare, recoverable from magnitude at 21x regime separation); a sustained tenant is
ordinary and wrong at any magnitude.

**Q3 — both directions, and they are not in tension.** All four contended3 readings, including the
three confident STEADY publications, are `WITHHELD` by `device_companion.certify` on
`foreign_sample_fraction 1.0` — evidence from outside the series. The companion is load-bearing
for two holes now, not one. Also: `contended` was a confident STEADY at n=8/2.1% high in the
retraction and now grades MARGINAL_TAIL — my floor working — **and** the clearest proof the floor is
necessary but not sufficient, since contended3 truncated satisfies every floor and publishes 10.99x
wrong.

**Two ERROR(instrument) of my own, both caught before publication, both recorded.** (1) A 1000x
units error: `gpu_steady_tail` takes `busy_us` and converts internally, I pre-divided. **Every
verdict, RSD and ratio is scale-invariant and did not move** — which is why it survived a first
reading. A units error that changes no verdict is the kind that gets published; caught by checking
soloA against its independently published 11.525 ms. (2) A reference built partly from *withheld*
MARGINAL_TAIL medians, giving 15.5159 ms — a number that exists nowhere, and a withheld median used
as a denominator is that median published, against my own §16 rule.

**Imported, not rebuilt:** Switch's `localise`/`per_inference_kernel_us`; my
`gpu_steady_tail`/`certify`. Reproduced 11.5252 / 11.7697 / 126.647 / 126.676 exactly.

**Artifacts:** `bench/results/probe_tenancy_signature.py`, `bench/results/tenancy_signature.json`,
`bench/test_tenancy_signature.py` (6 tests), `docs/PERF.md` §18, decision record
`niobe-per-dispatch-is-not-a-tenancy-signature.md`.

**Accepted `rust/src/trace.rs`.** Timestamp calibration and trace-event arithmetic; every
certification instrument I own consumes it, and this session is the argument — `gpu_ns` and the
us/ms conversions around it are exactly where a scale error hides without changing a verdict.

**Verification:** `pytest bench/ ci/ tests/ops/` one invocation — 485 passed, 336 skipped, 0
failed, ERROR(instrument) 0.
## 2026-08-02 — First real-node run: refused figure, and what fragmentation costs

Frame: `main` at `0baf660`, merged and rebuilt. DLL `E00C7F8B…` -> `47F66833…`.

**Task 1 refused.** Gate withheld the timing: CONTENDED, 7.73 foreign busy cores, 100% of samples
over threshold. No new device-clock figure. `12.1847 ms` stands as the last quotable one, and I now
always quote it with its context length (zero context, one token) because the roofline is not a
constant.

**The idle-clock specimen is the best one we have, and it is ours.** That refused run's device-clock
series is STEADY at 245.9149 ms with RSD 0.0717% — tighter than all 28 census traces — under
SOLE_TENANT with zero foreign GPU work, and it is 20.18x wrong. Both instruments that sound like a
pass, pass. Only the SM-clock record refuses it: 210.0 MHz min/median/mean AND max of 160 samples
against 3105 MHz boost. Every earlier specimen had foreign GPU work so tenancy could plausibly have
caught it; this one has none. That settles the companion's status. Also learned something new about
the causal chain: host contention showed up on the device axis as an idle clock, not as GPU
contention — a host problem with a device symptom the tenancy verdict cannot see.

**I got Task 2 wrong first and the shape is worth remembering.** I divided
`session_staging_upload_bytes` by inference count in two records and reported fragmentation had
*reduced* staging traffic 1.78x. That counter is cumulative and dominated by a one-time ~2185 MiB
weight upload; the records had 28 and 51 iterations; 51/28 = 1.82. The finding was the denominator.
The giveaway was that the "improvement" equalled the iteration ratio. R11 again, and it closed
beautifully with a believable mechanism while being backwards.

Fixed it with a slope instead of a quotient — two iteration counts per configuration, fixed cost
cancels, recovered constant becomes a check (agrees to 0.034%). Both arms on ONE binary, the fused
arm forced by handing the EP the DIVERGENT GQA proof key. That restores 355/363 in 1 island, exactly
the historical record, so the 33-way split is mechanically confirmed as the 32 declined GQA
instances. (Setting that env var to `1` is rejected — it takes keys, not booleans — and produces two
identical arms. ERROR(instrument), caught by the EP's own WARN, not by me.)

**Answer:** 33 islands is worse in exactly two currencies — host round-trips 2 -> 66 (33x) and
staging bytes 1.92x — and NOT worse in allocations, dispatches or high-water, which falsifies the
allocator/descriptor hypotheses rather than leaving them unmeasured. The marginal traffic is exactly
one extra host round-trip of the 64 KV tensors (+393,208 B out, +393,216 B back vs 393,216 B of KV),
and the fused readback slope lands on the model's declared outputs to the byte — the closure is a
measurement because that number comes from ONNX shapes, not from our counters. 786,424 B is 0.0376%
of the weight read, so in bytes it is negligible and Switch's shape holds.

Unpriced: 32 extra submissions/fence waits/drains per inference. My host-minus-GPU bound came out
vacuous (10.48 s) and was computed under an idle clock inflating the GPU term 20x — reported as
vacuous, not quoted. And the probably-dominant cost is in neither column: the 32 declined GQA nodes
now run on CPU. Execution-location, not boundary. Next thing to measure.

Open: `phase_containment` FAIL (one sub-record span outside every `vulkan.record` span) blocks any
phase-share reading from this run. New and unexplained.

Also accepted ownership of `rust/src/trace.rs`.

Shipped: `bench/results/probe_island_boundary_cost.py`, four `islandab_*` counter records,
`bench/test_island_boundary_cost.py` (13 tests), `docs/PERF.md` §19, decision to main's inbox.
## 2026-08-02 (later) — Contention is the baseline; counts become the primary instrument

Frame: `main` at `c1522e2`, merged and rebuilt. DLL hash UNCHANGED at `47F66833…` — the merge
carried no Rust change, so §19's figures stay in frame. Worth checking rather than assuming.

The box is shared with another team indefinitely. Wrote that into PERF.md §20 as policy: wall-clock
is STEADY_UNCERTIFIED by default, the companion refusing is the instrument working, and no plan may
contain "take the measurement when the box settles" because that step never completes.

**Promoted the byte model to a first-class instrument (`bench/ceiling.py`) — and found something
while wiring it.** The coordinator asked it to say UNOBSERVABLE about a context nobody measured.
But `kv_bytes(past_len)` is analytic, so "unmeasured context" was never the real refusal condition.
The real one is sharper: GroupQueryAttention is DECLINED on this build and runs on CPU, and **GQA is
the op that reads the KV cache**. So `island_bytes_phi35.json` charges KV bytes to a GPU roofline
that the GPU never incurs — 48 MiB at past_len 128 up to 3072 MiB at 8192, where they are 60.5% of
the modelled stream. That is a bound on a machine we are not running.

So the extent is `[0]`, and it reports UNOBSERVABLE — never a number, and pointedly never 0, since 0
would claim the traffic is free when it is merely elsewhere. The refusal still discloses the figure
it declined to publish so it stays auditable.

The satisfying part: at past_len 0 the KV term is exactly zero, so the bound IS admissible there —
and zero context is the only context we have ever run, and 12.1847 ms is the only quotable figure we
hold. **The one admissible bound and the one quotable figure sit at the same context.** That is why
the 67% comparison survives, and it is the reason rather than luck. Said so explicitly.

Also noticed and disclosed a harmless divergence: Switch quotes 67.1% (weights+scales, 8.179 ms), my
ceiling quotes 67.4% (by-context total including 9.52 MiB intermediates, 8.218 ms). Two floors 0.5%
apart, both correct about their own quantity. Flagged it so nobody reads it as a figure moving.

**`bench/clock_log.py`** — continuous tenancy+clock recorder reusing probe_gpustate's sampler (not a
second dialect for the same channel), with `window()` returning the shape certify() already
consumes. Inverts the workflow: record always, certify retrospectively when a run happens to land in
a quiet minute. Two refusals that must hold: <40 usable samples is UNOBSERVABLE because an
unrecorded window is not a quiet one; and a retrospective window declares itself weaker than an
in-run companion and may not upgrade a figure an in-run companion refused. While testing I found it
would raise on a truncated log line — fixed to refuse instead, since a harness that dies on its own
error path cannot report the error it found. Smoke test showed the board currently at 54.3% of boost
with zero foreign GPU work, which is exactly the opportunistic window this exists to catch.

**Screen, not a rule, for the cumulative-counter quotient.** On a permanently contended box run
lengths vary with whatever else is running, so denominators will differ between records by default
and my 51/28 = 1.82 artifact would reappear without anyone doing anything unusual. Text-decidable
screen over all of bench/, positive control built from the exact bad line, negative control on the
correct two-point construction. No offenders.

Shipped: `bench/ceiling.py`, `bench/clock_log.py`, `bench/test_ceiling.py` (21 tests),
`docs/PERF.md` §20, decision to main's inbox.

## 2026-08-02 — GQA claimed: re-derived the ceiling rather than flipping a flag

Frame: `main` @ `7c9d1b7`, rebuilt. DLL `47F668336A7BF6A9…` -> `3A9115417CD1A780…` (changed).

- **Found the stale-record defect in my own module before touching it.** `ceiling.py` printed
  the new DLL hash and, in the same paragraph, "GroupQueryAttention is declined on this build" —
  the claim status came from a record made by the previous binary. Artifact saved to
  `bench/_scratch/ceiling_stale_record_artifact.txt` before any edit. `load()` now hashes the DLL
  and raises `CeilingError` on an out-of-frame record; `bench/environment.py` writes
  `environment.build.sha256` so records can be checked at all.
- **Verified GQA on my own build, not from the message**: `subgraphs_live` 1, 355 of 363 claimed,
  MATCH, claim log `GroupQueryAttention x32 proven`.
- **Measured the KV byte term instead of assuming it.** `probe_kv_bytes_earned.py`, slope of
  slopes across past_len 0/128/512 x iters 5/25. Readback = **393,216 B per past token, ratio
  1.000000 on both segments, linearity spread 0.000000**. Upload **flat** at all three contexts ->
  `UNOBSERVABLE`, never 0. Falsifier for "past_len is wired" is an artifact: present.0 shape and
  argmax both move.
- **My first version of that probe averaged the two staging axes and printed `0.0`**, which read
  as a refutation of a term that is exact to the byte. Caught and corrected before commit. Two
  instruments, two worlds — do not collapse them to one number.
- **Split the extent in two.** `extent()` (DRAM bound describes this build) widened to the full
  grid; `binding_extent()` (and is the floor) stayed `[0]`, now for a measured reason. Published
  the transfer crossover per context rather than tuning a constant until the awkward contexts
  vanished — the threshold-episode discipline.
- **New result worth carrying: at past_len >= 2048 the inference is transfer-bound, not
  DRAM-bound, on any link that exists.** Conditional on the readback law past 512.
- Teeth: in-frame *declined* record still collapses the extent to `[0]`. Discharging a refusal
  once must not wire it open.
- `docs/PERF.md` §21; §19 annotated as superseded (33 islands no longer exist on `main`).
- Suite: `pytest bench/ ci/ tests/ops/` in one process — 586 passed, 3 failed, all three shown
  pre-existing by removing every file of mine and re-running (2 x Link's `ci/test_lane_checks.py`,
  1 x `tests/ops/test_harness_census.py::test_census_baseline_has_no_drift`). `ERROR(instrument): 0`.

📌 Team update (2026-08-02T22:37:04-07:00): Mouse's `mouse-counters-abi-mirror-equality` finding — `device_losses` was inserted mid-struct into `VulkanEpCounters` without a `COUNTERS_ABI_VERSION` bump; three ctypes mirrors kept the old layout and two other counters silently swapped meanings. Every counter reading you took through a ctypes mirror between `a52024f` and `4d47362` is suspect, including any of your link-bound/readback figures read that way. Mirrors must now assert exact `struct_size` equality. — decided by Mouse

📌 Team update (2026-08-02T22:37:04-07:00): Switch's `KV_CAN_STAY_DEVICE_RESIDENT` ruling changes what your link-bound-past-2048 measurement is a bound *on*: ORT permits binding an `OrtValue` in the EP's device memory as a graph output, bit-identical to unbound, so the round trip your 393,216 B/past-token figure describes is not a runtime limitation — it is `transfer.rs`'s own host-staging-authoritative invariant, now fixed as a per-span flag. The round trip is moved (paid whenever a caller asks for host bytes), not yet removed; your figure remains the correct measurement of the shipped (unbound) path, but it is no longer evidence of an ORT-side ceiling. — decided by Switch

📌 Team update (2026-08-03T04-55-00-07-00): Switch measured on the real Phi-3.5 graph (64 KV outputs, 6-step chain) that the shipping (host) KV lane OOMs at past context 4096 on the 8 GB discrete GPU while the device-resident lane completes — the round trip is a VRAM cost, not only the bandwidth cost your link-bound figure names. Also: ctx 8192 fails on both lanes, so the operating point the 82.2%-of-traffic KV figure is quoted at has never been reached on this hardware — bounds what your link-bound-past-2048 measurement can be extrapolated to at longer context. — decided by Switch
📌 Team update (2026-08-03T19:55:00-07:00): Switch's GLSL.std.450 table correction — the SIMT interpreter's extended-instruction table was wrong at slots 30/37/40/43. The silent-miscompute set is {37, 43} → celu, hardsigmoid, hardswish; a elu would NOT have been silently miscomputed (Switch corrected his own earlier headline on this point) — the actual discriminator is operand count, not op identity. Your weight-amplification measurement (re-derived at 1.000000 with three positive controls) is untouched by this and needs no re-run — it does not exercise the affected instruction slots. — decided by Switch
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
