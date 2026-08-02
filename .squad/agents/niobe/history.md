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

## 2026-08-01 — the harness died on someone else's WARN, and the exciting reading was the false one

**Merged local `main`, not `origin/main`.** `origin/main` is 21 commits behind and carries neither
Switch's barrier fix nor Tank's WARN; the branch I was told about is the local one (`1cd0b55`).
Worth knowing before the next merge: `rust/tools/probe_broken_commitment.py` — which `bench/` now
imports — exists only on local `main`.

### The crash was R12 and the fix was to borrow, not to write

`subprocess.run(..., text=True)` decoded the worker as UTF-8. ORT's sink writes UTF-16LE on Windows,
our narrow lines share the handle, and Tank's WARN goes through ORT's sink, so the reader thread
raised at byte `0xa7`. Then the error path raised on top of it: `proc.stderr.strip()` on `None`.
**A harness that dies on its own error path cannot report the error it found.**

Two instruments each correct about a different world, colliding as a crash rather than as a wrong
number — which is the good version of this failure. I took Tank's `decode_both` rather than writing
a third decoder; his docstring already records what each naive version got wrong, and a second
dialect for one channel is what Link refused to create for the verdict vocabulary. If his module is
absent the tail is `ERROR(instrument=stream_decoder)` — not a private fallback decode, because an
unreadable tail is an instrument error and a mangled string is evidence.

Capture is bytes now. `text=True` puts the decode in `subprocess`'s reader thread, which is the one
place a failure cannot be classified: it arrives as a traceback from inside the apparatus.
Instrument errors go to `instrument_errors`, never `refusals` (R13) — a refusal is a condition I
found, an instrument error is my not having looked.

### The A/B, and why interleaved

The machine was never quiet: 4.7–11.4 foreign cores all afternoon, and the survey named the sources
— the other agents' `copilot.exe`, `Code.exe`, Defender. Justin's project had indeed stopped; we
replaced it ourselves. **The gate refused end-to-end on both devices and I did not override it. Third
session running.**

So I measured what survives a loud machine. Switch's prediction — host moves sharply, device barely
— cannot be tested by "run old DLL, run new DLL, subtract", because host time is exactly what load
moves and the load is not constant across two runs minutes apart. Alternated the DLLs **A B A B** in
one sitting instead, each arm carrying its own survey. `bench/results/probe_barrier_ab.py`.

- **Host `record`: 16.412/20.344 ms → 3.780/3.687 ms median; leaf-only 810.0 → 139.0 ms over 43
  recordings (5.8×).** The load ordering scrambled — the second post run ran at *more* foreign load
  (8.12 cores) than the first pre run (5.06) and was still 4.3× cheaper. Load cannot make that shape.
- **Device: 13.3463 pre vs 13.3432 post on the matched pair (both n=43, both 100% coverage, back to
  back) — 0.02%.** No shader changed between the two DLLs, so a device movement would have had to be
  the barrier. There isn't one. **Both halves of his prediction hold.**
- His ~94 ns/struct falls out of my numbers independently: 86–106 ns over the same 147,618 structs.

### The ruling, and it cost me the headline

Morpheus asked for a call on a minimum-n floor. **Yes, and the floor is on coverage more than on n.**
The 2% RSD bar constrains the tail's *internal spread*, not its agreement with the device's true
rate, so five samples from a flat stretch of a wandering series clear it as easily as a settled
device. My own two bad tails: n=5/coverage 12% reading 37.562 ms against its run's warm mean of
26.412 (+42%), and n=7/coverage 15% reading 20.055 against 15.03 (+33%). Every good tail sat at
87–100% coverage and agreed with its warm mean inside 0.6%. A warmup ramp is a short *prefix*; a
device that never settled makes a short flat *suffix*. Floors: `n >= 8` **and** `coverage >= 50%`;
`MARGINAL_TAIL` otherwise, median parked in `withheld_median_ms`.

**Its first act was to refuse the reading that said the barrier fix improved GPU time by 33%.** That
was my exciting number, it was produced by a 5-sample tail, and it was false — under the floor the
two arms agree to 0.02%. Switch's `contended` row (n=8, 17% coverage, 2.1% above solo) is refused
by the same rule.

### Numbers of record, current `main`

- **NVIDIA RTX 4060: 13.3432 ms/inference GPU busy** (STEADY, n=43, coverage 100%, RSD 1.72%),
  `MATCH`, 355/363 nodes claimed, 1 island, 16,330 dispatch spans. Comparable to the same figure on
  the same box minutes apart, which is what §14.2 is. **Not** comparable to my 40.201 ms of
  2026-07-31 — the GEMV rewrite, the partitioner fusion and residency all landed in between, and
  dividing them credits four people's work to whichever one is under discussion. 40.201 ms is
  **retired as a baseline, not beaten by one.**
- **Intel Iris Xe: withheld, `NO_STEADY_TAIL`.** The series wanders 53.4–91.3 ms with no flat region.
  Same refusal and the same structural reason as last time: an iGPU shares its power budget with the
  loaded CPU cores, so the device clock is not contention-immune there.
- **End-to-end wall clock and the Vulkan/CPU ratio: withheld on both devices, gate `CONTENDED`.**
- The devices are not compared. `bench/compare.py`'s cross-device refusal stands.

199 tests in `bench/` pass (was 195). Three decision records filed. Not pushed.

**Left standing:** the pre-fix worktree at `../ep-vulkan-niobe-ab` (detached at `42deaba`) is the A/B
arm — remove it with `git worktree remove` when the comparison stops being interesting.

## 2026-08-01 (mid-day, reconstructed) — the device-state companion, logged late

This round is **not in this file** as it happened; the entry below is reconstructed from
`docs/PERF.md` §15 and `bench/device_state.py` so the chain is not broken. A Scribe run around this
time read 47 inbox records and deleted 50, and my own log entry went missing in the same window.

`DESIGN.md` §10.0 obligation 8: a device-clock figure is quotable only if it carries a
**device-state companion** — a tenancy verdict **and** a clock record — over the statistic's own
window. Built `bench/device_state.py` (five terminal verdicts, only `QUOTABLE` releases a number)
on top of Switch's `bench/results/probe_gpustate.py` sampler, imported rather than re-implemented.

Two things worth keeping:

- **I published dev0 #2 against myself.** `MARGINAL_TAIL`, median 12.187 ms withheld, on the lowest
  RSD I have ever recorded (0.086%) — and it agreed with the quotable 12.1847 ms to four
  significant figures. *A refusal is not a claim the number was wrong.* Coverage was 26%; the gate
  is about coverage, not about agreement, and letting agreement excuse coverage is how a gate stops
  being one.
- **I reproduced Switch's bug at my own call site.** `FOREIGN_GPU_WORK` in 93% of samples, against
  a PID holding 0.0 MiB, which was **my own worker** — `_run_worker` blocked, so the sampler
  learned the PID only after the child had exited. `ERROR(instrument)`, never a detection, exactly
  as his `_is_ours` docstring warned. Fixed with `Popen` + an `on_start` callback and a regression
  test that asserts the PID arrives while the process is still alive.

Also applied Switch's standard to my own 40.201 ms: it survives as a **regime**, not as a pairable
measurement, and I withdrew every "at most" argument I had used to put a floor under a difference —
**two upper bounds do not bound a difference from below.**

---

## 2026-08-01 (later) — Intel has no clock producer, and half a companion is not one

The task: obligation 8's only producer is `nvidia-smi`, so **every Intel figure is permanently
`UNCERTIFIED`** — and Intel is where the open question lives, because the 4.39× residual of the
13.52× kernel gap that memory bandwidth does not explain is a claim about Intel. The coordinator
handed me a lead, explicitly as a lead and not a solution: Windows' vendor-neutral
`\GPU Engine(*)\Running Time` / `Utilization Percentage`.

### What I built

- **`bench/win_gpu_counters.py`** — an in-process PDH sampler over the WDDM `\GPU Engine` counter
  set: LUID join from the Vulkan device name through `HKLM\SOFTWARE\Microsoft\DirectX`, per-PID
  cumulative-tick differencing, ancestry via Switch's `_is_ours`. **Tenancy only. It emits
  `clock: UNOBSERVABLE` in every record it will ever produce**, and its docstring says so before it
  says anything else.
- **`bench/results/probe_wingpu.py`** — the capability experiment. Samples the target adapter *and
  every other live adapter* as a negative control while a real Phi-3.5 pass runs.
- **`bench/test_win_gpu_counters.py`** — 22 tests. 38 with `test_device_state.py`; 237 in `bench/`.
- **`bench/device_state.py`** — the record is now **two independent axes**, tenancy and clock, each
  with its own producer, verdict and silence set. New states `TENANCY_ONLY`, `NO_PRODUCER`,
  `UNCERTIFIED_PARTIAL`; `compose()`, `empty_axis()`, `from_nvidia_record()`.
- **`docs/PERF.md` §16**, and the §15.2/§15.3 tables updated.

### The capability, measured rather than hoped

Intel arm, 8 inferences, `MATCH`, 71.7 s, 46 enumerations: our worker accrued **1.4292 s on
`engtype_3d` on the Intel LUID and 0 s on each of the three other live LUIDs**. `Code.exe` held
0.1296 s (0.18%, under the 1% bar) → `SOLE_TENANT`, clock `UNOBSERVABLE`.

The negative control is the load-bearing half. Our process holds counter *instances* on both
adapters, so "our PID is in the set" proves nothing; that engine time accrues **only** on the
adapter we ran on is what checks a registry-string join that would otherwise be silently wrong.

Second-order facts that came out of it: there is **no `engtype_Compute` node on either adapter** —
compute is scheduled on `3d`; **PID 4 (`System`) accrues Copy-engine paging time** on behalf of
whoever faulted (2.03 s on the NVIDIA arm), so counting it foreign would make the detector a
constant and it gets its own class; and on this hybrid laptop the panel hangs off the iGPU, so the
compositor's hold is `FOREIGN_GPU_WORK(display)` — **permanent, named as such rather than looking
like bad luck.**

### The verdict, and why it is not a pass

`TENANCY_ONLY` → **`UNCERTIFIED(partial_companion)`**, its own state at both levels, because
*bypassed / all-rejected / unobservable* sharing one `0` is Tank's lesson and this is the same
shape. **The failure the clock record exists to catch is a sole-tenant failure**: `base_b` was
verified sole tenant, 21.4× wrong, at the project's second-best RSD, because the board never left
210 MHz. A tenancy-only companion says `SOLE_TENANT` about that run and is *correct*.

And engine `Running Time` cannot be pressed into the clock role: it is a **duration**, so at a
lower clock the same kernel occupies the engine *longer* — it moves the same way as the figure it
would certify. A second copy of the quantity under certification, not a second quantity from
outside the series. That is the same-source falsifier that put 246.735 ms into the record.

The asymmetry that makes it safe to have at all: **it may subtract confidence and may never add
it.** `FOREIGN_GPU_WORK` without a clock is still a detection → `WITHHELD`. `SOLE_TENANT` without a
clock is not a pass. Implemented in `compose`, not merely stated.

### The reads-clean shape — three bugs, all of which made the record *cleaner*

1. `devices.by_index(1)` gave me **NVIDIA**: `vulkaninfo` enumerates 0=Intel, the EP's best-first
   order is 0=NVIDIA. `bench/devices.py` documents this trap and I walked into it. The sampler
   watched an untouched adapter → clean `SOLE_TENANT`.
2. Ancestry for all 204 instances cost **21.2 s/round**: 3 samples in 62 s → clean again. Cached
   per-PID and classified only PIDs with nonzero in-window ticks (21.2 s → 0.40 s).
3. **PDH caches its instance list per process** — my own worker was invisible for a whole 60 s run.
   Proved it with a poller using `PdhEnumObjectsW(..., bRefresh=TRUE)`, which sees the instances
   appear ~14 s in. Clean a third time.

R9 amendment 5 says ask which way a check moves when its subject is wrong; all three moved **with**
the reader's confidence, so none was fixable by tightening a threshold. Fixed at source and
backstopped by two interlocks that make the clean reading *unavailable*: `UNOBSERVABLE(self_not_witnessed)`
(a record must carry positive evidence it watched the right device) and a **blind-gap limit**
(>10× interval → `ERROR(instrument)`). The blind-gap interlock fired for real on the first NVIDIA
corroboration run — four samplers each re-enumerating, 12.6 s gap — and the fix was one shared
enumeration cache, not a looser limit.

### The ruling

**An Intel device-clock figure cannot be certified on this hardware. Permanent, not pending.**
Unlike Link's lavapipe `none_structural` there *is* a subject — the Iris Xe has a real clock that
really varies — so it gets its own classification, **`none_available`**: no `GPU *` counter set
carries MHz, no `root\wmi` GPU class exists here, `Win32_VideoController` has no core clock,
`nvidia-smi` exits 6, Vulkan has no clock query, and engine time is inadmissible as a proxy.
Reopening it needs a **producer**, not a better analysis — a Link/platform question, not on M0's
path.

**For Switch:** the 4.39× residual must be attacked with **counts and shapes, not clocks** —
which is §10.0.4's preference anyway, and now there is no alternative on Intel.

**The uncomfortable direction:** Morpheus's amendment 2 notes the iGPU shares its power budget with
loaded CPU cores, so it is *more* exposed to the clock failure than the discrete board — and it is
the device where we can never see it. The Iris Xe's `NO_STEADY_TAIL` refusals are **consistent
with** a wandering clock and are **not evidence of one**.

### Portability

Morpheus required a **record, not a tool**, and WDDM counters are as Windows-locked as `nvidia-smi`
is NVIDIA-locked. So the record is emitted **in full on every platform**, with `NO_PRODUCER` in the
axes nothing can fill — a missing key is indistinguishable from a key nobody thought to write; a
`NO_PRODUCER` axis is a statement. `winreg` is imported under a guard, `available()` is False
off-Windows, the Windows tests skip rather than fail, and
`test_the_record_is_emitted_in_full_where_no_producer_exists_at_all` holds the line.

### Corroboration, since one existed for once

On the 4060 both producers ran over the same window and both said `SOLE_TENANT` (`agree: true`,
peak 2010/3105 MHz). Obligation 7's shape, with the agreement **in the artifact**. It licenses
nothing about the clock axis and does not move Intel any closer to quotable.

### State

38 targeted tests pass; **237 in `bench/`, 0 failed**. Scratch prototypes deleted; the
`wingpu-idle-intel-dev1.json` artifact from the mis-joined run deleted rather than kept. Three
decision records filed. Committed on `squad/niobe`, not pushed.

**Still blocked on Justin:** `nvidia-smi --lock-gpu-clocks` + an elevated shell, which would close
the duty-cycle mechanism on the NVIDIA side. Not designed around, not blocked on.

📌 Team update (2026-08-01T17:16:56-07:00): Intel device-clock figures are permanently uncertifiable on this hardware (`none_available`, no producer exists and none of the available proxies are the right kind of quantity) — attack the Intel/NVIDIA residual with counts and shapes, not clocks — decided by Niobe


📌 Team update (2026-08-01T17:16:56-07:00): All wall-clock figures remain withdrawn; only counts, bytes and certified-companion device-clock figures are quotable — decided by Switch, Morpheus, Niobe, Link


📌 Team update (2026-08-01T17:16:56-07:00): `ledger_lookup` is the last `UNWIRED` mechanism in the instrument census (criterion 11); Mouse is building it — decided by Trinity, Mouse


## 2026-08-01 (evening) — R11 in a *selection*: the probe misled by omission while every field it printed was true

Small fix, and the size of the fix is not the size of the finding. Two `KEYS` lists and a print
line; the shape is new and worth carrying.

**The question.** Was `alloc_device_authoritative_spans = 0` in `bench/results/indexspace.json` a
real zero or R12 (a counter reporting `0` for an event that cannot occur in its frame)? Morpheus
ruled it **a real zero and he was right**: it is an `int`, not the string `"UNOBSERVABLE"`/
`"UNWIRED"`, so the type discipline had already answered; `alloc_device_residency_evaluations = 9`
from the same call site says the question was *asked* nine times and answered no; and
`ceiling = backed 9 − staged 9 = 0`, so zero is the only correct value. `UNOBSERVABLE` would have
been a **stronger and false** claim.

**The defect is the probe, not the counter.** `probe_indexspace.py`'s `KEYS` took
`alloc_device_backed_spans` and `alloc_device_authoritative_spans` and did **not** take
`alloc_staged_spans` or `alloc_device_authoritative_ceiling` — both emitted by `counters.rs`
(L934, L945) and simply not selected. So the artifact **genuinely did not contain** what a reader
needs to tell a measured zero from a pinned one. The coordinator's failure to interpret it was not
ignorance, and anyone who *had* interpreted it confidently would have been guessing.

**The shape, which is the part to keep.** R11 usually appears in a *name* that claims more than
the value supports. Here it appeared in a **selection**: a probe can mislead through what it omits
while every field it prints is individually true. Rule adopted in the file:

> a counter whose value is only interpretable against a companion key is not admissible without
> that companion key on the face of the same output.

### What I changed

- `probe_indexspace.py` `KEYS` gains `alloc_staged_spans`, `alloc_device_authoritative_ceiling`
  and — from the audit below — `alloc_allocations`. The printed line now shows
  `ceiling=0 (= backed 9 - staged 9)` and `authoritative_spans=0 over 9 residency evaluation(s)`,
  so the arithmetic is on the face of the output rather than reconstructable by someone who
  already knows to look.
- A `span_accounting()` note that **names which kind of zero it is**:
  `MEASURED_ZERO_AT_A_ZERO_CEILING` / `MEASURED_ZERO_BELOW_A_NONZERO_CEILING` / `UNWIRED_ZERO` /
  `NOT_A_NUMBER`. The middle one is the interesting state and it must never hide inside the same
  `0` as the first — same discipline as Tank's five states sharing one zero.
- **It reports; it does not judge.** Nothing in it feeds `checks`, so it cannot withhold
  `ONE_INDEX_SPACE`. An accounting note that could move the verdict would be a different
  instrument wearing this one's name.

### Re-run: the verdict did not move, and it should not have

`ONE_INDEX_SPACE`, both arms intact, all seven checks ok — allocator index `'1' → '0'` as the
selector moved `0 → 1`, matching the offered index on each arm, both `SHARED`, binds non-zero on
both. Identical values to the previous artifact (binds 6, allocations 9, staged 9, backed 9,
ceiling 0, authoritative 0, evaluations 9). **This changed what is reported, not what is
measured** — which is exactly what should be true of a reporting fix, and the confirmation is the
point of re-running rather than reasoning.

Worth noting what the newly-visible arithmetic actually says: **staged 9 of 9 spans, so every
device-backed span is a mirror and the ceiling is 0.** The §6.5 frame is shared and the residency
screen runs, but nothing is device-authoritative yet — visible now, invisible before.

### Audit: one instance of a shape is usually not one instance

| consumer | verdict |
|---|---|
| `probe_indexspace.py` | **two** instances — the reported `authoritative_spans` (fixed), and `alloc_device_backed_spans = 9` as a **bare numerator** with no `alloc_allocations` denominator: 9 of 9 and 9 of 900 are different findings and the extract could not tell them apart (fixed) |
| `probe_sec65.py` | **two more.** It printed `alloc_device_spans` — **a key that exists nowhere in the source**, so it printed `'<absent>'` on every run since it was written, and an always-absent key reads like a measurement that came back empty. Real name: `alloc_device_backed_spans`. And it printed `authoritative_spans` with no ceiling and no residency evaluations — the same uninterpretable zero. Both fixed |
| `probe_residency_screen.py` | clean — its `KEYS` already carries ceiling, evaluations, binds, backed and staged |
| `bench/phi35.py` `staging_label()` | clean — it reads staged **and** backed as a pair, and treats absent as its own `"unknown"` state that is explicitly *not* the same as device-backed |

`phi35.py` would classify this configuration as `mixed` ("both staged and device-backed spans
observed; attribution is ambiguous"), which is the correct reading of 9/9 and agrees with the
ceiling of 0.

242 tests pass in `bench/`. Committed on `squad/niobe`, not pushed. Scoped small on purpose: the
`MARGINAL_TAIL` census cross-check remains the more important of the two and did not collide with
this.
