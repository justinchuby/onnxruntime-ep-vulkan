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
Memory-pattern planner does not engage on run 1. From run 2 onward hands back interior pointers. 52 observed, `pointers_in_guard_band=0`. Gate: `epctl --check-counters <file> --require-dispatches 1`.

### Execution counters file is the instrument for "did anything execute"
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` — always-on JSON. `dispatches_executed > 0` is the only reliable indicator.

### `push_next` must rebind, never discard
`let _ = props2.push_next(..)` silently discards pNext chain. Rebind, never discard.

### First real execution: 45 ops Live, 161 nodes claimed on Phi-3.5
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. M0 not declared — open: validation positive control, CI lanes green.

### Performance metric is a TRIPLE
`(claimed_op_coverage, island_count, largest_island_flops)` per producer at version. Portability floor = §7.2. `SUBGROUP_SIZE_IS_GUARANTEED=False`.

---

## Turn 5 — 2026-07-30 — the first honest measurement (Phi-3.5, both devices)

### Coverage figures go stale fast — read the counter, never the briefing
I was told 161 claimed nodes. The run reported **257 claimed of 363 probed**. Always re-read
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` in the same run that produces the number.

### Island count is `subgraphs_live`, not the claimed-node count
They happened to be equal (257 == 257). Equality is a *coincidence to be falsified*, not a
definition. The falsifier: `compute_calls == subgraphs_live x inferences` — 7967 == 257 x 31
exactly, on both devices. Integer equality, no tolerance, free.

### The Intel iGPU gets SLOWER with warmup — a "take the min" convention would lie 4x
Iris Xe per-inference: 724 -> 695 -> 903 -> 1447 -> 2080 -> 2669 -> flat ~2790 ms. Monotone ramp
into steady state, not out of it. Added `stats.drift()` (first/second-half median ratio +
monotone fraction) and raised phi35 warmup default to 10. Spread cannot tell "noisy but stable"
from "moving steadily"; they demand opposite responses.

### Within-run spread is not run-to-run spread — carry both
With warmup 10 the within-run rsd is 1.7% (Intel) / 2.6% (NVIDIA), yet two whole runs minutes
apart differed by 28%. Added `--repeats` (default 3) launching whole processes.

### The CPU baseline is not a constant
218 ms then 665 ms for the same CPU-only session, minutes apart — page-cache pressure after a
2.2 GB model. Hence: each device's vulkan-vs-cpu delta must be measured back-to-back in ONE
process; `baseline_disagreement()` fires above 2x between workers.

### The counters file in this path carries no `alloc_*` keys
Only `abi_version, compile_calls, subgraphs_live, subgraphs_stub, compute_calls,
compute_failures, dispatches_executed`. So the staging label derives from **configuration**
(`ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` unset => staging-bound), with observation as a weaker
second ground. "unknown" would have invited the reader to assume the number was general.

### Timestamp verdict: inputs VERIFIED, arithmetic VERIFIED, end-to-end UNMEASURED
`bench/timestamp_audit.py` cross-checks `epctl --probe-loader` against `vulkaninfoSDK`. Agree on
both devices: Intel 52.0833 ns / 36 valid bits / UMA true (wrap 3579 s ~= 0.99 h);
NVIDIA 1.0 / 64 / UMA false. **No `VkQueryPool` exists yet**, so nothing end-to-end is measured.
Crucially: lavapipe and NVIDIA both report 1.0/64, so dropping BOTH the period scale and the
mask is green on the discrete GPU and green in CI while under-reporting every Intel duration by
52x. **The Iris Xe is the only instrument on this desk for that bug class, and CI has none.**

### The tracer is written and env-wired but NOT called
Verified empirically, not by reading: with `ONNXRUNTIME_EP_VULKAN_TRACE` and `TRACE_GPU=1` set,
a run that executed 257 islands over four inferences produced **no trace file**. Report adoption
as four facts (pinned / written / env-wired / invoked), never as one word.

### The number: we are 8-12x SLOWER, and that is the useful result
Intel 2790.7 vs 229.8 ms (12.1x); NVIDIA 1465.9 vs 185.9 ms (7.9x). Both `MATCH`, staging-bound.
Per island: >= 9.96 ms (Intel) / >= 4.98 ms (NVIDIA) — a lower bound, since the host delta nets
boundary cost against the GEMV saved.
**For Mouse:** fewer islands beats faster kernels by an order of magnitude right now.
**For Switch:** Intel costs ~2x per island *with no bus to cross* => argues for a fixed
per-submission cost (submit-and-wait per island), not per-boundary PCIe transfer. Hypothesis;
the §3 timestamps decide it.

### Tooling
- `.squad/decisions/inbox/` is **gitignored inside a worktree** — decision records must be
  written into the integration tree's inbox or they never reach the Scribe. `cargo ci` says so;
  git never will.
- Crate edition is 2024: `rustfmt --edition 2021 <file>` silently no-ops. Use `cargo fmt --all`.
- PowerShell `Select-String` piped after a native command can return nothing; redirect to a file.

---

## 2026-07-30 (evening) — the hypothesis died, and two defects found by instrument

### My per-submission hypothesis is FALSIFIED
`vulkan.submit` is **0.6%** on the discrete part. I reasoned from Intel costing ~2x per island
with no bus to cross, and I was right about the data and wrong about the stage. **Refusing to
assert it is what got the instrument built.** That is the whole of R9 in one episode: I got more
from being wrong with an instrument than I would have from being right with an argument.

### The real cost: `vulkan.record` is 68.9% (NVIDIA) and **98.8% of it is host staging memcpy**
~60 MiB re-copied into staging on *every* `Compute` call. The recording loop is worth 414 ms of a
33456 ms phase. Recording scales with **bytes, not dispatches** (implied bandwidth constant to
1.05x across a 10x dispatch-count change). Warmup decline is real and survives island identity;
NVIDIA flattens by inference 2, Intel has not flattened after 12.

### Defect: `ep.device_index` != `vkEnumeratePhysicalDevices` index
`engine.rs::probe_devices` is **sorted best-first**; `vulkaninfo`/`probe()`/`epctl` are in
enumeration order. Discrete outranks integrated, so `ep.device_index` 0 is NVIDIA while
`probe()[0]` is Intel. **My first results table named the wrong GPU on every row and every gate
stayed green.** A wrong-and-confident label is worse than a missing one.
**Rule I now hold:** never label a measurement by the index you passed in. Label it from a
fingerprint the *measured system itself* emitted. `devices.device_identity_check` does this from
`timestampPeriod`/`validBits` carried in the trace, and prints `UNIDENTIFIED DEVICE` rather than
a plausible guess. It refuses on ambiguity too — NVIDIA and lavapipe are both 1.0/64.

### Defect: GPU spans cannot be attributed to submissions by timestamp
`anchor_uncertainty_us` hits **314618 us (314 ms)** on Intel. Timestamp containment invented 14
"GPU busier than its own fence" violations on a build whose conversion arithmetic was *proven*
correct. Fixed by **ordinal** attribution (each subgraph consumes exactly `nodes` GPU spans),
with `sum(nodes) == len(gpu_spans) == dispatches_executed` asserted as integer equality (5457).
**Generalisable:** a derived coordinate whose error bar exceeds the thing being resolved must not
be used to resolve it. Durations are fine; *positions* are not evidence.

### Two things I retired rather than gated
`per_island_ms_lower_bound` (it went UP when islands fell 10x) and the vulkan/cpu ratio on an
unsteady baseline. Both for the same reason: **a number printed under a warning gets quoted
without the warning.** Refusal beats annotation. Also: untestable steadiness (< 4 samples) is
treated as unsteady — untested is not steady.

### The 52x trap is now closed end to end
`gpu_ns / period` must be a whole integer because ticks are integers. **Decisive only where
period != 1.0** — VACUOUS on NVIDIA and on lavapipe CI, and reported as a gap, never as a pass.
PASSED and decisive on Intel. Note GPU time is only 12.6%/43.9% of Compute, so a 52x error there
is *invisible in wall clock* — which makes the trap worse, not better.

### Tooling
- Traces are cheap to re-analyse and expensive to re-collect. A full 2-device run is ~40 min;
  both defects above were fixed and re-verified against the **stored traces**, no re-run.
- The timed pass must run with tracing OFF and the split come from a separate pass;
  `tracing_overhead_ratio` measured 1.02x on a 20-iter run and **7.89x** on a 2-iter one.
- Verdicts from short runs are honest but different (no steady state, huge tracing overhead).
  Do not quote a smoke run's `size_verdict`.

---

## 2026-07-30 evening — the machine is an uninstrumented variable, and it was the biggest one

### What happened
The coordinator captured a trace while six agents were compiling, then **ran a control**: same
device, same build, same test, quiet vs loaded. `vulkan.record` total went **19 460 ms -> 184 356 ms.
9.5x on load alone.** Every number this project has taken is in scope, including mine.

### The class of defect
`stats.drift` sees a baseline *moving*. It cannot see a baseline that is **uniformly wrong because
the machine was busy the whole time**. Same family as `dispatch_accounting` and `refuse_if_ep_absent`:
defects that raise nothing and leave every existing counter green.

### Three instruments, deliberately sharing no input
1. **Survey** (`contention.py`) — system-wide idle counter. Absolute; cannot miss a short-lived
   process. `busy = cores*wall - idle_delta`, **not** the sum of user+system+... — on Windows
   interrupt/dpc overlap with system and the naive sum over-counted (61.64 against a 60.06 capacity).
2. **Tachometer** — fixed-work integer spin, best-of-7. `min` is right because every error source is
   one-sided. Needs a persisted quiet reference; without one it reports **VACUOUS, never "pass"**.
3. **In-band signature** (`phases.contention_signature`) — the only one that works retroactively.
   **A Vulkan trace carries its own control**: host phase spans are wall-clock on a schedulable
   thread, GPU spans are differences of the device's own counter. The GPU does not care how many
   copies of rustc are running.

### Two defects in my own instruments, both caught on real data
- The survey named `System Idle Process` as a top consumer — the one thing on the box *not*
  competing with us. And its per-sample `children(recursive=True)` burned 9 CPU-seconds in 12,
  enough to trip its own threshold. Cache children; add `monitor_not_perturbing`.
- The signature v1 normalised per slot then took the **median across slots** — and called an RTX
  trace STABLE while slot 0 ran 12.48/70.19/12.59 ms and slot 5 ran 301/1156/374 ms. Stalls hit
  *some* islands; a median across slots averages away the thing the statistic exists to find.
  Rewritten per-slot: each slot's host spread against **that same slot's** GPU spread.

### `monitor_not_perturbing` must use thread CPU time, not wall time
On a saturated box a 5 ms sample takes 100 ms of wall clock, so the wall-time version fired on the
very condition it must be independent of. **A falsifier confounded with what it guards is not a control.**

### Verdicts on my own stored numbers — S9 is withdrawn
- NVIDIA: `HOST_SIDE_EXCURSIONS`, **20 of 33 slots (61%)**, worst 9.39x host against 1.0024x GPU.
- Intel: `NOT_STEADY`, 5.25x whole-inference spread.
Both non-quotable. I wrote the gate and my own numbers failed it.

### The inversion cannot be adjudicated, and that is the answer
Intel 807.2 vs 4060 1156.0 — I can neither confirm nor retract. It is a **cross-device comparison**
(`compare.py` exits 2 on one) *and* neither figure has a quiescence record. What I can say: the
ordering **reverses between runs of the same build**, both unmeasured. Stop quoting it — not
refuted, never established.

### The evidence did not outlive the next run
`_run_trace_pass` used a deterministic scratch path, so my own 3-iter smoke test **destroyed S9's
Intel trace** and its verdict had to be transcribed by hand. Traces are now copied to
`results/traces/` when `--out` is given. On this project re-analysing a stored trace has repeatedly
beaten a fresh run; 0.5 MB against 40 minutes.

### The spread of a machine is a lower bound until sampling stops
Tachometer was 2.08x when I first wrote it up, **2.65x (59.96 -> 159.19 ms)** by the end of the
watcher's run. Any single "how noisy is this box" figure is provisional in the direction of too small.

### Re-measurement is blocked, provably
23 watcher samples over the session, **zero quiet**, `loud=100%` on every one, foreign load
7.5-18.8 of 20 cores. `--require-quiet` refuses to *start* — failing in 15 s beats failing in 40 min.
Log kept at `bench/results/machine-load-2026-07-30.log` so the block is evidence, not a claim.

### For Switch
A 9.5x environmental swing swamps a 3x win **in either direction** — it can make a real
improvement look like a regression. Before and after must **both** carry `QUIET`.

### Standing rules this reinforced
- **Refuse, never warn.** Applied to contention exactly as to unsteady baselines.
- **Untested is not quiet**, mirroring untested is not steady.
- Two instruments disagreeing resolves **pessimistically**.
- Structural results (counts, integer identities) survive a contended run and are still printed.
  Only durations are withheld.
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

## 2026-07-30 late — two corrections from the coordinator; I checked both before applying them

### The one that did NOT apply to me, and applying it would have broken correct rows
Coordinator: "device labels are inverted, re-label any stored result." True of the world
(`select_device` indexes the **best-first sorted** list, so selector 0 = NVIDIA, 1 = Intel;
`instance.rs:536`, landed `bb885d9` 2026-07-29 — **before every run in PERF.md**, which is the check
that makes a label correction meaningful rather than a guess).

**But `bench/results/` was already right.** `devices.device_identity_check` labels each row from the
**timestamp fingerprint in that row's own trace** (Intel 52.0833/36, NVIDIA 1.0/64), not from the
index — `MATCH`, `name_may_be_quoted: true`, on every row. A blanket re-label would have **inverted
correct rows**. `docs/PERF.md` §6 was the thing that was wrong (prose predating the check).

**Rule:** a label must travel *with* the evidence, not beside it. An index asserts a convention; a
`timestampPeriod` is a property of the silicon appearing in the same artifact as the number.
Limit: 1.0/64 cannot separate NVIDIA from lavapipe — reported non-decisive, never as a pass.

### My own §6.4 premise was the mislabel
Published: "Intel pays 2x per island *while having no bus to cross*". Actually **the discrete part
paid 2x** — exactly what a staging round trip predicts, and utterly unsurprising. **That surprise is
what produced my fixed-per-submission hypothesis.** The instrument built to test it still paid for
itself (it closed the 52x trap and it is what measured upload), but link one was an artifact. I
kept the original paragraph unedited under a correction banner so the chain can be audited rather
than quietly repaired.

### The 68% was upload — and my own stored trace said so at zero measurement cost
`Phase::Record` brackets the host staging memcpy, which reports via a `ph:"C"` counter and emits
**no span**, so a `ph:"X"` aggregation is *structurally incapable* of seeing it.

```
vulkan.record  54389.02 ms  <- NOT A LEAF
  = 53635.57 ms upload (98.6%) + 753.46 ms actual command construction (1.2% of Compute)
upload = 1997.6 MiB / inference   <- matches Tank's independent probe to 5 significant figures
```

Two independent records in one trace (counters vs phase spans) that could have disagreed, and did
not. Re-analysing a stored trace beat a fresh run **again** — third time today.

### What I built: phases are a tree
`PHASE_CHILDREN`, `is_leaf_phase()`, `is_leaf`/`child_ms`/`leaf_ms` on every total,
`record_INCLUDING_upload` vs `record_excl_upload` in the share table, `<- NOT A LEAF` printed on the
same line as the total, and **`phase_leaf_accounting`** as the falsifier — `UNRESOLVED` when children
cannot be subtracted, *louder* when the non-leaf phase is also the largest (that is exactly when the
misreading is the natural one), `VACUOUS` when there is no non-leaf phase.

**With no transfer data `leaf_ms` is `None`, not the total.** Unknown is not equal to the parent.

### One restraint worth keeping
`host_phase_totals` does **not** derive the upload total itself — it consumes `record_scaling`'s
interval-containment attribution. Deriving it a second way (by transfer direction) was easy and
would have created two numbers that could disagree. **One number that can be checked beats two that
agree by luck.**

### The inversion is dead in both directions
After correct labelling the coordinator's run says NVIDIA is faster; **§6 says Intel is faster**.
Two runs, same build, opposite orderings, both failing the quiescence gate. It was never measured.

### The generalisation — and it is the day's real lesson
Three of the five defects here are the same shape: **wired, produces an artifact, and the
artifact's name misdescribes its content.** `record` is a real span with a real duration and a
caveat string asserting it is command-buffer recording. `index` is a real integer indexing a
different list than its reader assumes. Nothing is missing; nothing raises; a presence-check census
passes both.

> **A name is an assertion about content, and it needs a falsifier like any other assertion.**

`device_identity_check` (label vs evidence in the same artifact) and `phase_leaf_accounting`
(duration vs what its name claims) are the two instances in `bench/`.

### Still true
The contention guard stands. 9.5x inflation of `record` under load was correctly host CPU work —
a ~2 GB memcpy is precisely what degrades when six processes compile. `contention_signature`
compares host spread against GPU spread per slot and never depended on what the host time was
spent *on*.

### Note for Switch (in the decision inbox)
`rust/src/trace.rs:715` attaches "host: command-buffer recording; amortised across replays" to the
`record` span. **That caveat is now false and it ships inside every trace.** His file, not mine.

---

## 2026-07-30 night — admissibility: whether a stored number may be quoted at all

**Context.** Standing directive from Justin that performance is now continuous and first-class,
plus new rules R10/R11/R6-amendment-4. My charge: *"you own whether a performance claim is
admissible."*

**Merged `origin/main`** and immediately found what the merge would have done to me: `trace.rs` now
emits `desc_alloc`, `pipeline_lookup` and `cmd_upload` as real `ph:"X"` spans **inside**
`vulkan.record`. My sibling summation would have double-counted them — ~2x host inflation, every
share moving with it, nothing raising, no test failing. Same defect class as record-is-68%, one
level down.

**Built:**
- `phases.phase_nesting` — parenthood from **timestamp containment** (evidence), cross-checked
  against the `host/sub-record:` caveat (name). Red in both directions. Containment is primary
  because R11 says a name is not a definition; a rename cannot disable the check.
- `phases.sibling_phases` — union of the static table and the trace's own declaration. Asymmetric
  on purpose: table catches a missing caveat, declaration catches a child added after the table.
- `phases.upload_accounting` — the counters and the `cmd_upload` span measure the same memcpy.
  Prefer the span, corroborate with the counters, red on >25% disagreement, **VACUOUS not pass**
  with one instrument. Named after Tank's `alloc_device_upload_bytes` reading 0 while `cmd_upload`
  was 15.2 s.
- `phases.decomposition_identity` — R11. `internal_closure` is labelled **`WEAK` in the artifact**
  with the 99.0%-that-was-wrong quoted inline; only `external_closure` (harness `perf_counter`) can
  fire. No independent whole => `UNCHECKABLE`, `ok=False`.
- `stats.Sample.loop_wall_ms` + threading through `_run_trace_pass` so the external closure is real
  rather than theoretical.
- **`bench/admissible.py`** — the piece that did not exist. Every other guard runs at measurement
  time inside the producing process; a JSON file in `results/` carries none of them. Five gates,
  exit 1 on any inadmissible artifact. **Absence of a check is a refusal, not a default green.**
- `admissible.baseline_comparability` — the cross-artifact check. Neither GQA file is individually
  remarkable; the defect lives *between* them.

**Found.** The GQA pair: CPU baseline **6226.8 -> 345.2 ms, 18x**, across a Vulkan-only change.
Naive difference reads **5.44x speedup** — the best number this project has produced. Baseline-
normalised, Vulkan got **3.3x worse**. Both inadmissible. `post-gqa-dev1` (230.7 vs 254.0) would
have read as **the first win** and fails four of five gates.

**The guard's first real act was to refuse a result we wanted.** That is the test of whether it is
a guard or a decoration.

**State of the record: there is no admissible end-to-end performance number in this repository.**
Not a regression — the numbers were always this weak, the reporting has caught up.

**Judgements I would defend:**
- Containment over caveat as the primary nesting source. The alternative is a check a rename
  disables.
- `WEAK` written into the artifact rather than into the docs. Caveats travel with numbers or they
  do not travel.
- `NOT_A_RESULT` as a distinct grade. A false red costs a falsifier its authority as surely as a
  false green does, and I nearly shipped one against `timestamp-audit.json`.
- `WITHDRAWN` is not a failure. Re-flagging a retracted artifact forever teaches people to ignore
  the output.

**Still blocked on a quiet machine** (23 contention samples today, zero quiet): both-device
re-measurement, and the warmup-decline discriminator.

163 tests in `bench/` pass (was 141).
