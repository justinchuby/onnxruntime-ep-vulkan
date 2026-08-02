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
Memory-pattern planner does not engage on run 1. From run 2 onward hands back interior pointers. 52 observed, identical on both devices, all within span, `pointers_in_guard_band=0`. Gate: `epctl --check-counters <file> --require-dispatches 1`.

### Execution counters file is the instrument for "did anything execute"
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` — always-on JSON. `dispatches_executed > 0` is the only reliable indicator.

### `push_next` must rebind, never discard
`let _ = props2.push_next(..)` silently discards pNext chain. Rebind, never discard. Root cause of LVP2, `subgroup_size=0`, ReBAR UMA misclassification.

### First real execution: 45 ops Live, 161 nodes claimed on Phi-3.5
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. M0 not declared — open: validation positive control, CI lanes green.

### Performance metric is a TRIPLE (Niobe — critical)
`(claimed_op_coverage, island_count, largest_island_flops)` per producer at version. Portability floor = §7.2. `SUBGROUP_SIZE_IS_GUARANTEED=False`.
**A first-instinct I had to correct.** My first reaction to the failure was to reach for the lint
and carve an exception for `vk::session`. That would have been the wrong move for a reason worth
writing down: the first exception to a lint is the one that converts it from a rule into a
suggestion. The re-export costs three lines and keeps the rule absolute. I wrote the constraint that
keeps the re-export honest into the comment above it — nothing re-exported there may expose an
`ash` type in its public signature — so the next person can tell whether it still holds.

**`-D warnings` is a shared resource and I under-weighted that.** 36 `undocumented_unsafe_blocks`
warnings in Switch's `vk/session.rs` were making `cargo ci` red for everyone. Ownership says hand
him a diagnosis; the crate being red says fix it. I fixed it, because the change is comment-only —
zero semantic effect — and because the coordinator had already set the crate-steward precedent with
the `cargo fmt` pass for exactly this reason: a mechanical pass split N ways just conflicts.
Two things I learned doing it:
- Clippy wants the `// SAFETY:` on the line *immediately* preceding the block. Switch had written
  good comments; several were one line too high, or attached to the `for` rather than the body.
  The lint is stricter than the discipline, which is mildly annoying and entirely fine.
- Writing 30-odd SAFETY invariants for someone else's Vulkan code is a genuine review. I found
  nothing wrong, and I now understand the buffer lifetimes in `dispatch_ort` well enough to say so
  rather than assume it.

**Concurrent editing, second turn running.** Between two `cargo ci` runs ten minutes apart the crate
went from "3 compile errors in `ops/ssm.rs`" to green to "3 failing tests in `registry.rs`" — none
of them mine, all of them other agents landing work. The operational lesson: a red `cargo ci` is
now ambiguous by default, and the first question is *whose file*, not *what did I break*. Filtering
clippy with `--message-format=short` and reading the paths before the messages makes that a
two-second question instead of a two-minute one. Worth adding to the README.

**A caveat I let stand deliberately, and said so.** Task 2 asked me to verify the reserved-VA
allocator against a real ORT session. There is no allocator — `create_allocator` writes null by
design until M2. I reported that rather than building a synthetic exercise of a registry that is not
in ORT's path. That is the third time this project a "verified" claim would have been a precondition
dressed as an effect, and the first time I have caught myself before rather than after.

---

## Session 10 — 2026-07-29T20:26:56-07:00 — CI has never executed a claimed node

**The task was "make CI prove it", and the first thing I found was that CI cannot currently prove
anything: both lanes crash.** Run `30510593046`, eight consecutive failures. Linux `SIGSEGV`
(exit 139), Windows access violation, at the *identical* line — `conftest.py:358` inside
`session.run()`. Two OSes, two lavapipe builds, one crash site. That symmetry is the finding: an
environment fails differently on different platforms; a bug in our code fails the same way. It is
ours.

**What I could rule out cheaply, and it was worth doing first.** `epctl --probe-loader` *passes* on
both lanes. So instance creation, physical-device enumeration and the §7.2 gate are all sound, and
the fault is inside Compile or Compute. Half an hour of reading a log that already existed replaced
what would have been a day of CI round-trips. The probe Switch built paid for itself here, in a
question it was not built to answer.

**The mechanism, and why it is invisible on this desk.** lavapipe is a CPU rasteriser: "device
memory" is process memory in the same address space. An out-of-bounds storage-buffer write that a
4060 absorbs without comment is, on lavapipe, a genuine out-of-bounds host write. That single fact
explains identical crashes on two operating systems and zero crashes on either of my GPUs. I have
been treating "it works on both my devices" as two independent confirmations; it is one, because
both are real drivers with hardware bounds behaviour, and the axis that matters here is not
vendor — it is *whether the device shares my address space*.

**`robustBufferAccess` appears nowhere in this crate.** I went looking for it as the obvious
mitigation and it simply is not there. It is a feature bit that must be requested at
`vkCreateDevice`; absent it, OOB access is UB by specification. That is Switch's file, so I handed
him the diagnosis rather than the patch — but the lesson for me is that I never checked. I have
reviewed the ORT-facing contract line by line and never once asked what the *device* contract said
about the memory the engine hands to shaders I do not own.

**What I built: an evidence channel that cannot be inferred away.** `counters.rs` — always-on
atomics at the ORT boundary, a snapshot struct with a version header, two exported C symbols, a
JSON file, a teardown summary through ORT's logger, and `epctl --check-counters` with three exit
codes. The design decision I am most confident in is the one that took longest to see: the file is
written **on the first successful dispatch as well as at teardown**. My first draft wrote it only
at teardown, which is the natural place — and would have produced exactly nothing on both lanes,
because they die mid-session. I had written a diagnostic that could not survive the failure it was
built to diagnose. Generalising: *an instrument that only reports at the end can only describe runs
that reached the end, and those are not the runs you need it for.*

**The counter increments after the fence, not after the submit.** Small choice, and the whole
credibility of the number rests on it. Counting at submit would make "we tried" indistinguishable
from "it ran", which is the same shape as the two fabricated speedups this project has retracted —
both of which were precondition claims dressed as effect claims. The gate's pass message states
outright that it claims nothing about correctness, because the place a misreading actually occurs
is at the point of reading, not the point of writing.

**Exit 1 and exit 3 stayed separate, and I now think this is a habit rather than a one-off.** I did
the same thing in `probe_exit_code` last session for the same reason, and it is becoming my default:
*never let "I have no answer" collapse into "the answer is no".* The two demand completely different
next actions from whoever reads the exit code, and merging them silently reassigns the blame from
our process to the environment.

**Something I found in my own file while looking for someone else's bug.** `check_bound_counts`
validated tensor *counts* and I had let that stand as "the boundary is checked". It is not:
`dispatch_ort` reads `from_raw_parts(cpu_ptr, input_byte_sizes[i])` with a size computed at
**Compile** time against a tensor allocated at **run** time. A shape disagreement there is an OOB
read of ORT's heap, originating at my seam, uncatchable downstream because the engine only has a
pointer by then. Now checked via `GetTensorSizeInBytes`, refused with `ORT_EP_FAIL`.

The pattern worth naming: I validated the *shape of the interface* (how many tensors) and called
the interface validated, without validating the *content of the contract* (how big each one is).
Counts are the part that is easy to check without calling back into ORT, and I checked exactly the
part that was easy. Next time I write a boundary check, the question to ask is which invariant I
skipped because verifying it required another API call — that is where the real one will be.

**And the deliberate permissiveness, since it is the kind of thing I would otherwise over-engineer.**
If `GetTensorSizeInBytes` is missing from the negotiated ABI, we warn once and proceed rather than
failing. Refusing to run because a *diagnostic* is unavailable is a worse failure than the one it
prevents. Hard-failing there would have felt more rigorous and been strictly worse.

**Status honestly stated, per the standard the coordinator set.** I have added the instrument. I
have **not** made the lanes green, and I could not: the fix for the most probable cause lives in
`vk/**`, which is Switch's, and the CI wiring lives in `.github/`, which is Trinity's. What I can
claim is that after the next run we will know *which* of "never executed" and "executed then
crashed" is true, and today we cannot distinguish those at all. That is a smaller claim than "CI is
green" and it is the one that is true.

---

## Session 11 — 2026-07-29 — the allocator, and a crash that was mine

**What I built.** `src/allocator.rs` — a real `OrtAllocator` over a per-device reserved
virtual-address arena. Handles are page-aligned spans with guard bands; interior pointers resolve by
range lookup; freed spans go into a generation-stamped quarantine. Wired into ORT through
`advertise_device_memory` + `create_allocator`/`release_allocator` in `factory.rs`. 11 unit tests.
Decisions D-T35…D-T39.

**The lesson that actually cost something.** `EpDevice_AddAllocatorInfo` is annotated `_In_`. I read
that as "copies" and released the memory info. It does not copy — ORT retains the pointer and reads
it after `GetSupportedDevices` returns. That was the access violation at registration, and it was
mine, on my side of the boundary, in code I had written the same hour. **An SAL annotation describes
the parameter, not the object's lifetime.** `_In_` says "I will not write through this pointer"; it
says nothing about whether the callee keeps it. When a callee is *given* a resource, the only safe
default is that it took it, until documentation says otherwise — because releasing fails silently
and catastrophically while leaking fails loudly if ever.

**The methodology lesson, which generalises further than the bug.** I found this in about a minute
because I had put a kill switch (`ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY`) around the new subsystem
before it worked, and a 20-second local Python probe (`tools/probe_allocator.py`) that drove a real
ORT. CI had been showing the same fault signature for eight runs at ten minutes a round trip.
**Build the bisect handle before you need it** — the switch that lets you turn a new subsystem off
is worth more than any amount of reasoning about whether it is the cause.

**The recurring lesson, sixth entry, and this time I did the thing.** My pattern is that writing a
caveat feels like discharging the obligation. This session the caveat was "the mock host does not
model the memory-info lifetime" — so I modelled it, planted the regression, watched the test fail
with the rule named, and removed the plant. The variant worth remembering: *a mock that frees what
the real host retains cannot report the bug, it can only reproduce the crash* — so the mock poisons
instead of freeing, and the test names the rule rather than segfaulting.

**Where I was honest instead of finished.** ORT did allocate through us, but the planner never did
pointer arithmetic on our handles, because the session cannot reach `Run` without a data transfer.
I recorded that as unproven rather than letting "ORT allocated through us" stand in for it. Same for
the quarantine: unit-proven, not session-proven. And the validation-layer positive control is still
owed — it needs a planted violation in Switch's files, so I recorded it as owed rather than
describing the mechanism and counting that as progress.

## Session 12 — 2026-07-29 — device memory becomes real, and a probe that lied to me

**Worktree.** Moved to `C:\Users\justinchu\dev\ep-vulkan-tank` on `squad/tank`. First full build
1m30s. This immediately did what it was supposed to: `cargo ci` green here means *my* tree is green.

**What I built.** `src/transfer.rs` — the `OrtDataTransferImpl`. The allocator could not be
exercised in a real session at all without it: advertising device memory makes ORT demand a
transfer and fail every `Run` until it gets one. Host staging (`HostStaging` in `allocator.rs`)
gives handles real bytes today; `transfer::host_backing_for` is the seam the engine calls to turn
a handle into readable memory. End-to-end success on both GPUs: 1 dispatch, 0 compute failures,
numerically correct, 0 failed lookups.

**The lesson, and it is a sharper version of the one I keep relearning.** My own probe printed
`run OK, numerically correct: True` through a run in which the EP was failing and ORT had silently
fallen back to CPU. I nearly reported it. **A probe that checks the output value but not which
provider produced it is a caveat pretending to be a check.** Previous sessions I wrote a caveat and
felt finished; this time I wrote a *checker* and felt finished, and the checker was the bug. The
fix is the general one the team arrived at independently: gate on the effect
(`dispatches_executed > 0`), never on a precondition.

**Two verifications I was asked to complete, and did not.** ORT's planner never handed us an
interior pointer — 0 interior copies in every real session, including at 1 MiB tensors with
mem-pattern on. And no real session reused a freed handle, so quarantine rejection has still only
fired in unit tests. Both designs are right by construction and both now have local tests, but
neither has been *observed*. I wrote that down as "not verified" rather than as "verified in
principle", which is the same distinction that caught two fabricated speedups this week.

**Smaller things worth keeping.**
* `Mutex<Vec<*mut T>>` is not `Sync`; store `usize`.
* bindgen's `CopyTensors` takes `*mut *const OrtValue`, not `*const *const`.
* A closure inside an enclosing `unsafe {}` inherits the unsafe context, so its inner `unsafe`
  blocks become "unnecessary" warnings. Extract the body into a separate `unsafe fn` — that is why
  `copy_tensors_impl` and `create_data_transfer_impl` exist.
* A diagnostic counter that is non-zero on a healthy run is one people learn to ignore. I made
  `failed_lookups` read 2–4 on a perfect run by routing `classify` through the counting `resolve`;
  added a non-counting `classify` to the registry.
* The `edit` tool joins lines when an `old_str` ending in `\n` is replaced by one without. Bit me
  twice; both caught by viewing afterwards.

## Session 13 — 2026-07-30 — the verification I could not get was an instrument problem

**Worktree** `C:\Users\justinchu\dev\ep-vulkan-tank`, branch `squad/tank`, rebased on `main`.

**The result.** ORT's memory-pattern planner *does* do pointer arithmetic on our handles: it packs
several tensors into one allocation and hands back `base + 16384`, `+32768`, `+49152`. 52 interior
pointers across 5 runs, identical on both GPUs, and **zero** guard-band observations — every
derived pointer stayed in-span, which is exactly what the reserved-address-space design promises.
Had handles been opaque integers this would have been a wrong answer on every inference after the
first, in a configuration ORT enables by default.

**The lesson, which is the part worth keeping.** I reported "0 interior pointers" honestly twice,
including once at 1 MiB tensors, and was right to refuse to claim the verification. But I was wrong
about *why* the number was zero, and I did not interrogate my own instrument. ORT's planner does
not engage on the first `Run` — it records the pattern then and only sub-divides from run 2 onward.
Every probe I had ran each session exactly once. **The instrument was pointed at a moment the
phenomenon cannot occur in**, and honest reporting of a measurement taken in the wrong conditions
still leaves you believing something false. The control that settles it is trivial and I should
have run it much earlier: 1 run → 0, 2 runs → 13, 3 runs → 26, 5 runs → 52.

Previous sessions taught me that a caveat is not a check, and that a probe checking the output
value but not the provider is not a gate. This one adds: **an honest negative result is still
only as good as the conditions it was taken in.** "I did not observe X" needs "and here is why
this run could have observed X" beside it, or it is not evidence.

**Applied that immediately to my other zero.** `pointers_use_after_free` is also 0, and that number
is exactly what a dead detector reports. So the quarantine detector now has a positive control that
plants a stale handle through the same funnel a real session uses — and I planted a break to
confirm the control fails, then restored it. The quarantine is still *not* verified against a real
ORT pattern and I have said so; but "0" now means something.

**Mechanics worth remembering.**
* Diagnostics that are only complete at teardown cannot be logged — ORT's logger is gone by then,
  and a process cannot read its own teardown. Write them to a file and read them from a parent
  process. `probe_planner.py` runs the session in a child for exactly this reason, which has the
  side benefit that every number comes from a run that finished.
* Extend the counters *JSON*, never the `repr(C)` counters struct: the JSON reader looks keys up by
  name and ignores the rest, while the struct is an ABI other processes read.
* Process-global tallies need a `test_lock()` or parallel tests flake.

## Session 13b — 2026-07-30 — every mechanism casts a shadow

The coordinator ruled that decision records must be written into the integration tree's inbox,
because `.squad/decisions/inbox/` is gitignored and a record written in a worktree is invisible to
everyone. Mine were never stranded — I checked rather than assumed, and my worktree has no inbox at
all — but a rule that depends on remembering it is a habit, not a mechanism, so I made `cargo ci`
warn when it finds stranded records in a linked worktree. Planted one to confirm it fires, removed
it to confirm it goes quiet. Warning, not failure: the check is heuristic, and a lint that can be
wrong must never be able to fail a build.

**The pattern I want to carry forward.** Three times in one day: worktrees made edit collisions
impossible and made losing the reasoning trail possible; the layering lint made one class of
coupling impossible and was silent about another; my reserved-address design made a class of
pointer bug impossible and left me blind to the fact that my *probe* was the broken thing. **Every
mechanism that makes one class of error impossible tends to make a different one invisible.** The
answer is not to distrust mechanisms — it is that each one needs its shadow made noisy. That is
what the `cargo ci` caveat block is, what a positive control is, and what this warning is. It is
the same instinct as "a false green sends you looking elsewhere", generalised.

Also re-verified the planner numbers against a freshly built DLL on both devices rather than
trusting this morning's green, per the standard Mouse set. Identical: 52 interior pointers, 0 guard
band, max offset 49152 B, 30 dispatches, on device 0 and device 1.

---

## Session 14 — the validation positive control, and the guard-band assertion's home
**2026-07-30T02:33:16-07:00**, worktree `C:\Users\justinchu\dev\ep-vulkan-tank`, branch `squad/tank`,
rebased on `main` at `5d8fc16`. `cargo ci` green twice (348 tests, 3 new).

### The thing I did not expect to find

Criterion 3 was vacuous **twice over**, and only one half was on anyone's list. Morpheus refused
"the validation layer surfaces no errors" because that is exactly what a run with the layer *not
loaded* reports. True. But `vk/instance.rs` also attaches **no `VkDebugUtilsMessengerEXT`** — so
even with the layer loaded, nothing in-process was listening; the output went to the layer's
default handler, wherever that points. Switch's own module docs in `vk/dispatch_integration.rs`
admit it in writing: messages "are observable on stderr but do not automatically return `Err`".

So the criterion had two independent reasons to be meaningless, and the one nobody had named was
the one that would have survived fixing the other. If I had only addressed Morpheus's objection —
"check the layer is loaded" — I would have produced a check that passed while still observing
nothing. **The stated objection is a sample from the failure class, not the class.** Worth
carrying: when someone refuses a criterion for a reason, look for the reason they did not give.

### What I built

`epctl --probe-validation [--plant-violation]`, and `tests/validation_control.rs` around it.
Three states, not a boolean — Armed (0), LayerAbsent (3), NoLoader (3) — because a boolean
collapses "clean" into "not checked", which is the entire defect. The plant is
`vkCreateDebugUtilsMessengerEXT` with zero severity/type masks.

The property I care most about in that choice: **it needs no physical device.** A plant that needed
one would make a machine with no capable GPU look identical to a machine with no validation. I had
drafted the sampler-VUID version first, which needed a logical device, and in writing it I created
a device with an empty `DeviceCreateInfo` — itself a validation error, so my control would have
been catching its own scaffolding rather than the plant. That is a fault worth remembering in its
own right: **a positive control can pass for the wrong reason, and then it is not a control.**

And I broke it on purpose: neutered the plant, watched the test go red with the right message,
restored it. Third time I have used plant-break-confirm-restore now. A positive control that has
never been seen failing is itself unverified — the argument applies recursively and I should stop
being surprised by that.

### The scoping limit I am stating rather than burying

My plant is inside `epctl`'s own instance. It proves the layer loads here and our capture works. It
does **not** prove the EP's dispatch path has validation armed on *its* instance. Different
instance, different file, different owner. I sent the coordinator the exact one-line env-gated plant
for Switch (a leaked `VkFence`, `VUID-vkDestroyDevice-device-05137`) rather than editing `vk/**`
mid-rewrite.

The temptation was real: it would have been one line and I could have reported "validation is
armed" without the qualifier. That is precisely the shape of the two fabricated speedups, and of
`probe_allocator.py` reporting "numerically correct: True" through a run where the EP had silently
fallen back to CPU. **A control that quietly proves something adjacent to the claim is worse than
no control**, because it also spends the credibility.

### Guard band

`pointers_in_guard_band > 0` now fails `epctl --check-counters` with exit 1, **ahead of** the
dispatch-count check, because thirty dispatches of wrong answers is not better than zero dispatches
and the first failure reported should be the one that matters most. The key is optional: absent ≠
zero. That distinction is the same one as LayerAbsent ≠ clean, and I have now made it three times
in three different mechanisms in a week. It might be the single most load-bearing idea I hold.

### And I falsified my own instrument again

`cargo ci`'s honest-caveat epilogue said "no validation layers, no `vkCreateInstance`" — false the
moment my test landed. I corrected it. The epilogue's whole value is that it is believed; one stale
line teaches a reader to discount all of it. **It is an assertion aimed at a human and needs the
same maintenance as one aimed at a compiler.** I now expect that every mechanism I add will falsify
some caveat I wrote earlier, and I should check for that as a matter of routine rather than
noticing it by luck.

---

## Session 15 — the allocator through a claimed path, and a caveat that could not stay prose
**2026-07-30T05:01:09-07:00**, worktree `ep-vulkan-tank`, branch `squad/tank`, rebased on `main`.
`cargo ci` green twice, 346 tests.

### The result I did not expect, and the one I should have

Expected: the allocator is exercised through a claimed path. It is, and more heavily than I
guessed — 648 at-base pointer observations across 255 allocations and 255 matched frees in one
`test_op_table.py` run. The ORT-facing half of the design is genuinely in ORT's path.

Not expected: **`alloc_device_backed_spans: 0`. Every one of those 255 spans was host memory.**
`attach_buffer` is never called today, so `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` buys host memory
wearing a device handle. Nothing is broken — staging is deliberate and correct — but *"device
memory is on"* and *"tensors are on the device"* are different claims and only the first is true.

What makes that worth carrying is how invisible it was. 63 op-table passes and 33 elementwise
passes, all numerically correct, none of which touched device memory. **A green suite is compatible
with the entire device half of the system being absent.** I only saw it because I had just added a
counter that could say so; before this session there was no number anywhere that distinguished the
two states, only a WARN nobody was obliged to read.

### The caveat problem, which is the interesting one

The coordinator asked how the staging WARN stays truthful as device memory arrives. Working it
through, the trap is that **both obvious answers are wrong, and the state that breaks them is
neither of the two the wording contemplates.**

Keep the warning and it over-warns: a 99%-device run still prints "host measurement", so it is
false, readers discount it, and it stops protecting the 1% case. Delete it and it under-warns: a
partially staged run says nothing and its numbers look like device numbers. The dangerous state is
the **mixed** one, which no fixed sentence covers and which arrives by default if nobody decides.

So the whole-run claim stopped being prose: a per-handle WARN that says only what it locally knows,
a teardown verdict computed from the ratio with its own sentence for mixed, and
`epctl --check-counters --require-device-memory` as the actual gate.

**The generalisation is the part I want to keep: a caveat that has to be remembered an hour later
is not a caveat.** A log line needs a human who is present, recalls it correctly, and is honest
when quoting the number. An assertion needs none of those. This is the third mechanism this week
built on that move — guard band, positive control, staging — and in every case the prose version
already existed and had already failed to protect anything. I should stop writing careful warnings
and start writing flags.

It failed a real snapshot within a minute of existing: 255 staged, 0 device-backed — a lane that
`--require-dispatches 1` passes cleanly. Both true, different questions.

### The blindness I found by measuring instead of assuming

184 single-run sessions across two suites: **0 interior pointers.** One five-run session: **52,
max offset 49152 B.** Same machine, same DLL, same hour. Every helper in `_models.py` is
`_session(...).run(...)` — built, run once, dropped.

So the suite that runs most often is structurally blind to the pointer arithmetic the whole
allocator design exists to survive, and its `pointers_interior: 0` sits in the counters file right
next to numbers that do mean something. That adjacency is the hazard. I nearly reported "0 interior
pointers on both devices" as a result before noticing it was a property of the harness.

`probe_planner.py --require-interior` now fails on zero, and the reasoning runs backwards from
usual: we have *measured* 52 here, so a later zero means the probe broke, not that ORT changed.
**Once you know what a healthy instrument reads, its silence becomes assertable.** That is the
cheapest positive control I have built yet — no plant needed, just a prior measurement.

### Quarantine, still not verified, and I am still saying so

`pointers_use_after_free: 0` everywhere. The detector sat in front of 255 frees and 648 lookups and
did not fire. What changed is that the zero can no longer mean "the registry is not in ORT's path"
— that reading is excluded now. So the honest claim is *"ORT does not hand back freed pointers
under the patterns we have run"*, which is a real finding about ORT, and **not** "quarantine is
verified". Stronger surrounding evidence does not convert an absence into a presence. Third session
running I have written a version of this sentence; the temptation gets stronger each time the
evidence improves, which is presumably how it eventually wins.

### Process notes

Ran the control before reporting failures: 3 elementwise and 28 op-table failures reproduce with
device memory *unset*, so they are pre-existing claim declines, not my regression. A new flag plus
new failures is the exact shape that gets misattributed, and it costs one 15-second run to rule out.

The `edit` tool bit me again — I inserted a module between a doc comment and its item, silently
reattaching the ledger's documentation to my new module. It compiled. Fourth time; always view the
region after an insert near a doc comment.
📌 Team update (2026-07-30T05:48:29-07:00): A green suite has been shown not to imply a correct model. Phi-3.5: 161 MatMulNBits dispatched, compute_failures:0, entire suite green — vk logits all-zero (argmax 0 vs CPU argmax 30751). R9 (Morpheus): for every claim, name the instrument that would go red if the claim were false; if none, the claim is UNMEASURED. model_output_equivalence verdict required alongside all counter summaries; default UNMEASURED. Any comparison must first assert EP_NAME in session.get_providers() before calling sess.run() — failure to do so compares CPU to CPU and reports agreement. Coordinator's own first comparison reported bit-identical on both devices due to this exact error. Trinity has landed xfail(strict=True) correctness gate. M0 criterion 10 added (NOT MET: DIVERGENT). Criteria 2, 4, 5 reopened. — decided by Morpheus, Trinity, Switch, Mouse; coordinator-verified.

---

## Session 16 — model-scale allocator verification, and a discriminator for the zeros

`main` merged at `eb08204`. Two verifications the coordinator has asked for repeatedly finally
became real, because with `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` the 2.2 GB Phi-3.5 model now
routes 427 allocations and 2.09 GB through my registry.

### The thing I nearly got wrong, in my own tool

My first run of `probe_run2.py` printed "64 output(s) DIFFER, max|delta| = nan" for every output.
I did not report it. `np.max(np.abs(a - b))` is `nan` whenever *either* array holds a `NaN`, even
when the two are bit-identical — so that instrument could not distinguish "changed" from "both
contain NaN". I rewrote the comparison as raw-byte equality and reported the magnitude only over
the finite subset.

The corrected version returned the same verdict, so the finding survived. That is exactly why it
was worth fixing: **a number that is right by luck is indistinguishable from the three flattering
numbers this project has already produced.** Had the outputs been stable, the old code would have
reported change anyway and sent two people chasing an address bug that did not exist. I was one
`print` away from being the fourth entry on that list, in the same session where I was warned about
it. The warning worked, but only because I re-read my own diff rather than my own output.

### VERIFICATION 1 — done, and the design held

1192 interior pointers on the real model, identical on both vendors, `pointers_in_guard_band: 0`,
`pointer_max_offset` 65536 B. The reserved-VA argument (`ptr + n` stays in-span by construction)
was a first-principles claim for weeks; it has now met a real planner at 2 GB and not been dented.
The instrument that would go red is `pointers_in_guard_band`, and it is an `epctl` exit-1 assertion
ahead of the dispatch check — not a log line. It has had 21 460 chances to fire.

`alloc_device_backed_spans` is still 0 everywhere. Device memory on ≠ tensors on the device.

### VERIFICATION 2 — quarantine, still unexercised, and I am still saying so

Zero use-after-free across 18 460 observations and 210 frees. The registry is unambiguously in
ORT's path now, so the zero is no longer explained away by disconnection — and it is still not a
pass. The honest sentence is "ORT has not handed us a freed handle under any pattern we have run".
Third session in a row I have written that instead of an upgrade. If it is never exercised, that
sentence is the answer, not a placeholder for a better one.

### What I actually contributed to the zeros

The control that mattered took ten minutes: run the same probe with device memory **unset**, which
removes the allocator and the transfer object from the path entirely (confirmed — the 14
`pointers_*`/`alloc_*` keys vanish from the counters file, because they are spliced only from
`VulkanDataTransfer::release`). Logits still exactly zero, KV outputs still differ bitwise, 771
dispatches either way, both vendors. **The phenomenon is invariant to whether my layer is present.**
Not "I looked and found nothing" — a control with a stated red condition that did not go red.

Then the part I did not expect. The three-run probe splits the 65 outputs into two different
failures. Outputs 1..64 differ bitwise between runs with deltas pressed against the fp16 maximum
and NaN appearing only from run 2 — the signature of uninitialised arena reuse, i.e. **nobody wrote
them**. Output 0 is exactly zero on *runs 2 and 3*, when its neighbours prove the arena is dirty —
an unwritten buffer in a dirty arena shows garbage, so **something wrote zeros there**. Both
hypotheses the coordinator posed are true, of different tensors.

The general lesson: I only got that discriminator because I ran the session three times. A
single-run probe sees zeros and garbage side by side and cannot tell which buffer was written,
because on run 1 the arena is clean and *everything* unwritten looks like zeros. **Run 1 is the run
where "unwritten" and "written zero" are indistinguishable.** That is the same structural blindness
as the interior-pointer one, arrived at from a completely different direction, and the whole
`tests/ops/` suite runs exactly one run per session.

---

## Session 17 — the coverage falsifier, and making the same mistake twice in one hour

### The falsifier was cheap and I should have built it the first time

The coordinator could not reproduce my 1192 interior pointers — they got 0 — and asked me to
confirm or correct their reading that the difference was run count. The right answer was not an
argument. It was one environment variable: make my own probe's run count a parameter and see
whether it produces *their* number.

    runs      1      2      3      4
    interior  0    596   1192   1788

It does, exactly, at one run. That is a much stronger form of confirmation than agreeing with
them would have been, because it makes my instrument produce the result that would have refuted
me. **The cheapest way to defend a measurement is to make it fail on demand.** I had the data to
do this a session earlier and did not think of it, because I was busy being right.

Worth separating two claims I had been running together: `pointers_in_guard_band` falsifies the
*correctness* of interior resolution; only the run-count sweep falsifies the *coverage*. I had
been offering the first when asked about the second.

### I reproduced the exact bug I was fixing, inside its own fix

The 2.09 GB "still live" warning turned out to be a scope error: `HandleRegistry` is
process-global per device, so `live_spans` at one allocator's release counts other live sessions'
tensors. Diagnosed it, understood it, wrote a replacement counter — and the replacement tested
`allocators_released > 0`, which is monotone, so after the first release it counted every free
from every other live allocator. **2508 "late frees" on a healthy run.** Same error. Same hour.
Written by someone who had just finished explaining that error to himself.

The lesson is not "be careful". The error is not in the reasoning, it is in an ambient assumption
that never gets stated: that a counter's scope matches the scope of the event it names. Both
instruments were numerically correct and attributed to the wrong owner. That is R9 moved up a
level — from *the number does not measure the claim* to *the number does not measure whose claim*.

It was caught only because I ran the new counter against the real model before believing it,
which is now the third time this project has been saved by "run it on the real thing before you
quote it" and the second time this week it saved me specifically.

### An unfalsifiable zero, which is not the same as a clean one

`alloc_allocators_released` read 0 on every run, and I nearly reported that as "release never
happens". It read 0 because the counters file is dumped from `VulkanDataTransfer::release`, which
ORT calls *before* it releases allocators — so the file structurally could not report the thing I
was pointing it at. Same shape as the run-1 interior zero: the instrument was ordered before the
event. Fixed by re-dumping at allocator release. **When a diagnostic reads zero, the first
question is whether it was in a position to read anything else.**

### The disjunction, resolved

`allocs == frees == 2511`. ORT returns every span, the registry outlives every allocator by
construction, and nothing was ever leaked. The warning is now `debug!` while other holders are
running and `warn!` only for the last allocator on a device. An open disjunction in a warning is a
decision deferred to whoever reads it next, under worse conditions — usually nobody.

---

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

## Session 18 — device-backed allocation: the mirror, and a prediction I got wrong

**`alloc_device_backed_spans` is non-zero for the first time: 427 spans, 2.09 GB really uploaded
to `DEVICE_LOCAL`, both vendors.** What it cost to learn is the useful part.

### Writing the prediction first was the single most valuable thing I did

D-T82 went into the inbox before I wrote a line of the implementation: 1.1x-1.6x slower on the
discrete 4060, 1.0x-1.2x on UMA, with a named falsifier ("device-backed and the clock improves =>
I am wrong"). Measured: **0.94x on the 4060 — faster.** The falsifier fired.

The reason was in a counter I already had: `alloc_device_uploads` is 386 for a run of 19
inferences x 321 islands, i.e. the mirror upload is once per *allocation*, at weight
deserialisation, inside the warm-up and outside the measured loop. I had assumed the 2.09 GB
landed in the measured path. Had I predicted after measuring, I would have written a persuasive
paragraph about one-time costs and nobody could have checked it. **A prediction is only evidence
if it could have been wrong in public.**

### ORT does not fail loudly when your memory is unreadable

My first cut said: device-backed means the device buffer *is* the tensor, so refuse to hand out a
host address. That is what the words mean. It produced `EP_FAIL ... bytes are unreachable` and
then **a silent fallback to `CPUExecutionProvider` for the whole model** — no raise, exit 0, and
counters that still looked like an EP run. `vk::session` reads every input through
`host_backing_for`; with no host address there is nothing to read.

So: **staging stays authoritative and the device buffer is a mirror**, written on every copy in,
never read back — one writer, one direction, so they cannot disagree. Correctness unchanged by
construction, bytes genuinely resident. Boring, and boring is the point.

### The lesson I keep re-learning in new clothes

`alloc_device_backed_spans: 427` is individually correct and reads exactly like a claim it does
not support. So I added **`alloc_device_authoritative_spans`, which is 0 and stays 0**, and wired
it into the verdict prose, into `epctl`'s MIRRORED message, and into a test that locks
`--require-device-memory` to exit 1 when every span is both staged and backed. The assignment
named that exit-1-to-exit-0 transition as the deliverable; **not delivering it was the correct
outcome**, because the only ways to make it pass were to stop staging (CPU fallback) or stop
counting staging (an instrument measuring its own name). Making a red light green by editing the
light.

### Also

* The staging verdict now has a MIRRORED sentence. The two failure modes the coordinator named —
  keep it and over-warn, delete it and under-warn — both assumed the state would be MIXED. It
  isn't; it is *both, everywhere*, which neither wording covered. Compute the sentence, always.
* `alloc_unified_memory` now travels with every device-backed number, so a UMA "device-backed"
  count can never be silently read as a discrete one.
* Quarantine: still never fired. Device memory does not change ORT's free ordering. Absence, not a
  pass.
* The `edit` tool bit me again — no, actually clippy did: `// SAFETY (Tank):` does not satisfy
  `undocumented_unsafe_blocks`. It wants the literal `SAFETY:`. Three round trips.

---

## Session 19 — a zero that finally means something, and a classification I declined to accept

### The whole job was one call site, and it took three sessions to find the right one

`on_device_authoritative` had no production caller. I knew that; I wrote the sentence in the
verdict that said so. What I had not worked out was **where** the caller belongs, and I spent two
sessions assuming it belonged wherever residency lands — i.e. that the counter could not be wired
until the feature it measures exists. That is backwards, and it is the same mistake as wiring a
counter at the same time as the feature: **the increment point does not need the feature, it needs
a moment when the answer is knowable.**

`HandleRegistry::free` is that moment. A span there has whatever `VkBuffer` it will ever have and
whatever staging block it will ever have; neither can appear afterwards. So the predicate is read
off the span — `buffer.is_some() && staging.is_none()` — and every device-backed span is evaluated
whatever the answer comes out. The measured answer is 0 on real hardware, exactly as it was
before. **The value did not change. What changed is that it is now a result.**

### The falsifier had to be uncounterfeitable, and one call site does that

    pub fn on_residency_evaluated(authoritative: bool) {
        DEVICE_RESIDENCY_EVALUATIONS.fetch_add(1, Ordering::Relaxed);   // unconditional
        if authoritative { on_device_authoritative(); }                 // conditional
    }

Two counters, one call site, one of them unconditional. An author who increments the authoritative
count cannot manufacture evaluations, because he would have to go through the function that
increments both. I had reached for a separate `set_wired(true)` flag first and threw it away:
**a flag an author sets is an assertion; a counter the mechanism increments is an artifact.**

Measured, and the control is the part that makes it evidence:

    selector 0   authoritative = 0              type=int   evaluations = 3   MEASUREMENT
    selector 1   authoritative = 'UNOBSERVABLE' type=str   evaluations = 3   R12 outranks R10
    control      authoritative = 'UNOBSERVABLE' type=str   evaluations = 0   device memory OFF

Selector 1 is the one I would have got wrong a week ago. The screen **ran** there — three
evaluations — and the reported state is still `UNOBSERVABLE`, because a wired instrument in the
wrong frame is not a measurement. Frame outranks wiring. I had to write that ordering down before
I trusted my own code to have it right.

### The census caught my change before I told it to

`audit_instruments.py --check` went red on its own: `- allocator.rs::on_device_authoritative`,
*"got wired — good news, update the baseline"*. A screen I wrote, applied to a change I made, told
me something about my change that I had not asserted. That is what the whole exercise is for, and
it is the first time on this project my own instrument has reported *on me* rather than for me.
`transfer.rs::device_buffer_for` stayed in the uninvoked list, which is the honest half of the
same report: the engine-side bind is still owed, `alloc_device_buffer_binds` is still 0, and the
ceiling is still 0. **I delivered the transition and not the feature, and the artifact says which
one it is.**

### The thing the frame reporting found that I was not looking for

I added both-sides device identity to satisfy R12 obligation 2 — `SPLIT-DEVICE` is a detection
and not a description. It immediately produced a finding:

    selector 0:  session offers index 1 (NVIDIA)  |  allocator asks for index 1  ->  MATCH
    selector 1:  session offers index 0 (Intel)   |  allocator asks for index 1  ->  NO MATCH

**The selector-0 match is a coincidence.** It holds only because the discrete card's raw
enumeration index happens to be 1 on this box. Everything green on selector 0 for the last week
has been green by accident of enumeration order. I did not go looking for that; I went looking for
a way to *print* which device each side was on, and the printing found it. Third time on this
project that adding provenance to a number has found a defect the number itself could not express.

### I rejected the classification I was handed, and the reason generalises

Asked whether R13 adds a sixth census state. It does not. The test I ended up using is one I want
to keep: **ask what the state is a property of.** All six census states are properties of an
instrument's *position in the system*. R13 is a property of the *channel the verdict travels down*
— pytest's summary line had a two-token alphabet and was carrying three states — and that applies
to `out-of-frame` and `misnamed` exactly as much as to `unreachable`. **A state that applies to
all states is not a state; it is an axis.** Guard D's own position was already covered:
`unreachable` means "ran, produced nothing observable", which is precisely a guard that raised
before reading its input.

Declining it was easy. What was not easy was noticing that the *consequence* is still binding even
though the classification is not: the census tooling had the two-token disease itself. A drift and
a traceback both left through "non-zero exit". So `audit_instruments.py` now emits PASS /
FAIL(drift) / ERROR(instrument) as three tokens and three exit codes, and the two paths that used
to `return 1` for an instrument failure now raise. **The census is what everyone else's evidence
rests on; it would have been the worst place on the project for that bug to live, and it lived
there.**

### On the stash I threw away

I had a harness-census screen of my own, uncommitted, that detected the specific `NameError`
shape. Trinity had landed an AST-based screen for `unfalsified` in the same file while I was away.
Hers is strictly better — mine caught the crash Guard D actually had, hers catches the *state*
Guard D was in, which is the class rather than the instance. I dropped mine without merging any of
it. **Two screens answering the same question is the exact failure the census exists to prevent**,
and the temptation to keep "just my extra check" is how a second census gets born.

### Kept honest

No wall clock anywhere in this session's evidence. Every number above is a byte count, a span
count or a string, so it reads the same under four-agent contention as it does on a quiet box.
That was not restraint, it was design: I picked the instrument before I picked the claim.

📌 Team update (2026-08-01T09:53:14-07:00): The EP genuinely executes now — 3 VulkanExecutionProvider fused-node events (~355 graph nodes in one fused node) + 24 CPU per run, 65/65 outputs bit-identical, argmax 30751 matching CPU; coverage figures are execution, not offer. All wall-clock figures including 3.1x/3.7x are withdrawn under R13 pending device-clock measurement. Switch holds exclusive claim on device-clock measurement while agents run in parallel. — decided by Scribe

---

## 2026-08-01 — Tank — the broken-commitment WARN, through ORT's own sink, with a control that bites

Ruling 2 specified the mechanism; my job was to build it and then to make it *falsifiable*. The
mechanism is small. The control is the deliverable, and the control is what took the session.

### What was built

`disclose_broken_commitment` sits in `ep.rs::compute`, immediately after `guard_ffi_status`, so a
panic converted to `ORT_EP_FAIL` and a normally-returned non-OK status leave through the same
door. On any non-OK status it names the subgraph, every node and op_type in it, a condition token,
the error text, and states that CPU re-execution will follow and that `get_providers()` will still
list us.

**Why every non-OK return here is a broken commitment, mechanically rather than by policy.** A
node declined at partition time never becomes part of a fused node and therefore never gets a
`Compute`. Ruling 2's narrow scope — never-claimed ops falling back is the plan and must not
produce per-node noise — is enforced by the *position of the call site*, not by a predicate that
someone has to keep correct as the code moves. The 258 dynamic-shape declines on Phi-3.5 cannot
reach this function. There is nothing to filter, so there is no filter to get wrong.

**Why it does not go through the `log` crate.** `log::warn!` reaches ORT only if our env-controlled
`LevelFilter` lets it. A disclosure that an environment variable can switch off is not a
disclosure — it is a default. `warn_through_ort_sink` calls `forward_to_ort` directly, so
`RUST_LOG` has no vote. Ruling 2 said "no opt-out" and an opt-out that exists but is off by
default is still an opt-out.

`forward_to_ort` now returns whether `Logger_LogMessage` was actually invoked, so
`broken_commitment_warn_channel` can distinguish `ORT_SINK` (every WARN delivered) from
`PRIVATE_LOG_ONLY` (at least one reached nobody). A WARN in our own log is invisible to exactly
the audience that matters, and until now we had no way to tell the two apart.

### The two-polarity control, and the proof the good-run polarity actually asserts absence

`rust/tools/probe_broken_commitment.py` runs each polarity in a fresh child process, captures the
raw bytes, and judges. Restored build: **device 0 PASS, device 1 PASS**.

An assertion nobody has seen fail is a hope. So I mutated the mechanism and re-ran:

- **Mutation A** — `if status.is_null() { return false; }` replaced by `if false`, so the WARN
  fires on OK statuses too. The **negative** polarity failed, on two independent grounds: a
  `BROKEN COMMITMENT` line through ORT's sink on a successful run, and
  `broken_commitments=1, expected the integer 0`.
- **Mutation B** — unconditional `return false` at the top of the disclosure body. The
  **positive** polarity failed on three grounds: ORT's sink emitted lines and none carried the
  marker, `broken_commitments=0 compute_calls=1`, and `broken_commitment_warn_channel` read
  `UNOBSERVABLE` where `ORT_SINK` was required.

Artifact: `bench/results/broken-commitment-mutation-controls.json`, alongside the restored-build
PASS in `broken-commitment-control.json`. **The good run's silence is now a measured silence: I
have made it speak, deliberately, and the negative polarity caught it.**

Two further things the negative polarity asserts that a naive one would not. It requires
`dispatches_executed != 0`, because the silence of an EP that executed nothing is not a result —
that is yesterday's `UNATTRIBUTED` incident wearing a different hat. And it requires the integer
`0` for `broken_commitments`, never the token: a CPU-only run cannot forge a clean bill of health,
because with `compute_calls == 0` the field reads `UNOBSERVABLE` and the assertion fails.

### The instrument error that first printed as a detection — a live R13 specimen

The probe reported **FAIL on both devices** for a WARN that had been delivered correctly the whole
time. ORT's default sink on Windows writes **wide characters** to stderr; a UTF-8 decode renders
`[W:onnxruntime:...]` as spaced-out letters and every grep over it misses. My witness could not
read the channel it was watching, and it reported that as a property of the mechanism.

I then made the same class of error twice more, which is the part worth recording. First I
concatenated the two captured streams with a one-byte separator, which shifts the second stream's
UTF-16LE alignment by one byte and turns all of it to mojibake — that version reported
FAIL for a line a direct run showed plainly. Then, decoding each stream from offset zero, I hit
the same problem *within* one stream: our own narrow stderr line is written to the same handle and
has odd length, so every wide line after it is off by one. The decoder now reads the bytes four
ways — UTF-8, UTF-16LE at both alignments, and NUL-stripped — because a witness that can read only
one alignment reports absence for something present.

I also burned real time on a dead end that should not be repeated: `CreateEnvWithCustomLogger`
(vtable entry 4, `ORT_API_VERSION = 28`) called through ctypes before importing onnxruntime
returns a null status and the callback never fires. ORT Python does not honour a pre-created
singleton's sink here. Removed.

**R13 consequence, now built in.** If the positive polarity sees *no* line from ORT's sink at all —
not even ORT's own error for the failure we planted — the verdict is
`ERROR(instrument=ort_sink_not_observable_in_this_host)` with exit code 4, never `FAIL`. A blind
witness cannot produce a detection. That branch exists because I lived in it for most of a
session, believing a correct mechanism was broken.

### Is the new observable in-frame at the moment it must be read?

Yes, and deliberately so, because I made exactly the opposite mistake last round. `record_broken_
commitment` calls `dump_if_requested()` **at the instant of the event**, not at teardown. A broken
commitment is followed by ORT's silent fallback and, in both real incidents, by a session that
never reaches an orderly shutdown. A counter that can only be read at a moment which no longer
occurs is `UNOBSERVABLE` by construction — the out-of-frame state — and a "0 broken commitments at
shutdown" gate would have been the R12 hazard I was warned about, reintroduced by me, in my own
file, one round after I found it in someone else's.

### RAI-011 — what I handed Mouse

`viable_islands_retained == 0` meant *gate bypassed* and *all islands rejected* indistinguishably.
It is now three-valued: `UNWIRED` (no cluster reached the decision point), `UNOBSERVABLE`
(clusters seen, but every one bypassed the gate — the event cannot have occurred in this frame),
or an integer (the gate ran, so `0` means all-rejected and is a real detection). A companion token
`net_benefit_gate` reads `UNWIRED|BYPASSED|EVALUATED|MIXED`, with the raw
`clusters_seen`/`evaluations`/`bypasses` triple beside it — an increment can forge a number, but
it cannot forge a type.

On the probe's own single-fused-island model the artifact reads
`viable_islands_retained="UNOBSERVABLE", net_benefit_gate="BYPASSED", clusters_seen=1,
evaluations=0`. **That is RAI-011 reproducing on the bench, not a defect in the probe** — it is
precisely the Phi-3.5 single-island shape, and it now prints as its own condition instead of as a
zero. I wired the one call site I own, `ep.rs::GetCapability`; if `partition.rs` has a second
entry point into the same decision, it needs `counters::record_net_benefit_decision(evaluated)`
too, and that half is Mouse's.

### Still open, and not mine to close

`transfer.rs::device_buffer_for` remains uninvoked and `alloc_device_buffer_binds` stays 0 until
Switch's engine binds. I delivered the transition, not the feature, and I am not building his half.
My reciprocal ask to Trinity — `HARNESS_INSTRUMENT_FILES` omits `ops/conftest.py` — is still open.

### Kept honest

No wall-clock figure anywhere in this session's evidence, and no timing threshold added. Every
number above is a count of events, lines or bytes. Switch holds the device clock; nothing here
competes for it, and nothing here changes meaning under contention.

### Two more things this session found, both in my own files

**The verdict vocabulary could drift silently, and now cannot.** `tests/ops/_verdict.py` and
`counters.rs` each hold their own copy of `MATCH / DIVERGENT / UNMEASURED / UNATTRIBUTED /
SPLIT-FRAME`, and `extract_equivalence` maps anything it does not recognise to `UNMEASURED`. So a
token renamed on the Python side would have arrived here as *"no comparison was performed"* — a red
turned into a shrug, the two-token disease again, with no test able to notice.
`counters::tests::verdict_vocabulary_cannot_drift_from_the_python_harness` reads the Python file and
asserts set equality, plus that both JSON keys match. **Mutation control:** changing
`SPLIT-FRAME` to `SPLIT_FRAME` in `counters.rs` makes it fail and prints both sets. A missing or
unparsable `_verdict.py` panics as `ERROR(instrument=…)` — a cross-language check that skips when it
cannot find its subject would let every future drift pass.

**A latent test race that my change exposed rather than caused.** After adding disclosure to the
real `Compute` path, `counters_record_what_they_claim_to_record` began failing intermittently: the
three pre-existing `ep` tests that drive `compute` now legitimately record broken commitments while
it asserts on those very statics. Then, with the timing changed, two `allocator` tests started
racing on `ONNXRUNTIME_EP_VULKAN_QUARANTINE_SPANS` — one took `ledger::test_lock()`, the other set
and removed the same process-wide variable without it, and the loser read the wrong bound. Both are
now under the one shared lock. A pristine tree passed eight consecutive runs, so the second race was
mine to expose and mine to fix; the fix is in `allocator.rs`, which is my file. **Eight consecutive
clean runs after, 424 tests.** I am recording this because "it passed" and "it passes" are different
claims and only the second one is worth anything from a suite with process-global state.

### 2026-08-01 addendum — the load was misattributed, and my evidence is unaffected by the correction

The coordinator withdrew his attribution of the machine load: it is a second development project of
Justin's running CPU **and GPU** tests, not squad orchestration. **Nothing in this session's
evidence moves**, and that is a property of the instruments rather than luck:

- The two-polarity control's verdicts are counts of events and the presence or absence of a string
  on a channel. `broken_commitments`, `broken_commitment_warn_channel`, `fault_injection`,
  `dispatches_executed != 0` — none of them has a time term, so there is no reading of them that a
  foreign process can move.
- The probe was in fact run under this load, twice, on both devices, and returned PASS both times.
  That is not a claim that it is contention-proof; it is the weaker and true claim that its inputs
  do not include the clock.
- The mutation controls are the same shape: a WARN either appeared on ORT's sink or it did not.

**The part worth keeping.** The coordinator's error was stopping at the first cause consistent with
the observation — two `copilot` processes were consistent with "mine", so he stopped looking. That
is the same failure as §6.5 "closing" on selector 0: the observation agreed with the hypothesis for
a reason the hypothesis did not name. My own version of it this session was the UTF-16 witness — I
read FAIL, it was consistent with "the WARN is broken", and I spent most of a session inside that
reading before checking whether the *witness* could see. **A confirming reading and a working
instrument look identical until you try to make the instrument say the other thing**, which is
exactly why the negative polarity of a control is the half that has to be mutated.

No wall-clock figure and no timing threshold anywhere in this session's work, so there is nothing
here for the device-clock question to invalidate. Switch's `gpu_steady_tail()` question and
Niobe's Intel `NO_STEADY_TAIL` competing explanation are theirs; I hold no measurement that bears
on either.

### STOP POINT 2026-08-01T11:39 — read this first if you are resuming as Tank with no memory

**Everything is committed.** Worktree `C:\Users\justinchu\dev\ep-vulkan-tank`, branch `squad/tank`,
commit `bce87cd`, on top of `main` at `17c2fab`. Working tree clean, nothing pushed, nothing
mid-flight. There is no half-integrated code and no feature flag to finish wiring.

**Done, and verified:**
- The broken-commitment WARN through ORT's own sink (`ep.rs::disclose_broken_commitment` →
  `logging::warn_through_ort_sink` → `Logger_LogMessage`), bypassing the `log` crate so no
  environment variable can suppress it.
- **Both polarities of the control, including the one that must not be skipped.** The good-run
  polarity is written *and mutation-tested*: Mutation A makes the WARN fire on successful runs and
  the negative polarity catches it on two independent grounds. `bench/results/broken-commitment-
  mutation-controls.json`. It is not a printed opinion; it has been made to fire and to fall
  silent on demand.
- `PASS` on device 0 and device 1, twice, the second time after the load correction.
- RAI-011 tokenisation of `viable_islands_retained`, and the verdict-vocabulary drift test against
  `tests/ops/_verdict.py`.

**No measurement I took today is inadmissible, because I took none that could be.** Every figure in
this session's artifacts is a count of events, a token, or a string on a channel. There is no wall
clock, no device clock, no timing threshold, and nothing to re-take in a quiet window. Nothing of
mine needs a `machine_quiescence` label because nothing of mine has a time term to contaminate.

**Next step, in priority order, for the fresh session:**
1. **Mouse's half of RAI-011.** I wired `counters::record_net_benefit_decision(evaluated)` only at
   the `ep.rs::GetCapability` cluster loop, the one site I own. If `partition.rs` reaches the
   net-benefit decision by any other path, that path is unwired and RAI-011 reproduces inside its
   own fix — R10's exact shape. Handed over in
   `.squad/decisions/inbox/tank-net-benefit-gate-observable.md`. **Ask Mouse; do not audit his file
   yourself.**
2. **The real failure conditions are still only unit-covered.** The probe's positive polarity plants
   a synthetic failure (`condition=planted-control`). `failure_condition_token` classifies allocator
   and shape conditions and is tested, but no *real* OOM or empty-tensor-shape failure has yet gone
   through the live path end to end. That is the next falsifier worth building, and it needs no
   timing either.
3. **`transfer.rs::device_buffer_for` is still uninvoked** and `alloc_device_buffer_binds` still
   reads 0 until Switch's engine binds. Not mine to build. Coordinate; do not build his half.
4. **Open ask to Trinity:** `HARNESS_INSTRUMENT_FILES` omits `ops/conftest.py`.

**One trap that will cost you an hour if you rediscover it:** ORT's default logging sink on Windows
writes UTF-16LE to stderr, and our own narrow line shares the handle, so wide lines can sit at
either byte alignment. `probe_broken_commitment.py::decode_both` reads the bytes four ways for that
reason. If you "simplify" it to a single decode, the probe will report `FAIL` for a WARN that was
delivered correctly — which is exactly what it did to me. Also: `CreateEnvWithCustomLogger` via
ctypes before importing onnxruntime returns a null status and never fires its callback. Dead end,
already walked.

📌 Team update (2026-08-01T17:16:56-07:00): Intel device-clock figures are permanently uncertifiable on this hardware (`none_available`, no producer exists and none of the available proxies are the right kind of quantity) — attack the Intel/NVIDIA residual with counts and shapes, not clocks — decided by Niobe


📌 Team update (2026-08-01T17:16:56-07:00): All wall-clock figures remain withdrawn; only counts, bytes and certified-companion device-clock figures are quotable — decided by Switch, Morpheus, Niobe, Link


📌 Team update (2026-08-01T17:16:56-07:00): `ledger_lookup` is the last `UNWIRED` mechanism in the instrument census (criterion 11); Mouse is building it — decided by Trinity, Mouse

