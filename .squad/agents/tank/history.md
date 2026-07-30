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
