# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Runtime & FFI — ORT plugin EP C ABI, sys/ep/factory, build & packaging
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- SUMMARIZED by Scribe 2026-07-29T09:00:39-07:00 — full session details in decisions.md -->

### [SUMMARY] Sessions 1–7: crate foundation, version policy, logging crash fix, cargo ci (2026-07-28–2026-07-29)

**M0 crate foundation (sessions 1–2):**
- `onnxruntime_ep_c_api.h` has no include guard. `rust/wrapper_ort.h` must include ONLY `onnxruntime_c_api.h`.
- Bindings via bindgen over vendored headers (`third_party/onnxruntime/`, tag v1.28.0). Wrong vtable field order = silent UB. CI: every runner needs LLVM/libclang.
- `clippy::undocumented_unsafe_blocks`: `// SAFETY:` must be on the line immediately preceding `unsafe {`. Generated bindings: allow at `mod ort` level.
- `GetSupportedDevices` with no Vulkan: return success + zero devices. `GetCapability`: decline all nodes inside a control-flow subgraph body (non-null `Graph_GetParentNode`) or ORT raises `INVALID_GRAPH`.
- `GetSessionConfigEntry`: two-call protocol; release status even on success. Getting this wrong leaks `OrtStatus` per option per session.
- `Logger_LogMessage` `file_path` is `_In_z_` — ORT dereferences unconditionally; u16 on Windows, u32 on Unix. Was passing `null()` — caused the first real ORT load to crash.
- cdylib exports exactly `CreateEpFactories` and `ReleaseEpFactory`. 37 tests at M0 baseline.

**OrtEp/OrtEpFactory vtable field order (ORT 1.28 — authoritative):**
- `OrtEp`: ort_version_supported, GetName, GetCapability, Compile, ReleaseNodeComputeInfos, GetPreferredDataLayout, ShouldConvertDataLayoutForOp, SetDynamicOptions, OnRunStart, OnRunEnd, CreateAllocator, CreateSyncStreamForDevice, GetCompiledModelCompatibilityInfo, GetKernelRegistry, IsConcurrentRunSupported, Sync, CreateProfiler, IsGraphCaptureEnabled, IsGraphCaptured, ReplayGraph, GetGraphCaptureNodeAssignmentPolicy, GetAvailableResource, OnSessionInitializationEnd, GetDefaultMemoryDevice, ReleaseCapturedGraph.
- `OrtEpFactory`: ort_version_supported, GetName, GetVendor, GetSupportedDevices, CreateEp, ReleaseEp, GetVendorId, GetVersion, ValidateCompiledModelCompatibilityInfo, CreateAllocator, ReleaseAllocator, CreateDataTransfer, IsStreamAware, CreateSyncStreamForDevice, GetHardwareDeviceIncompatibilityDetails, **CreateExternalResourceImporterForDevice**, GetNumCustomOpDomains, GetCustomOpDomains, InitGraphicsInterop, DeinitGraphicsInterop, SelectBestModelCandidate.

**Version negotiation (session 2):** Compile/ship against ORT 1.28 (`ORT_API_VERSION=28`); min host 1.24; exclude 1.27 (null-allocator PrePack + deleter lifetime bugs). Write negotiated version (not compiled-against) into every `ort_version_supported` field.

**OQ-3 — reserved VA registry (session 3, adopted by Morpheus):** ORT pointer arithmetic on allocator return values — synthetic tokens break. `VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` on Windows; `mmap(PROT_NONE, MAP_NORESERVE)` on POSIX. Unique spans; stray dereference = MMU fault. No BDA at all.

**C1/C2 linting (session 3):** C1 bans the contrib domain VALUE (not the comparison). C2 drift alarm: assert against `all_specs()` linked data, not source text. `SchemaBaseline` inside `ContribSchema` (not a parallel table). C2 item 7: fingerprint audit CI job; non-release baseline rows may not go `Live`.

**`cargo ci` xtask (session 4, D-T13):** `rust/xtask/` package + `rust/.cargo/config.toml` alias. Runs fmt → clippy → build → test. `--release` flag. `ALLOW_MISSING_GLSLC=1` set automatically when no SDK. Zero deps. Lesson: CI verification must be an artefact, not a habit.

**Null `file_path` crash fix (session 5, D-T14):** `Logger_LogMessage` `file_path: *const wchar_t` is `_In_z_`. Always pass real NUL-terminated string with two `cfg` branches. Bug manifested as a crash at the first `log::warn!` after `CreateEp`. Lesson: SAL annotations (`_In_z_` vs `_In_opt_z_`) are contract text — read the implementation on ambiguity.

**OrtLogger lifetime bug (session 5, D-T15):** `CreateEp` overwrote the process-default logger permanently. `ReleaseEp` now calls `restore_default_ort_logger()`.

**`tests/host_registration.rs` mock ORT (session 5, D-T16):** Zeroed vtable with Rust callbacks asserting `_In_z_` non-null/NUL-terminated, `_Outptr_` written, `OrtStatus` released exactly once. Verified adversarially (plant the bug, test fails). Blind spot documented: cannot catch packaging faults (missing exports, wrong crate-type).

**`tests/cdylib_load.rs` (session 6, D-T18):** dlopens the shipped cdylib, resolves exports by name as ORT does. `libloading` added as `[dev-dependencies]`. 272 tests, `cargo ci --release` green.

**`ort::wchar_t` Linux fix (session 7):** `ORTCHAR_T` behind a single `cfg`-selected alias. `tests/portability.rs` added. Lesson: writing a caveat in a caveats section discharges the feeling of owing something about it — the countermeasure must be structural: the commit either closes the gap or explains in the caveat itself why it was rejected.

**External resource importer (OQ-13, post-M2):** `OrtEpFactory::CreateExternalResourceImporterForDevice` landed ORT 1.24 (not 1.28). Does NOT answer OQ-3. `sys::importer_seam` names all types so upstream rename = build failure. Teardown order: ORT handles → importer → deinit → `vkDeviceWaitIdle` → Vulkan.

---

## Cross-agent context appended (2026-07-29T09:00:39-07:00) — first-hardware round

📌 **Standard-domain LLM rows registered (2026-07-29, Mouse D-M6-04):** `ai.onnx::Attention`, `ai.onnx::RMSNormalization`, `ai.onnx::RotaryEmbedding` all registered at `OPSET_STD_LLM = 23`. Without these, a Qwen3 built by Justin's own `onnx-genai-models` (mobius builder) would have declined ~5 nodes/layer × 28 layers. Tank's `GetCapability` path must correctly handle these standard-domain rows — they are not contrib-domain and must not be filtered by any `com.microsoft` domain check.

📌 **Niobe's `onnx-runtime-tracer` is now a dependency (2026-07-29, Niobe D-N1):** Pin: `0.1.0-dev.5, default-features = false`. The absolute UNIX-microsecond clock in this crate is critical for correct overlay of plugin cdylib spans onto the host timeline. If Tank's `Cargo.toml` patches or replaces this pin, coordinate with Niobe — wrong clock semantics silently corrupt the trace.

📌 **Vulkan SDK at `C:\VulkanSDK\1.4.350.0` (2026-07-29):** Not on default PATH. `cargo ci` sets `ALLOW_MISSING_GLSLC=1` automatically when no SDK is found; for full builds including shader compilation, prefix the SDK `bin/` directory explicitly.

📌 **`rustfmt --edition 2021` silently no-ops on this edition-2024 crate (2026-07-29, D-T12):** Always use `cargo fmt --all` or the `cargo ci` xtask.
## Session 9 — 2026-07-29T10:50:02-07:00 — the seam lands, and the lint catches its first real violation

**What I actually did.** Finished the `Compile` → `Compute` seam reconciliation, then spent most of
the turn on something I did not expect: getting `cargo ci` back to green after three agents landed
into the same crate at once.

**The layering lint caught a real violation, from an agent, on the first run after his edit.**
Switch's integration put `use crate::vk::session::{...}` at the top of `ep.rs`. That is layering
rule 4.3 — the ABI boundary layer must not name the Vulkan layer. `tests/layering.rs` failed on it
immediately. The fix is a `pub(crate) use` re-export in `engine.rs`, so `ep.rs` names
`crate::engine::{CompileRecorder, CompiledKernel, VulkanSession}` and the seam is declared in one
place. Recorded as D-T27/D-T28.

The lesson worth keeping: I argued for the lint in D-T9 on principle. This is the first time it
earned its keep on live code, and the shape of the catch is instructive — a *single `use` line*
inside a 1600-line file, added in good faith by someone doing exactly what the coordinator asked.
No reviewer catches that reliably. Mechanical enforcement is not about distrust; it is about the
fact that architecture erodes one plausible import at a time.

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
