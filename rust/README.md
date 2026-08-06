# `onnxruntime-ep-vulkan` — Rust crate

The Vulkan compute **plugin execution provider** for ONNX Runtime, built as a standalone shared
library that ONNX Runtime loads at runtime. Nothing here links against ONNX Runtime, and nothing
here links against the Vulkan SDK.

| | |
|---|---|
| Crate | `onnxruntime-ep-vulkan` |
| Artifact base name | `onnxruntime_vulkan_ep` |
| ORT ABI targeted | **1.28.0** (`ORT_API_VERSION` 28) — see [version policy](#version-policy) |
| Minimum supported ORT | **1.24** (`ORT_API_VERSION` 24) |
| EP registration name | `VulkanExecutionProvider` |
| Vendor | `onnxruntime-ep-vulkan` |
| Crate version scheme | `0.<ORT_API_VERSION>.<patch>` |
| Edition / MSRV | 2024 / 1.85 |

**Status: M0.** The crate builds, loads, negotiates the ABI, enumerates devices and reports
capability. It claims **zero nodes** — by design. Every model runs on CPU, correctly. The engine
(`src/engine.rs`) and the op registry (`src/ops/`) are documented seams awaiting their owners.

---

## Before you report work complete: `cargo ci`

```sh
cd rust
cargo ci
```

**Run this before saying a change is done.** It runs, in CI's own order, exactly the checks
CI's Rust lanes run:

| # | Check | Mirrors |
|---|---|---|
| 1 | `cargo fmt --all -- --check` | job `format` |
| 2 | `cargo clippy --workspace --all-targets -- -D warnings` | job `build-test-{linux,windows}` |
| 3 | `cargo build` | job `build-test-{linux,windows}` |
| 4 | `cargo test` (includes the layering lint and the capability-dump suite) | job `build-test-linux` |

```sh
cargo ci --list     # show the checks and which CI job each mirrors, without running them
cargo ci --fix      # same, but rustfmt rewrites instead of complaining
cargo ci --release  # build and test optimised, as CI does (slower; catches release-only faults)
```

It runs **every** check even after one fails, so a single invocation shows you every problem.

### Why this exists

CI was red for four consecutive runs and nobody noticed. Every agent ran `cargo build`,
`cargo clippy` and `cargo test`, saw green, and reported green — `cargo fmt --check` was
never in that loop. "Green" meant *"the commands I happened to remember passed"*. That is a
verification gap, not bad luck, and the fix is an artefact rather than a habit: `cargo ci` **is
the list**. If CI gains a Rust check, add it to `CHECKS` in `xtask/src/main.rs` in the same
commit.

Clippy is run with `--workspace`, which is deliberately one notch stricter than CI: the tool
that tells you CI will be green must not itself be the dirty thing.

### It also warns about stranded decision records

`.squad/decisions/inbox/` is gitignored by design — the Scribe merges it into `decisions.md` on
the integration tree. That was fine while everyone shared one checkout, and became a trap the day
we moved to per-agent worktrees: **a decision record written in a worktree is invisible to `main`,
to the Scribe and to every other agent, and nothing in git reports it.** One was found stranded
that way within hours of the switch.

So `cargo ci` names them. It is a warning rather than a failure — the check infers "linked
worktree" from `.git` being a file rather than a directory, and a heuristic that can be wrong must
not be able to fail a build, or people learn to ignore it.

Worth naming as a category, because it is the third instance today: **isolation that is good for
code is bad for shared state, and the failure is silent.** Worktrees made edit collisions
impossible and quietly made lost reasoning possible. Every mechanism that makes one class of error
impossible tends to make a different one invisible, which is an argument for making the new one
noisy rather than for abandoning the mechanism.

### It finds an installed Vulkan SDK, and works without one

`glslc` is looked for in three places, in order: `$VULKAN_SDK/bin/glslc`, then `PATH`, then — on
Windows — the highest-versioned `C:\VulkanSDK\<version>\Bin\glslc.exe`. The third case exists
because the LunarG Windows installer sets neither `VULKAN_SDK` machine-wide nor `PATH`, so an SDK
can be installed and still invisible to the build. `build.rs` and the xtask apply the same order,
and `build.rs` emits a `cargo:warning` naming the compiler it fell back to.

That third lookup is a *parity* fix, not a convenience. Without it, a box with the SDK installed
still builds with zero shaders, and the tests that assert a Live op has a compiled shader variant
fail locally while passing in CI. A tool that is red for a reason CI is not is worse than no tool:
it teaches people to ignore it.

If none of the three finds a compiler, `cargo ci` sets
`ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1` for you so `build.rs` does not abort, and says so in
its caveats. It also sets `LIBCLANG_PATH` for bindgen if it can find a libclang and you have not
set one.

### The rustfmt edition preflight

Before running anything, `cargo ci` reads `edition` out of `Cargo.toml` and checks that this
toolchain's `rustfmt` accepts it, refusing to run (exit 2) if it does not.

This guards a genuinely nasty footgun: **`rustfmt --edition 2021` against this edition-2024 crate
does not fail.** It parses what it can, silently leaves alone what it cannot, and reports success.
You get a green formatting check locally and a red one in CI, with no diff to explain it. The
normal path is already correct — `cargo fmt` passes the manifest's edition through — but a
toolchain older than the edition would still produce the silent-success behaviour, so it is a hard
failure here rather than a caveat somebody has to remember.

### What it *cannot* verify

`cargo ci` prints this on success too, because it matters more than the word "passed":

- **No shader has executed.** DESIGN.md §9.1.2: no GLSL in this repository has ever run on any
  device, real or software. Everything `cargo ci` checks is host-side Rust logic — claim
  predicates, translation, layering, FFI shape.
- **Without a Vulkan SDK, no shader is even compiled.** A GLSL syntax error is invisible
  locally; CI's Linux and Windows lanes are the first thing that compiles them.
- **No Vulkan device is touched** — no lavapipe, no validation layers, no `vkCreateInstance`.
- **No Python lane** — `tests/ops` (op correctness against the ORT CPU oracle, barrier parity,
  claim diagnostics, no-ICD fallback) needs a real ONNX Runtime and is not run.
- **One OS only.** CI builds Linux *and* Windows; a `cfg(unix)` path that does not compile is
  invisible from a Windows machine.

`cargo ci` green means CI's *Rust* lanes should pass. **It does not mean the EP works.**

Note the second half of that sentence carefully: on 2026-07-29 the plugin was loaded by a real
ONNX Runtime for the first time and killed the host process with an access violation, while the
crate had 268 passing tests and a green `cargo ci`. See
[the mock-ORT-host test](#the-mock-ort-host-test) for what now covers that gap and what still
does not.

### How it is wired

`rust/.cargo/config.toml` defines `ci = "run --quiet --package xtask --"`; the sequence lives in
the `xtask` package (`rust/xtask/`), which has **zero dependencies** so it cannot fail for a
reason of its own on a fresh clone. `rust/Cargo.toml` declares `default-members = ["."]`, so a
bare `cargo build` / `cargo test` — and CI's `--manifest-path rust/Cargo.toml` invocations —
still mean "the EP crate only" and are completely unaffected by the workspace.

`cargo ci` builds debug, for speed; CI builds `--release`. Use `cargo ci --release` when you want
the same profile CI uses.

### When it is red, ask *whose file* before asking *what did I break*

Several agents edit this crate at once, so a red `cargo ci` is ambiguous by default. Two runs ten
minutes apart on 2026-07-29 went from three compile errors in `ops/ssm.rs`, to fully green, to
three failing tests in `registry.rs`, with none of those changes coming from the agent running it.
The fastest way to find out is to read the *paths* before the messages:

```powershell
cargo clippy --workspace --all-targets --message-format=short 2>&1 | Select-String '^src|^tests'
```

One line per finding, file and line first. If the paths are not yours, re-run in a minute rather
than starting a debugging session on someone else's half-landed work.

The one exception is `-D warnings`: a warning in *anyone's* file turns the whole lane red, so it is
a shared resource rather than a private one. Comment-only fixes (a missing `// SAFETY:`, a
`#[allow]` with a stated reason) are reasonable to land across an ownership boundary; anything that
changes meaning is not.

---

## The mock-ORT-host test

[`tests/mock_ort/mod.rs`](tests/mock_ort/mod.rs) is a **hand-built ONNX Runtime**: a zeroed
`OrtApi`, `OrtEpApi` and `OrtApiBase` with the slots we depend on filled in by Rust callbacks. It
drives the exact sequence a real ORT performs during `register_execution_provider_library`:

```
CreateEpFactories
  → GetName / GetVendor / GetVersion / GetVendorId
  → GetSupportedDevices          (with fake CPU and GPU OrtHardwareDevices)
  → CreateEp / ReleaseEp         (and a deliberately invalid two-device CreateEp)
  → ReleaseEpFactory
```

The point is not that these calls succeed. The point is that **every mock callback checks ORT's
own SAL annotations and fails the test if we violate one**: `_In_z_` strings must be non-null and
NUL-terminated at the platform's `ORTCHAR_T` width, `_Outptr_` out-parameters must be written
before a success return, every `OrtStatus` handed out must be released exactly once, and
`OrtKeyValuePairs` must not leak. It also asserts that a log record emitted while the EP is
registered actually arrives at the host's logger — the round trip, not just the call.

Two test binaries drive that one host:

| Test | How it reaches the plugin | What only it can catch |
|---|---|---|
| [`tests/host_registration.rs`](tests/host_registration.rs) | linked **rlib** | shares the plugin's `log` crate, so it can force a record through the bridge on demand |
| [`tests/cdylib_load.rs`](tests/cdylib_load.rs) | `dlopen`s the built **cdylib** and resolves the entry points **by name**, as ORT does | packaging faults: a missing `#[unsafe(no_mangle)]` export, a wrong `crate-type`, an unresolvable dependent DLL |

`cdylib_load` sets `ONNXRUNTIME_EP_VULKAN_VERBOSE=1` before loading, because a loaded library has
its own private copy of `log` that the test cannot write to — raising the plugin's own level makes
it emit the "loaded" line at the end of `CreateEpFactories`, which forces the same logger round
trip that access-violated in CI.

### Why it exists

The plugin's first-ever load by a real ONNX Runtime ended in a Windows access violation inside
`register_execution_provider_library`. The cause was one argument: `forward_to_ort` passed `NULL`
for `Logger_LogMessage`'s `file_path`. ORT annotates it `_In_z_`, **not** `_In_opt_z_`, and on
Windows the implementation does `onnxruntime::ToUTF8String(file_path)` — a `std::wstring`
constructed from the pointer, which dereferences `NULL` unconditionally. Our side was flawless: we
never touched it. No amount of testing *our* code could have found it, because the bug was in
what we told ORT to do.

That is the general shape of every FFI bug worth having a test for, so the mock host asserts the
*host's* contract rather than our behaviour.

### What it cannot catch

It is not ONNX Runtime. It checks that we honour the contracts ORT's headers document, not that
ORT's implementation is happy with us, and it never creates a Vulkan device or runs a shader.
**"`cargo ci` is green" and "the plugin works in ORT" remain unrelated claims** — CI's Python lane
is the only thing that proves the second.

---

## Building

### Prerequisites

| Tool | Required? | Why |
|---|---|---|
| Rust ≥ 1.85 | **yes** | edition 2024 |
| **libclang** (LLVM) | **yes** | `bindgen` parses the vendored ORT headers at build time |
| ONNX Runtime install | no | the headers are vendored; nothing links against ORT |
| Vulkan SDK / `glslc` | **yes** | `rust/shaders/glsl/` now has 10 `.comp` shaders; `build.rs` compiles them at build time |
| Vulkan loader | at runtime only | absent loader ⇒ zero devices advertised, not an error |

Install libclang:

```powershell
# Windows
winget install LLVM.LLVM
$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'   # only if bindgen cannot find it
```

```bash
# Debian / Ubuntu
sudo apt-get install -y libclang-dev
# macOS
brew install llvm      # or rely on the Xcode CLT libclang
```

### Build

```powershell
cd rust
cargo build                  # debug
cargo build --release        # release
```

Artifacts land in `target/{debug,release}/`:

| OS | File |
|---|---|
| Windows | `onnxruntime_vulkan_ep.dll` (+ `.dll.lib`, `.pdb`) |
| Linux | `libonnxruntime_vulkan_ep.so` |
| macOS | `libonnxruntime_vulkan_ep.dylib` |

Two C symbols are exported, and only two: `CreateEpFactories` and `ReleaseEpFactory`.

### Windows: MSVC environment via `vcvars64.bat`

A plain `cargo build` from a PowerShell prompt that never ran a Visual Studio developer shell
setup script will fail in the linker (`cl.exe`/`link.exe` not on `PATH`, `INCLUDE`/`LIB` unset).
The standard fix — Visual Studio's own `vcvars64.bat` — is a **`cmd.exe` batch file**, and the
naive way to combine it with a custom build in one line is fragile on this toolchain, for two
distinct, verifiable reasons — **not** because a real child process fails to inherit variables
`set` earlier in the same `cmd /c "..."` chain (it does; a `set X=1` followed later in the same
chain by a genuine child process, e.g. another `cmd /c echo %X%` or `cargo build`, sees `X`
correctly — confirmed by direct test on this toolchain):

```powershell
# FRAGILE — avoid. Two real hazards, neither of which is "child processes don't inherit
# `set` mutations":
#
# 1. Parse-time %VAR% expansion: cmd.exe expands every %VAR% token in a single command line
#    ONCE, before executing any part of that line. `set PATH=%VULKAN_SDK%\Bin;%PATH%` chained
#    after `call vcvars64.bat` in the *same* line reads the pre-vcvars values of %VULKAN_SDK%/
#    %PATH% (or a literal, unexpanded token if undefined) — confirmed: `set X=1 && echo %X%` in
#    one line prints the literal text `%X%`, not `1`. Delayed expansion (`setlocal
#    enabledelayedexpansion` + `!VAR!`) works around this but is easy to omit.
# 2. PowerShell/cmd quoting: building a long `cmd /c "..."` argument from PowerShell requires
#    getting nested double-quotes (for paths containing spaces) and `$`/backtick escaping (so
#    PowerShell does not interpolate before cmd ever sees the string) exactly right; a small
#    quoting mistake drops or corrupts arguments with no error message, only a confusing build
#    failure or a wrong environment.
cmd /c 'call vcvars64.bat && set PATH=%VULKAN_SDK%\Bin;%PATH% && cargo build --release'
```

The proven recipe instead **captures** `vcvars64.bat`'s resulting environment and **applies it
natively inside the current PowerShell process**, using PowerShell's own `$env:`/
`[System.Environment]` mechanisms instead of `%VAR%`/chained-`cmd` syntax — sidestepping both
hazards above entirely, since nothing is expanded or quoted through `cmd.exe` more than once:

```powershell
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$envDump = cmd /c "call `"$vcvars`" >nul 2>&1 && set"
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

# Set anything project-specific the same way — natively in this process, never chained through cmd.
$env:VULKAN_SDK    = "C:\VulkanSDK\1.4.350.0"
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:PATH          = "$env:USERPROFILE\.cargo\bin;$env:VULKAN_SDK\Bin;$env:PATH"

cd rust
cargo build --release
```

Adjust the `vcvars64.bat` and Vulkan SDK paths to the versions actually installed
(`vswhere.exe -latest -property installationPath` finds the Visual Studio install path if it is
not the default). This recipe is for a fresh interactive PowerShell session — a developer
machine or a squad worktree — that never ran a VS developer shell; **CI's `windows-latest`
runner does not run this recipe** (its images ship with the MSVC toolchain already reachable
from a plain PowerShell session, so `.github/workflows/ci.yml` never calls `vcvars64.bat`). If a
build fails with an MSVC toolchain error, verify with `$env:INCLUDE` / `$env:LIB` non-empty in
the *same* shell that runs `cargo build` before assuming the code is at fault.

### Verify

```powershell
cargo build
cargo clippy --all-targets -- -D warnings
cargo test
cargo test --test layering        # the layering lint on its own
```

All four must be clean before a change lands.

---

## Where the ORT headers come from

The plugin-EP C ABI headers are **vendored** in
[`../third_party/onnxruntime/include/`](../third_party/onnxruntime/include/), taken verbatim from
`microsoft/onnxruntime` tag `v1.28.0` (commit `da9b5e364c465de65c49d91e696cd6485270757f`). MIT
licence, reproduced alongside them. Full provenance and the re-vendoring procedure are in
[`../third_party/onnxruntime/PROVENANCE.md`](../third_party/onnxruntime/PROVENANCE.md).

`build.rs` resolves the include directory in this order, so you can point at a local ORT checkout
when you need to test an unreleased ABI:

1. `$ORT_INCLUDE_DIR`
2. `$ORT_HOME/include`
3. `third_party/onnxruntime/include` *(default)*

### bindgen, not hand-written bindings

The ORT plugin-EP ABI is **experimental and still moving** — `OrtEp` and `OrtEpFactory` both gained
fields in 1.23, 1.24 and 1.28. That is exactly the argument people usually make *for* hand-writing
bindings ("the API is unstable, keep it under our control"), and it is why we do the opposite:

* These are `#[repr(C)]` **vtables**. Getting a field *order* wrong does not fail to compile and
  does not fail to load. It calls the wrong function pointer with the wrong arguments — silent
  undefined behaviour, discovered later, somewhere else. `OrtApi` alone has several hundred
  function-pointer fields. Transcribing that by hand, repeatedly, is a bet we lose eventually.
* bindgen derives the layout from the same bytes ORT was compiled from, so field order cannot
  drift from the header.
* Vendoring the headers recovers everything the hand-written camp actually wants: builds are
  byte-reproducible, no network access and no ORT install are needed, and an ORT bump becomes a
  reviewable diff of two header files.

The cost is a libclang dependency on build machines. That is a CI provisioning line, paid once.

### Version policy

Three numbers, and conflating them is how plugin EPs corrupt themselves:

| | value | meaning |
|---|---|---|
| `ORT_API_VERSION_EXPECTED` | 28 | what we compile and ship against |
| `ORT_API_VERSION_MIN` | 24 | oldest host we will run against |
| negotiated version | 24–28 | what the host in front of us actually serves |

`sys::check_api_version()` asks `GetApi(28)` and walks down to `GetApi(24)`, taking the first
version the host serves. Below 24 the plugin **refuses to load** with a message naming both the
requirement and the host's version.

Running below 28 is safe because `OrtApi`, `OrtEp` and `OrtEpFactory` are append-only — version *v*
is a prefix of 28. But that safety is an *obligation* (never touch a field added after *v*), not a
property, so two mechanisms discharge it:

* The **negotiated** version, not 28, is written into `OrtEpFactory::ort_version_supported`,
  `OrtEp::ort_version_supported` and friends — so ORT stops reading our vtables exactly where its
  own header stops describing them. This is the same signal ORT uses on its side of the boundary.
* Every optional entry point is gated by `NegotiatedApi::supports(since::*)`, one named constant
  per feature (today: `since::EXTERNAL_RESOURCE_IMPORTER = 24`). A test asserts the gate returns
  **false** at version 23 — a gate that only ever says yes would pass every other test.

Three compile-time assertions back this up: the vendored header's `ORT_API_VERSION` is 28, the
crate minor version is 28, and the floor does not exceed the ceiling.

**Why ship against 1.28 specifically.** 1.27 has a critical plugin-EP defect — a null allocator in
`PrePack` plus a deleter lifetime bug — that would hit us directly. 1.27 is excluded on purpose,
not merely superseded.

**Why 1.24 as the floor.** `OrtEpFactory` has existed since 1.22, but 1.24 is where the surface we
depend on settled, and ORT's own bridge uses `ort_version_supported < 24` as its compatibility
line. We never have to reason about the 1.22/1.23 layouts.

The plugin EP API is **still experimental in 1.28** — it did not graduate. That is why every raw
ORT type stays behind `src/sys.rs`: a vtable change should be a one-file fix.

### Zero-copy IO binding (bound, not implemented)

`OrtEpFactory::CreateExternalResourceImporterForDevice` (ORT 1.24+) is the OS-handle-based external
memory import path: the caller allocates `VkDeviceMemory` **with `VkExportMemoryAllocateInfo`**,
exports it via `vkGetMemoryWin32HandleKHR` or `vkGetMemoryFdKHR`, and the EP re-imports it as a
zero-copy tensor. Timeline-semaphore import for GPU↔GPU sync rides the same interface.

M0 **binds the seam without implementing it**: `sys::importer_seam` names
`OrtExternalResourceImporterImpl`, the handle base structs, and the four Vulkan-specific enum
values ORT defines (`..._VK_MEMORY_WIN32`, `..._VK_MEMORY_OPAQUE_FD`, and the two timeline
semaphore types). The factory slot is left `None` — which ORT reads as "cannot import external
memory", and which is true today — behind an already-written `supports()` gate. Adding the
implementation is one new `importer.rs` plus one line; nothing else moves. The contract to match is
ORT's own `nv_vulkan_test.cc`.

This does **not** answer OQ-3 (what our `Alloc()` returns to ORT's pointer-based allocator API when
a Vulkan allocation is a `VkBuffer` + offset). Different direction, different memory, different
owner. That one is still open.

#### Integration contract for callers

Zero-copy import is **not a transparent optimization**. It imposes a precondition on the caller,
and a caller who did not plan for it cannot opt in after the fact:

* The buffer must be created with `VkExternalMemoryBufferCreateInfo` in its `pNext`, **and** its
  memory allocated with `VkExportMemoryAllocateInfo` in `pNext`. Both. Memory that was not
  allocated as exportable cannot be imported — there is no retrofit.
* Handle types are `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT` on Windows and
  `..._OPAQUE_FD_BIT` on Linux. **DMA-BUF is not supported**: ORT defines no
  `ORT_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD`, and its own reference test asserts as much.
* Call `CanImportMemory` (and `CanImportSemaphore`) before committing. A device whose driver lacks
  `VK_KHR_external_memory_win32` / `_fd` answers `false`, and that is the caller's cue to fall
  back — not a bug.
* **OS handle ownership is asymmetric.** On Linux, importing takes ownership of the fd; the caller
  must not close it. On Windows the `HANDLE` is *not* transferred; the caller retains it and must
  `CloseHandle` after import.
* **Teardown order is fixed**: release every ORT handle (`ReleaseExternalMemoryHandle`,
  `ReleaseExternalSemaphoreHandle`), then the importer, then `DeinitGraphicsInteropForEpDevice`,
  then `vkDeviceWaitIdle`, and only then destroy Vulkan objects (buffer views → buffers → memory
  → semaphores → queue → device → instance).

---

## Inspecting a build: `epctl`

```console
$ cargo run --bin epctl -- --dump-capabilities
$ cargo run --bin epctl -- --dump-capabilities --json   # for CI diffing
```

Prints every registered op with its opset window, dtypes, live/staged status, backing shader
template, and — for contrib (`com.microsoft`) rows — the ORT release its claim predicate was
written and verified against (`DESIGN.md` §1.4 constraint **C2**), followed by a grouped list of
the reasons rows are staged.

It creates no Vulkan instance, loads no ORT, and touches no device: the output is a property of
the *binary*, so it can be captured in CI, diffed across commits, and attached to a bug report
from a machine that cannot run the EP at all. Default-domain rows report `n/a (opset-versioned)`
in the baseline column on purpose — their compatibility contract is the opset window, and a
baseline there would dilute the signal on the rows that need one.

---

## Real-model validation without Python: `ort-model-runner`

`rust/modelrunner` is a second, host-only workspace member. It loads a real ONNX model, runs it on
the CPU EP and on this EP, and proves the Vulkan run was *real* and *correct* — with no Python
interpreter, no PyPI wheel and no third-party crate involved at any point.

```console
$ cargo build --release -p ort-model-runner
$ ./target/release/ort-model-runner --list-models
$ ./target/release/ort-model-runner --check-model-agreement mnist-12 \
    --ort-lib /path/to/onnxruntime.dll \
    --out bench/results/rust-model-runner/mnist-12.json
PASS mnist-12
```

### Why it is Rust and why it has no dependencies

The Python probes under `rust/tools/` are the reason this exists. They are the only thing that ever
ran a real model end to end, and they need `onnx`, `onnxruntime`, `numpy` and a working index. On a
machine where the package index is unreachable — an air-gapped runner, a locked-down corporate host,
a fresh container behind an egress policy — none of them can execute, and the project's only
real-model evidence becomes unavailable exactly when someone needs to check a claim.

So SHA-256, JSON, and the deterministic input PRNG are implemented in-tree and tested against
published vectors. A runner whose purpose is "works where PyPI is blocked" must not be one
`cargo fetch` away from the same failure. The only non-`std` code it uses is `libloading`, and only
to `dlopen` ONNX Runtime — the same thing ORT's own C API examples do, and the reason the EP crate
itself never links against `onnxruntime`.

### The six guards

Every one must hold for `PASS`. Each is written into the evidence JSON with its reason, whether it
held or not, because a guard that is silent when it passes cannot be audited later.

| Guard | Asks | Witness |
| --- | --- | --- |
| `model_identity_pinned` | Are these the bytes we said they were? | `bench/results/model_provenance.json` — size **and** SHA-256 |
| `vulkan_ep_device_present` | Did the plugin register an `OrtEpDevice`? | `GetEpDevices` on the real environment |
| `vulkan_ep_in_session` | Was the EP selected for *this* session? | `SessionOptionsAppendExecutionProvider_V2` succeeded against that device |
| `vulkan_executed_nodes` | Did ORT attribute executed nodes to us? | **ORT's own profile JSON** — `cat == "Node"`, tallied by provider |
| `vulkan_dispatched_work` | Did the GPU do anything? | the EP's counters snapshot, `dispatches_executed > 0` |
| `outputs_agree` | Is the answer right? | the CPU EP's outputs for the same bytes, under a written-in-advance tolerance |

The fourth guard is the point of the tool. `rust/tools/probe_model_output_agreement.py` documents a
dispatch guard and does not implement one: it checks `"VulkanExecutionProvider" in
session.get_providers()`, and that list is fixed at session-create time, so it is `True` whenever
the EP was merely *requested*. A run in which every node fell back to CPU passes it.

The two witnesses are deliberately unequal. ORT's profile is **primary** because it is produced
outside the frame under question — ORT decides what ran where, and it has no stake in this EP's
claims. Our own counter is **corroborating** only, because it is inside that frame. When they
disagree, the evidence records `split_frame` and the run does not pass; it is not resolved in the
EP's favour.

### Tolerance is policy, not a knob

`compare.rs` holds a table of per-model `rtol`/`atol` with a written rationale for each. A model
that is not in the table is refused rather than compared against a default — an unreviewed default
tolerance is how a real numerical regression becomes a green run. `--rtol`/`--atol` override it,
must be given together, and are stamped into the evidence as `tolerance.source = "cli"` so a reader
can see the comparison was loosened.

Inputs are generated from a fixed-seed SplitMix64 stream (the stream itself is pinned by a unit
test), so two runs of the same model on the same build compare the same numbers. Free dimensions
default to 1 and every resolution is recorded; `--free-dim` pins them explicitly and `--input`
feeds real bytes for models whose inputs are interdependent.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | `PASS` — all six guards held |
| 1 | `FAIL(condition=…)` — a claim about the EP is false (outputs disagreed, nothing dispatched, the pin did not match) |
| 2 | `ERROR(instrument=…)` — the harness could not ask the question (no ONNX Runtime, ambiguous library, unreadable model) |
| 3 | `UNSUPPORTED(reason=…)` — the model is outside what this runner can drive, stated as such and never as a pass |

`UNSUPPORTED` is a real state, not a polite failure. Phi-3.5-mini resolves and hashes correctly and
then cannot be driven with generated inputs, because `GroupQueryAttention` requires a KV cache
consistent with `seqlens_k`. The runner reports exit 3 with the model's exact path and SHA-256 still
stamped in the evidence, so the identity is on record even though the comparison is not.

### Finding ONNX Runtime

Searched in order: `--ort-lib`, `ORT_MODEL_RUNNER_ORT_LIB`, `ORT_HOME`, `ONNXRUNTIME_DIR`, the
repository `.venv`, then the loader path. **Two different libraries found is an error, not a
choice.** The API version is gated to the same `ORT_API_VERSION_MIN`/`ORT_API_VERSION_EXPECTED`
constants the EP itself pins, so the runner cannot drift from the plugin it is testing.

On Windows this matters immediately: `C:\Windows\System32\onnxruntime.dll` is ORT 1.17.1 on many
machines and wins the loader search. The runner refuses it by version and names `--ort-lib` in the
refusal rather than loading it and failing later in a way that looks like an EP defect.

### What it is not

It is not a benchmark and it does not measure speed. It is not a replacement for `tests/ops`, which
covers per-op semantics at a granularity no whole-model run reaches. And its host-free lane
(`cargo test -p ort-model-runner`, wired into both build lanes) cannot see the guards that need a
device — those are only claimed by a real run, and such runs are committed under
`bench/results/rust-model-runner/` with an artifact frame that says which commit and which GPU
produced them.

---


A green test suite and an executed dispatch are unrelated claims. A lane where every op declines,
or every test skips, or every node quietly falls back to CPU under our provider's name, passes its
assertions and runs nothing on a device. This project has already had to retract two fabricated
speedups produced in exactly that state. `DESIGN.md` M0 criterion 8 therefore requires **a non-zero
executed-dispatch count per lane, reported** — not inferred.

`src/counters.rs` is that channel. It is always on (six relaxed atomics on paths that already do a
GPU submit; the cost is unmeasurable) because a diagnostic you have to remember to enable makes a
lane that *forgot* look identical to a lane that *executed nothing*.

### What the number means

`dispatches_executed` is incremented **after `dispatch_ort` returns success**, and `dispatch_ort`
submits and then waits on a fence. So a non-zero count means a command buffer reached a device and
the device finished it.

It deliberately claims nothing about correctness. "A dispatch executed" and "the answer is right"
are separate claims, and conflating them is how a provider that declined every node reported a
1.45× speedup. Numerical agreement is the differential test's job.

The other five counters exist to tell failure modes apart when the count is zero:

| Counter | Zero-dispatch diagnosis |
|---|---|
| `compile_calls` | 0 → ORT never asked us to compile anything; nothing was claimed |
| `subgraphs_live` | 0 with `subgraphs_stub` > 0 → we claimed nodes but had no device or produced no kernels |
| `compute_calls` | 0 with `subgraphs_live` > 0 → Compile succeeded, the session never ran the node |
| `compute_failures` | > 0 → we ran and returned a status; the log has the reason |

### Reading it

Two paths, because the interesting failure is a process that dies mid-session:

```console
# 1. A snapshot file, written on the FIRST successful dispatch and again at factory teardown.
$ set ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE=%RUNNER_TEMP%\ep-counters.json
$ pytest tests/ops
$ cargo run --bin epctl -- --check-counters %RUNNER_TEMP%\ep-counters.json --require-dispatches 1
```

The early write is deliberate: both CI lanes currently crash inside `session.run()`, and a
teardown-only snapshot would tell us nothing about how far they got. With the early write, a
surviving file distinguishes "never executed anything" from "executed something and then
corrupted memory".

```c
/* 2. In-process, for a host or a test that wants the live numbers. */
size_t OrtEpVulkanGetExecutionCounters(void* out, size_t out_bytes);  /* returns bytes written */
void   OrtEpVulkanResetExecutionCounters(void);
```

The struct leads with `struct_size` and `abi_version` so a reader validates before trusting, and
growth is append-only. A short buffer gets a prefix and an honest byte count; a buffer too small
for the header gets zero.

### Exit codes, and why there are three

| Code | Meaning |
|---|---|
| 0 | at least `--require-dispatches N` dispatches executed |
| 1 | the lane reported, and the number was below the requirement |
| 2 | usage error |
| 3 | the lane **did not report** — file missing, truncated, or from a counters ABI we do not understand |

1 and 3 are different codes on purpose, the same reasoning as `--probe-loader`. "Executed nothing"
is a real, attributable answer. "Did not report" is the absence of one, and almost always means the
process died before it could write. A crashed lane must not be able to look like any kind of
answer, and a truncated file must not parse as a zero.

At teardown the EP also logs a one-line summary through ORT's logger, and emits a `WARN` when the
count is zero — so the signal is in the log even for a lane that never set the environment
variable.

### The contract for CI (Trinity)

The counters file is a stable, additive contract, so a workflow can assert on it directly rather
than inferring execution from a pass count:

* **The gate:** `epctl --check-counters <file> --require-dispatches 1`. Exit 0 / 1 / 3 mean what
  the table above says, and the distinction between 1 and 3 is the point — a lane that crashed
  before reporting must not be able to look like a lane that reported zero.
* **The keys** are looked up by name and unknown keys are ignored, so the document can grow
  without breaking a reader. Present today: `abi_version`, `compile_calls`, `subgraphs_live`,
  `subgraphs_stub`, `compute_calls`, `compute_failures`, `dispatches_executed`, and — added
  after the planner verification — `pointers_observed`, `pointers_host`, `pointers_at_base`,
  `pointers_interior`, `pointers_in_guard_band`, `pointers_use_after_free`,
  `pointer_max_offset`.
* **`pointers_in_guard_band > 0` is a hard failure, and `--check-counters` now asserts it.** It
  means ORT derived a pointer that ran off the end of one of our allocations — a
  silent-wrong-answer bug in any allocator design that cannot detect it. The check sits **ahead of**
  the `--require-dispatches` comparison and outranks it, because thirty dispatches producing wrong
  answers is not a better outcome than zero dispatches. Exit code 1, with the address trace in the
  `*.trace.txt` written beside the counters file.
  The key is **optional**: a snapshot from a build without the ledger does not carry it, and
  **absence must not be read as zero** — so a missing key passes and a present non-zero one fails.
* **`alloc_staged_spans > 0` disqualifies a timing**, and `--require-device-memory` asserts it. See
  [Keeping the staging caveat truthful](#keeping-the-staging-caveat-truthful).
* `ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` must be set **before** the process that loads the EP
  starts; the file is written on the first successful dispatch and again at teardown.

---

## Keeping the staging caveat truthful

When a device handle has no `VkBuffer` behind it, its contents are held in host memory. That is
correct — it is the near half of the real copy path — but it means the tensor never reached the
device, so no timing from such a run is a device measurement.

The original mechanism for saying so was a one-shot `WARN` ending *"any timing from this run is a
host measurement"*. **That sentence cannot survive contact with real device memory**, and it fails
in both directions:

* **Keep it and it over-warns.** A run that is 99% device-backed still prints "host measurement".
  The warning is then wrong, readers learn to discount it, and it stops protecting the 1% case it
  was written for. A caveat that is always printed carries no information.
* **Delete it and it under-warns.** Staging does not stop the day device memory lands; it stops
  *per allocation*. A partially staged run would then say nothing at all, and its numbers would
  look exactly like device numbers. This is the worse failure of the two.

So the whole-run claim is no longer prose. Three things replaced it:

1. **The per-handle WARN says only what it knows** — *this* handle has no `VkBuffer`, so *its*
   contents are in host memory — and explicitly defers the whole-run question to teardown.
2. **A teardown verdict computed from the ratio** (`allocator::tally::staging_verdict`), with a
   distinct sentence for the **mixed** state that no fixed wording covered and that is precisely
   where we are heading: *"neither a host measurement nor a device one; an average over two
   different memories, comparable with neither."*
3. **An assertion, so nobody has to remember a log line.** `epctl --check-counters
   --require-device-memory` fails the lane unless every handle was device-backed. A snapshot with
   no allocation tally **cannot answer** and exits 3 rather than passing — absent keys are not
   zero. Set the flag on any lane that quotes a number.

The counters file carries `alloc_allocations`, `alloc_frees`, `alloc_bytes`,
`alloc_high_water_bytes`, `alloc_device_backed_spans`, `alloc_staged_spans` and
`alloc_staged_bytes` to support this. They come from a process-global tally rather than from
`AllocStats`, because an `AllocStats` lives inside a `VulkanAllocator` that ORT releases on its own
schedule — a snapshot published at release would be correct only when the teardown order happened
to favour us, and would silently write zeros otherwise.

**As of today every one of those runs reports `alloc_device_backed_spans: 0`.** The ORT-facing half
of the allocator is real and in the path; the `VkBuffer` behind each handle is not attached yet, so
`ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` currently buys host memory wearing a device handle. The
flag above is what will notice the day that changes, and the day it half-changes.

---

## Proving the validation layer is actually watching

```console
$ cargo run --bin epctl -- --probe-validation                     # expect: VALIDATION ARMED, exit 0
$ cargo run --bin epctl -- --probe-validation --plant-violation   # expect: EPCTL-VALIDATION-CAUGHT lines
$ cargo test --test validation_control                            # the harness that asserts both
```

M0 criterion 3 originally read "the Vulkan validation layer surfaces no errors". That was refused,
correctly, because **"no errors surfaced" is precisely what a run with the validation layer *not
loaded* reports.** A green lane and an absent check are indistinguishable.

Investigating it found the gap was wider than the objection stated. The engine requests
`VK_LAYER_KHRONOS_validation` but attaches no `VkDebugUtilsMessengerEXT`, so even when the layer
*is* loaded, nothing in-process observes its output — it goes to the layer's default handler,
wherever that points on a given machine. A clean run was uninformative twice over: the layer might
not be there, **and we were not listening**.

`--probe-validation` closes both halves. It creates an instance with the layer enabled *and* a
debug messenger attached, and reports three states apart rather than collapsing them into a
boolean:

| State | Exit | Meaning |
| --- | --- | --- |
| `VALIDATION ARMED` | 0 | Layer installed, enabled, and its output is reaching our callback. **Only this state licenses any claim about validation cleanliness.** |
| `VALIDATION LAYER ABSENT` | 3 | Loader present, layer not installed. The absence of an answer, not a failing one. |
| `NO VULKAN LOADER` | 3 | Ditto. |

`--plant-violation` is the positive control, the same mechanism the [layering lint](#layering-lint)
uses for criterion 7: it deliberately calls `vkCreateDebugUtilsMessengerEXT` with zero
`messageSeverity` and `messageType` masks
(`VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-requiredbitmask`). Every message the
layer hands back is printed with the literal marker `EPCTL-VALIDATION-CAUGHT:`.

That violation was chosen for four properties, all of which matter:

1. It is a **stateless parameter check**, so any build of the layer catches it on any ICD,
   including lavapipe on a CI runner with no GPU.
2. Nothing is allocated, bound, submitted or executed, so it cannot corrupt anything.
3. It needs **no logical device and no physical device.** This is the important one: a plant that
   needed a device would make *a machine with no capable GPU* look identical to *a machine with no
   validation* — the exact conflation the control exists to prevent.
4. It exercises the debug-utils extension itself, so a pass proves the capture path is live rather
   than merely that an instance was created.

`tests/validation_control.rs` asserts both directions — the planted violation must be caught, and
the clean run must be silent. A control that fires unconditionally is as useless as one that never
fires. The failure message on the positive control says so in as many words, because the tempting
misreading of that red is "our code is clean"; it is not, it is "the check does not work, so no
green from it means anything."

**Skips are loud, and CI can forbid them.** On a machine with no layer both tests skip with an
explanation rather than passing. Set `ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION=1` — which CI
should — and the skip becomes a failure, so a lane that quietly loses the layer cannot keep
reporting green forever.

**Scope, stated honestly.** The plant lives inside `epctl`'s *own* instance. Passing proves the
layer is loadable here and our capture works. It does **not** prove the EP's dispatch path has
validation armed on *its* instance — that one is created in `vk/instance.rs` and needs its own
env-gated plant. That gap is named rather than papered over; a control that quietly proves
something adjacent to the claim is the failure mode this whole section is about.

---

## The 2.09 GB "still live" warning: it was a scope error in the instrument

For a while the allocator printed, on the real model, on both vendors:

```
WARN: 322 device handle(s) (2093838336 B) were still live when the allocator was released.
ORT frees what it allocated, so this is either a leak on our side or a tensor the session outlived.
```

That wording is honest and useless. **An open disjunction in a warning is a decision deferred to
whoever reads it**, which in practice means it is decided later, by someone with less context — or
never, because a warning that has always been there reads as furniture. 2.09 GB is not furniture.

It is neither branch. Measured on Phi-3.5 under pytest, both devices:

| | |
| --- | --- |
| `alloc_allocations` | 2511 |
| `alloc_frees` | **2511** |
| `alloc_frees_after_release` | 0 |

**ORT hands back every span.** Three things were true at once and only their combination looked
alarming:

1. `HandleRegistry` is **process-global per device** (`factory::REGISTRIES`), shared by every
   allocator and data transfer for that device. `registry.stats().live_spans` read at *one*
   allocator's release counts spans that other, still-running sessions own. That run released seven
   allocators; summed, they reported 1257 "still live" spans, every one of which was freed later.
   The warning named this allocator as the owner of its neighbours' memory.
2. The registry **outlives every allocator by construction** — `REGISTRIES` holds an `Arc` for the
   process lifetime — so the hazard the warning gestured at, a late `Free` landing in a torn-down
   registry, cannot occur. Nothing is reclaimed at allocator release.
3. The counters file was written only from `VulkanDataTransfer::release`, which ORT calls **before**
   releasing allocators, so the file could never have shown any of this. It reported
   `alloc_allocators_released: 0` no matter what happened afterwards — an unfalsifiable zero rather
   than a clean one. It is now rewritten at allocator release too.

The warning is now scoped: `debug!` while other holders of the shared registry are still running,
`warn!` only for the last allocator on a device, where "still live" finally means what it says.

### The replacement counter reproduced the same bug, and that is worth recording

`alloc_frees_after_release` was written to close the disjunction. Its first version tested
`allocators_released > 0`, which is **monotone** — so once any one allocator went away, every
subsequent `Free` from every *other* live allocator counted as late. It reported **2508 late frees
on a run where nothing was wrong**: the identical scope error, reproduced inside its own
replacement, in the same hour. It now requires `allocators_live == 0`, which is the condition that
means something — a `Free` arriving when no allocator of ours exists is a span nobody is left to
own. `epctl --check-counters` fails on it unconditionally, ranked with the guard band.

---

## Loading the plugin

### Python

```python
import onnxruntime as ort

ort.register_execution_provider_library(
    "VulkanExecutionProvider",
    r"C:\path\to\onnxruntime_vulkan_ep.dll",   # or lib....so / .dylib
)

sess = ort.InferenceSession(
    "model.onnx",
    providers=["VulkanExecutionProvider", "CPUExecutionProvider"],
)
```

In M0 every node falls through to CPU. That is the expected result, and
`ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` will tell you exactly why, node type by node type.

### C / C++

```c
OrtEnv* env = /* ... */;
g_ort->RegisterExecutionProviderLibrary(env, "VulkanExecutionProvider", ORT_TSTR("onnxruntime_vulkan_ep.dll"));
```

---

## The device allocator

`src/allocator.rs` implements the `OrtAllocator` ORT uses when it decides a tensor should live in
device memory. It is **opt-in**: set `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1`. See *Why it is off by
default* below — the reason is measured, not cautious.

### What the pointer is

ORT's allocator API is pointer-based: `Alloc` returns a `void*` and ORT may hand back
`returned_ptr + offset`. Vulkan has no pointers — it has `VkBuffer` plus an offset. The pointer we
return is therefore a **handle**, and it is deliberately not an integer: it is a page-aligned span
of **real reserved virtual address space**, carved from one large `PROT_NONE` /
`MEM_RESERVE`-only region per device.

That choice is the whole design:

* **`ptr + n` stays in-span by construction** for `n` below the requested size, so ORT's
  memory-pattern planner does its arithmetic and we can still resolve the result to
  `(handle, offset)` by range lookup.
* **The address is unreadable.** Any code that mistakes the handle for memory and dereferences it
  faults immediately at the exact instruction, instead of silently reading someone else's tensor.
  Unreadability is the safety property, not an accident of the implementation.
* **A guard band separates spans**, so the one-past-the-end pointer `ptr + size` lands in a hole
  rather than on the next allocation.
* Lookups are bounded by the **requested** size, not the page-rounded size — accepting the rounding
  slack would quietly permit a short read past the tensor's end.
* Reuse from the free list is **exact-size only**, for the same reason.

`HandleRegistry::attach_buffer(addr, BufferView)` is the seam where the engine layer binds a real
`VkBuffer` behind a handle. The registry never touches Vulkan itself.

### When a handle goes bad

`resolve` fails with one of three named causes, and each is logged in prose:

| `LookupError` | Meaning |
|---|---|
| `NotAHandle` | the pointer is outside every reservation — a host pointer reached a device path |
| `InGuardBand` | the pointer is inside the arena but between spans — an overrun by less than a page |
| `Freed { freed_at_generation }` | the span was freed and is still in quarantine |

### Quarantine is a window, not a proof

Freed spans are held in a FIFO (`ONNXRUNTIME_EP_VULKAN_QUARANTINE_SPANS`, default 4096) and stamped
with the generation at which they were freed, so a use-after-free is *reported* rather than aliased
onto a live tensor. When the FIFO overflows, the oldest address space is reused and that guarantee
lapses for it. `AllocStats::quarantine_retired` counts exactly that: **non-zero means the window was
exhausted and a stale handle could now alias.** It is reported rather than hidden because a
detection window that silently closes is worse than none.

### Stats

`GetStats` reports nine keys. Two matter to other people:

* **`MaxInUse`** (`AllocStats::high_water_bytes`) is the *peak*, not the current value. This is what
  the `MatMulNBits` P6 assertion — "no dequantised weight is ever materialised in device memory" —
  reads: a weight that is allocated and freed before the check still shows up in the high-water mark.
* **`QuarantineRetired`** — see above.

### Why it is off by default

Advertising `OrtDeviceMemoryType_DEFAULT` is a package deal. Once ORT knows we have device memory it
requires a registered `OrtDataTransferImpl` to move tensors in and out, and without one **every
session fails at `Run`**:

```
There's no data transfer registered for copying tensors from
  Device:[DeviceType:0 MemoryType:0 VendorId:0 DeviceId:0 Alignment:0] to
  Device:[DeviceType:1 MemoryType:0 VendorId:4318 DeviceId:1 Alignment:4096]
```

Measured on both local devices. The data transfer cannot be written until handles are backed by real
`VkBuffer`s, which is the engine layer's side. So the allocator ships complete and unit-proven, and
stays behind a switch until its partner exists. The switch is also a bisect handle: flipping it took
a ten-minute CI round trip down to a one-minute local answer when device memory first crashed
registration.

### The lifetime rule that cost us a crash

`EpDevice_AddAllocatorInfo` is annotated `_In_`, which reads like a copy. It is not — the
`OrtEpDevice` **retains** the `OrtMemoryInfo` pointer and ORT dereferences it after
`GetSupportedDevices` returns, while it is still inside
`register_execution_provider_library`. Releasing it there is an access violation. We leak it
deliberately: one small object per device per registration, and ORT offers no way to hand it back.

`tests/host_registration.rs` now enforces this. The mock host retains every attached memory info,
`ReleaseMemoryInfo` **poisons instead of freeing**, and the scenario re-reads them after
`GetSupportedDevices` returns — where ORT does. Re-introduce the release and the test names the
rule instead of segfaulting. It also asserts one info per advertised device, so it cannot pass
vacuously.

---

## Data transfer and host staging

Advertising `OrtDeviceMemoryType_DEFAULT` is a package deal. The moment we do, ORT requires an
`OrtDataTransferImpl` and every `Run` fails with

> There's no data transfer registered for copying tensors from Device:[DeviceType:0 …] to
> Device:[DeviceType:1 … VendorId:… Alignment:4096]

`src/transfer.rs` supplies it. `CanCopy` claims only copies with at least one end in our memory;
`CopyTensors` classifies both ends against **every** registry — not just the one the
`OrtMemoryDevice` names, because a mislabelled side would otherwise be `memcpy`d from an
unreadable reserved page — and then moves bytes.

A copy needs real bytes behind a handle, and no `VkBuffer` is attached yet (that is the engine's
seam, `HandleRegistry::attach_buffer`). So each span lazily acquires **host staging**: an ordinary
host allocation that stands in for device memory. This is not throwaway work — a CPU→device Vulkan
copy goes through host-visible staging anyway, so it is the near half of the real path, built
first. Once a `VkBuffer` is attached, `host_bytes` deliberately **refuses** and the copy becomes
the engine's.

Staging must never be mistaken for a device result, so it is loud in three places:

* a one-shot `WARN` per registry saying explicitly that any timing from that run is a host
  measurement;
* `AllocStats::staging_spans` / `staging_live_bytes`;
* the allocator's release summary — `HOST STAGING: … tensors on those handles never reached
  device memory`.

The whole path is behind `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` and off by default.

`transfer::host_backing_for(ptr, len)` is the single public seam: the engine calls it after
`GetTensorMutableData` to turn an EP-owned handle into readable bytes. Without it the process
access-violates the instant ORT places a subgraph input in our memory.

`transfer::copy_counters()` reports copies, bytes, and how many landed at a non-zero offset into
a handle.

---

## Device-backed allocation: what `alloc_device_backed_spans: 427` does and does not mean

For the whole life of this project that counter read **0**. It is now non-zero, and the exact
extent of what that buys is the point of this section.

**What is real.** With `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1`, every span the handle registry
carves now also gets a real `VkBuffer` in `DEVICE_LOCAL` memory, and every `CopyTensors` *into* a
handle is mirrored across `vkCmdCopyBuffer`. Measured on the real 2.2 GB Phi-3.5 model:
`alloc_device_backed_spans: 427`, `alloc_device_uploads: 386`,
`alloc_device_upload_bytes: 2 094 231 552`. Two gigabytes really crossed into device memory, on
both vendors.

**What is not real yet, and the counter that says so.** `alloc_device_authoritative_spans` is
**0**. Every device-backed span *also* keeps its host staging block, and staging stays
authoritative. That is not a hedge — it is forced. `vk::session` resolves every kernel input
through `transfer::host_backing_for` and binds buffers it allocated itself. The first cut of this
work made a device-backed handle refuse to produce a host address, on the theory that the device
buffer was now the tensor's home. ORT's answer, on the first real model run:

```
EP_FAIL : ... Deserialize tensor model.layers.31.mlp.down_proj.MatMul.weight_Q4 failed.
VulkanExecutionProvider: copy 0/1 failed: could not obtain backing memory for device handle ...
```

and, when that was worked around, `input 1 is bound to device handle 0x… and its bytes are
unreachable`, followed by a silent fallback to `CPUExecutionProvider` for the whole model. So the
design is a **mirror**: the device buffer is written on every copy in and never read back, which
makes host and device incapable of disagreeing. `transfer::device_buffer_for(ptr, len)` is the
seam that ends the mirror — when the engine binds the returned `(BufferView, offset)` instead of
re-uploading its own buffer, `alloc_device_authoritative_spans` becomes the thing that moves.

**The offset in that tuple is not optional.** ORT's planner sub-divides one span across several
tensors, so binding a mirrored buffer at offset 0 for an interior pointer overwrites a
neighbouring tensor. `a_copy_into_a_device_backed_handle_mirrors_the_bytes_at_the_right_offset`
asserts on the bytes the provider was handed, not on a counter, and fails if the offset is lost.

### Why `epctl --check-counters --require-device-memory` still exits 1

It should. Every span is device-backed *and* every span is staged, so the flag's question — "were
this run's tensors where you asked them to be?" — is still answered no. `epctl` now names the
mirrored state explicitly rather than reporting it as a partial one, and
`every_span_device_backed_does_not_pass_while_every_span_is_also_staged` locks that in. The flag
passing was proposed as this work's deliverable; it turned out that making it pass would have
required either the fallback-to-CPU failure above, or lying about which memory the engine reads.

### The timing, and a prediction that was wrong

Recorded in `tank-device-memory-prediction.md` **before** any of this was implemented: device
backing on its own would make Phi-3.5 **1.1×–1.6× slower on the discrete RTX 4060** and
1.0×–1.2× slower on the UMA Iris Xe, because `vk::session` re-allocates and re-uploads every
input on every one of the 6099 `Compute` calls and device backing adds bus traffic without
removing any of that.

Measured on this build, `bench/phi35.py --device N --iters 8`, medians of three whole-process
repeats. **UMA and discrete are reported separately and are not comparable — `bench/compare.py`
refuses cross-device comparison by design.**

| device | device memory OFF | device memory ON | ratio |
| --- | --- | --- | --- |
| 0 — Intel Iris Xe (UMA) | 2921 ms (2782–3339) | 2958 ms (2916–2976) | 1.01× |
| 1 — RTX 4060 (discrete) | 2320 ms (2103–2345) | 2173 ms (2075–2184) | **0.94×** |

The discrete prediction is **falsified**: it did not get slower, it got slightly faster, and the
intervals nearly overlap. The reason is visible in a counter I already had:
`alloc_device_uploads` is **386** for a run of 19 inferences × 321 islands. The mirror upload
happens once per *allocation*, at weight deserialisation — it is not in the measured inference
loop at all, so the 2.09 GB crosses the bus during warm-up and never again. My prediction assumed
the added traffic landed in the measured path; it does not. The UMA prediction survives only
because its range included 1.0, which is not much of a survival.

**None of these numbers is a device-memory measurement.** Both configurations run the same
host-staged inference loop; the ON column additionally pays a one-time mirror. The lever that
would change the shape of this table is the engine binding `device_buffer_for`, and it is not
this work.

---

## What ORT's planner actually does with our handles

This was an argument for a long time and is now a measurement.

> **The op suite cannot observe any of this, and that is a property of the suite, not of the
> allocator.** Every helper in `tests/ops/_models.py` is `_session(model, providers).run(...)` — a
> session built, run **once**, and dropped. Measured on 2026-07-30, same machine, same DLL, both
> devices:
>
> | | sessions | runs each | interior pointers | max offset |
> | --- | --- | --- | --- | --- |
> | `test_elementwise.py` | 67 | 1 | **0** | 0 B |
> | `test_op_table.py` | 117 | 1 | **0** | 0 B |
> | `tools/probe_planner.py` | 1 | 5 | **52** | 49152 B |
> | `tools/probe_run2.py` (Phi-3.5, 2.2 GB) | 1 | 3 | **1192** | 65536 B |
>
> **The run-count sweep is the falsifier for that reading**, and it was run: the *same* probe on
> the *same* model and device, varying only `PROBE_RUNS`, gives
>
> | runs | 1 | 2 | 3 | 4 |
> | --- | --- | --- | --- | --- |
> | `pointers_interior` | **0** | 596 | 1192 | 1788 |
> | `alloc_frees` | 101 | 103 | 105 | 107 |
>
> Exactly 596 interior pointers per run after the first, and **0 at one run** — which reproduces
> the pytest suite's zero exactly. So 1192 is a property of the run count, not of the probe. If it
> were a property of the probe, `PROBE_RUNS=1` would still have shown interior pointers.
>
> So the suite that runs most often is structurally blind to the interior-pointer path, and its
> zero is not evidence of anything. `probe_planner.py --require-interior` is the standing check
> that the instrument still reaches the path at all: we have measured that it sees 52 interior
> pointers here, so a later zero means the *probe* broke, not that the planner stopped doing
> arithmetic. Without that flag the regression reads as a clean bill of health.

**It does pointer arithmetic on them, and only from the second run of a session onward.**

ORT's memory-pattern planner does not engage on the first `Run`. It *records* the allocation
pattern during run 1, and from run 2 onward it makes **one** allocation per pattern and hands out
sub-ranges of it. Every probe we had ran each session exactly once, which is why the interior
counter read 0 for so long: the instrument was pointed at a moment the phenomenon cannot occur in.

`tools/probe_planner.py` is the probe that catches it — static shapes, `enable_mem_pattern`, and
several runs of one session. The run count is the whole experiment:

| runs | `pointers_interior` | `pointer_max_offset` |
|------|--------------------|----------------------|
| 1    | 0                  | 0 B                  |
| 2    | 13                 | 49152 B              |
| 3    | 26                 | 49152 B              |
| 5    | 52                 | 49152 B              |

Thirteen derived pointers per run after the first, on **both** the RTX 4060 and the Iris Xe. The
trace says exactly what happened: six 16 KiB tensors were packed into a **single 64 KiB handle**
and ORT handed back `base + 16384`, `base + 32768`, `base + 49152`.

```
INTERIOR 0x1e65dcbc000 = handle 0x1e65dcb8000 + 16384 (span 65536 B)
INTERIOR 0x1e65dcc0000 = handle 0x1e65dcb8000 + 32768 (span 65536 B)
INTERIOR 0x1e65dcc4000 = handle 0x1e65dcb8000 + 49152 (span 65536 B)
```

**`pointers_in_guard_band` was 0 across every run on both devices.** That is the result that
matters: every derived pointer stayed inside the span it was derived from, which is what the
reserved-address-space design promises by construction. Had handles been opaque integers, each of
those 52 pointers would have been an unrecognisable value — not a crash, a *wrong answer*.

### At model scale, on the real 2.2 GB model

`tools/probe_run2.py` runs Phi-3.5-mini three times in one session with
`ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1`, so 427 allocations and 2.09 GB of real model tensors go
through the registry. Measured 2026-07-30, both devices:

| | device 0 (Intel Iris Xe) | device 1 (RTX 4060) |
| --- | --- | --- |
| `pointers_observed` | 9552 | 8908 |
| `pointers_interior` | **1192** | **1192** |
| `pointers_in_guard_band` | **0** | **0** |
| `pointers_use_after_free` | 0 | 0 |
| `pointer_max_offset` | 65536 B | 65536 B |
| `alloc_allocations` | 427 | 427 |
| `alloc_device_backed_spans` | **0** | **0** |

The interior count being *identical* across two vendors is what a deterministic planner should
produce, and is a weak self-check that the ledger is counting the planner rather than the driver.

**The instrument that would go red if the reserved-VA claim were false is
`pointers_in_guard_band`**, and it is not a log line: `epctl --check-counters` returns
`OutOfBounds` (exit 1) on any non-zero value, *before* it checks the dispatch count. It has now
had 21 460 opportunities across two vendors and has not fired.

**It also runs the session more than once on purpose, for a second reason.** On run 1 the arena is
freshly zeroed, so a tensor nobody wrote and a tensor written with zeros are indistinguishable.
From run 2 the arena is dirty, and the two separate cleanly: an unwritten buffer shows garbage, a
zero-written buffer shows zeros. Any probe that runs a session once cannot tell those apart.

**`probe_run2.py` asserts `EP_NAME in session.get_providers()` before it compares anything**, and
that gate has already earned its place — registering the library under a name the session does not
then request makes ORT print `Unknown Provider Type ... Falling back to CPUExecutionProvider` and
**not raise**, so the comparison silently becomes CPU-versus-CPU and agrees perfectly.

Its run-to-run diff compares **raw bytes**, not `max|a - b|`. `numpy` returns `nan` for the latter
whenever either side holds a `NaN` — including for two bit-identical arrays — so a magnitude-based
diff cannot distinguish "changed" from "both contain NaN". Magnitudes are reported separately and
only over the finite subset.

### The ledger

`allocator::ledger` classifies every pointer that crosses back to us and tallies it by
`LookupError` taxonomy: at-base, interior, guard-band, use-after-free, host. It observes both
endpoints of every `CopyTensors` and every `GetTensorMutableData` result the engine resolves.
It does **not** see arithmetic ORT performs internally and never shows us, so a zero means "ORT
never handed us a derived pointer", not "ORT never computed one".

The numbers are written to `$ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` at teardown as
`pointers_observed`, `pointers_host`, `pointers_at_base`, `pointers_interior`,
`pointers_in_guard_band`, `pointers_use_after_free`, `pointer_max_offset`, with the verbatim trace
beside it in `*.trace.txt`. They are written to the file rather than logged because by teardown
ORT's logger is usually already gone — and a process cannot read its own teardown, which is why
`probe_planner.py` runs the session in a child and reads what it left behind.

### Quarantine: armed, never fired, and proven able to fire

`pointers_use_after_free` was **0** in every real session, now including the 2.2 GB model: 18 460
pointer observations and 210 frees across both vendors. The registry is unambiguously in ORT's
path — 1192 of those observations were interior pointers it derived itself — so the zero can no
longer be dismissed as "the detector is not connected". It still is not a pass. The claim this
supports is precisely:

> ORT has not presented us with a freed handle under any allocation pattern we have run.

That is not "quarantine is verified", and it must not be written up as such.
`the_quarantine_detector_fires_when_a_stale_handle_is_presented` is the positive control that
separates the two: it frees a handle and presents it through the same `classify` funnel a real
session uses, and requires the ledger to count it and `host_bytes` to refuse it. Break the
detector and that test fails — verified by planting the break. Generation-stamped rejection is
therefore proven only against frees in an order *we* chose.

**And multi-run does not improve the odds — it makes them worse.** The obvious next question is
whether ORT's *free* ordering changes from run 2 the way its allocation pattern demonstrably does,
which would mean quarantine had been unexercised in the only regime where it could plausibly fire.
Measured, same sweep: free traffic grows by **2 per additional run** while interior-pointer
derivation grows by **596**. That is the planner working as designed — from run 2 it makes one
allocation per pattern and stops allocating and freeing almost entirely. So the multi-run regime is
the one *least* likely to present a stale handle, not the most. Quarantine's exercise gap is not a
gap that more runs will close.

---

### Diagnostics must read zero when nothing is wrong

`classify` probes every registry, so a miss is the *expected* answer for every host pointer.
Routing that through `resolve` — which counts misses — made `failed_lookups` non-zero on a
perfectly healthy run. `HandleRegistry::classify` is the non-counting twin, and `failed_lookups`
is reserved for lookups that *should* have succeeded. A diagnostic that is non-zero when nothing
is wrong is one people learn to ignore.

---

## Environment variables

| Variable | Effect |
|---|---|
| `ONNXRUNTIME_EP_VULKAN_VERBOSE=1` | verbose EP logging through ORT's logger |
| `ONNXRUNTIME_EP_VULKAN_TRACE=1` | per-node trace during capability and compile |
| `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` | log every node the EP declined **and why**, aggregated by op type |
| `ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE=<path>` | write the execution-counter snapshot here on the first successful dispatch and at factory teardown; read it back with `epctl --check-counters` |
| `ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION=1` | turn "the Vulkan validation layer is unavailable here" from a loud skip into a failure, so a lane cannot silently drop its own positive control. CI should set this |
| `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` | **opt-in.** Advertise a device allocator to ORT, so ORT allocates tensors through us. Off by default — see *The device allocator* below |
| `ONNXRUNTIME_EP_VULKAN_VA_RESERVE_MIB=<n>` | size of the reserved virtual-address arena per device (default 65536 MiB, halved until the OS agrees) |
| `ONNXRUNTIME_EP_VULKAN_QUARANTINE_SPANS=<n>` | how many freed handles are held before their address space is reused (default 4096) |
| `RUST_LOG=debug` | Rust-side `log` filtering, independent of ORT's level |
| `LIBCLANG_PATH` | build only — where `bindgen` finds libclang |
| `ORT_INCLUDE_DIR` / `ORT_HOME` | build only — override the vendored headers |
| `VULKAN_SDK` | build only — where `build.rs` looks for `glslc` |
| `ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1` | build only — allow a build with shaders present but no compiler |

Session-option equivalents (`ep.vulkan.*`) are read in `src/ep.rs` and take precedence over the
environment.

---

## Layering lint

`DESIGN.md` §4.2 states two rules that protect the op layer:

1. The ORT C ABI never appears in `src/ops/`.
2. Raw Vulkan — and `unsafe` — never appears in `src/ops/`.

Both are enforced mechanically by [`tests/layering.rs`](tests/layering.rs):

```powershell
cargo test --test layering
```

(It also runs as part of `cargo ci` — see [Before you report work complete](#before-you-report-work-complete-cargo-ci).)

It scans `src/ops/**/*.rs` for a forbidden vocabulary (`crate::sys`, `Ort*`, `ash`, `vk::`, `Vk*`,
`unsafe`, …) after stripping comments and string literals, so documentation that *names* a
forbidden token is not a false positive — `src/ops/mod.rs` names all of them on purpose. A
mirror-image check keeps `ash`/`vk::` out of the ORT boundary modules.

The lint is itself tested: several cases run the scanner over deliberately planted violations and
assert it catches each one, so a refactor that neuters the detector fails too. It was also verified
against a real planted file under `src/ops/`, which produced seven findings before being removed.

#### It has caught a real violation

On 2026-07-29 the mirror-image check failed on `ep.rs:28` — `use crate::vk::session::{...}`, added
by the engine owner while integrating the compiled-session types into `Compile`. A single `use`
line in a 1600-line file, added in good faith; not the kind of thing review catches reliably.

The fix was not to relax the lint. `engine.rs` now re-exports the three names the boundary layer
needs:

```rust
pub(crate) use crate::vk::session::{CompileRecorder, CompiledKernel, VulkanSession};
```

so `ep.rs` says `use crate::engine::{...}` and the module dependency table (`DESIGN.md` §4.3) holds
as written: `ep.rs` → `engine` → `vk`. The condition that keeps the re-export honest rather than a
laundering trick is stated above it in `engine.rs` — **nothing re-exported there may expose an
`ash` type in its public signature.** While that holds, `ep.rs` has no path to a raw Vulkan handle,
which is the property the rule exists to protect.

### Contrib domain (constraint C1)

`DESIGN.md` §1.4 **C1** forbids any domain-wide contrib opt-in: the registry key *is* the
allowlist. The same lint enforces it by banning the contrib domain as a **value** in non-test
code — both `"com.microsoft"` as a bare string and `Domain::Ms` as a variant — with exactly one
exemption, the `Domain::Ms => "com.microsoft"` arm of `Domain::as_str` that defines the spelling.

Banning the value rather than enumerating comparison forms is what makes it airtight: `==`, `!=`,
`matches!`, `if let` and `starts_with` all become unwritable at once, and there is no third
spelling to forget. Fully-qualified names such as `"com.microsoft::MatMulNBits"` remain
permitted — they name one op, which is precisely what C1 asks for instead of a domain predicate.

Test modules are out of scope on purpose: C1's own regression test has to fabricate a contrib
node, and a lint that forbade that would forbid the proof. The runtime half — fabricate
`com.microsoft::NotARealOp`, assert an ordinary `not-registered` decline plus a correct CPU
fallback run — is an M-tier regression test in Trinity's harness.

**Why a test rather than a `deny` attribute or an xtask.** A lint attribute cannot express "this
identifier must not appear in this directory"; `ash` is a legitimate dependency of the crate, and
there is no built-in way to forbid it per-module. An xtask is a second binary to build and invoke
on every CI lane. A test is already run by `cargo test`, so it cannot be forgotten — CI cannot be
green without it, and a contributor sees the failure locally before pushing.

*CI wiring is Trinity's.* This crate owns the check and the local command; `.github/workflows/`
needs `cargo test --all-targets` (which includes it) plus LLVM/libclang on every runner.

---

## Portability lint

[`tests/portability.rs`](tests/portability.rs), run by `cargo test --test portability` and by
`cargo ci`. It is the layering lint's companion for a different failure class: **code that
compiles on the machine it was written on and cannot compile anywhere else.**

| Rule | What it enforces |
|---|---|
| **P1** | A binding that exists on only some targets may only be named by a `cfg`-gated definition. Today that list is one entry, `ort::wchar_t`. |
| **P2** | Every `#[cfg(windows)]` item has a `#[cfg(not(windows))]` sibling in the same file. |

It also prints the crate's entire platform-conditional surface and fails if it grows past a
reviewable size, on the theory that a small surface consolidated behind aliases is worth more than
a lint chasing `cfg` spread through the crate.

**Why it exists.** `ORTCHAR_T` is `wchar_t` on Windows and `char` elsewhere, so bindgen emits
`ort::wchar_t` only on Windows. `tests/mock_ort/mod.rs` named it directly, which compiled here and
broke the Linux lane outright — masking everything behind it, including the lavapipe result we
were actually waiting for. `cargo ci` had already printed a caveat saying precisely this could
happen. This file is that caveat converted into a mechanism.

**Why not just cross-compile locally?** Because it does not work on a Windows box, and that was
measured rather than assumed. `rustup` has the Linux and macOS std libraries installed here and
`cargo check --target x86_64-unknown-linux-gnu` gets all the way through every dependency —
bindgen even re-targets clang correctly — before dying in `build.rs` on `'stdlib.h' file not
found`. Clang's own builtin headers can be supplied via `BINDGEN_EXTRA_CLANG_ARGS` (that fixes
`stdbool.h`), but `stdlib.h` is glibc's, and having it means vendoring or downloading a Linux
sysroot on every dev box. Rejected as infrastructure we would not maintain; see D-T20 in
`.squad/decisions/inbox/tank-m0-foundation.md`. A lint is not a compiler, and `cargo ci` says so
in its own output.

---

## Module map

| Module | Owner | What it is |
|---|---|---|
| `src/lib.rs` | Tank | crate root, `guard_ffi_status` panic guard, the two exported C symbols |
| `src/sys.rs` | Tank | generated ORT bindings, version gates, status helpers |
| `src/factory.rs` | Tank | `OrtEpFactory` vtable, device enumeration and correlation |
| `src/ep.rs` | Tank | `OrtEp` vtable, session options, capability reporting |
| `src/logging.rs` | Tank | Rust `log` → ORT logger bridge |
| `src/registry.rs` | Mouse | the op table; `NodeView`, the ABI→safe-Rust translation point |
| `src/engine.rs` | **Switch** | **stub** — the Vulkan engine seam and shared vocabulary |
| `src/ops/mod.rs` | **Mouse** | **stub** — op handlers |
| `build.rs` | Tank | bindgen + GLSL→SPIR-V pipeline |

### FFI discipline

Non-negotiable in this crate, and visible in every file:

* Every exported `extern "C"` function body runs inside `guard_ffi_status`, which catches unwinds
  and converts them to an `OrtStatus`. **No panic crosses the FFI boundary** — unwinding into
  ORT's C++ is UB.
* Every `unsafe` block carries a `// SAFETY:` comment stating the invariant it relies on.
  `clippy::undocumented_unsafe_blocks` is `warn`, and CI runs with `-D warnings`, so an
  undocumented block fails the build.
* No `unwrap()` / `expect()` anywhere near the ABI boundary. Null pointers from ORT are checked,
  not assumed.
* `panic = "unwind"` in **both** dev and release profiles — `panic = "abort"` would stop the guards
  from working.

---

## What is stubbed, and for whom

**Switch** (`src/engine.rs`) — replace `probe_devices()` with real `vkEnumeratePhysicalDevices`
(its contract: never fails, returns sorted best-first, returns empty when there is no loader), add
the `vk/` tree, implement `DispatchContext`, and add SPIR-V shaders under `shaders/glsl/` for
`build.rs` to compile.

**Mouse** (`src/registry.rs`, `src/ops/`) — add rows to `REGISTRY` and the matching handlers. The
invariant to preserve: a node is claimed *only if* it can actually be translated. The layering lint
is on and will reject an op handler that reaches for `sys` or `ash`.

**Tank**, next — the allocator and data-transfer vtable slots (M2) and the external resource
importer when zero-copy IO binding is wanted. `src/sys.rs` carries the bound seam and a marked
TODO showing exactly where that slots in.

---

## The `Compile` → `Compute` seam

**`Compile`** walks each fused subgraph ORT hands us and reads every body node into an owned
`engine::NodeDesc` (op type, domain, since-version, attributes, and per-edge name / dtype / shape),
producing an `engine::Plan`. The plan's `inputs` and `outputs` come from the **fused node**, not
from the subgraph body, because that is the order ORT binds tensors in at `Compute` time — taking
them from the body produces a list that looks right and is indexed wrong. The plan borrows nothing
from ORT: once `Compile` returns, no graph pointer survives anywhere.

Each node's registry `translate` handler then runs against a `CompileRecorder` — a `DispatchContext`
that records instead of dispatching — baking a `Vec<CompiledKernel>`. Input/output byte sizes and
output shapes are resolved from the plan at the same time. All of it lands in a
`SubgraphComputeInfo`, which is what ORT holds.

**`Compute`** checks that the kernel context binds exactly as many tensors as the subgraph was
compiled for, then calls `VulkanSession::dispatch_ort`. The count check is not defensive noise: the
compiled counts come from the fused node and the bound counts come from ORT, and if they ever
disagree every index past the mismatch names a different tensor than the compiled plan believes —
a wrong answer rather than an error.

### Failure is a status, never a null

ORT reads a **null return from `Compute` as success.** A `Compute` that fails quietly therefore
reports success and leaves ORT's output tensors holding whatever was in them — a silent wrong
answer, which from the host's point of view is worse than a crash and indistinguishable from a
working EP. `SubgraphComputeInfo` carries the `OrtApi` for exactly this reason, and
`ep::tests::a_compute_that_cannot_dispatch_returns_a_status_not_a_silent_success` pins it.

A subgraph with no device or no recorded kernels gets a *stub* compute-info, which returns a status
saying so rather than writing nothing and returning success.

One consequence worth stating plainly: once an op is claimable, a `Compile` or `Compute` failure
**fails session creation** rather than falling back to CPU. That is the right trade — correct and
loud beats fast and wrong — but it means the first claimable op and a working dispatch path need to
land together.

---

