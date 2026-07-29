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

### It works without a Vulkan SDK

If `glslc` is not found (neither `$VULKAN_SDK/bin/glslc` nor on `PATH`), `cargo ci` sets
`ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1` for you so `build.rs` does not abort. It also
sets `LIBCLANG_PATH` for bindgen if it can find a libclang and you have not set one.

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
| Vulkan SDK / `glslc` | not yet | only needed once `shaders/glsl/` is non-empty |
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

## Environment variables

| Variable | Effect |
|---|---|
| `ONNXRUNTIME_EP_VULKAN_VERBOSE=1` | verbose EP logging through ORT's logger |
| `ONNXRUNTIME_EP_VULKAN_TRACE=1` | per-node trace during capability and compile |
| `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` | log every node the EP declined **and why**, aggregated by op type |
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

**Tank**, next — real `Compile`, the allocator and data-transfer vtable slots (M2), a proposal for
OQ-3 (opaque-handle registry vs `VK_KHR_buffer_device_address`) for Morpheus to decide, and the
external resource importer when zero-copy IO binding is wanted. `src/sys.rs` carries the bound
seam and a marked TODO showing exactly where that slots in.
