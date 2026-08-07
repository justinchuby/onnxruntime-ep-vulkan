# onnxruntime-ep-vulkan

[![CI](https://github.com/justinchuby/onnxruntime-ep-vulkan/actions/workflows/ci.yml/badge.svg)](https://github.com/justinchuby/onnxruntime-ep-vulkan/actions/workflows/ci.yml)

> **CI is the only place shaders execute.** A red CI lane means we have zero empirical
> evidence that the EP works. A red badge blocks all merges — not by a GitHub branch
> protection rule (not yet configured), but by a hook that reads the badge for you.
> See [`.github/CI_POLICY.md`](.github/CI_POLICY.md) for the full policy.
>
> **Install the hook once per clone: `python ci/install_hooks.py`.** It points
> `core.hooksPath` at the tracked [`.githooks/`](.githooks) directory, whose
> `pre-merge-commit` runs [`ci/check_main_is_green.py`](ci/check_main_is_green.py) and
> refuses the merge when `main` is red, or when the colour could not be read at all —
> because "I could not ask" is UNOBSERVABLE, not green. `MAIN_COLOUR_ACK=red` and
> `MAIN_COLOUR_ACK=unread` let you merge anyway; neither is silent, and both have to be
> typed. `python ci/install_hooks.py --check` says whether the hook is live in this clone.
>
> **Why this replaced "run `gh run list --limit 5` before landing a commit".** That
> instruction was correct and it was followed zero times in a window where `main` was red
> for at least ten consecutive pushes. Every merge in that window was verified locally —
> lib tests, clippy, `cargo fmt --check`, the proof ledger — and every report quoted those
> green numbers, which are a complete-looking answer to a different question. A notice is
> read once, at a moment when nobody is merging. The reading now happens at the moment it
> is about.
>
> **The badge is red as of this writing, and has been for at least the last ten pushes to
> `main`** (`python ci/check_main_is_green.py`). Read the notice above as a rule the
> project is currently in violation of, not as a description of its state.

A **cross-platform Vulkan compute execution provider for ONNX Runtime**, written in Rust and
shipped as an out-of-tree **plugin EP**. It is loaded by a stock, unmodified ONNX Runtime through
the plugin-EP C ABI — no ORT fork, no ORT rebuild, no link against `libonnxruntime`.

## Status

> **Every number in this section is re-derived from a named artifact or symbol in this tree, at a
> named commit. Where a figure is a *computation* rather than a *measurement*, it says so.**
> A README that lists only wins is a marketing document, and the first reader who checks one line
> of it will correctly distrust the rest — so the second half of this section is as long as the
> first. [`docs/DESIGN.md`](docs/DESIGN.md) §0 is the same statement one level down, for a reader
> who is going to work on the code.
>
> **Re-derived: 2026-08-04, at `4ee8c9c` (`main` = `b8679a4` plus one docs merge); the op-table row
> re-derived 2026-08-05 at `ba2da4c` from a fresh `epctl --dump-capabilities --json` run, which is
> where 96/78 and the field name `has_kernel` come from.** Sources:
> `epctl --dump-capabilities --json`; `bench/results/{op_census_phi35_r13_after,
> op_census_mnv2_r14_final, op_census_bert_r14_final, op_census_gptoss20b_r13_after,
> phi35_claim_reading_summary, criterion10-dev0, lane-identity-armA-arena0,
> paired_ratio_dev0, _ledger_counters}.json`; `evidence/proof_ledger.jsonl`;
> `ci/check_ledger_census.py`; `cargo test --lib`; `docs/PLATFORMS.md` §7.26.3.
> **A README with no commit and no artifact list is a citation with no subject** — it cannot be
> checked, only believed, and it will be stale again within days. Check it: every claim below
> names where it came from.

### What it does

**It loads into a stock ORT and claims most of a real decoder.** On Phi-3.5-mini-instruct (ONNX,
fp16, int4 `MatMulNBits`, external data, 2.2 GB), `op_census_phi35_r13_after.json` reads
`graph.nodes: 366`, `claimed_nodes: 355`, `declined_nodes: 8`. ORT fuses them into **one island**:
`phi35_claim_reading_summary.json` — `islands_offered: 1`, `viable_islands_retained: 1`,
`ledger_hits: 355`, `unproven_forms_claimed: 0`. Attribution is ORT's own profiler, not ours:
`criterion10-dev0.json` records `executed_by = {VulkanExecutionProvider: 3, CPUExecutionProvider: 24}`
over three inferences — one island execution per run.

**It runs two model classes, not one.** Since `Conv` (f32) landed, the EP is no longer
decoder-only: `op_census_mnv2_r14_final.json` reads **98 of 105** claimed on MobileNetV2-12, a
convolutional image classifier that was **0 of 105** before. The 7 declines are one contiguous
`Shape→Gather→Unsqueeze→Concat→Reshape` classifier tail, not seven scattered holes.

| Model | Graph nodes | Claimed | Artifact |
|---|---|---|---|
| Phi-3.5-mini-instruct (int4, fp16) | 366 | **355** | `op_census_phi35_r13_after.json` |
| gpt-oss-20b (int4 MoE, fp16) | 374 | **293** | `op_census_gptoss20b_r13_after.json` |
| MobileNetV2-12 (f32) | 105 | **98** | `op_census_mnv2_r14_final.json` |
| BERT-SQuAD-12 (f32) | 1167 | **480** of 1274 decisions | `op_census_bert_r14_final.json` |

BERT records more decisions than the graph has nodes because ORT calls `GetCapability` more than
once; the number quoted is the decision count, which is what the census counts.

**Its op table is 96 rows, of which 78 carry a kernel.** Read from `epctl --dump-capabilities
--json` (`rust/src/bin/epctl.rs`), not by counting `ops!` lines: **46 `live`, 32 `ready`, 18
`staged`** — the staged rows are described and claim-tested and decline at runtime with a named
blocker in the dump's own `staged_reason` field. **The 78 is the count of rows whose boolean
`has_kernel` field is `true`, which is *not* `status == "live"` (46).** The row used to spell `live`
twice with two meanings — a boolean for *this row has a kernel* and a status token that is the
deprecated `OpStatus::Live` alias — so a reader checking this sentence against the field named
`live` got 46; the boolean is now `has_kernel` and the derivation is checkable against the field it
names ([`docs/DESIGN.md`](docs/DESIGN.md) §8.9.25 ruling 6, landed). **This count has been
misstated more than once, including by earlier revisions of this file; the dump is the only reading
of it that is not a hand tally.** `OpStatus::Live` is a **deprecated alias** of `OpStatus::Ready`
(`rust/src/registry.rs::OpStatus`) and neither status grants a claim — **claimability is a ledger
fact, not a table field** ([`docs/DESIGN.md`](docs/DESIGN.md) §8.9).

**Nothing is claimed without a proof under its own key.** `evidence/proof_ledger.jsonl` holds
**129 entries** (its own header line, `entry_count: 129`), each recording the device, the ORT
build, the tolerance, the artifact and its sha256, the nodes claimed and dispatches executed on the
proof run, the shader stems dispatched, and three digests over those stems — SPIR-V, source
closure, and specialisation — so that a compiler change, a kernel change and a specialisation
change are distinguishable from one another. A form with no entry declines with `[unproven]`.
**All 129 entries now carry `spec_digest`** (`_ledger_counters.json::ledger_specialisation_unrecorded_entries: 0`),
closing a debt that stood at 102 of 121 two days ago: `SPEC-UNRECORDED` was the ledger's own word
for *"this proof does not name the specialisation constant it was proven under"*, and a
specialisation constant is resolved at pipeline creation and changes the code the driver emits, so
an entry without it named its subject only partly.

**The KV cache can decline the host round trip.** `kv_chain_readback-{nvidia,intel}.json` both
read `ROUND_TRIP_REMOVED`: steady-state device→host traffic falls from **1792 to 0 bytes per step**
at identical dispatch counts, on both an RTX 4060 and an Iris Xe. No wall-clock appears in that
artifact, by design. **None of it reaches a user today** — see below.

**The conservatism is tested, not asserted.** Zero Vulkan devices are advertised when there is no
ICD and when the build carries no shaders (`criterion4_icd_witness-dev{0,1}.json`,
`criterion5_shaderless_witness-dev{0,1}.json`); the layering lint runs in CI over permanently
planted violations (`rust/tests/layering.rs::detects_planted_ort_abi_violations`); and the Rust
library suite is **574 passed / 0 failed / 4 ignored** (`cargo test --lib` at `4ee8c9c`).

### What is not true yet

**M0 is not met.** Five of its twelve criteria are met (3, 4, 5, 6, 7) and seven are not (1, 2, 8,
9, 10, 11, 12), per the criterion-by-criterion table in [`docs/DESIGN.md`](docs/DESIGN.md) §10.
Criterion 1 is red *today* — CI's clippy step, above. **This is an early implementation.**

**The device-memory path ships OFF, so none of the KV work above reaches a user.**
`factory::device_memory_enabled` is a read of the `ENV_DEVICE_MEMORY` environment variable and
nothing else; unset means false. A default run reports `alloc_device_frame: "OFF"` and, in its own
words, `"no device-memory provider exists in this process, so the allocator side is on no VkDevice
at all"` (`_ledger_counters.json`). A capability behind an opt-in flag is a capability the project
has, **not one the product ships**.

**There is no timing of any kind here, and the design that was going to produce it is refuted —
not withheld.** The paired interleaved A/B alternation assumed contention is common-mode across
the two arms. `paired_ratio_dev0.json` returns `PAIRING_FAILS(apparatus_asymmetry)`: `cpuload`
moves the ratio 0.560×, `gpuload` 0.722×; `paired_ratio_dev1.json` and
`paired_ratio_resident_dev0.json` fail the same way — four control failures. Two are worse than a
failed control: under foreign **GPU** load our own arm gets *faster* (`vk_lift_x: 0.771`), which no
contention model predicts, and the granularity that would make contention symmetric is the same
granularity that manufactures the device-axis asymmetry, **so the failure is not tunable**. No
ratio is published and none is withheld: the apparatus does not measure what it claims to.

**Output agreement with the CPU EP is `DIVERGENT`, on both devices.** Over all 65 model outputs,
`criterion10-dev0.json` records **62 of 65 within tolerance**, a **median of 1 fp16 ULP**,
`argmax_cpu == argmax_vk` and `top10_overlap = 10`. Three are outside: the logits head at a median
of **12 ULP**, and the last layer's key and value at **4 ULP** each. It must not be closed by
moving `atol` — [`docs/DESIGN.md`](docs/DESIGN.md) §8.9.24 rules the predicate satisfiable at every
representable fp16 value with ≥ 20.48 element-ULPs of margin, so each failing element failed by
more than twenty representable steps at its own magnitude. **Which side is wrong is under
investigation and not yet answered.**

**At a 4096-token context the shipping lane cannot run at all.** `lane-identity-armA-arena0.json`
records verdict `ERROR(instrument)` with `gpu-allocator failed to allocate 14155776 bytes for
'ep_in_427': Out of memory`, `compute_failures: 1`, `dispatches_executed: 0` — **and
`exit_code: 0`.** The same lane at past lengths 512 and 1024 runs clean (`ctx_device_loss.json`:
8875 and 5325 dispatches, 355 claimed nodes, zero failures). Two separate defects: the allocation
failure, and **a failed inference that reports success at the process level**, which is the one a
user meets first.

**The ledger's loss invariant is FAILING, and the cause is two retirement registers.**
`ci/check_ledger_census.py` reports `172 ever proven = 129 present now + 0 retired + 43 VANISHED`
and exits non-zero. Those 43 keys *were* retired, deliberately and with reasons, into
`evidence/retired_proof_keys.json` — which is the file `rust/tools/gen_proof_ledger.py` reads. The
census reads `evidence/proof_retired.json`, which **does not exist**, and the two files do not even
share a schema (a list of `{key, reason}` against an object keyed by proof key with
`owner`/`date`/`reason`). So the generator says *43 retired* and the census says *43 vanished*
about the same 43 keys. **The screen written to catch a silent deletion is currently failing on a
deliberate one**, which is the same class of defect it exists to detect, and it is open.

**Windows is one desk, and Linux is red.** Every GPU result above is Windows, on local hardware,
on two desktop GPUs (Intel Iris Xe, NVIDIA RTX 4060). The Linux/lavapipe op suite reads **8 failed
/ 633 passed / 50 skipped / 3 xfailed** ([`docs/PLATFORMS.md`](docs/PLATFORMS.md) §7.26.3, where
the 8 are decomposed by cause rather than quoted as a total: 3 host provisioning, 1 unexplained
instrument error, 2 real `Asin`/`Acos` divergence on llvmpipe, 1 closed, 1 a declared accepted
red). M0 criterion 2 says *green*; green is not what either lane reports. **Android and macOS have
never run at all** — every Android and macOS entry in [`docs/PLATFORMS.md`](docs/PLATFORMS.md) §5
is explicitly marked untested, and CI has no GPU hardware of any kind.

**No performance claim is made anywhere in this repository.** The development machine is
permanently contended by several agents.

| | |
|---|---|
| Registered EP name | `VulkanExecutionProvider` |
| Artifact | `libonnxruntime_vulkan_ep.so` / `onnxruntime_vulkan_ep.dll` / `libonnxruntime_vulkan_ep.dylib` |
| ORT ABI | plugin-EP C ABI · built against ONNX Runtime **1.28** · minimum runtime API **1.24** *(`epctl --dump-capabilities --json`: `ort_built_against 1.28.0`, `ort_api_version 28`, `ort_minimum 1.24.0`, `ort_api_version_min 24`)* |
| Backend | Vulkan compute · GLSL → SPIR-V · [`ash`](https://github.com/ash-rs/ash) |
| Target platforms | `Windows · Linux · Android · macOS (MoltenVK)` — **Windows executes claimed nodes on real GPUs, on one desk. Linux/lavapipe has executed claimed nodes since 2026-07-30 ([`docs/PLATFORMS.md`](docs/PLATFORMS.md) §7.7) and its op suite is red (§7.26.3). Android and macOS have zero coverage on any hardware; every Android and macOS entry in [`docs/PLATFORMS.md`](docs/PLATFORMS.md) §5 is explicitly marked untested.** |
| Device requirement | **Vulkan 1.1 core + a compute queue.** No required extensions — see [`docs/DESIGN.md`](docs/DESIGN.md) §7. |
| Target hardware | NVIDIA · AMD · Intel · Adreno · Mali — *none of these is covered by CI today; CI has no GPU hardware at all. The only executing lanes are two desktop GPUs on one development machine and the lavapipe software rasterizer.* |
| Operator domains | `ai.onnx` and `com.microsoft` — the contrib domain is in scope because the ORT GenAI model builder emits contrib ops directly; see [`docs/DESIGN.md`](docs/DESIGN.md) §1.4 for the claim-safety constraints |
| Op table | **96 rows — 46 `live`, 32 `ready`, 18 `staged`; 78 carry a kernel (`has_kernel == true` on the dump row, which is not `status == "live"`).** Status is not permission to claim: **claimability is a ledger fact** (`evidence/proof_ledger.jsonl`, 129 entries), not a table field. |

## How it works

```
ONNX graph
   └─ GetCapability   claim only node forms the Vulkan translator implements exactly
   └─ fuse            maximal convex connected clusters
   └─ Compile         build a plan · prepack + upload constant weights once · warm pipelines
   └─ Compute         record (or replay) one command buffer · one submission · fence
Everything unclaimed runs on ORT's CPU EP.
```

Conservative claiming with clean CPU fallback is a hard requirement, not a stopgap: an unclaimed
op is always correct, and a wrongly-claimed one is silently wrong.

## Building

**Prerequisite: the Vulkan SDK, or `glslc` on `PATH`.** Shaders are compiled from GLSL to SPIR-V at
build time and embedded in the library; there is no checked-in SPIR-V, deliberately — a checked-in
binary that drifts from its source would silently change what runs
([`docs/DESIGN.md`](docs/DESIGN.md) §7.8). A build without `glslc` fails with a message naming what
to install.

```powershell
cargo build --release        # from rust/
cargo test                   # lib + layering + capability-dump tests
```

`ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1` builds a **shader-less** artifact for lint-only and
docs-only lanes. It can create no compute pipeline, it advertises no device, and it must never be
shipped.

**A wheel is a checked-in binary from the consumer's point of view**, which is the same
hazard §7.8 rules on, one level out. It is answered three ways, all in
`python/build_wheel.py` and all tested in `tests/packaging/test_wheel_provenance.py`:
nothing binary enters the tree (the staged directory is gitignored, and `git check-ignore`
is the test's authority, not a reading of `.gitignore` by eye); the wheel carries a
`_provenance.json` naming the commit, whether that tree was dirty, its own sha256, and a
digest over all 17 shader sources with the file count beside it; and the packaging step
**refuses an artifact containing zero SPIR-V modules**, which enforces §7.8 condition 4
against the shipped bytes rather than against whichever shell happened to set the escape
hatch. That detector is exercised in both states — 210 modules in the shipping artifact,
0 in a shader-less one.

A digest match proves the shipped bytes are the recorded bytes. It does **not** prove they
were compiled from the named commit; only a rebuild at that commit does, which is why the
digest and the commit are separate fields with separate meanings.

## Installing and using it

There are two consumption paths and both are tested. **Pick the first if you have the
Vulkan SDK and want a wheel; pick the second if you already build this repository.**

### 1. Build a wheel and install it

```powershell
cargo build --release                 # from rust/ — needs the Vulkan SDK / glslc
python python/build_wheel.py          # stages the cdylib + a provenance record
pip install python/dist/onnxruntime_ep_vulkan-*.whl
```

```python
import onnxruntime as ort
import onnxruntime_ep_vulkan

onnxruntime_ep_vulkan.register_execution_provider_library()
sess = ort.InferenceSession(model, providers=onnxruntime_ep_vulkan.providers())
onnxruntime_ep_vulkan.assert_ep_selected(sess)   # ORT will not raise for you — see below
```

`python -m onnxruntime_ep_vulkan --check` registers the EP, runs a four-element `Add` on
it, and reports whether the EP was actually selected.

**Nothing is published to PyPI, deliberately** — no release process is in scope, and a
`pip install onnxruntime-ep-vulkan` from an index would not work today. The wheel is built
locally from a source tree you control.

**This path is demonstrated, not asserted.** `python/verify_cleanroom.py` creates a fresh
venv *outside* this repository, installs only the wheel, and — with the repository
unreachable — imports, registers, runs, and asserts the EP was selected. Its `pip install`
of `onnx`/`numpy` into that venv defaults to the approved proxy index
(`https://packagefeedproxy.microsoft.io/pypi/simple`, override with `--index-url` or
`$ONNXRUNTIME_EP_VULKAN_PYPI_INDEX_URL`) because public PyPI is blocked in the sandboxes
this tool is actually run from during EP development (issue #40). If an override embeds
credentials (`user:pass@host/...`), those credentials reach `pip` alone, in the argv passed
directly to it — the tool's own progress echo and its persisted
`cleanroom_install_dev0.json` record always redact any URL userinfo to a fixed
`REDACTED@host` placeholder first, never the raw value (issue #55).
`bench/results/cleanroom_install_dev0.json` records the run: `verdict: PASS`,
`session_providers: ["VulkanExecutionProvider", "CPUExecutionProvider"]`,
`artifact_inside_site_packages: true`, on Windows / RTX 4060 / ORT 1.28.0. **It has been
demonstrated on this platform and no other**; there is no wheel CI matrix and none is
claimed.

### 2. Use the ORT API directly, from a source checkout

No package needed. This is what the repository's own 26 test files do:

```python
import onnxruntime as ort

ort.register_execution_provider_library(
    "VulkanExecutionProvider",
    r"C:\path\to\rust\target\release\onnxruntime_vulkan_ep.dll",   # must be ABSOLUTE
)
sess = ort.InferenceSession(model, providers=["VulkanExecutionProvider", "CPUExecutionProvider"])
assert "VulkanExecutionProvider" in sess.get_providers()          # ORT will not check this
```

### Four things about that one ORT call, all measured

`bench/results/consumption_surface_dev0.json` — six cases, each in its own subprocess
because plugin registration is process-global state. These are why path 1 exists; a wrapper
around one call would otherwise not be worth a package.

| Measured | Consequence |
|---|---|
| A relative library path resolves against ORT's own `capi` directory, **not** the caller's CWD | the absolute path is mandatory, and nothing tells you |
| The registration name is never checked against the library — it registered fine as `"NotOurNameAtAll"` and advertised its GPUs under that name | the name at registration and the name in `providers=[...]` must agree, enforced by nobody |
| Registering the same name twice **raises** | the call is not safe to run twice in one process |
| **A session asking for an unregistered EP name does not raise.** It warns, falls back to CPU, and returns numerically correct results | the natural failure is a session that silently never touches the GPU |

The last row is the one that matters. A reader of an earlier revision of this file was told
to `import onnxruntime_ep_vulkan` — a package that did not exist — got `ModuleNotFoundError`,
and the obvious fix (delete the import, keep the providers list) produced a working session
running entirely on CPU with no error. `assert_ep_selected` and the bare `assert` above are
the answer to that; both check the session's own `get_providers()`, which is the only
reliable reading, because warnings are routinely filtered or lost.

`assert_ep_selected` asserts the EP was *selected for the session*. It does not assert that
any node was claimed or that any dispatch executed — ORT can select an EP that claims
nothing, and `claimed_nodes` is not what executes (see the census caveat above).

## Documentation

| Document | Owner | Contents |
|---|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Morpheus | **Architecture of record.** Goals and non-goals, ORT integration, crate layout, module boundaries, execution flow, tensor/memory model, Vulkan baseline, op strategy, testing, milestones, open questions. |
| [`docs/ENGINE.md`](docs/ENGINE.md) | Switch | Vulkan runtime: device and context, memory and allocators, shader strategy, pipelines and descriptors, command submission and synchronization. |
| [`docs/PLATFORMS.md`](docs/PLATFORMS.md) | Link | Platform and driver support matrix, Vulkan version reality per platform, capability detection, toolchains, CI lanes. |
| [`docs/OP_COVERAGE.md`](docs/OP_COVERAGE.md) | Mouse | **Authoritative op-coverage plan** — 174 ops across 16 families, driven by model families (LLM/Qwen3.5 → int4 → MoE → multimodal → linear attention → conv), with model-level exit criteria per tier. Ratified by Morpheus 2026-07-28. |
| `docs/THIRD_PARTY.md` | Rai | Third-party licence compliance and attribution requirements for adapted code. |
| [`python/README.md`](python/README.md) | Niobe | The consumer-facing Python shim: what it does, why one ORT call needs a package, and how the wheel reconciles with §7.8's no-checked-in-binary rule. |
| [`docs/PERF.md`](docs/PERF.md) | Niobe | Performance record and methodology. **No wall-clock ratio is published here**; every quantity carries a provenance class (SPECIFICATION, MEASUREMENT, or MODEL) and the paired-alternation apparatus that would produce a timing is refuted, not pending. |
| `docs/OP_ARCHITECTURE.md` | Mouse | Op registry design and the per-op claim contract. *(does not exist)* |
| `docs/BENCHMARKS.md` | Niobe | *(does not exist — superseded by `docs/PERF.md`)* |

## Design in one paragraph

This project deliberately mirrors the architecture of
[`onnxruntime-mlx`](https://github.com/justinchuby/onnxruntime-mlx), a proven ONNX Runtime plugin
EP, with Vulkan replacing MLX as the backend. The registry, claim/fuse/compile/run pipeline, module
split, FFI ownership model, and repository layout are the same. The one structural difference that
drives everything else: MLX runs on Apple unified memory, while Vulkan has explicit, non-unified
device memory — so this EP owns an allocator, a data-transfer implementation, weight prepacking,
and a real Vulkan engine layer that MLX supplied for free. Every deliberate divergence is
enumerated in [`docs/DESIGN.md`](docs/DESIGN.md) §12.

## License

MIT — see [`LICENSE`](LICENSE).
