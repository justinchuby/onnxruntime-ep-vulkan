# Vulkan EP benchmark harness

> A benchmark without a baseline and a variance number is a rumour.

Owner: Niobe (performance). Layout mirrors `onnxruntime-mlx/bench/` so the two projects stay
recognisably siblings; the differences are the ones a GPU with explicit transfers forces.

**Read `docs/DESIGN.md` §9.1.2 first.** No shader in this repository has ever executed on any
device. Nothing in this directory contains a performance number, and none may be added until the
corresponding op is green in `tests/ops/` on a real GPU (`DESIGN.md` §9.2, last bullet). The
device *facts* below are properties of the hardware read from the driver, not measurements of
our code.

Pin `onnxruntime>=1.28`. This is enforced: see the refusals below.

## Files

| File | What it is |
|---|---|
| `bench.py` | The runner. Vulkan vs the ORT CPU EP on the same machine, same process, same ORT build. |
| `cases.py` | The cases, built from `tests/ops/_models.py` so the benchmark cannot drift from what is tested. |
| `producers.py` | Who built the graph. Makes it impossible to name a case after a model family its producer did not export, and refuses base-vs-PR comparisons across producers. |
| `portability.py` | The §7.2 admission floor (16 KiB shared, 256 invocations). Answers whether a measured configuration is selectable on every device the EP admits — and refuses to blend UMA and discrete transfer models. |
| `devices.py` | Per-device facts (`timestampPeriod`, `timestampValidBits`, UMA vs discrete, shared memory, subgroup size) read from `vulkaninfo`. Decides whether two numbers may be compared at all. |
| `stats.py` | Median + robust spread (MAD, IQR, p05/p95, `rsd`) and the noise gate for comparisons. |
| `environment.py` | Device / driver / OS / CPU / ORT / build-flags record stamped onto every result. |
| `compare.py` | base-vs-PR Markdown report. Flags a delta only when it exceeds the samples' own noise, and refuses cross-device comparisons outright. |
| `transfer_calibration.py` | The byte staircase that replaces `TransferModel`'s placeholder constants with measured ones. Per device, per transfer class. |
| `test_harness.py` | Self-tests for the statistics and the environment record. No GPU, no EP. |
| `test_plausible_but_wrong.py` | Tests named for the *plausible but wrong* reading each one prevents, using this machine's real device values as fixtures. |
| `pinned_bytes.py` | The provenance authority (issue #78). Admits an immutable pin, hashes the bytes on disk, walks the whole ONNX graph for external weights, confines and hashes them, and derives a verdict. Every refusal raises with a named `reason`; nothing here reports a verdict as a field. |
| `path_screen.py` | The public-artifact screen. Refuses to serialise a record that carries a filesystem path from this machine, by shape rather than by allowlist. |

## The two GPUs in this laptop are not interchangeable

```
python bench/devices.py
```

|  | Intel Iris Xe | NVIDIA RTX 4060 Laptop |
|---|---|---|
| `timestampPeriod` | **52.0833 ns/tick** | 1.0 ns/tick |
| `timestampValidBits` | **36** (wraps ~hourly) | 64 |
| max compute shared memory | 32 KiB | 48 KiB |
| transfer class | UMA | discrete (+ a BAR window, which is *not* UMA) |

So: a hardcoded `timestampPeriod = 1.0` under-reports the Xe by 52×; a 48 KiB tile config is not
even selectable on the Xe; and an affine transfer model fitted on one is meaningless on the
other. The harness therefore **requires `--device N`** on this machine and refuses to compare
across devices. Full detail and the reasoning: `docs/PERF.md` §1.4.

The Iris Xe is also the closest thing here to the mobile case — Adreno and Mali are UMA too, and
Intel's Vulkan implementation is the stricter of the two, so it is the more useful device to
find bugs on even though the 4060 is the more useful device to find speed on. Benchmark both.

## Run it

```powershell
# Windows — the Vulkan SDK must be on PATH so `vulkaninfo` can report device facts
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.350.0"
$env:ONNXRUNTIME_VULKAN_EP_LIB = "rust\target\release\onnxruntime_vulkan_ep.dll"

python bench\bench.py --out pr-4060.json --label PR --device 1   # discrete
python bench\bench.py --out pr-xe.json   --label PR --device 0   # UMA / mobile-like
```

```sh
# Linux / macOS
ONNXRUNTIME_VULKAN_EP_LIB=rust/target/release/libonnxruntime_vulkan_ep.so \
  python bench/bench.py --out pr.json --label PR --device 0
```

`--print-env` prints the environment record and the per-device facts, then exits — run it first
on any new machine, and check that the device you think you are benchmarking is the device
listed. Omitting `--device` on a multi-device machine is an error, not a default.

Regression check (same device on both sides — enforced):

```sh
python bench/compare.py --base base-4060.json --pr pr-4060.json
```

Device study (deliberately different devices, no verdict issued):

```sh
python bench/compare.py --base pr-xe.json --pr pr-4060.json --cross-device-study
```

Pin `onnxruntime>=1.28` (Tank + Fact Checker's version policy; 1.27 has the null-allocator
`PrePack` bug that produces NaN/Inf on fp16, and its loader rejects the plugin's API version 28
outright). The harness refuses to produce a Vulkan column under an older ORT rather than
recording a CPU run under our provider's name.

## What the numbers are

`session.run` wall time is **end-to-end host latency**: staging upload, command-buffer record
(first run) or replay, `vkQueueSubmit`, the fence wait, and readback. That is what a user waits
for. It is **not** GPU kernel time, and on an explicit, asynchronous API those are different
numbers — see `docs/PERF.md` §1.

For kernel time, run again with `--gpu-timestamps`. That turns on `VkQueryPool` timestamp
queries and the trace export; the GPU spans land on their own device lane in the Chrome trace.
Timestamp writes perturb the command buffer, so that run's latency must not be quoted as
steady-state latency — the harness labels which run is which and so should you.

## What is reported, and what is refused

Every case carries, per `OP_COVERAGE.md` §7.3:

`island_count` · `largest_island_nodes` · `largest_island_flops` ·
`boundary_bytes_per_inference` · `boundary_time_fraction` · the `declined` histogram keyed by
`deny!` reason.

`largest_island_flops` is the metric of record. `claimed_node_fraction` is a diagnostic and is
explicitly **not** a target: a change that claims more nodes while shrinking the fused region is
a regression.

Refusals built into the harness:

* **No `--device N` on a multi-device machine is a hard error.** The EP's own scoring prefers
  discrete, so an unpinned run silently benchmarks the 4060 while the reader assumes whichever
  device they had in mind.
* **An ORT older than 1.28 produces no Vulkan column at all.** The plugin is built against API
  28; an older loader rejects it and ORT then runs everything on its CPU EP under our provider
  name. See "the 1.70× that wasn't" below.
* **A case the EP did not claim has no Vulkan number.** It ran on the CPU EP under our name.
  `speedup_end_to_end` is `null`, the row is marked ⛔, and `--fail-on-unclaimed` makes it fatal
  in CI. CPU fallback is always numerically correct, which is exactly why a wall-time table can
  hide it.
* **A delta inside the noise is not a regression.** `compare.py` requires a delta to exceed both
  the threshold and twice the worse sample's robust spread. A harness that cries wolf gets
  ignored.
* **Two runs from different devices are not compared.** Not warned about — *refused*, exit 2,
  no table. `--cross-device-study` prints a labelled side-by-side with no verdict instead.
* **A result file that does not name its device cannot be compared to anything.** "We forgot to
  record it" must not degrade into "assume it is the same device".
* **Rows with different tile configs are not compared.** Two unknowns are *unknown*, not equal.
* **A case cannot be named after a model family its producer did not export.** Building a case
  called `qwen3_decoder_layer` out of hand-written ops raises `ProducerProvenanceError` at
  construction, before any timing exists. Op coverage is relative to a producer
  (`OP_COVERAGE.md` §4.18); so is a benchmark artefact. Justin's `mobius` builder emits
  `ai.onnx::Attention`@23 / `RMSNormalization` / `RotaryEmbedding`; the ORT GenAI builder emits
  the `com.microsoft` contrib graph. `MatMulNBits` is the only op they agree on.
* **Two runs built by different producers are not compared.** Refused, exit 2, no table.
  `--cross-producer-study` prints a labelled side-by-side with no verdict. A file with no
  recorded producer is refused for the same reason a file with no recorded device is.
* **A number from a configuration a floor device cannot select is not the EP's number.**
  `DESIGN.md` §7.2 admits devices with **16 KiB** of shared memory and 256 invocations. Both
  GPUs here are 2–3× above that, so "it fits on the smaller local GPU" is not portability
  evidence — the Iris Xe is our UMA proxy for Adreno/Mali, not a proxy for their shared-memory
  budget. Every row carries a `portability` verdict; only `portable` is quotable, and today
  every row is `unknown` because the engine does not report its configuration yet.
* **UMA and discrete transfer models are never blended.** The blended affine constants would
  land plausibly between the two and describe neither part.
* **Nothing is extrapolated.** A number that was not measured is `null`.
* **A model is identified by its bytes, not by its filename** (issue #78). For a `pinned`
  model, `resolve_model` refuses unless *every* one of these holds: the pin is complete and
  correctly typed; the file's SHA-256 and size equal the pinned values; the source state is one
  that carries verified bytes; an independently-produced sidecar names the same digest; the
  graph's external-data scan actually ran; and the number of external files it found equals the
  number the pin declared. The verdict is a **derived property** of those recorded fields, not a
  stored one, so no record can say `provenance_ok: true` while the metadata beside it disagrees.
  Anything short of all of it raises `ModelUnavailable` carrying
  `REFUSED(instrument=<reason>)` — never a benchmark of unidentified bytes, and never a
  silent skip. Nothing imports `onnxruntime` on that path: a provenance check that had already
  loaded the runtime would be deciding whether to trust bytes it handed over already.

### What "verified" costs, and where it stops

* **MiniLM is provenance-only.** It is in `PROVENANCE_ONLY`, not in `MODELS`, so it never
  reaches the timed matrix. This repository has no reviewed latency claim about MiniLM and this
  change does not create one; `probe_real_model_latency.py` iterates `MODELS` and a test locks
  that it does not iterate `ALL_MODELS`.
* **The second witness is independent by construction.** `bench/results/pinned-bytes/` holds
  witnesses produced by a procedure *other than* `pinned_bytes.py` — a digest a module both
  writes and then checks proves nothing. That directory deliberately is not
  `bench/results/rust-model-runner/`, which holds readings taken by `rust/modelrunner` against a
  built EP and is stamped by an artifact frame whose subject is that DLL.
* **MiniLM is deliberately absent from `bench/results/model_provenance.json`.** That file is the
  *download* contract for differential-test subjects consumed by `rust/modelrunner`; an entry
  there claims a subject with a verification lane behind it, and MiniLM has none.
* **The path screen is structural, and says so.** An ONNX node name (`/encoder/layer.0/MatMul`)
  and a POSIX path (`/srv/models/minilm`) are not distinguishable as strings. The screen resolves
  that by *position*: the POSIX-absolute exemption is granted only under declared graph-name
  keys, and drive-letter, UNC, device and home-macro findings are never exempt anywhere. A
  private POSIX path smuggled in under a graph-name key is the known limitation, and it is
  stated rather than papered over.

### The 1.70× that wasn't

The first time this harness ran on real hardware (2026-07-29, RTX 4060, release EP build), the
host ORT was 1.27, which rejects the plugin's API version 28. Every node ran on ORT's CPU EP
while the column was still labelled Vulkan. It collected:

```
matmulnbits_q4_b32_K4096_N4096   vulkan= 1.361 ms    cpu= 2.311 ms      → 1.70x
```

1.70× is above the OQ-12 pass bar, on the OQ-12 anchor case, on a real discrete GPU. It is also
two samples of the same CPU code separated by noise (rsd 38.6% and 62.5%, both flagged). The
claim gate suppressed it; the version gate was added afterwards so the column is not produced at
all. Full write-up: `docs/PERF.md` §5.1.

### And then the 1.45× that wasn't

ORT 1.28 landed later the same day, so the plugin now **loads**: the EP enumerates both devices
and its claim predicates run. Every op still declines — `Add` reports *"is in the op table but not
enabled: its compute shader compiles but has never executed on a device, so claiming it would be a
bet"* — so every "vulkan" column is still the CPU EP. On the RTX 4060:

```
add_fp32_4096x1024               vulkan= 0.858 ms    cpu= 1.247 ms      → 1.45x
```

Same shape, different route in. The claim gate marked it ⛔ NOT CLAIMED with
`speedup_end_to_end: null`. Two manufactured speedups in one day, both caught. The gates are not
theoretical.

## Expected shapes of results (not predictions — reading instructions)

Small elementwise cases will be **slower than the CPU EP**. One `Add` over 1024 floats pays a
submit, a fence wait and two boundary crossings to do work the CPU finishes in microseconds.
That is expected, must be labelled, and must not be hidden (`DESIGN.md` §9.2). The size
staircase exists so the crossover point is measured rather than assumed — and that crossover is
precisely what the MVS policy in `rust/src/ops/partition.rs` needs in order to stop declining
work it should take, or taking work it should decline.

## Calibrating the partition cost model

```sh
python bench/transfer_calibration.py --out calib-4060.json --device 1
python bench/transfer_calibration.py --out calib-xe.json   --device 0
```

Sweeps a doubling byte staircase and fits `fixed_ns + bytes / bytes_per_ns` — the same estimator
as `TransferModel::fit` — then prints a Rust literal stamped with the device, the driver and the
**transfer class** it came from. `--device` is required on this machine: the Iris Xe is UMA (an
"upload" may be a mapping, so `fixed_ns` dominates and `bytes_per_ns` looks enormous — a real
property of the part, not a fast link) while the 4060 crosses PCIe. Applying one model to the
other is worse than the placeholder it replaces.

The constants in `partition.rs` (`SAFETY = 3.0`, `min_nodes = 4`, the 64 KiB floor) are
placeholders set at 3× because the cost model is crude; they are replaced by measurement, per
device, behind review, with the device named in the comment. A fit with R² < 0.9 prints a
warning and should not be pasted anywhere.

## CI

`bench.yml` (Trinity/Link own the workflow files) posts an informational base-vs-PR table. It
does **not** gate: shared-runner timings are noise, and lavapipe is a software rasteriser whose
timings say nothing about a GPU. A flagged regression is a prompt to re-measure locally on real
hardware, not a verdict.
