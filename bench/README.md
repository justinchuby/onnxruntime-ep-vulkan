# Vulkan EP benchmark harness

> A benchmark without a baseline and a variance number is a rumour.

Owner: Niobe (performance). Layout mirrors `onnxruntime-mlx/bench/` so the two projects stay
recognisably siblings; the differences are the ones a GPU with explicit transfers forces.

**Read `docs/DESIGN.md` §9.1.2 first.** No shader in this repository has ever executed on any
device. Nothing in this directory contains a performance number, and none may be added until the
corresponding op is green in `tests/ops/` on a real GPU (`DESIGN.md` §9.2, last bullet).

## Files

| File | What it is |
|---|---|
| `bench.py` | The runner. Vulkan vs the ORT CPU EP on the same machine, same process, same ORT build. |
| `cases.py` | The cases, built from `tests/ops/_models.py` so the benchmark cannot drift from what is tested. |
| `devices.py` | Per-device facts (`timestampPeriod`, `timestampValidBits`, UMA vs discrete, shared memory, subgroup size) read from `vulkaninfo`. Decides whether two numbers may be compared at all. |
| `stats.py` | Median + robust spread (MAD, IQR, p05/p95, `rsd`) and the noise gate for comparisons. |
| `environment.py` | Device / driver / OS / CPU / ORT / build-flags record stamped onto every result. |
| `compare.py` | base-vs-PR Markdown report. Flags a delta only when it exceeds the samples' own noise, and refuses cross-device comparisons outright. |
| `transfer_calibration.py` | The byte staircase that replaces `TransferModel`'s placeholder constants with measured ones. Per device, per transfer class. |
| `test_harness.py` | Self-tests for the statistics and the environment record. No GPU, no EP. |
| `test_plausible_but_wrong.py` | Tests named for the *plausible but wrong* reading each one prevents, using this machine's real device values as fixtures. |

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
`PrePack` bug that produces NaN/Inf on fp16). The harness records the ORT version it ran under,
so a result taken on the wrong one is identifiable after the fact rather than silently wrong.

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
* **Nothing is extrapolated.** A number that was not measured is `null`.

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

## Expected shapes of results (not predictions — reading instructions)

Small elementwise cases will be **slower than the CPU EP**. One `Add` over 1024 floats pays a
submit, a fence wait and two boundary crossings to do work the CPU finishes in microseconds.
That is expected, must be labelled, and must not be hidden (`DESIGN.md` §9.2). The size
staircase exists so the crossover point is measured rather than assumed — and that crossover is
precisely what the MVS policy in `rust/src/ops/partition.rs` needs in order to stop declining
work it should take, or taking work it should decline.

## Calibrating the partition cost model

```sh
python bench/transfer_calibration.py --out calib.json
```

Sweeps a doubling byte staircase and fits `fixed_ns + bytes / bytes_per_ns` — the same estimator
as `TransferModel::fit` — then prints a Rust literal. The constants in `partition.rs`
(`SAFETY = 3.0`, `min_nodes = 4`, the 64 KiB floor) are placeholders set at 3× because the cost
model is crude; they are replaced by measurement, per device, behind review, with the device
named in the comment. Calibrate per device: a UMA integrated GPU and a discrete GPU do not share
an affine model.

## CI

`bench.yml` (Trinity/Link own the workflow files) posts an informational base-vs-PR table. It
does **not** gate: shared-runner timings are noise, and lavapipe is a software rasteriser whose
timings say nothing about a GPU. A flagged regression is a prompt to re-measure locally on real
hardware, not a verdict.
