# Vulkan vs CUDA — gap attribution

workload: `prefill_1`  
verdict: **GPU_TIME_MEASURED**

- untraced median: 28.0215 ms
- traced median: 28.4271 ms
- tracer overhead ratio: 1.014x (host phases inflate, device time does not)
- CUDA median: 15.0470 ms

## Where one warm inference goes

Anchor: `vulkan.compute_call`

| region | ms | inside |
|---|---:|---|
| traced wall (one `session.run`) | 28.427 | — |
| outside `vulkan.compute_call` (ORT + harness) | 1.017 | traced wall |
| `vulkan.compute_call` (whole Compute callback) | 27.410 | traced wall |
| `vulkan.subgraph` (inside `dispatch_ort`) | 27.377 | `vulkan.compute_call` |
| sibling phases (record+submit+fence_wait) | 24.163 | `vulkan.subgraph` |
| unattributed inside `vulkan.subgraph` | 3.345 | `vulkan.subgraph` |
| **outside `vulkan.subgraph`, inside the callback** | 0.036 | `vulkan.compute_call` |

All terms are traced-run warm-call medians, in milliseconds. Do not subtract these traced-axis terms from `untraced_median_ms`; the traced run is 1.014x the untraced one.

`outside_subgraph`: MEASURED, UNATTRIBUTED. This instrument reports the size of the region and which side of `vulkan.subgraph` it falls on. It does not name its cause.

**Read the table as terms, not as a sum.** these are independently-taken medians, not a partition: sibling_phases_ms + unattributed_in_subgraph_ms = 27.508 ms against subgraph_ms 27.377 ms, a residual of -0.131 ms. The median of a sum is not the sum of the medians unless every warm call splits the same way, so this residual is a property of the statistic and NOT unaccounted-for time. Do not report it as a gap.

## Device time

- GPU kernel time per run: **19.1730 ms** (basis: warm_call_median, 25 warm calls, 9230 timestamped spans across all calls)
- `gpu_share_untraced_bound` and `host_ms_per_run_residual` are **withdrawn**: both combined this traced-process device median with the untraced process's wall clock, which is a subtraction across two runs, not a decomposition of one. Their operands are still published above and in `untraced_median_ms`. No replacement is offered here and nothing was re-measured.

| kernel | device ms (warm-call median) | dispatches/call |
|---|---:|---:|
| `q_gemv_matmul_nbits_f16` | 16.618 | 161 |
| `gqa_f16` | 1.464 | 32 |
| `skip_simplified_layer_norm_f16` | 0.801 | 64 |
| `ew_binary_mul_f16` | 0.178 | 64 |
| `ew_unary_sigmoid_f16` | 0.089 | 32 |
| `simplified_layer_norm_f16` | 0.010 | 1 |
| `gather_f16` | 0.006 | 1 |

## Host phases per Compute call — cold vs steady

Cumulative totals are not shown as a per-run figure: on this EP the first `Compute` uploads the whole weight set and carries almost the entire `cmd_upload` cost, so `total / calls` describes no regime that actually occurs.

| phase | cold call 0 (ms) | warm median (ms) |
|---|---:|---:|
| `fence_wait` | 197.822 | 20.609 |
| `record` | 1004.338 | 3.286 |
| `submit` | 0.176 | 0.158 |
| **sibling total** | 1202.336 | 24.163 |

| nested phase | cold call 0 (ms) | warm median (ms) | inside |
|---|---:|---:|---|
| `cmd_upload` | 966.812 | 0.132 | `record` |
| `desc_alloc` | 3.567 | 0.442 | `record` |
| `pipeline_lookup` | 11.310 | 0.007 | `record` |
| `record` residual (vkCmd* calls) | 22.649 | 2.710 | `record` |

- GPU device time, cold call 0: 22.491 ms
- GPU device time, warm median: 19.173 ms

## Host phases, cumulative over every call (provenance only)

| phase | ms | calls |
|---|---:|---:|
| `fence_wait` | 729.793 | 26 |
| `record` | 1097.339 | 26 |
| `submit` | 4.770 | 26 |
| **sibling total** | 1831.902 | |

| nested phase | ms | calls | inside |
|---|---:|---:|---|
| `cmd_upload` | 970.810 | 26 | `record` |
| `desc_alloc` | 17.083 | 9230 | `record` |
| `pipeline_lookup` | 12.807 | 9230 | `record` |
| `record` residual (vkCmd* calls) | 96.639 | | `record` |

## Transfers

- readback: 11.3 MiB over 26 transfer(s)
- upload: 2214.1 MiB over 26 transfer(s)

## Command-buffer reuse: **RERECORDED_EVERY_CALL**

warm calls allocate a median of 355 descriptor sets against 355 on the cold call — the command buffer is rebuilt from scratch on every inference, so per-dispatch host cost is paid every time rather than once.

_(`ep.path` instants are absent from this EP build, so this is inferred from per-dispatch recording work rather than read from a marker; an absent marker is not evidence of replay.)_

## Per-op-type comparison unavailable

The Vulkan EP fuses its claimed island into a single ORT profile node, so ORT's profile has no per-op Vulkan rows to intersect with CUDA's. Per-kernel Vulkan time comes from the GPU-timestamp table above instead.
