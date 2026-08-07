# Vulkan vs CUDA — gap attribution

workload: `prefill_1`  
verdict: **GPU_TIME_MEASURED**

- untraced median: 26.2063 ms
- traced median: 34.0796 ms
- tracer overhead ratio: 1.300x (host phases inflate, device time does not)
- CUDA median: 15.5207 ms

## Where one warm inference goes

Anchor: `vulkan.compute_call`

| region | ms | inside |
|---|---:|---|
| traced wall (one `session.run`) | 34.080 | — |
| outside `vulkan.compute_call` (ORT + harness) | 1.398 | traced wall |
| `vulkan.compute_call` (whole Compute callback) | 32.682 | traced wall |
| `vulkan.subgraph` (inside `dispatch_ort`) | 32.627 | `vulkan.compute_call` |
| sibling phases (record+submit+fence_wait) | 25.708 | `vulkan.subgraph` |
| unattributed inside `vulkan.subgraph` | 6.437 | `vulkan.subgraph` |
| **outside `vulkan.subgraph`, inside the callback** | 0.056 | `vulkan.compute_call` |

All terms are traced-run warm-call medians, in milliseconds. Do not subtract these traced-axis terms from `untraced_median_ms`; the traced run is 1.300x the untraced one.

`outside_subgraph`: MEASURED, UNATTRIBUTED. This instrument reports the size of the region and which side of `vulkan.subgraph` it falls on. It does not name its cause.

**Read the table as terms, not as a sum.** these are independently-taken medians, not a partition: sibling_phases_ms + unattributed_in_subgraph_ms = 32.145 ms against subgraph_ms 32.627 ms, a residual of +0.482 ms. The median of a sum is not the sum of the medians unless every warm call splits the same way, so this residual is a property of the statistic and NOT unaccounted-for time. Do not report it as a gap.

## Device time

- GPU kernel time per run: **18.9043 ms** (basis: warm_call_median, 13 warm calls, 4970 timestamped spans across all calls)
- `gpu_share_untraced_bound` and `host_ms_per_run_residual` are **withdrawn**: both combined this traced-process device median with the untraced process's wall clock, which is a subtraction across two runs, not a decomposition of one. Their operands are still published above and in `untraced_median_ms`. No replacement is offered here and nothing was re-measured.

| kernel | device ms (warm-call median) | dispatches/call |
|---|---:|---:|
| `q_gemv_matmul_nbits_f16` | 16.485 | 161 |
| `gqa_f16` | 1.402 | 32 |
| `skip_simplified_layer_norm_f16` | 0.764 | 64 |
| `ew_binary_mul_f16` | 0.167 | 64 |
| `ew_unary_sigmoid_f16` | 0.085 | 32 |
| `simplified_layer_norm_f16` | 0.009 | 1 |
| `gather_f16` | 0.006 | 1 |

## Host phases per Compute call — cold vs steady

Cumulative totals are not shown as a per-run figure: on this EP the first `Compute` uploads the whole weight set and carries almost the entire `cmd_upload` cost, so `total / calls` describes no regime that actually occurs.

| phase | cold call 0 (ms) | warm median (ms) |
|---|---:|---:|
| `fence_wait` | 196.624 | 20.672 |
| `record` | 1230.592 | 4.724 |
| `submit` | 0.302 | 0.222 |
| **sibling total** | 1427.518 | 25.708 |

| nested phase | cold call 0 (ms) | warm median (ms) | inside |
|---|---:|---:|---|
| `cmd_upload` | 1189.755 | 0.164 | `record` |
| `desc_alloc` | 3.691 | 0.754 | `record` |
| `pipeline_lookup` | 12.924 | 0.032 | `record` |
| `record` residual (vkCmd* calls) | 24.222 | 3.673 | `record` |

- GPU device time, cold call 0: 22.707 ms
- GPU device time, warm median: 18.904 ms

## Host phases, cumulative over every call (provenance only)

| phase | ms | calls |
|---|---:|---:|
| `fence_wait` | 488.135 | 14 |
| `record` | 1296.556 | 14 |
| `submit` | 3.155 | 14 |
| **sibling total** | 1787.846 | |

| nested phase | ms | calls | inside |
|---|---:|---:|---|
| `cmd_upload` | 1192.049 | 14 | `record` |
| `desc_alloc` | 13.953 | 4970 | `record` |
| `pipeline_lookup` | 15.127 | 4970 | `record` |
| `record` residual (vkCmd* calls) | 75.427 | | `record` |

## Transfers

- readback: 6.1 MiB over 14 transfer(s)
- upload: 2200.5 MiB over 14 transfer(s)

## Command-buffer reuse: **RERECORDED_EVERY_CALL**

warm calls allocate a median of 355 descriptor sets against 355 on the cold call — the command buffer is rebuilt from scratch on every inference, so per-dispatch host cost is paid every time rather than once.

_(`ep.path` instants are absent from this EP build, so this is inferred from per-dispatch recording work rather than read from a marker; an absent marker is not evidence of replay.)_

## Per-op-type comparison unavailable

The Vulkan EP fuses its claimed island into a single ORT profile node, so ORT's profile has no per-op Vulkan rows to intersect with CUDA's. Per-kernel Vulkan time comes from the GPU-timestamp table above instead.
