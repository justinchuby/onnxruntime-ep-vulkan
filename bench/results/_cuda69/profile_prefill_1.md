# Vulkan vs CUDA — gap attribution

workload: `prefill_1`  
verdict: **GPU_TIME_MEASURED**

- untraced median: 30.3160 ms
- traced median: 31.5258 ms
- tracer overhead ratio: 1.040x (host phases inflate, device time does not)
- CUDA median: 15.7097 ms

## Where one warm inference goes

Anchor: `vulkan.compute_call`

| region | tier | ms | inside |
|---|---|---:|---|
| traced wall (one `session.run`) | — | 31.526 | — |
| outside `vulkan.compute_call` (ORT + harness) | — | 0.363 | traced wall |
| `vulkan.compute_call` (whole Compute callback — the anchor) | 0 | 31.163 | traced wall |
| `vulkan.bind_check` (the three bind checks) | 1 | 0.035 | `vulkan.compute_call` |
| `vulkan.subgraph` (the dispatch region, inside `dispatch_ort`) | 1 | 31.092 | `vulkan.compute_call` |
| **unattributed inside the callback** | 1 | 0.016 | `vulkan.compute_call` |
| tier-2 phases (record+submit+fence_wait) | 2 | 25.687 | `vulkan.subgraph` |
| unattributed inside `vulkan.subgraph` | 2 | 5.187 | `vulkan.subgraph` |

Each tier sums against its own parent. Rows from different tiers are not additive: tier-2 time is already inside `vulkan.subgraph`.
- tier 1: `compute_call = bind_check + subgraph + unattributed_in_call`
- tier 2: `subgraph = record + submit + fence_wait + unattributed_in_subgraph`

All terms are traced-run warm-call medians, in milliseconds. Do not subtract these traced-axis terms from `untraced_median_ms`; the traced run is 1.040x the untraced one.

`unattributed_in_call_ms`: MEASURED, UNATTRIBUTED. This instrument reports the size of the region and which side of `vulkan.subgraph` it falls on. It does not name its cause, and it is not bind-check time: `bind_check_ms` is measured separately and is reported above.

## Cross-check: ORT's own measurement of the same callback

| arm | ORT fused-node warm median | that arm's wall median | outside the fused node | counters in timed region |
|---|---:|---:|---:|---|
| traced | 31.859 ms | 31.526 ms | -0.333 ms | `first_run_only` |
| untraced | 30.379 ms | 30.316 ms | -0.063 ms | `first_run_only` |

Each row subtracts within **one** process. ORT's profiler shares no code with our tracer, so this is the only line in the report that is not the EP measuring itself.

- anchor `vulkan.compute_call`: **31.163 ms** vs ORT's **31.859 ms** — **-2.18%**
- inner `vulkan.subgraph`: 31.092 ms vs ORT's 31.859 ms — **-2.41%**

## Device time

- GPU kernel time per run: **19.0020 ms** (basis: warm_call_median, 13 warm calls, 4970 timestamped spans across all calls)
- share of untraced wall: **62.7%** — **cross-run**: traced-process device median over untraced-process wall. Not a bound. Two device medians taken on this machine on the same afternoon differ by 9.6% (12.17456 / 13.347296 ms), and the untraced arm emits no device timestamps at all, so this cannot be checked here.
- host-side residual (traced axis): **12.5238 ms** — traced wall median minus traced device median, both from the same process. The untraced-axis version of this number (`host_ms_per_run_residual`) is withdrawn: it subtracted across two runs.

| kernel | device ms (warm-call median) | dispatches/call |
|---|---:|---:|
| `q_gemv_matmul_nbits_f16` | 16.613 | 161 |
| `gqa_f16` | 1.346 | 32 |
| `skip_simplified_layer_norm_f16` | 0.767 | 64 |
| `ew_binary_mul_f16` | 0.167 | 64 |
| `ew_unary_sigmoid_f16` | 0.084 | 32 |
| `simplified_layer_norm_f16` | 0.009 | 1 |
| `gather_f16` | 0.006 | 1 |

## Host phases per Compute call — cold vs steady

Cumulative totals are not shown as a per-run figure: on this EP the first `Compute` uploads the whole weight set and carries almost the entire `cmd_upload` cost, so `total / calls` describes no regime that actually occurs.

| phase | cold call 0 (ms) | warm median (ms) |
|---|---:|---:|
| `bind_check` | 0.042 | 0.035 |
| `fence_wait` | 193.493 | 20.677 |
| `record` | 1297.414 | 4.947 |
| `submit` | 0.850 | 0.175 |
| **sibling total** | 1491.799 | 25.722 |

| nested phase | cold call 0 (ms) | warm median (ms) | inside |
|---|---:|---:|---|
| `cmd_upload` | 1228.650 | 0.128 | `record` |
| `desc_alloc` | 8.852 | 0.789 | `record` |
| `pipeline_lookup` | 16.068 | 0.356 | `record` |
| `record` residual (vkCmd* calls) | 43.844 | 3.683 | `record` |

- GPU device time, cold call 0: 20.151 ms
- GPU device time, warm median: 19.002 ms

## Host phases, cumulative over every call (provenance only)

| phase | ms | calls |
|---|---:|---:|
| `bind_check` | 0.886 | 14 |
| `fence_wait` | 480.823 | 14 |
| `record` | 1375.193 | 14 |
| `submit` | 3.398 | 14 |
| **sibling total** | 1860.300 | |

| nested phase | ms | calls | inside |
|---|---:|---:|---|
| `cmd_upload` | 1230.862 | 14 | `record` |
| `desc_alloc` | 21.314 | 4970 | `record` |
| `pipeline_lookup` | 21.051 | 4970 | `record` |
| `record` residual (vkCmd* calls) | 101.966 | | `record` |

## Transfers

- readback: 6.1 MiB over 14 transfer(s)
- upload: 2200.5 MiB over 14 transfer(s)

## Command-buffer reuse: **RERECORDED_EVERY_CALL**

warm calls allocate a median of 355 descriptor sets against 355 on the cold call — the command buffer is rebuilt from scratch on every inference, so per-dispatch host cost is paid every time rather than once.

_This says re-recording happens and that it costs something per dispatch. It does not say it is the largest remaining cost, and nothing measured here ranks it against the other candidates. The counters A/B below moves a term of its own, and the cross-workload deltas are non-uniform, so a ranking would need a per-candidate ablation that has not been run._

_(`ep.path` instants are absent from this EP build, so this is inferred from per-dispatch recording work rather than read from a marker; an absent marker is not evidence of replay.)_

## Per-op-type comparison unavailable

The Vulkan EP fuses its claimed island into a single ORT profile node, so ORT's profile has no per-op Vulkan rows to intersect with CUDA's. Per-kernel Vulkan time comes from the GPU-timestamp table above instead.
