# Vulkan vs CUDA — gap attribution

workload: `prefill_1`  
verdict: **GPU_TIME_MEASURED**

- untraced median: 46.9645 ms
- traced median: 59.3647 ms
- tracer overhead ratio: 1.264x (host phases inflate, device time does not)

## Where one warm inference goes

Anchor: `vulkan.compute_call`

| region | ms | inside |
|---|---:|---|
| traced wall (one `session.run`) | 59.365 | — |
| outside `vulkan.compute_call` (ORT + harness) | 3.257 | traced wall |
| `vulkan.compute_call` (whole Compute callback) | 56.108 | traced wall |
| `vulkan.subgraph` (inside `dispatch_ort`) | 30.509 | `vulkan.compute_call` |
| sibling phases (record+submit+fence_wait) | 25.316 | `vulkan.subgraph` |
| unattributed inside `vulkan.subgraph` | 5.224 | `vulkan.subgraph` |
| **outside `vulkan.subgraph`, inside the callback** | 25.269 | `vulkan.compute_call` |

All terms are traced-run warm-call medians, in milliseconds. Do not subtract these traced-axis terms from `untraced_median_ms`; the traced run is 1.264x the untraced one.

`outside_subgraph`: MEASURED, UNATTRIBUTED. This instrument reports the size of the region and which side of `vulkan.subgraph` it falls on. It does not name its cause.

## A/B experiment: ARM A - counters dump RETAINED inside the timed region

- paired with: `bench/results/_cuda69/profile_prefill_1.json (ARM B, counters first_run_only)`
- single variable: ONNXRUNTIME_EP_VULKAN_BENCH_KEEP_COUNTERS=1. Same release DLL, same workload prefill_1, same machine, runs minutes apart on 2026-08-07.
- what it isolates: outside_subgraph_ms: 25.269 ms here vs 0.056 ms in arm B, while vulkan.subgraph barely moves (30.509 vs 32.627). The counters dump lands after dispatch_ort returns, which is exactly where the outer bracket sees it and the inner one does not.
- device time moved too: 20.136 ms here vs 18.904 ms in arm B - 6.5% apart between two runs of the same build. Cross-run device comparisons carry at least this spread.
- SCOPE: prefill_1 only. Cross-workload deltas are non-uniform and are not pure counter costs, so this must not be extrapolated to other workloads.
- This measures one term. It does not rank the counters dump, command buffer re-recording, or anything else against each other.

## Cross-check: ORT's own measurement of the same callback

| arm | ORT fused-node warm median | that arm's wall median | outside the fused node | counters in timed region |
|---|---:|---:|---:|---|
| traced | 57.118 ms | 59.365 ms | 2.247 ms | `all_runs_INFLATES_TIMING` |
| untraced | 45.908 ms | 46.964 ms | 1.056 ms | `all_runs_INFLATES_TIMING` |

Each row subtracts within **one** process. ORT's profiler shares no code with our tracer, so this is the only line in the report that is not the EP measuring itself.

- anchor `vulkan.compute_call`: **56.108 ms** vs ORT's **57.118 ms** — **-1.77%**
- inner `vulkan.subgraph`: 30.509 ms vs ORT's 57.118 ms — **-46.59%**

## Device time

- GPU kernel time per run: **20.1360 ms** (basis: warm_call_median, 13 warm calls, 4970 timestamped spans across all calls)
- share of untraced wall: **42.9%** — **cross-run**: traced-process device median over untraced-process wall. Not a bound. Two device medians taken on this machine on the same afternoon differ by 9.6% (12.17456 / 13.347296 ms), and the untraced arm emits no device timestamps at all, so this cannot be checked here.
- host-side residual (traced axis): **39.2287 ms** — traced wall median minus traced device median, both from the same process. The untraced-axis version of this number (`host_ms_per_run_residual`) is withdrawn: it subtracted across two runs.

| kernel | device ms (warm-call median) | dispatches/call |
|---|---:|---:|
| `q_gemv_matmul_nbits_f16` | 17.551 | 161 |
| `gqa_f16` | 1.457 | 32 |
| `skip_simplified_layer_norm_f16` | 0.821 | 64 |
| `ew_binary_mul_f16` | 0.183 | 64 |
| `ew_unary_sigmoid_f16` | 0.095 | 32 |
| `simplified_layer_norm_f16` | 0.009 | 1 |
| `gather_f16` | 0.006 | 1 |

## Host phases per Compute call — cold vs steady

Cumulative totals are not shown as a per-run figure: on this EP the first `Compute` uploads the whole weight set and carries almost the entire `cmd_upload` cost, so `total / calls` describes no regime that actually occurs.

| phase | cold call 0 (ms) | warm median (ms) |
|---|---:|---:|
| `fence_wait` | 196.248 | 20.662 |
| `record` | 1360.004 | 4.546 |
| `submit` | 0.356 | 0.143 |
| **sibling total** | 1556.608 | 25.316 |

| nested phase | cold call 0 (ms) | warm median (ms) | inside |
|---|---:|---:|---|
| `cmd_upload` | 1291.584 | 0.153 | `record` |
| `desc_alloc` | 4.994 | 0.786 | `record` |
| `pipeline_lookup` | 20.117 | 0.037 | `record` |
| `record` residual (vkCmd* calls) | 43.309 | 3.593 | `record` |

- GPU device time, cold call 0: 22.722 ms
- GPU device time, warm median: 20.136 ms

## Host phases, cumulative over every call (provenance only)

| phase | ms | calls |
|---|---:|---:|
| `fence_wait` | 478.693 | 14 |
| `record` | 1418.540 | 14 |
| `submit` | 2.196 | 14 |
| **sibling total** | 1899.429 | |

| nested phase | ms | calls | inside |
|---|---:|---:|---|
| `cmd_upload` | 1293.885 | 14 | `record` |
| `desc_alloc` | 15.072 | 4970 | `record` |
| `pipeline_lookup` | 20.649 | 4970 | `record` |
| `record` residual (vkCmd* calls) | 88.934 | | `record` |

## Transfers

- readback: 6.1 MiB over 14 transfer(s)
- upload: 2200.5 MiB over 14 transfer(s)

## Command-buffer reuse: **RERECORDED_EVERY_CALL**

warm calls allocate a median of 355 descriptor sets against 355 on the cold call — the command buffer is rebuilt from scratch on every inference, so per-dispatch host cost is paid every time rather than once.

_This says re-recording happens and that it costs something per dispatch. It does not say it is the largest remaining cost, and nothing measured here ranks it against the other candidates. The counters A/B below moves a term of its own, and the cross-workload deltas are non-uniform, so a ranking would need a per-candidate ablation that has not been run._

_(`ep.path` instants are absent from this EP build, so this is inferred from per-dispatch recording work rather than read from a marker; an absent marker is not evidence of replay.)_
