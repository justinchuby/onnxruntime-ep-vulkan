# Vulkan vs CUDA — gap attribution

workload: `prefill_1`  
verdict: **GPU_TIME_MEASURED**

- untraced median: 71.0281 ms
- traced median: 78.8161 ms
- tracer overhead ratio: 1.110x (host phases inflate, device time does not)

## Where one warm inference goes

Anchor: `vulkan.compute_call`

| region | tier | ms | inside |
|---|---|---:|---|
| traced wall (one `session.run`) | — | 78.816 | — |
| outside `vulkan.compute_call` (ORT + harness) | — | 2.156 | traced wall |
| `vulkan.compute_call` (whole Compute callback — the anchor) | 0 | 76.660 | traced wall |
| `vulkan.bind_check` (the three bind checks) | 1 | 0.129 | `vulkan.compute_call` |
| `vulkan.subgraph` (the dispatch region, inside `dispatch_ort`) | 1 | 38.259 | `vulkan.compute_call` |
| **unattributed inside the callback** | 1 | 36.392 | `vulkan.compute_call` |
| tier-2 phases (record+submit+fence_wait) | 2 | 27.490 | `vulkan.subgraph` |
| unattributed inside `vulkan.subgraph` | 2 | 9.654 | `vulkan.subgraph` |

Each tier sums against its own parent. Rows from different tiers are not additive: tier-2 time is already inside `vulkan.subgraph`.
- tier 1: `compute_call = bind_check + subgraph + unattributed_in_call`
- tier 2: `subgraph = record + submit + fence_wait + unattributed_in_subgraph`

All terms are traced-run warm-call medians, in milliseconds. Do not subtract these traced-axis terms from `untraced_median_ms`; the traced run is 1.110x the untraced one.

`unattributed_in_call_ms`: MEASURED, UNATTRIBUTED. This instrument reports the size of the region and which side of `vulkan.subgraph` it falls on. It does not name its cause, and it is not bind-check time: `bind_check_ms` is measured separately and is reported above.

## Cross-check: ORT's own measurement of the same callback

| arm | ORT fused-node warm median | that arm's wall median | outside the fused node | counters in timed region |
|---|---:|---:|---:|---|
| traced | 77.745 ms | 78.816 ms | 1.071 ms | `all_runs_INFLATES_TIMING` |
| untraced | 65.774 ms | 71.028 ms | 5.254 ms | `all_runs_INFLATES_TIMING` |

Each row subtracts within **one** process. ORT's profiler shares no code with our tracer, so this is the only line in the report that is not the EP measuring itself.

- anchor `vulkan.compute_call`: **76.660 ms** vs ORT's **77.745 ms** — **-1.40%**
- inner `vulkan.subgraph`: 38.259 ms vs ORT's 77.745 ms — **-50.79%**

## Device time

- GPU kernel time per run: **20.1790 ms** (basis: warm_call_median, 13 warm calls, 4970 timestamped spans across all calls)
- share of untraced wall: **28.4%** — **cross-run**: traced-process device median over untraced-process wall. Not a bound. Two device medians taken on this machine on the same afternoon differ by 9.6% (12.17456 / 13.347296 ms), and the untraced arm emits no device timestamps at all, so this cannot be checked here.
- host-side residual (traced axis): **58.6371 ms** — traced wall median minus traced device median, both from the same process. The untraced-axis version of this number (`host_ms_per_run_residual`) is withdrawn: it subtracted across two runs.

| kernel | device ms (warm-call median) | dispatches/call |
|---|---:|---:|
| `q_gemv_matmul_nbits_f16` | 17.579 | 161 |
| `gqa_f16` | 1.471 | 32 |
| `skip_simplified_layer_norm_f16` | 0.822 | 64 |
| `ew_binary_mul_f16` | 0.191 | 64 |
| `ew_unary_sigmoid_f16` | 0.096 | 32 |
| `simplified_layer_norm_f16` | 0.009 | 1 |
| `gather_f16` | 0.006 | 1 |

## Host phases per Compute call — cold vs steady

Cumulative totals are not shown as a per-run figure: on this EP the first `Compute` uploads the whole weight set and carries almost the entire `cmd_upload` cost, so `total / calls` describes no regime that actually occurs.

| phase | cold call 0 (ms) | warm median (ms) |
|---|---:|---:|
| `bind_check` | 0.322 | 0.129 |
| `fence_wait` | 451.415 | 20.907 |
| `record` | 1630.168 | 6.416 |
| `submit` | 0.349 | 0.201 |
| **sibling total** | 2082.254 | 27.623 |

| nested phase | cold call 0 (ms) | warm median (ms) | inside |
|---|---:|---:|---|
| `cmd_upload` | 1562.359 | 0.241 | `record` |
| `desc_alloc` | 4.756 | 0.899 | `record` |
| `pipeline_lookup` | 18.709 | 0.361 | `record` |
| `record` residual (vkCmd* calls) | 44.344 | 4.964 | `record` |

- GPU device time, cold call 0: 20.207 ms
- GPU device time, warm median: 20.179 ms

## Host phases, cumulative over every call (provenance only)

| phase | ms | calls |
|---|---:|---:|
| `bind_check` | 1.851 | 14 |
| `fence_wait` | 757.592 | 14 |
| `record` | 1735.023 | 14 |
| `submit` | 3.622 | 14 |
| **sibling total** | 2498.088 | |

| nested phase | ms | calls | inside |
|---|---:|---:|---|
| `cmd_upload` | 1566.880 | 14 | `record` |
| `desc_alloc` | 22.449 | 4970 | `record` |
| `pipeline_lookup` | 24.668 | 4970 | `record` |
| `record` residual (vkCmd* calls) | 121.026 | | `record` |

## Transfers

- readback: 6.1 MiB over 14 transfer(s)
- upload: 2200.5 MiB over 14 transfer(s)

## Command-buffer reuse: **RERECORDED_EVERY_CALL**

warm calls allocate a median of 355 descriptor sets against 355 on the cold call — the command buffer is rebuilt from scratch on every inference, so per-dispatch host cost is paid every time rather than once.

_This says re-recording happens and that it costs something per dispatch. It does not say it is the largest remaining cost, and nothing measured here ranks it against the other candidates. The counters A/B below moves a term of its own, and the cross-workload deltas are non-uniform, so a ranking would need a per-candidate ablation that has not been run._

_(`ep.path` instants are absent from this EP build, so this is inferred from per-dispatch recording work rather than read from a marker; an absent marker is not evidence of replay.)_
