# Vulkan EP vs ORT CUDA EP — issue #69

device      : ['NVIDIA RTX A1000'] driver=573.44 cuda_driver=12.8
iters/warmup: 8/3  repeats=1

## Arm admissibility

Fallback shares: `time` gates admissibility (fusion-invariant); `nodes` is the fusion-blind profile-node share; `graph` is unclaimed/probed original graph nodes and exists only for the fusing Vulkan EP.

| workload | arm | verdict | ORT | profile nodes | fb-time | fb-nodes | fb-graph | median ms | RSD% |
|---|---|---|---|---|---|---|---|---|---|
| prefill_1 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.4% | 0.9% | - | 15.5994 | 1.54 |
| prefill_1 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.2% | 88.9% | 2.2% | 45.5135 | 7.66 |
| prefill_1 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 104.3611 | 1.72 |
| prefill_1 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 100.4198 | 1.04 |

## Vulkan vs CUDA

| workload | vulkan med ms | cuda med ms | speedup (vk over cuda) | 95% CI | verdict |
|---|---|---|---|---|---|
| prefill_1 | 45.5135 | 15.5994 | 0.343 | [0.327, 0.367] | CUDA_FASTER |