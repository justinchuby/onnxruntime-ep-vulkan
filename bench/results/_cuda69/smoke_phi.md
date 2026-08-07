# Vulkan EP vs ORT CUDA EP — issue #69

device      : ['NVIDIA RTX A1000'] driver=573.44 cuda_driver=12.8
iters/warmup: 6/2  repeats=1

## Arm admissibility

| workload | arm | verdict | ORT | nodes | cpu-fallback | median ms | RSD% |
|---|---|---|---|---|---|---|---|
| prefill_1 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.9% | 15.4653 | 1.30 |
| prefill_1 | vulkan | SPLIT_FRAME | 1.28.0 | 9 | 88.9% | 59.0449 | 14.04 |

## Vulkan vs CUDA

| workload | vulkan med ms | cuda med ms | speedup (vk over cuda) | 95% CI | verdict |
|---|---|---|---|---|---|
| prefill_1 | - | - | - | - | UNMEASURED |