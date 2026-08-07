# Vulkan EP vs ORT CUDA EP — issue #69

device      : ['NVIDIA RTX A1000'] driver=573.44 cuda_driver=12.8
iters/warmup: 8/3  repeats=1

## Arm admissibility

| workload | arm | verdict | ORT | nodes | cpu-fallback | median ms | RSD% |
|---|---|---|---|---|---|---|---|
| mobilenetv2_b1 | vulkan | SPLIT_FRAME | 1.28.0 | 8 | 87.5% | 26.4927 | 9.76 |
| mobilenetv2_b1 | cpu_host | ADMISSIBLE | 1.28.0 | 61 | 0.0% | 3.4473 | 2.64 |

## Vulkan vs CUDA

| workload | vulkan med ms | cuda med ms | speedup (vk over cuda) | 95% CI | verdict |
|---|---|---|---|---|---|
| mobilenetv2_b1 | - | - | - | - | UNMEASURED |