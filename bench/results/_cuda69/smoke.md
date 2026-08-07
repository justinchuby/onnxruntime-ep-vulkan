# Vulkan EP vs ORT CUDA EP — issue #69

device      : ['NVIDIA RTX A1000'] driver=573.44 cuda_driver=12.8
iters/warmup: 8/3  repeats=1

## Arm admissibility

| workload | arm | verdict | ORT | nodes | cpu-fallback | median ms | RSD% |
|---|---|---|---|---|---|---|---|
| mobilenetv2_b1 | cuda | ADMISSIBLE | 1.26.0 | 104 | 2.9% | 3.1860 | 3.61 |
| mobilenetv2_b1 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 61 | 0.0% | 2.6464 | 6.61 |

## Vulkan vs CUDA

| workload | vulkan med ms | cuda med ms | speedup (vk over cuda) | 95% CI | verdict |
|---|---|---|---|---|---|
| mobilenetv2_b1 | - | - | - | - | UNMEASURED |