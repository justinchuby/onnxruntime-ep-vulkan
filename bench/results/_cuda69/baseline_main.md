# Vulkan EP vs ORT CUDA EP — issue #69

device      : ['NVIDIA RTX A1000'] driver=573.44 cuda_driver=12.8
iters/warmup: 20/5  repeats=1

## Arm admissibility

Fallback shares: `time` gates admissibility (fusion-invariant); `nodes` is the fusion-blind profile-node share; `graph` is unclaimed/probed original graph nodes and exists only for the fusing Vulkan EP.

| workload | arm | verdict | ORT | profile nodes | fb-time | fb-nodes | fb-graph | median ms | RSD% |
|---|---|---|---|---|---|---|---|---|---|
| prefill_1 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.3% | 88.9% | 2.2% | 44.6052 | 8.80 |
| prefill_1 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.5% | 0.9% | - | 15.3512 | 2.16 |
| prefill_1 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 103.1126 | 1.62 |
| prefill_1 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 100.8967 | 1.09 |
| prefill_2 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.3% | 88.9% | 2.2% | 55.0329 | 7.46 |
| prefill_2 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.3% | 0.9% | - | 99.8442 | 0.39 |
| prefill_2 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 246.1648 | 6.71 |
| prefill_2 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 256.9395 | 3.75 |
| prefill_4 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.3% | 88.9% | 2.2% | 80.6384 | 8.83 |
| prefill_4 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.3% | 0.9% | - | 100.5115 | 0.57 |
| prefill_4 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 258.6734 | 2.85 |
| prefill_4 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 262.2053 | 4.06 |
| prefill_8 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.2% | 88.9% | 2.2% | 123.2234 | 3.36 |
| prefill_8 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.3% | 0.9% | - | 101.0225 | 0.36 |
| prefill_8 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 313.5016 | 0.79 |
| prefill_8 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 312.2163 | 2.00 |
| prefill_16 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.1% | 88.9% | 2.2% | 224.2485 | 1.83 |
| prefill_16 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.4% | 0.9% | - | 102.3743 | 0.45 |
| prefill_16 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 385.6828 | 2.30 |
| prefill_16 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 379.5397 | 3.64 |
| prefill_32 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.1% | 88.9% | 2.2% | 485.9558 | 1.48 |
| prefill_32 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.3% | 0.9% | - | 103.2343 | 0.19 |
| prefill_32 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 562.8393 | 2.80 |
| prefill_32 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 553.9727 | 1.74 |
| prefill_64 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.1% | 88.9% | 2.2% | 1118.8228 | 0.60 |
| prefill_64 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.3% | 0.9% | - | 107.1766 | 0.29 |
| prefill_64 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 830.5308 | 1.56 |
| prefill_64 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 827.2062 | 1.10 |
| prefill_128 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.0% | 88.9% | 2.2% | 3026.5414 | 0.72 |
| prefill_128 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.3% | 0.9% | - | 120.8462 | 0.35 |
| prefill_128 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 1472.5192 | 1.62 |
| prefill_128 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 1526.3665 | 2.42 |
| decode_past128 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.3% | 88.9% | 2.2% | 108.4679 | 8.27 |
| decode_past128 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.5% | 0.9% | - | 30.8595 | 4.06 |
| decode_past128 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 124.0096 | 3.08 |
| decode_past128 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 131.5649 | 2.78 |
| decode_past512 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.1% | 88.9% | 2.2% | 387.5502 | 5.67 |
| decode_past512 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.5% | 0.9% | - | 66.3688 | 0.97 |
| decode_past512 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 155.8330 | 5.83 |
| decode_past512 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 150.5695 | 5.93 |
| decode_past1024 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.1% | 88.9% | 2.2% | 649.2843 | 4.70 |
| decode_past1024 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.5% | 0.9% | - | 125.1212 | 4.54 |
| decode_past1024 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 204.0578 | 4.67 |
| decode_past1024 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 197.7816 | 2.64 |
| decode_past2048 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.0% | 88.9% | 2.2% | 1251.4938 | 2.53 |
| decode_past2048 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.4% | 0.9% | - | 228.7212 | 3.87 |
| decode_past2048 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 291.9106 | 4.37 |
| decode_past2048 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 283.5706 | 2.64 |
| mobilenetv2_b1 | vulkan | ADMISSIBLE | 1.28.0 | 8 | 1.3% | 87.5% | 6.7% | 26.7310 | 6.29 |
| mobilenetv2_b1 | cuda | ADMISSIBLE | 1.26.0 | 104 | 0.4% | 2.9% | - | 3.2532 | 3.32 |
| mobilenetv2_b1 | cpu_host | ADMISSIBLE | 1.28.0 | 61 | 0.0% | 0.0% | - | 3.2057 | 6.03 |
| mobilenetv2_b1 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 61 | 0.0% | 0.0% | - | 2.7123 | 3.11 |
| minilm_seq16 | vulkan | ADMISSIBLE | 1.28.0 | 207 | 1.3% | 88.4% | 54.1% | 429.8015 | 3.91 |
| minilm_seq16 | cuda | ADMISSIBLE | 1.26.0 | 183 | 0.3% | 1.1% | - | 4.5244 | 2.19 |
| minilm_seq16 | cpu_host | ADMISSIBLE | 1.28.0 | 183 | 0.0% | 0.0% | - | 6.6872 | 2.26 |
| minilm_seq16 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 183 | 0.0% | 0.0% | - | 6.4291 | 7.07 |
| minilm_seq64 | vulkan | ADMISSIBLE | 1.28.0 | 207 | 1.6% | 88.4% | 54.1% | 461.6033 | 5.30 |
| minilm_seq64 | cuda | ADMISSIBLE | 1.26.0 | 183 | 0.3% | 1.1% | - | 4.5586 | 1.85 |
| minilm_seq64 | cpu_host | ADMISSIBLE | 1.28.0 | 183 | 0.0% | 0.0% | - | 9.0408 | 4.50 |
| minilm_seq64 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 183 | 0.0% | 0.0% | - | 8.8061 | 5.79 |
| minilm_seq128 | vulkan | ADMISSIBLE | 1.28.0 | 207 | 1.7% | 88.4% | 54.1% | 502.7959 | 4.20 |
| minilm_seq128 | cuda | ADMISSIBLE | 1.26.0 | 183 | 0.3% | 1.1% | - | 4.4469 | 2.73 |
| minilm_seq128 | cpu_host | ADMISSIBLE | 1.28.0 | 183 | 0.0% | 0.0% | - | 10.0541 | 4.97 |
| minilm_seq128 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 183 | 0.0% | 0.0% | - | 10.0469 | 5.85 |

## Output equivalence (vs CPU EP reference, derived ULP budget)

| workload | arm | verdict | max ULP | budget | max abs diff | top-k agree |
|---|---|---|---|---|---|---|
| prefill_1 | vulkan | DIVERGENT | 17402 | 32 | 0.0390625 | 1.0 |
| prefill_1 | cuda | DIVERGENT | 19102 | 32 | 0.078125 | 1.0 |
| prefill_1 | cpu_cuda_rt | MATCH | 0 | 32 | 0.0 | 1.0 |
| prefill_2 | vulkan | DIVERGENT | 17949 | 32 | 0.0625 | 1.0 |
| prefill_2 | cuda | DIVERGENT | 18630 | 32 | 0.09375 | 1.0 |
| prefill_2 | cpu_cuda_rt | MATCH | 0 | 32 | 0.0 | 1.0 |
| prefill_4 | vulkan | DIVERGENT | 17949 | 32 | 0.0625 | 0.75 |
| prefill_4 | cuda | DIVERGENT | 18630 | 32 | 0.09375 | 0.75 |
| prefill_4 | cpu_cuda_rt | MATCH | 0 | 32 | 0.0 | 1.0 |
| prefill_8 | vulkan | DIVERGENT | 17949 | 32 | 0.0625 | 0.75 |
| prefill_8 | cuda | DIVERGENT | 18630 | 32 | 0.09375 | 0.5 |
| prefill_8 | cpu_cuda_rt | MATCH | 0 | 32 | 0.0 | 1.0 |
| prefill_16 | vulkan | DIVERGENT | 17949 | 32 | 0.0625 | 0.75 |
| prefill_16 | cuda | DIVERGENT | 18630 | 32 | 0.09375 | 0.5625 |
| prefill_16 | cpu_cuda_rt | MATCH | 0 | 32 | 0.0 | 1.0 |
| prefill_32 | vulkan | DIVERGENT | 17949 | 32 | 0.140625 | 0.6875 |
| prefill_32 | cuda | DIVERGENT | 20317 | 32 | 0.15625 | 0.5625 |
| prefill_32 | cpu_cuda_rt | MATCH | 0 | 32 | 0.0 | 1.0 |
| prefill_64 | vulkan | DIVERGENT | 17949 | 32 | 0.140625 | 0.609375 |
| prefill_64 | cuda | DIVERGENT | 20317 | 32 | 0.140625 | 0.5625 |
| prefill_64 | cpu_cuda_rt | MATCH | 0 | 32 | 0.0 | 1.0 |
| prefill_128 | vulkan | DIVERGENT | 17949 | 32 | 0.140625 | 0.546875 |
| prefill_128 | cuda | DIVERGENT | 20028 | 32 | 0.1875 | 0.4765625 |
| prefill_128 | cpu_cuda_rt | MATCH | 0 | 32 | 0.0 | 1.0 |
| decode_past128 | vulkan | DIVERGENT | 25315 | 32 | 0.5625 | 0.0 |
| decode_past128 | cuda | DIVERGENT | 28906 | 32 | 1.765625 | 0.0 |
| decode_past128 | cpu_cuda_rt | MATCH | 0 | 32 | 0.0 | 1.0 |
| decode_past512 | vulkan | DIVERGENT | 996 | 32 | 0.390625 | 0.0 |
| decode_past512 | cuda | DIVERGENT | 25490 | 32 | 0.65625 | 0.0 |
| decode_past512 | cpu_cuda_rt | MATCH | 0 | 32 | 0.0 | 1.0 |
| decode_past1024 | vulkan | DIVERGENT | 25323 | 32 | 0.484375 | 0.0 |
| decode_past1024 | cuda | DIVERGENT | 27494 | 32 | 0.8046875 | 0.0 |
| decode_past1024 | cpu_cuda_rt | MATCH | 0 | 32 | 0.0 | 1.0 |
| decode_past2048 | vulkan | DIVERGENT | 22892 | 32 | 0.265625 | 1.0 |
| decode_past2048 | cuda | DIVERGENT | 27645 | 32 | 0.8359375 | 0.0 |
| decode_past2048 | cpu_cuda_rt | MATCH | 0 | 32 | 0.0 | 1.0 |
| mobilenetv2_b1 | vulkan | DIVERGENT | 10640 | 128 | 1.0967254638671875e-05 | - |
| mobilenetv2_b1 | cuda | DIVERGENT | 8197048 | 128 | 0.008611798286437988 | - |
| mobilenetv2_b1 | cpu_cuda_rt | MATCH | 0 | 128 | 0.0 | - |
| minilm_seq16 | vulkan | DIVERGENT | 87552 | 128 | 2.294778823852539e-06 | - |
| minilm_seq16 | cuda | DIVERGENT | 1900180699 | 128 | 0.0027505159378051758 | - |
| minilm_seq16 | cpu_cuda_rt | DIVERGENT | 86528 | 128 | 4.0531158447265625e-06 | - |
| minilm_seq64 | vulkan | DIVERGENT | 150528 | 128 | 2.7418136596679688e-06 | - |
| minilm_seq64 | cuda | DIVERGENT | 1929382906 | 128 | 0.0018224716186523438 | - |
| minilm_seq64 | cpu_cuda_rt | DIVERGENT | 74752 | 128 | 3.0994415283203125e-06 | - |
| minilm_seq128 | vulkan | DIVERGENT | 319488 | 128 | 3.516674041748047e-06 | - |
| minilm_seq128 | cuda | DIVERGENT | 1932753927 | 128 | 0.0023512840270996094 | - |
| minilm_seq128 | cpu_cuda_rt | DIVERGENT | 260096 | 128 | 3.337860107421875e-06 | - |

## Vulkan vs CUDA

| workload | vulkan med ms | cuda med ms | speedup (vk over cuda) | 95% CI | ORT-version bracket | verdict |
|---|---|---|---|---|---|---|
| prefill_1 | - | - | - | - | - | NOT_EQUIVALENT |
| prefill_2 | - | - | - | - | - | NOT_EQUIVALENT |
| prefill_4 | - | - | - | - | - | NOT_EQUIVALENT |
| prefill_8 | - | - | - | - | - | NOT_EQUIVALENT |
| prefill_16 | - | - | - | - | - | NOT_EQUIVALENT |
| prefill_32 | - | - | - | - | - | NOT_EQUIVALENT |
| prefill_64 | - | - | - | - | - | NOT_EQUIVALENT |
| prefill_128 | - | - | - | - | - | NOT_EQUIVALENT |
| decode_past128 | - | - | - | - | - | NOT_EQUIVALENT |
| decode_past512 | - | - | - | - | - | NOT_EQUIVALENT |
| decode_past1024 | - | - | - | - | - | NOT_EQUIVALENT |
| decode_past2048 | - | - | - | - | - | NOT_EQUIVALENT |
| mobilenetv2_b1 | - | - | - | - | - | NOT_EQUIVALENT |
| minilm_seq16 | - | - | - | - | - | NOT_EQUIVALENT |
| minilm_seq64 | - | - | - | - | - | NOT_EQUIVALENT |
| minilm_seq128 | - | - | - | - | - | NOT_EQUIVALENT |