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

## Output equivalence (vs CPU EP reference)

Budget is in **ULP at the reference tensor's peak magnitude** — see `EQUIVALENCE`/`ROUNDING_DEPTH_BOUND` in this module for why raw ULP and absolute tolerance were both rejected. `regime` is the precision the arm's provider declares it computes fp32 ops in; the CUDA EP defaults to **TF32** (10 mantissa bits, not 23), which both widens its budget and is part of why it is fast. `top-k` is conditioned on rows whose reference ranking the numerics can resolve; `UNRESOLVED` means the check had no power on this input and abstained rather than passing.

| workload | arm | verdict | regime | max peak-ULP | budget | raw ULP | max abs diff | top-k | resolvable rows |
|---|---|---|---|---|---|---|---|---|---|
| prefill_1 | vulkan | MATCH | float32 | 5.00 | 128 | 17402 | 0.0390625 | UNRESOLVED | -/1 |
| prefill_1 | cuda | MATCH | tf32 | 10.00 | 128 | 19102 | 0.078125 | UNRESOLVED | -/1 |
| prefill_1 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/1 |
| prefill_2 | vulkan | MATCH | float32 | 2.00 | 128 | 17949 | 0.0625 | MATCH | -/2 |
| prefill_2 | cuda | MATCH | tf32 | 3.00 | 128 | 18630 | 0.09375 | MATCH | -/2 |
| prefill_2 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/2 |
| prefill_4 | vulkan | MATCH | float32 | 2.00 | 128 | 17949 | 0.0625 | MATCH | -/4 |
| prefill_4 | cuda | MATCH | tf32 | 3.00 | 128 | 18630 | 0.09375 | MATCH | -/4 |
| prefill_4 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/4 |
| prefill_8 | vulkan | MATCH | float32 | 2.00 | 128 | 17949 | 0.0625 | MATCH | -/8 |
| prefill_8 | cuda | MATCH | tf32 | 3.00 | 128 | 18630 | 0.09375 | MATCH | -/8 |
| prefill_8 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/8 |
| prefill_16 | vulkan | MATCH | float32 | 2.00 | 128 | 17949 | 0.0625 | MATCH | -/16 |
| prefill_16 | cuda | MATCH | tf32 | 3.00 | 128 | 18630 | 0.09375 | MATCH | -/16 |
| prefill_16 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/16 |
| prefill_32 | vulkan | MATCH | float32 | 4.50 | 128 | 17949 | 0.140625 | MATCH | -/32 |
| prefill_32 | cuda | MATCH | tf32 | 5.00 | 128 | 20317 | 0.15625 | MATCH | -/32 |
| prefill_32 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/32 |
| prefill_64 | vulkan | MATCH | float32 | 4.50 | 128 | 17949 | 0.140625 | MATCH | -/64 |
| prefill_64 | cuda | MATCH | tf32 | 4.50 | 128 | 20317 | 0.140625 | MATCH | -/64 |
| prefill_64 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/64 |
| prefill_128 | vulkan | MATCH | float32 | 4.50 | 128 | 17949 | 0.140625 | MATCH | -/128 |
| prefill_128 | cuda | MATCH | tf32 | 6.00 | 128 | 20028 | 0.1875 | MATCH | -/128 |
| prefill_128 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/128 |
| decode_past128 | vulkan | MATCH | float32 | 36.00 | 128 | 25315 | 0.5625 | UNRESOLVED | -/1 |
| decode_past128 | cuda | MATCH | tf32 | 113.00 | 128 | 28906 | 1.765625 | UNRESOLVED | -/1 |
| decode_past128 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/1 |
| decode_past512 | vulkan | MATCH | float32 | 12.50 | 128 | 996 | 0.390625 | UNRESOLVED | -/1 |
| decode_past512 | cuda | MATCH | tf32 | 21.00 | 128 | 25490 | 0.65625 | UNRESOLVED | -/1 |
| decode_past512 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/1 |
| decode_past1024 | vulkan | MATCH | float32 | 31.00 | 128 | 25323 | 0.484375 | UNRESOLVED | -/1 |
| decode_past1024 | cuda | MATCH | tf32 | 51.50 | 128 | 27494 | 0.8046875 | UNRESOLVED | -/1 |
| decode_past1024 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/1 |
| decode_past2048 | vulkan | MATCH | float32 | 17.00 | 128 | 22892 | 0.265625 | UNRESOLVED | -/1 |
| decode_past2048 | cuda | MATCH | tf32 | 53.50 | 128 | 27645 | 0.8359375 | UNRESOLVED | -/1 |
| decode_past2048 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/1 |
| mobilenetv2_b1 | vulkan | MATCH | float32 | 23.00 | 128 | 10640 | 1.0967254638671875e-05 | - | -/- |
| mobilenetv2_b1 | cuda | MATCH | tf32 | 18060.25 | 1048576 | 8197048 | 0.008611798286437988 | - | -/- |
| mobilenetv2_b1 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | - | -/- |
| minilm_seq16 | vulkan | MATCH | float32 | 4.81 | 128 | 87552 | 2.294778823852539e-06 | - | -/- |
| minilm_seq16 | cuda | MATCH | tf32 | 5768.25 | 1048576 | 1900180699 | 0.0027505159378051758 | - | -/- |
| minilm_seq16 | cpu_cuda_rt | MATCH | float32 | 8.50 | 128 | 86528 | 4.0531158447265625e-06 | - | -/- |
| minilm_seq64 | vulkan | MATCH | float32 | 5.75 | 128 | 150528 | 2.7418136596679688e-06 | - | -/- |
| minilm_seq64 | cuda | MATCH | tf32 | 3822.00 | 1048576 | 1929382906 | 0.0018224716186523438 | - | -/- |
| minilm_seq64 | cpu_cuda_rt | MATCH | float32 | 6.50 | 128 | 74752 | 3.0994415283203125e-06 | - | -/- |
| minilm_seq128 | vulkan | MATCH | float32 | 7.38 | 128 | 319488 | 3.516674041748047e-06 | - | -/- |
| minilm_seq128 | cuda | MATCH | tf32 | 4931.00 | 1048576 | 1932753927 | 0.0023512840270996094 | - | -/- |
| minilm_seq128 | cpu_cuda_rt | MATCH | float32 | 7.00 | 128 | 260096 | 3.337860107421875e-06 | - | -/- |

## Vulkan vs CUDA

| workload | vulkan med ms | cuda med ms | speedup (vk over cuda) | 95% CI | ORT-version bracket | verdict |
|---|---|---|---|---|---|---|
| prefill_1 | 44.6052 | 15.3512 | 0.344 | [0.328, 0.359] | [0.968, 0.987] | CUDA_FASTER |
| prefill_2 | 55.0329 | 99.8442 | 1.814 | [1.752, 1.906] | [1.017, 1.082] | VULKAN_FASTER |
| prefill_4 | 80.6384 | 100.5115 | 1.246 | [1.197, 1.307] | [0.995, 1.035] | VULKAN_FASTER |
| prefill_8 | 123.2234 | 101.0225 | 0.820 | [0.808, 0.836] | [0.989, 1.004] | CUDA_FASTER |
| prefill_16 | 224.2485 | 102.3743 | 0.457 | [0.452, 0.459] | [0.974, 0.998] | CUDA_FASTER |
| prefill_32 | 485.9558 | 103.2343 | 0.212 | [0.211, 0.215] | [0.965, 1.000] | CUDA_FASTER |
| prefill_64 | 1118.8228 | 107.1766 | 0.096 | [0.095, 0.096] | [0.988, 1.002] | CUDA_FASTER |
| prefill_128 | 3026.5414 | 120.8462 | 0.040 | [0.040, 0.040] | [1.022, 1.047] | CUDA_FASTER |
| decode_past128 | 108.4679 | 30.8595 | 0.285 | [0.264, 0.305] | [1.040, 1.073] | CUDA_FASTER |
| decode_past512 | 387.5502 | 66.3688 | 0.171 | [0.165, 0.176] | [0.926, 1.004] | CUDA_FASTER |
| decode_past1024 | 649.2843 | 125.1212 | 0.193 | [0.187, 0.201] | [0.939, 0.989] | CUDA_FASTER |
| decode_past2048 | 1251.4938 | 228.7212 | 0.183 | [0.178, 0.187] | [0.949, 0.994] | CUDA_FASTER |
| mobilenetv2_b1 | 26.7310 | 3.2532 | 0.122 | [0.118, 0.125] | [0.822, 0.867] | CUDA_FASTER |
| minilm_seq16 | 429.8015 | 4.5244 | 0.011 | [0.010, 0.011] | [0.882, 0.990] | CUDA_FASTER |
| minilm_seq64 | 461.6033 | 4.5586 | 0.010 | [0.010, 0.010] | [0.950, 1.020] | CUDA_FASTER |
| minilm_seq128 | 502.7959 | 4.4469 | 0.009 | [0.009, 0.009] | [0.941, 1.079] | CUDA_FASTER |