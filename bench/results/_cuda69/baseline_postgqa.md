# Vulkan EP vs ORT CUDA EP — issue #69

device      : ['NVIDIA RTX A1000'] driver=573.44 cuda_driver=12.8
iters/warmup: 20/5  repeats=1

## Arm admissibility

Fallback shares: `time` gates admissibility (fusion-invariant); `nodes` is the fusion-blind profile-node share; `graph` is unclaimed/probed original graph nodes and exists only for the fusing Vulkan EP.

| workload | arm | verdict | ORT | profile nodes | fb-time | fb-nodes | fb-graph | median ms | RSD% |
|---|---|---|---|---|---|---|---|---|---|
| prefill_1 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.6% | 88.9% | 2.2% | 31.1925 | 17.12 |
| prefill_1 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.4% | 0.9% | - | 14.9398 | 1.97 |
| prefill_1 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 94.4201 | 6.37 |
| prefill_1 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 91.6546 | 2.29 |
| prefill_2 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.6% | 88.9% | 2.2% | 35.0422 | 11.87 |
| prefill_2 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.3% | 0.9% | - | 99.8697 | 0.24 |
| prefill_2 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 234.4328 | 3.43 |
| prefill_2 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 229.3144 | 3.88 |
| prefill_4 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.6% | 88.9% | 2.2% | 60.0972 | 6.30 |
| prefill_4 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.3% | 0.9% | - | 100.3196 | 0.17 |
| prefill_4 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 243.7095 | 3.90 |
| prefill_4 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 251.5097 | 3.31 |
| prefill_8 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.4% | 88.9% | 2.2% | 98.1496 | 2.47 |
| prefill_8 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.3% | 0.9% | - | 101.8422 | 0.68 |
| prefill_8 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 386.3807 | 3.20 |
| prefill_8 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 365.5069 | 4.67 |
| prefill_16 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.3% | 88.9% | 2.2% | 180.7126 | 2.18 |
| prefill_16 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.3% | 0.9% | - | 103.7508 | 0.28 |
| prefill_16 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 515.9519 | 9.08 |
| prefill_16 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 480.9844 | 7.86 |
| prefill_32 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.2% | 88.9% | 2.2% | 358.8358 | 2.35 |
| prefill_32 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.3% | 0.9% | - | 105.1090 | 0.48 |
| prefill_32 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 609.7335 | 2.94 |
| prefill_32 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 610.9087 | 3.69 |
| prefill_64 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.1% | 88.9% | 2.2% | 705.6162 | 1.01 |
| prefill_64 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.3% | 0.9% | - | 109.2432 | 1.06 |
| prefill_64 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 958.8918 | 2.51 |
| prefill_64 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 939.0857 | 2.23 |
| prefill_128 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.1% | 88.9% | 2.2% | 1435.2203 | 0.68 |
| prefill_128 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.2% | 0.9% | - | 126.0233 | 0.41 |
| prefill_128 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 1620.8005 | 2.72 |
| prefill_128 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 1606.9887 | 1.56 |
| decode_past128 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.4% | 88.9% | 2.2% | 90.6427 | 7.59 |
| decode_past128 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.4% | 0.9% | - | 34.4468 | 1.86 |
| decode_past128 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 118.1033 | 2.47 |
| decode_past128 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 116.4235 | 1.54 |
| decode_past512 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.1% | 88.9% | 2.2% | 409.3482 | 4.78 |
| decode_past512 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.4% | 0.9% | - | 65.1779 | 1.62 |
| decode_past512 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 147.7788 | 3.42 |
| decode_past512 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 145.3538 | 1.88 |
| decode_past1024 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.1% | 88.9% | 2.2% | 739.8482 | 4.25 |
| decode_past1024 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.5% | 0.9% | - | 112.2497 | 5.13 |
| decode_past1024 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 192.9178 | 1.38 |
| decode_past1024 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 189.5845 | 2.67 |
| decode_past2048 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.1% | 88.9% | 2.2% | 1354.8363 | 2.21 |
| decode_past2048 | cuda | ADMISSIBLE | 1.26.0 | 331 | 0.5% | 0.9% | - | 205.8484 | 1.04 |
| decode_past2048 | cpu_host | ADMISSIBLE | 1.28.0 | 459 | 0.0% | 0.0% | - | 278.7425 | 3.41 |
| decode_past2048 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 459 | 0.0% | 0.0% | - | 281.1868 | 3.71 |
| mobilenetv2_b1 | vulkan | ADMISSIBLE | 1.28.0 | 8 | 1.9% | 87.5% | 6.7% | 8.8867 | 8.32 |
| mobilenetv2_b1 | cuda | ADMISSIBLE | 1.26.0 | 104 | 0.4% | 2.9% | - | 3.2677 | 3.31 |
| mobilenetv2_b1 | cpu_host | ADMISSIBLE | 1.28.0 | 61 | 0.0% | 0.0% | - | 2.6820 | 4.78 |
| mobilenetv2_b1 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 61 | 0.0% | 0.0% | - | 2.3369 | 8.19 |
| minilm_seq16 | vulkan | SPLIT_FRAME | 1.28.0 | 207 | 13.5% | 88.4% | 54.1% | 11.2953 | 3.56 |
| minilm_seq16 | cuda | ADMISSIBLE | 1.26.0 | 183 | 0.4% | 1.1% | - | 4.5859 | 13.98 |
| minilm_seq16 | cpu_host | ADMISSIBLE | 1.28.0 | 183 | 0.0% | 0.0% | - | 4.9239 | 4.81 |
| minilm_seq16 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 183 | 0.0% | 0.0% | - | 4.5829 | 15.93 |
| minilm_seq64 | vulkan | SPLIT_FRAME | 1.28.0 | 207 | 15.4% | 88.4% | 54.1% | 16.7971 | 4.55 |
| minilm_seq64 | cuda | ADMISSIBLE | 1.26.0 | 183 | 0.3% | 1.1% | - | 4.5826 | 3.77 |
| minilm_seq64 | cpu_host | ADMISSIBLE | 1.28.0 | 183 | 0.0% | 0.0% | - | 6.9797 | 6.34 |
| minilm_seq64 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 183 | 0.0% | 0.0% | - | 6.4483 | 9.97 |
| minilm_seq128 | vulkan | SPLIT_FRAME | 1.28.0 | 207 | 14.8% | 88.4% | 54.1% | 22.3687 | 3.24 |
| minilm_seq128 | cuda | ADMISSIBLE | 1.26.0 | 183 | 0.4% | 1.1% | - | 4.3127 | 59.74 |
| minilm_seq128 | cpu_host | ADMISSIBLE | 1.28.0 | 183 | 0.0% | 0.0% | - | 9.4050 | 6.57 |
| minilm_seq128 | cpu_cuda_rt | ADMISSIBLE | 1.26.0 | 183 | 0.0% | 0.0% | - | 8.6902 | 7.01 |

## Output equivalence (vs CPU EP reference)

Budget is in **ULP at the reference tensor's peak magnitude** — see `EQUIVALENCE`/`ROUNDING_DEPTH_BOUND` in this module for why raw ULP and absolute tolerance were both rejected. `regime` is the precision the arm's provider declares it computes fp32 ops in; the CUDA EP defaults to **TF32** (10 mantissa bits, not 23), which both widens its budget and is part of why it is fast. `top-k` is conditioned on rows whose reference ranking the numerics can resolve; `UNRESOLVED` means the check had no power on this input and abstained rather than passing.

| workload | arm | verdict | regime | max peak-ULP | budget | raw ULP | max abs diff | top-k | resolvable rows |
|---|---|---|---|---|---|---|---|---|---|
| prefill_1 | vulkan | MATCH | float32 | 3.00 | 128 | 15644 | 0.0234375 | MATCH | -/1 |
| prefill_1 | cuda | MATCH | tf32 | 16.75 | 128 | 19946 | 0.130859375 | UNRESOLVED | -/1 |
| prefill_1 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/1 |
| prefill_2 | vulkan | MATCH | float32 | 2.00 | 128 | 15678 | 0.0625 | MATCH | -/2 |
| prefill_2 | cuda | MATCH | tf32 | 2.71 | 128 | 20300 | 0.084716796875 | MATCH | -/2 |
| prefill_2 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/2 |
| prefill_4 | vulkan | MATCH | float32 | 3.00 | 128 | 15678 | 0.09375 | MATCH | -/4 |
| prefill_4 | cuda | MATCH | tf32 | 4.00 | 128 | 20300 | 0.125 | MATCH | -/4 |
| prefill_4 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/4 |
| prefill_8 | vulkan | MATCH | float32 | 3.00 | 128 | 15678 | 0.09375 | MATCH | -/8 |
| prefill_8 | cuda | MATCH | tf32 | 5.00 | 128 | 20300 | 0.15625 | MATCH | -/8 |
| prefill_8 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/8 |
| prefill_16 | vulkan | MATCH | float32 | 3.50 | 128 | 20141 | 0.109375 | MATCH | -/16 |
| prefill_16 | cuda | MATCH | tf32 | 5.00 | 128 | 21368 | 0.15625 | MATCH | -/16 |
| prefill_16 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/16 |
| prefill_32 | vulkan | MATCH | float32 | 3.50 | 128 | 20141 | 0.109375 | MATCH | -/32 |
| prefill_32 | cuda | MATCH | tf32 | 9.12 | 128 | 19782 | 0.28515625 | MATCH | -/32 |
| prefill_32 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/32 |
| prefill_64 | vulkan | MATCH | float32 | 3.50 | 128 | 20141 | 0.109375 | MATCH | -/64 |
| prefill_64 | cuda | MATCH | tf32 | 9.12 | 128 | 19782 | 0.28515625 | MATCH | -/64 |
| prefill_64 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/64 |
| prefill_128 | vulkan | MATCH | float32 | 3.50 | 128 | 20141 | 0.109375 | MATCH | -/128 |
| prefill_128 | cuda | MATCH | tf32 | 9.25 | 128 | 23141 | 0.2890625 | MATCH | -/128 |
| prefill_128 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/128 |
| decode_past128 | vulkan | MATCH | float32 | 8.12 | 128 | 21367 | 0.126953125 | UNRESOLVED | -/1 |
| decode_past128 | cuda | MATCH | tf32 | 10.25 | 128 | 22395 | 0.16015625 | UNRESOLVED | -/1 |
| decode_past128 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/1 |
| decode_past512 | vulkan | MATCH | float32 | 18.50 | 128 | 24654 | 0.578125 | MATCH | -/1 |
| decode_past512 | cuda | MATCH | tf32 | 31.75 | 128 | 1703 | 0.9921875 | UNRESOLVED | -/1 |
| decode_past512 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/1 |
| decode_past1024 | vulkan | MATCH | float32 | 37.00 | 128 | 24624 | 0.578125 | MATCH | -/1 |
| decode_past1024 | cuda | MATCH | tf32 | 106.50 | 128 | 28893 | 1.6640625 | UNRESOLVED | -/1 |
| decode_past1024 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/1 |
| decode_past2048 | vulkan | MATCH | float32 | 36.00 | 128 | 25060 | 0.5625 | MATCH | -/1 |
| decode_past2048 | cuda | MATCH | tf32 | 86.00 | 128 | 27980 | 1.34375 | UNRESOLVED | -/1 |
| decode_past2048 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | MATCH | -/1 |
| mobilenetv2_b1 | vulkan | MATCH | float32 | 20.00 | 128 | 5876 | 9.5367431640625e-06 | - | -/- |
| mobilenetv2_b1 | cuda | MATCH | tf32 | 24912.00 | 1048576 | 12307168 | 0.01187896728515625 | - | -/- |
| mobilenetv2_b1 | cpu_cuda_rt | MATCH | float32 | 0.00 | 128 | 0 | 0.0 | - | -/- |
| minilm_seq16 | vulkan | MATCH | float32 | 5.75 | 128 | 91520 | 2.7418136596679688e-06 | - | -/- |
| minilm_seq16 | cuda | MATCH | tf32 | 4052.00 | 1048576 | 1907369518 | 0.0019321441650390625 | - | -/- |
| minilm_seq16 | cpu_cuda_rt | MATCH | float32 | 9.12 | 128 | 31360 | 4.351139068603516e-06 | - | -/- |
| minilm_seq64 | vulkan | MATCH | float32 | 6.75 | 128 | 323584 | 3.2186508178710938e-06 | - | -/- |
| minilm_seq64 | cuda | MATCH | tf32 | 4201.25 | 1048576 | 1916461764 | 0.002003312110900879 | - | -/- |
| minilm_seq64 | cpu_cuda_rt | MATCH | float32 | 5.50 | 128 | 183200 | 2.6226043701171875e-06 | - | -/- |
| minilm_seq128 | vulkan | MATCH | float32 | 8.00 | 128 | 2113536 | 3.814697265625e-06 | - | -/- |
| minilm_seq128 | cuda | MATCH | tf32 | 5002.44 | 1048576 | 1913981017 | 0.002385348081588745 | - | -/- |
| minilm_seq128 | cpu_cuda_rt | MATCH | float32 | 7.25 | 128 | 1335296 | 3.4570693969726562e-06 | - | -/- |

## Vulkan vs CUDA

| workload | vulkan med ms | cuda med ms | speedup (vk over cuda) | 95% CI | ORT-version bracket | verdict |
|---|---|---|---|---|---|---|
| prefill_1 | 31.1925 | 14.9398 | 0.479 | [0.426, 0.548] | [0.936, 0.996] | CUDA_FASTER |
| prefill_2 | 35.0422 | 99.8697 | 2.850 | [2.671, 3.034] | [0.955, 1.009] | VULKAN_FASTER |
| prefill_4 | 60.0972 | 100.3196 | 1.669 | [1.653, 1.740] | [1.008, 1.058] | VULKAN_FASTER |
| prefill_8 | 98.1496 | 101.8422 | 1.038 | [1.020, 1.049] | [0.918, 0.985] | VULKAN_FASTER |
| prefill_16 | 180.7126 | 103.7508 | 0.574 | [0.566, 0.579] | [0.852, 1.103] | CUDA_FASTER |
| prefill_32 | 358.8358 | 105.1090 | 0.293 | [0.286, 0.294] | [0.968, 1.017] | CUDA_FASTER |
| prefill_64 | 705.6162 | 109.2432 | 0.155 | [0.154, 0.156] | [0.962, 0.999] | CUDA_FASTER |
| prefill_128 | 1435.2203 | 126.0233 | 0.088 | [0.087, 0.088] | [0.977, 1.010] | CUDA_FASTER |
| decode_past128 | 90.6427 | 34.4468 | 0.380 | [0.367, 0.394] | [0.969, 0.995] | CUDA_FASTER |
| decode_past512 | 409.3482 | 65.1779 | 0.159 | [0.153, 0.162] | [0.959, 1.009] | CUDA_FASTER |
| decode_past1024 | 739.8482 | 112.2497 | 0.152 | [0.150, 0.158] | [0.976, 0.995] | CUDA_FASTER |
| decode_past2048 | 1354.8363 | 205.8484 | 0.152 | [0.150, 0.155] | [0.979, 1.041] | CUDA_FASTER |
| mobilenetv2_b1 | 8.8867 | 3.2677 | 0.368 | [0.338, 0.381] | [0.806, 0.945] | CUDA_FASTER |
| minilm_seq16 | - | - | - | - | - | UNMEASURED |
| minilm_seq64 | - | - | - | - | - | UNMEASURED |
| minilm_seq128 | - | - | - | - | - | UNMEASURED |