# Vulkan EP vs ORT CUDA EP — issue #69

device      : ['NVIDIA RTX A1000'] driver=573.44 cuda_driver=12.8
iters/warmup: 20/5  repeats=1

## Arm admissibility

Fallback shares: `time` gates admissibility (fusion-invariant); `nodes` is the fusion-blind profile-node share; `graph` is unclaimed/probed original graph nodes and exists only for the fusing Vulkan EP.

| workload | arm | verdict | ORT | profile nodes | fb-time | fb-nodes | fb-graph | median ms | RSD% |
|---|---|---|---|---|---|---|---|---|---|
| prefill_1 | vulkan | ADMISSIBLE | 1.28.0 | 9 | 0.5% | 88.9% | 2.2% | 60.5193 | 13.82 |

## Output equivalence (vs CPU EP reference)

Budget is in **ULP at the reference tensor's peak magnitude** — see `EQUIVALENCE`/`ROUNDING_DEPTH_BOUND` in this module for why raw ULP and absolute tolerance were both rejected. `regime` is the precision the arm's provider declares it computes fp32 ops in; the CUDA EP defaults to **TF32** (10 mantissa bits, not 23), which both widens its budget and is part of why it is fast. `top-k` is conditioned on rows whose reference ranking the numerics can resolve; `UNRESOLVED` means the check had no power on this input and abstained rather than passing.

| workload | arm | verdict | regime | max peak-ULP | budget | raw ULP | max abs diff | top-k | resolvable rows |
|---|---|---|---|---|---|---|---|---|---|

## Vulkan vs CUDA

| workload | vulkan med ms | cuda med ms | speedup (vk over cuda) | 95% CI | ORT-version bracket | verdict |
|---|---|---|---|---|---|---|
| prefill_1 | - | - | - | - | - | UNMEASURED |