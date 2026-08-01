### 2026-08-01: The 13.5x GEMV gap is not a hardware-only prediction
**By:** Fact Checker
**What:** Treat the RTX 4060 Laptop / local Iris Xe theoretical memory-bandwidth ratio as 3.08x
(256 / 83.2 GB/s), not 13.5x. The measured 13.52x leaves a 4.39x residual. Assign Switch a
portability investigation: packed/vector loads, multiple output accumulators, and a
capability-gated subgroup reduction while preserving the shared-tree fallback.
**Why:** The local i7-13800H uses LPDDR5 configured at 5200 MT/s. Batch-1 int4 GEMV is
weight-bandwidth dominated. Shared-memory capacity and workgroup limits do not constrain the
current 1 KiB, 128/256-thread shader; UMA contention and shader structure remain plausible.

### 2026-08-01: Intel device timestamps are valid but noisy under CPU load
**By:** Fact Checker
**What:** Correct the statement that the iGPU device clock is not contention-immune. The Intel
52.0833 ns/tick counter is a 19.2 MHz reference timer; CPU load changes GT work per tick, not the
tick conversion. Keep `NO_STEADY_TAIL`, because the measured workload genuinely did not stabilize.
**Why:** Vulkan defines timestampPeriod as nanoseconds per increment; calibrated timestamps require
monotonicity across power events. Intel's Gen11+ driver derives timestamp frequency from a platform
reference crystal. A changing calibrated counter slope would be ERROR(instrument); observed duration
variation with a stable slope is valid performance noise.

### 2026-08-01: Subgroups are an optional GEMV fast path, not a portability requirement
**By:** Fact Checker
**What:** Keep the subgroup-free shared-tree kernel mandatory. Add optional subgroup-arithmetic and
hybrid variants without assuming width 32; use gl_NumSubgroups and subgroup invocation IDs.
**Why:** llama.cpp builds shared-memory, hybrid, and subgroup-only `mul_mat_vec` variants and selects
the shared fallback when subgroup arithmetic is unavailable. Its larger structural advantage is
packed/vector loads and multiple register accumulators, which can also be used without subgroups.
