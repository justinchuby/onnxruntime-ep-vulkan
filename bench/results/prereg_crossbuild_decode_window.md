# Pre-registration — issue #96, cross-build decode KV-length sweep

**Registered by:** Switch (Vulkan Compute Engineer), independently of PR #95's author.
**Registered at:** 2026-08-08 (local), before the first timed iteration of this instrument.
**Instrument:** `bench/results/probe_crossbuild_decode_window.py`
**Artifact it will produce:** `bench/results/crossbuild_decode_window.json`

This file is hashed and its digest is embedded in the artifact. Nothing below may be changed
after the first timed iteration; if it is, the artifact's digest no longer matches this file and
the run is void.

## 1. The question

Issue #96 reports that Phi-3.5 decode at `past = 128` is ~14% slower on `85fbda2` (the GQA
workgroup-size change, PR #72) than on its parent `c96e7d9`, in all three whole-process repeats,
while `past = 1024` and prefill `M = 1` show nothing. Three mechanisms are proposed there and
none is tested. This instrument answers one question only:

> **Is there a KV-length window in which the compiled `85fbda2` library is slower at Phi-3.5
> decode than the compiled `c96e7d9` library, and if so, where are its edges?**

It does **not** attribute a mechanism. Attribution is a separate pass (§7) and a separate claim.

## 2. Arms

| arm | tree | what it is |
|---|---|---|
| `candidate` | `85fbda29a92e0e99c3895be8b13664d4ee670c50` | the GQA local-size change, as landed |
| `baseline` | `c96e7d94ff706d26ee6a1bd9bb084c0ade426820` | the single parent of `0cfa362` (PR #72) |

Both trees are built from clean detached worktrees with `cargo build --release`, each with its own
`target/`. The two `.dll` files' sha256 and byte lengths are recorded in the artifact. A record
whose `ep_library_sha256` does not match its arm's declared library digest is refused.

Ancestry is verified in this run, not quoted: `git log -1 --format=%P 0cfa362` must report exactly
`c96e7d9…`, and `git log --oneline c96e7d9..85fbda2` must report exactly two commits.

## 3. Workloads (fixed here, in this order)

Measured in this order within each repeat; the two arms of a workload are always adjacent in time.

| # | workload | GQA present? | `gqa_local_size` predicts | role |
|---|---|---|---|---|
| 1 | MobileNetV2 `N=1` | no | — | drift control (no-treatment) |
| 2 | MobileNetV2 `N=16` | no | — | drift control (no-treatment) |
| 3 | Phi-3.5 prefill `M=1`, `past=0` | yes | local 1, 32 groups | treatment, empty cache |
| 4 | Phi-3.5 decode `past=32` | yes | local 1, 32 groups | treatment |
| 5 | Phi-3.5 decode `past=64` | yes | local 1, 32 groups | treatment |
| 6 | Phi-3.5 decode `past=128` | yes | local 1, 32 groups | treatment (the reported point) |
| 7 | Phi-3.5 decode `past=256` | yes | local 1, 32 groups | treatment |
| 8 | Phi-3.5 decode `past=512` | yes | local 1, 32 groups | treatment |
| 9 | Phi-3.5 decode `past=1024` | yes | local 1, 32 groups | treatment |

`past = 2048` is **excluded, and excluded here rather than after seeing a result**: one feed dict
at 2048 is 32 layers x 2 x [1, 32, 2048, 96] fp16 = 805 MB, which changes the host-side transfer
regime the sweep is trying to hold fixed, and the window question is already bracketed by 512 and
1024. If the sweep's own runtime turns out to be well under an hour it is still not added,
because adding a point after seeing the others is the thing this document exists to prevent.

MobileNetV2 contains no `GroupQueryAttention` node, so the landed change **cannot** touch it. Its
spread is drift and nothing else. That is what makes it the band's source (§5).

## 4. Per-record protocol

One OS process per `(workload, arm, repeat)`. ORT registers an EP library process-globally, so a
process is the only unit in which "which build produced this number" has an unambiguous answer.

* **3 whole-process repeats**, fixed here. Nothing is re-run to move a number.
* **Arm order flips per repeat**: `repeat % 2 == 0` runs candidate first, else baseline first.
* **5 warmups discarded, 20 timed iterations**, on the same session object.
* **Equivalence, per record**: the process computes its own CPU-EP reference and classifies the
  Vulkan outputs against it (`real_model.classify_outputs`). The outputs digest is taken again
  *after* the timed pass and must be unchanged.
* **Path witness, per record**: `pipeline_variants` from the EP's own counters file, written at
  pipeline-creation time from the resolved specialisation vector. Candidate must show
  `gqa_f16:1`; baseline must show `gqa_f16:` (no constant). On the no-GQA controls neither arm
  may show any `gqa_f16` key at all.
* **Identity, per record**: EP `.dll` sha256, model sha256 + bytes + resolver, device name from
  the EP's own `running_device_names`, pid, and non-overlapping process spans.
* **Environment**: every inherited `ONNXRUNTIME_EP_VULKAN_*` variable is stripped from the child
  environment except the counters-file path the driver itself sets. Both arms therefore run
  production defaults; no tuning variable is set on either arm.

## 5. Admissibility, and the band — both fixed before the first timed iteration

A record is **refused** — and a refused record carries **no** `speed` key, structurally — if any
of these fails, re-derived from the written record by a pure function:

1. equivalence verdict missing, or not `MATCH`;
2. outputs digest missing/empty, or changed across the timed pass;
3. model digest missing, or disagreeing with the recorded provenance pin;
4. no path witness (the EP wrote no counters file), or `compute_failures > 0`, or
   `dispatches_executed == 0`;
5. the arm's `ep_library_sha256` not equal to that arm's declared library digest;
6. the observed `gqa_f16` witness key not equal to the key its arm and workload require.

A **pair** (one workload, one repeat, two arms) is refused if either record is refused, or if the
two arms report the **same** `gqa_f16` witness key on a GQA workload — two arms that cannot be
told apart by the production path are not two arms.

**The band is a rule, not a number, and the rule is fixed here:**

> `band = max(0.05, H)` where `H` is the largest per-workload half-range
> `(max(ratios) - min(ratios)) / 2` over the **no-GQA drift controls** (workloads 1 and 2).

`0.05` is the floor. `H` is measured from workloads that provably cannot have been affected by the
change, in this same run, on this same box.

**Ratio convention:** `ratio = baseline_median_ms / candidate_median_ms`, per repeat, arms paired
within a repeat. `> 1` means the candidate is faster. This is PR #95's convention, kept so the
two runs are directly comparable.

**Verdicts**, applied per workload to its per-repeat ratios:

* `FASTER` — **every** repeat's ratio `> 1 + band`.
* `SLOWER` — **every** repeat's ratio `< 1 - band`.
* `INDETERMINATE` — the median is outside the band but the repeats disagree in sign.
* `NEUTRAL` — median inside the band and no repeat outside `[1 - 2·band, 1 + 2·band]`.
* `REFUSED` — fewer than 3 paired repeats survive §5, or an arm-identity check failed.

A `SLOWER` or `FASTER` verdict is additionally reported as `INSIDE-DRIFT` if its worst repeat lies
inside the no-GQA controls' own observed per-repeat ratio envelope `[min, max]`. That is a
disclosure, not an override: the verdict stands as the rule produced it.

## 6. Window claim — what would count

The issue's hypothesis 3 is that the regression occupies a KV-length window. This run may claim a
window only if **both**:

* at least one interior length is `SLOWER` by §5, and
* at least one length on each side of it (or the sweep's end) is not `SLOWER`,

and the claim names the exact measured edges. A single slow point with everything else neutral is
reported as a **single slow point**, not as a window with interpolated edges.

## 7. Attribution pass (separate claim, separate artifact section)

At `past ∈ {64, 128, 256}`, both arms, 3 repeats, one process each, with the EP's **pre-existing**
tracer enabled (`ONNXRUNTIME_EP_VULKAN_TRACE`, `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1`). This is the
same instrument `docs/PERF.md` §26.4 used and it is present, unchanged, in both trees.

Reported per inference: `vulkan.gpu.*` device spans (per kernel, from `VkQueryPool` timestamps),
and the host phases `compile`, `prepack`, `record`, `desc_alloc`, `pipeline_lookup`, `cmd_upload`,
`upload`, `submit`, `fence_wait`, `readback`. Spans carrying `nested_in` are excluded from the
phase sum so a child's cost is not attributed twice.

**Tracing changes absolute latency.** Nothing from this pass may be quoted as a wall-clock result;
it is only ever used to say *where* a difference sits, and only after §5 has said whether there is
one.

No instrumentation from PR #94 (unmerged) is used anywhere in this run.

## 8. Exclusivity

The run holds the machine's existing OS byte-range GPU lock for its whole duration. Policy is
**wait, never kill**: if the lock is held, this instrument blocks until it is released and records
how long it waited. Nothing is terminated. The lock record (`ACQUIRED` → `RELEASED`, wait and hold
durations, whether anything was killed) is embedded in the artifact with its absolute path
reduced to a basename.

## 9. What this run may not conclude

* It may not name a mechanism. §7 localises; it does not explain.
* It may not compare against CUDA, or against any execution provider other than this one's own
  earlier build. No CUDA number appears in the artifact or in anything written from it.
* It may not be used to justify a production shader change. If a literal-1 decode variant is
  supported by the evidence, the output is a recommendation to issue #90, not a merge.
