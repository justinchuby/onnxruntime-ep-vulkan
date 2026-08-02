### 2026-08-02: Criterion 10 closure is not supported by the scope named in the criterion
**By:** Fact Checker
**What:** Advisory finding: the checked-in artifacts accurately carry the harness's derived
`MATCH`, and a zero-attribution CPU-vs-CPU run cannot reach that token. However, criterion 10
names `model_output_equivalence = MATCH`; the harness compares only output 0 (logits) to CPU.
Outputs 1–64 are compared only between the three Vulkan runs. The evidence therefore does not
establish CPU equivalence for all model outputs and does not see deterministic stale, zeroed, or
otherwise consistently wrong KV outputs. Rate the closure ❌ Contradicted as written, without
purporting to move Morpheus's tally.
**Why:** `tests/ops/test_criterion10.py::_compare_run_to_cpu` indexes only `vk_out[0]` and
`cpu_out[0]`; `outputs_compared = 65` is merely `len(run)`. The constructor's `MATCH` is genuine
but inherits the comparison's narrower scope. Settlement is a rebuilt, independently generated
three-run record that compares all 65 Vulkan outputs against all 65 CPU outputs per run, records
per-run attribution/success, preserves the raw profile, and binds artifact/model, source commit,
ORT/EP binary digests, and one-session identity into the record.

## Verification ratings

- ✅ **The lane artifacts say `verdict = MATCH`.** This is the canonical vocabulary used for
  `model_output_equivalence`; `write_equivalence_record` writes the same derived token under that
  exact counters key. Falsifier: a parser showing the lane `verdict` is produced by a different
  constructor or vocabulary.
- ❌ **They do not show full model-output equivalence.** CPU comparison uses only logits and only
  argmax/top-10/non-zero checks; even logits are not elementwise tolerance-gated. The 64 KV outputs
  are only required to repeat bit-for-bit across Vulkan runs. Falsifier: source or a raw comparison
  record demonstrating CPU comparison of every output for every run.
- ⚠️ **Three Vulkan profile events are aggregate session attribution, not per-run attribution.**
  Three is compatible with one fused island in each of three runs, but the profile has no run IDs
  and `uniformly_attributed` proves only `count >= runs && count % runs == 0`. Falsifier: per-run
  profile/counter intervals showing exactly one completed island in each run.
- ✅ **The current source constructs one `InferenceSession` and invokes it three times.**
  ⚠️ The checked-in artifact does not authenticate that source, binary, or one-session provenance;
  its statement is the harness's word. Falsifier: a record-bound session ID plus source/binary
  digests, or raw trace boundaries proving the three calls share one session.
- ✅ **Zero ORT-profile attribution cannot construct `MATCH`.** The adversarial CPU-only tests and
  derived constructor enforce `UNATTRIBUTED`. Falsifier: a test producing `MATCH` with
  `VulkanExecutionProvider == 0`.
- ⚠️ **A positive ORT node event is not by itself proof of successful output contribution.**
  ORT's `KernelScope` emits the provider-tagged node event in its destructor before the executor
  checks `status`; a failed attempt can therefore be profiled. EP counters increment only after a
  successful dispatch, but are process-global, are not reset in this test, and can witness partial
  work before a later island failure. Falsifier: a completion-tagged per-run witness linked to the
  returned outputs, or a demonstrated ORT invariant that failed/retried nodes cannot appear in this
  successful run's profile.

## Strongest case against closure

The advance-fixed condition prevents post-hoc bar movement; it does not make an under-scoped
instrument discharge broader words. ORT profiling is independently implemented but shares the
same process, session, aggregate frame, and fallback behavior. The EP counters are implementation
owned and establish successful dispatches, not returned-output provenance. Two devices reduce the
chance of a device-specific fault, but repeat the same harness, parser, comparison scope, model,
and constructor, so they do not independently test systematic blindness. Three runs are too short
to characterize cache exhaustion, and deterministic unwritten outputs pass the cross-run identity
check indefinitely.

## Thirty-day pre-mortem

Criterion 10 reopens because a longer persistent session exhausts the weight cache or arena after
run 3; ORT records a failed Vulkan attempt and returns CPU-fallback logits; KV outputs remain
deterministically stale/zero across runs; or a harness/source mismatch means the checked-in record
cannot be reproduced. Tonight's evidence would miss all four: it stops at three, aggregates
attribution, CPU-checks only logits, and does not bind executable provenance.
