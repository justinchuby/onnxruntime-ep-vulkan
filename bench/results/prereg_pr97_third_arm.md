# Pre-registration addendum: PR #97 as a third arm (issue #96)

**Author:** Switch · **Written:** 2026-08-09 · **Status:** frozen on publication.

This is an **addendum**, not a revision. `prereg_crossbuild_decode_window.md`
(sha256 `4d1834e231c2bd28a41eb8323b18163cecd529268e6843683cd7b5fec1fe544d`) and the two artifacts
it produced are **closed and unchanged**. Nothing below alters, reinterprets, or re-scores the
baseline-vs-candidate sweep, which completed and was published before PR #97 existed as a
comparison request.

The digest of *this* file is published to issue #96 **before the first third-arm process runs**,
under the same discipline.

---

## 1. What is being added, and why it is comparable

| arm | tree | role |
|---|---|---|
| 1 · baseline | `c96e7d94ff706d26ee6a1bd9bb084c0ade426820` | pre-#72 |
| 2 · candidate | `85fbda29a92e0e99c3895be8b13664d4ee670c50` | #72 spec-constant change |
| 3 · pr97 | `fc1b163548a307e3b1be6d996673842203ebba25` | PR #97, decode KV-parallel |

Comparability is a fact about the trees, checked before building, not an assumption:

* `git diff 85fbda2 8701812c -- rust/src rust/shaders rust/Cargo.toml rust/Cargo.lock` is
  **empty**. Arm 2's library is, in compiled terms, `main`'s library.
* `fc1b163`'s single parent is `8701812c`, and its compiled delta is 6 files:
  `gqa_decode_f16.comp` (new, 379 lines), `attention.rs`, `counters.rs`,
  `ops/common/templates.rs`, `ops/common/variants.rs`, `vk/session.rs`.

Therefore **arm 2 ↔ arm 3 isolates PR #97 exactly**, and arm 1 ↔ arm 3 additionally carries #72.
Both pairings are declared here; the arm 2 ↔ arm 3 pairing is the one that answers the question.

Library digests, fixed before timing:

| arm | sha256 (16) | bytes |
|---|---|---|
| baseline | `655a247c29a85858` | 2,560,000 |
| candidate | `3b210b016179011a` | 2,563,072 |
| pr97 | `705c52d239e1d470` | 2,580,992 |

## 2. Protocol: unchanged, and unchanged *by construction*

The third arm is measured with **`bench/results/probe_crossbuild_decode_window.py`, byte-identical
to the file that produced the frozen artifacts**. It is not edited. The third arm is realised as
two additional **pairings** of the existing two-arm instrument, not as a new N-arm rewrite,
precisely so that the code which produced the frozen result cannot drift:

1. `--baseline-lib` = arm 2, `--candidate-lib` = arm 3 → **isolates #97**
2. `--baseline-lib` = arm 1, `--candidate-lib` = arm 3 → places #97 against pre-#72

Everything else is inherited verbatim from prereg §2–§6: the same 9 workloads (MobileNetV2 `N=1`
and `N=16`; Phi-3.5 prefill `M=1 past=0`; decode `M=1` at `past ∈ {32,64,128,256,512,1024}`),
3 whole-process repeats, one process per `(workload, arm, repeat)`, interleaved arm order,
5 warmups / 20 timed, per-process median, the same equivalence / witness / provenance /
library-identity gates, the same refusal discipline (a refused record keeps no speed field), the
same exclusive GPU lock with **wait — never kill**, the same band rule
`max(0.05, max no-GQA control half-range)`, and the same ratio convention
`baseline_median_ms / candidate_median_ms`, so **> 1 means the second-named arm is faster**.

`past = 2048` remains excluded for the reason given in prereg §3.

## 3. Consequences of #97 that are declared *before* timing

PR #97 does not modify `gqa_f16.comp`. It adds a **separate** shader and takes it
**unconditionally when `seq_len == 1`**. Three things follow, and all three are stated now
rather than discovered later:

**(a) The witness vocabulary changes, and the requirement is strengthened, not relaxed.**
On arm 3 every `seq_len == 1` workload produces a `gqa_decode_f16:<W>` witness, not `gqa_f16:*`.
The frozen gate takes one expected key per arm; #97's key varies with length. The third-arm
driver therefore sets that expectation per `(arm, workload)` before each process and **refuses any
record whose witness is not exactly the predeclared string**. A missing, extra, or unexpected
witness key is a refusal, not a footnote.

**(b) The expected `W` is derived here, independently, before any run.**
From `gqa_decode_kv_parallel_with`: `W` is the largest power of two `≤ 16` such that
`(past_len_max + seq_len) / W ≥ 32`, and `past_len_max` is the declared `past_key` capacity, which
`bench/real_model.py:phi35_feeds` sets to `case.past`. Computed by hand from that rule alone —
**not** read from PR #97's results, documentation, or claims:

| workload | `past_len_max + 1` | predeclared `W` | predeclared arm-3 witness |
|---|---|---|---|
| prefill `M=1 past=0` | 1 | 1 | `gqa_decode_f16:1` |
| decode `past=32` | 33 | 1 | `gqa_decode_f16:1` |
| decode `past=64` | 65 | 2 | `gqa_decode_f16:2` |
| decode `past=128` | 129 | 4 | `gqa_decode_f16:4` |
| decode `past=256` | 257 | 8 | `gqa_decode_f16:8` |
| decode `past=512` | 513 | 16 | `gqa_decode_f16:16` |
| decode `past=1024` | 1025 | 16 | `gqa_decode_f16:16` |

If a measured witness disagrees with this table, the record is **refused as `INCOMPARABLE`** and
the disagreement is reported. It is not silently adopted.

Note the consequence for `past = 32`: `W = 1`, which by #97's own construction is the serial
geometry. `past = 32` is therefore an **internal control** — if arm 3 shows a large speed change
there, it is not KV parallelism doing it.

**(c) Prefill `M=1` is no longer a null for arm 3.** It has `seq_len == 1`, so it takes the new
kernel too. It stays in the sweep as a declared **treatment** row for arm 3. Only MobileNetV2
remains a genuine no-GQA control, and the band continues to come from MobileNetV2 alone — the same
choice, for the same reason, as in the frozen prereg.

## 4. The questions, and the decision rules, fixed now

**Q0 — is there a `past = 128` whole-model regression for #97 to resolve?**
Already answered by the frozen sweep: **no**. Arm 1 ↔ arm 2 at `past = 128` is `1.001×`,
`NEUTRAL`, and the window claim is `NO-SLOW-LENGTH`. **The verdict `RESOLVES-P128-REGRESSION` is
therefore unreachable in this investigation**, and is declared unreachable *before* the third arm
runs so that no later reading can quietly award it. What remains live is whether #97 is a
whole-model *improvement*.

Per length, on the arm 2 ↔ arm 3 pairing, exactly one of:

| verdict | rule |
|---|---|
| `WHOLE-MODEL-FASTER` | **every** repeat's ratio > `1 + band`, and the row's equivalence and witness gates all passed |
| `WHOLE-MODEL-SLOWER` | **every** repeat's ratio < `1 − band` |
| `KERNEL-ONLY` | the decode-GQA device-time share drops by more than the untouched-control spread at that length, **and** the whole-model verdict is `NEUTRAL` |
| `NEUTRAL` | neither of the above; the row sits inside the band |
| `INCOMPARABLE` | any gate refused: equivalence, witness (including a `W` disagreement), provenance, or library identity |

`KERNEL-ONLY` requires the kernel claim and the whole-model claim to be measured on the **same**
runs, and requires the kernel movement to exceed the movement of kernels #97 cannot touch. A
kernel ratio alone never earns it.

**Q1 — does the GQA device time actually fall?** Attribution pass at `past = 128`, and at `64` and
`256` if they yield admissible pairs, with the existing merged tracer on **both** arms of the
pairing. Because the kernel is renamed, arm 2's `gqa_f16` and arm 3's `gqa_decode_f16` are compared
as a **share of a composition**, not as a same-name ratio; that is a weaker statement and is
labelled as such in the artifact. The untouched-kernel control set is unchanged
(`q_gemv_matmul_nbits_f16` and the five others).

**Q2 — is #97's `W = 1` claim of bit-identical output true?** #97 documents
`ONNXRUNTIME_EP_VULKAN_GQA_DECODE_KV_PARALLEL=1` as restoring the serial geometry with
*"bit-identical output, not merely close"*. That is falsifiable, so it will be falsified or not:
arm 3 at default `W` vs arm 3 with the override set to 1, output digests compared at every length.
**This is the only place any `ONNXRUNTIME_EP_VULKAN_*` variable is deliberately set**; the sweep's
env hygiene (strip all of them from every child) is otherwise untouched, and this H-test is run
outside the sweep and never contributes a speed figure to it.

## 5. Independence

PR #97's `bench/results/real_model_gqa_decode_kv_parallel.json`, its `docs/PERF.md` diff, and its
claimed kernel speedups **have not been read and will not be read** before this investigation's own
numbers are recorded. `rust/src/ops/attention.rs` and `gqa_decode_f16.comp` were read — that is
reading the *implementation* in order to derive §3(b) independently, which is verification, not
importing a result. Link is not consulted. Link's branch is not edited, fetched into a working
tree, or pushed to; arm 3 is built in a **detached** worktree at the exact head.

## 6. What this addendum may not conclude

* It cannot award `RESOLVES-P128-REGRESSION`; see Q0.
* It cannot speak to lengths whose equivalence gate refuses on either arm. The frozen sweep already
  lost `past ∈ {32, 64, 256}` to the repo's standing `PHI35_MAX_PROB_DELTA = 0.02` budget, and the
  same budget applies here. If those lengths refuse again, `W ∈ {1, 2, 8}` — including the `W = 1`
  internal control — go unmeasured, and that loss is reported rather than worked around.
* It cannot speak to any device other than this one, nor to the driver's final machine code.
* Three repeats support a direction, never a magnitude.
* A `NEUTRAL` whole-model result is not evidence that the kernel did not get faster; it is evidence
  that the model did not. Those are different claims and the artifact keeps them apart.
