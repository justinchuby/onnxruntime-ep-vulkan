# The int8 KV error budget — written **before** the first run

**Switch, 2026-08-03.** Nothing in this file was derived from any int8 measurement, because at the
time of writing none exists. It is here so the numbers below can be *wrong* rather than *explained*.
The probe is `bench/results/probe_kv_int8_budget.py`; the record it writes is
`bench/results/kv_int8_budget-dev{0,1}.json`.

Same discipline as the two rounds before it: 5,512,528,520 B at ctx 8192 and
804,126,720 B = 2045 x 393,216 at ctx 2048 were both written down first and both landed.

---

## 0. Why an error budget comes before a kernel

The KV ledger's third lever is int8 (the second, the present copy, is banked and measured). But
**int8 is a correctness change wearing a bandwidth change's clothes**, and this project has a
specific reason to be careful: Trinity established that at the final RMSNorm **Vulkan is bit-exact
against float64 while ORT's CPU EP is the 1-ULP side** — our accuracy story is presently *better*
than the reference in at least one place. Criterion 10 is open with three failing outputs at
**12 ULP on the logits and 4 ULP on layer 31's key and value**. Quantising the KV cache moves
exactly those tensors.

So: measure what int8 costs on the outputs criterion 10 already measures, in the unit criterion 10
already uses, **using Trinity's instrument** (`tests/ops/_models.ulp_residual`, and the
median/p99/max/cancellation distribution her consumer records beside it) — not a second one.

## 1. The model of the error

int8 symmetric, round-to-nearest, per group of `N` values, scale `s = max|x| / 127` stored fp16.
Quantisation error is uniform on `+/- s/2`, so its RMS is `s / sqrt(12) = max|x| / 440`.

fp16 relative spacing is between `2^-11` and `2^-10` — call it `7.3e-4` averaged over the mantissa.
So for an element of magnitude `|x|`:

```
ULP(x)  ~=  (max|x| / 440) / (|x| * 7.3e-4)  =  3.11 * (max|x| / |x|)
```

**The whole budget is that ratio.** ULP is scale-free *per element*; quantisation noise is scale-free
*per group*. An element well below its group's max pays the group's absolute error against its own
much finer spacing. This is why the granularity ladder is the experiment.

For Gaussian data the group max over `N` samples is about `2.4 sigma (N=32)`, `2.7 sigma (N=96)`,
`3.6 sigma (N=3072)`, and the median magnitude is `0.674 sigma`. Real KV is *not* Gaussian — the
published outlier-channel behaviour makes `max/median` larger, which moves every number below up.

## 2. Predictions — ULP, against the fp16 CPU-EP oracle on identical inputs

Granularity is defined on the **newest token's slice** of one layer's K or V, shape
`(1, 32 kv_heads, 1, 96)` = 3072 values, because that is what a real kernel quantises once and keeps:

| name | scales per token per tensor | group size N |
|---|---|---|
| `per_tensor` | 1 | 3072 |
| `per_head` | 32 | 96 |
| `per_block32` | 96 | 32 |

`per_head` **is** the per-head-group scale of the general case. See §5.

| lane | median ULP on K/V | p99 ULP | max ULP | median ULP on logits |
|---|---|---|---|---|
| `fp16` control (criterion 10 today) | 1–3 | 1–3 | ~4 (K/V), ~12 (logits) | 1–3 |
| int8 `per_block32` | **10–40** | 10^2 | > 10^3 | 5–50 |
| int8 `per_head` | **12–60** | 10^2–10^3 | > 10^3 | 5–50 |
| int8 `per_tensor` | **30–200** | 10^3 | > 10^4 | 10–200 |
| int4 `per_head` (16x coarser) | **200–1000** | 10^4 | > 10^4 | 10^2–10^3 |

**The headline prediction, stated so it can fail:** int8 raises the K/V ULP residual by **one to two
orders of magnitude** over the 4 ULP criterion 10 measures today, at every granularity tested. `max`
will be dominated by cancellation elements and — per Trinity's own R11 on her instrument — **cannot
acquit or condemn on its own**; the medians are the object.

**Therefore, predicted before the fact: no ULP-denominated criterion 10 passes with an int8 KV
cache, at any granularity, and no tolerance exists that both admits int8 and still catches a real
defect in the fp16 path.** If that holds, the question is not "what atol" — it is *which observable*,
and that is Morpheus's ruling, not a number I get to pick. §4.

Secondary predictions:
- **Top-1 token agreement** against the fp16 oracle over a fixed 12-step chain: int8 >= 11/12 at
  every granularity; int4 `per_tensor` is where I expect the first disagreement.
- **Depth.** ULP median vs `past_len` is predicted **flat** for int8 — each token is quantised once,
  errors do not compound, and attention averages over more tokens as depth grows. A *rising* slope
  would be a finding (compounding through the residual stream), a falling one would be averaging.
- **`per_block32` beats `per_head` beats `per_tensor`**, monotonically, on the medians.

## 3. Predictions — bytes (class: MODEL, per Niobe's provenance rule; not a measurement)

Derived from `bench/results/island_bytes_phi35.json` (kv_cache 3072 MiB, total 5078.37 MiB at
past 8192) and from the arena's **measured** 5,512,528,520 B footprint at ctx 8192. Scales stored
fp16.

Footprint at ctx 8192 (non-KV term held at the measured `5,512,528,520 - 8192*393,216 =
2,291,303,048 B`):

| lane | KV bytes/token | footprint at ctx 8192 | vs fp16 arena |
|---|---|---|---|
| fp16 arena (**measured**) | 393,216 | **5,512,528,520** | 1.000 |
| int8 `per_tensor` | 196,736 | 3,902,964,360 | 1.412 |
| int8 `per_head` | 200,704 | 3,935,470,216 | 1.401 |
| int8 `per_block32` | 208,896 | 4,002,579,080 | 1.377 |
| int4 `per_head` | 102,400 | 3,130,163,848 | 1.761 |

On the modelled **stream** at ctx 8192: int8 `per_head` 5078.4 -> 3574.4 MiB = **1.421x**; int4
`per_head` -> 2806.4 MiB = **1.809x**.

**A disagreement with the ledger, recorded before running rather than after.** The KV ledger this
work is sequenced against quotes **2.21x (present copy), 3.17x (int8), 4.06x (int4)**. I cannot
reproduce any of the three from any artifact in this tree, on any baseline I can construct
(footprint, modelled stream, KV-only, with or without the present write). The figures above are what
`island_bytes_phi35.json` and the arena's measured footprint actually support. **If int8 lands near
1.4x rather than 3.17x, the ledger is what was wrong, and this paragraph is why that is not a
retrofit.**

## 4. What "correct" means for a quantised cache — deliberately NOT decided here

Bit-identical is unavailable by construction. The standing rule is that **a criterion may not be
hardened because it is about to pass nor narrowed because it has just failed**, and the same rule
forbids its mirror: *I do not get to pick a tolerance my result happens to pass.* So this round picks
none. It measures, and files the question with the tolerance owner (Morpheus), with Trinity's
`atol`-is-absolute-on-a-growing-scale finding already in front of him.

The three shapes the ruling could take, so the question is a choice and not an open field:

1. **ULP band on K/V, widened.** Rejected in advance by me: any band admitting int8 (10^1–10^2 ULP)
   is ~30x looser than the fp16 path's own residual, so it stops catching fp16 defects. A tolerance
   that admits the change under test and nothing else is a tolerance chosen by its answer.
2. **A different observable on the logits** — top-1 agreement, top-k set agreement, or KL against the
   fp16 oracle — with the K/V tensors explicitly out of the oracle comparison for quantised lanes,
   *and said so in the verdict*. This is the shape I expect to be right, and it is a ruling because it
   changes what criterion 10 claims, not how tightly it claims it.
3. **Two criteria.** fp16 lane keeps criterion 10 unchanged; a quantised lane gets its own named
   criterion with its own observable, and the two are never quoted as one number.

## 5. What is untested, named in advance

`Nq / Nkv = 1.00` on Phi-3.5. It is **4x on Llama-3**. Every granularity above is defined on the KV
head axis, so `per_head` here is a per-head-group scale over a group of one query head. On a real GQA
model one KV head's quantisation error is shared by four query heads, and a scale that is degenerate
here is load-bearing there. **This is the lever where tuning to this model would hurt most.** No run
in this round exercises a non-unit grouping, and no number in it may be quoted for a 4x model.

Second: this probe quantises at the **host** boundary, so it models the *storage* error exactly and
models **nothing** about a kernel's own rounding on write or any change to accumulation order. It is
therefore a **lower bound** on a real int8 kernel's residual, never an estimate of it.

## 6. Note for Mouse, before he finds it

The arena introduced **no** specialisation constant — `pipeline_variants` shows `"gqa_f16:"` with an
empty list, push constant plus binding topology only, witnessed by `kv_cache_convention`. **An int8
KV kernel almost certainly does introduce one** (element type and/or block size as a spec constant on
`gqa_f16`), and if it does, his `spec_digest` must see it. Saying so here rather than letting him
discover it.

## 7. Residual carried in from the arena round

An overrun past the arena's capacity is **dropped by the shader guard, not refused**, because the
true past length lives in `seqlens_k` on device. If int8 work goes back into `gqa_f16.comp`, making
that silent drop observable is worth more than a percent of bandwidth.

---

**RESOLVED (2026-08-03, Switch).** The three ledger figures are **RETRACTED**, not corrected. See
`docs/PERF.md` §23.4 and `bench/results/probe_kv_lever_ledger.py`, which re-derives every lever from
artifacts at run time and classes each one. `2.21` sits 0.21 from the naive KV-term-only fp16/int8
ratio 2.0 and `4.06` sits 0.06 from the naive fp16/int4 ratio 4.0 — consistent with an **axis error**
(KV-term savings quoted as whole-system savings) rather than an arithmetic one. `3.17` fits nothing,
so even that does not explain all three. The paragraph above stands as written and was right.
