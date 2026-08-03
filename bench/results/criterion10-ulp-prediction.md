# Criterion 10 residual — the unit, predicted before it was measured

**Recorded:** 2026-08-02T17:36-07:00, on `abf3b3e`, **before any ULP number existed.**
**Author:** Trinity. **Prediction:** Morpheus's, quoted from his ruling.
**DLL:** `918E8FF5…` → rebuilt `523A07C1…`.

## The defect being fixed

Not the kernels. `atol` is an **absolute** bound applied to tensors whose scale grows with
depth. §10.0.4 says prefer the ratio; our own criterion-10 gate violates it.

Morpheus, having checked the premise of the question he was asked before reasoning about
it: of the 65 per-output residuals, **64 are exact negative powers of two** and the 65th is
`3 × 2⁻⁹` — small integer multiples of the fp16 ULP. KV magnitude grows with depth, the ULP
grows with it, so the absolute residual rises with depth **for a correct implementation**.
The monotone curve is a plot of magnitude, not a defect.

fp16 here is a *storage* format and never was an accumulation format: `q_gemv.comp`
accumulates in fp32 "regardless of storage", both f16 layer-norms are "all arithmetic is
fp32", and `gqa_f16.comp` carries `float q[128], k_new[128], acc[128]`, `float dot`, and an
online softmax entirely in `float`.

## The prediction, on record before the instrument exists

> Express the residual **in ULPs**. **Predicted flat at 1–3 across all 32 layers.**
> **Flat ⇒ no defect; a step ⇒ a located one.**

Both outcomes are informative and neither is the one I am hoping for. A step at some layer
is the *better* result: it is a located defect rather than an absence.

## Why this is not a relaxation, stated before the numbers can bias the claim

A 1-ULP residual at layer 0 passes comfortably today on an absolute `atol` that is generous
at small magnitudes. **In ULPs it has nowhere to hide.** Fixing the unit may make the gate
*tighter*.

This is also why it does not fall under Switch's earlier refusal. He declined to replace
`atol` with a scale-set tolerance because his proposal lost to the incumbent at 9.3×
against a 10× bar, missing "1 ULP added everywhere". A ULP-denominated residual is exactly
the observable that catches "1 ULP added everywhere", so it should beat both.

## Why ULP and not relative error — the correction that is in my file

`max_rel_diff` **is not monotone and must stop being the headline.** Layer 2's key reads
**0.4559**, above every layer from 3 to 30, because it is attained at near-zero elements.
Mouse fixed one wrong denominator; there was a second, and it is in the criterion. It
misled two people today.

ULP does not inherit that failure, and the reason is worth stating because it is the whole
argument for the unit: float spacing **floors at the denormal spacing** (`6e-8` for fp16),
so a near-zero element cannot manufacture a large ULP count the way it manufactures a large
relative error. Relative error divides by something that goes to zero; ULP divides by
something that does not.

## Constraints this work is bound by

- **Criterion 10 stays open on the unit alone.** The reopening ground — the all-zero KV
  defect — is measured absent; arms (b), (c), (d) are discharged. **`DIVERGENT` is honest
  right now and must not be flipped by moving `atol`.** The verdict changes when the unit
  is right, or it does not change.
- **GQA's 1.37× margin stays open.** Its proposed remedy was already in place, which is not
  a reason to close it.
- `argmax 30751` and top-10-overlap-10 are **declined on arithmetic, not scepticism**: that
  is **one token**, and N=1 is not a stated N.

## Morpheus's sentence, declined as a rule for the fifth time

> An observable that is true whatever happens cannot convict; an observable that degrades
> whatever happens cannot acquit.

`max_rel_diff` on fp16 tensors with near-zero elements is the second half of that sentence.

---

## RESULT — appended after measurement, 2026-08-02

See `bench/results/criterion10_ulp-dev0.json` and `-dev1.json`. Summary is written into
this file by `tests/ops/test_criterion10_ulp.py` rather than by hand.

---

## RESULT — measured 2026-08-02, both devices

**Binary:** `onnxruntime_vulkan_ep.dll` SHA256 `7F55C0C1CD68FD227A528804FEF4CDD5D195E1C4B901F7B84275C221F6F88FF2`
(before this merge+rebuild it was `523A07C194D916651A1E58824B06B233DAA750BD408DA574D1DF568E0781D9E0`; the
hash is recorded either side because a stale binary told this project the wrong story three
times today).
Artifact: `phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx`, 3 consecutive runs of one
session, 65 outputs, both selectors.

**Headline statistic is `median_ulp_diff`, not `max_ulp_diff`, and that changed while I built it.**
A cancellation element inflates the ULP *max* by exactly the mechanism that inflates
`max_rel_diff` — verified on a synthetic specimen and asserted in
`tests/ops/test_criterion10_ulp.py`. Swapping one max for another would have reinstated the
artefact in a fresh unit (R11: a decomposition that appears to close is the hardest kind of
wrong). Real evidence for that: output 0's `max_ulp_diff` reads **255658** on dev0 and
**242090** on dev1 while its median reads **12** on both. The max is not a residual; it is a
count of how badly one cancelled element behaves.

### The prediction

> flat at 1–3 across all 32 layers. Flat ⇒ no defect; a step ⇒ a located one.

### The measurement

`median_ulp_diff`, per output, **byte-identical on NVIDIA RTX 4060 and Intel Iris Xe**:

```
out  0 (logits) : 12
out  1.. 4      : 0,0,0,1
out  5..48      : 0/1 throughout
out 49..60      : 1..2
out 61,62       : 3, 3
out 63,64       : 4, 4
```

**Verdict on the prediction: confirmed for the KV cache, and falsified in two places — both
of which are reported rather than rounded away.**

1. **The KV cache is flat.** 62 of 64 KV outputs sit at 0–3 ULP, baseline median **1**. The
   absolute residual over the same outputs rises monotonically with depth; in ULPs it does
   not. **That is the ruling's claim, measured: the absolute curve was a plot of magnitude.**
2. **A smooth drift, not a step.** Outputs 63 and 64 read **4** — one ULP above the predicted
   ceiling. There is no discontinuity anywhere in the curve: it climbs 1 → 2 → 3 → 4 across
   32 layers with no layer at which it jumps. That is what accumulation across depth looks
   like in a correct implementation, and it is *not* a located defect. It is recorded as an
   exceedance anyway, because the alternative was to widen the band after seeing it.
3. **Output 0 — the logits — is the step, and it is located.** Median **12 ULP**, `12×` the
   KV baseline, on both vendors, on all three runs. **This is the entire return on changing
   the unit.** Under `max_abs_diff` the logits were also the worst output (`0.0625`), and
   that read as *the logits are the largest tensor, so of course they carry the largest
   absolute residual*. In ULPs magnitude is already in the denominator — and output 0 is
   **still** an order clear of every other output. It is not a big tensor. It is a defect,
   and it is in the head, not in the layers.

**Vendor-independence is the strongest single fact here.** The 65 medians are identical
element-for-element across two different vendors' drivers and two different GPU
architectures. Whatever produces the 12, it is arithmetic in our kernels — not a driver, not
a precision mode, not a scheduling artefact. Owner: Mouse/Switch. The candidate is the final
vocabulary projection (`MatMulNBits`, hidden 3072 → vocab 32064, the longest reduction in the
graph and the only node downstream of output 0 that is not also upstream of the KV outputs).

### What did NOT change, deliberately

- **`atol` is untouched.** `within` is still `np.allclose` on the incumbent tolerance, and
  `tests/ops/test_criterion10_ulp.py::test_the_ulp_statistic_does_not_move_the_pass_fail_decision_yet`
  fails loudly if anyone changes that quietly.
- **The verdict is still `DIVERGENT`** on both devices, as the ruling requires. Criterion 10
  stays open. Nothing was flipped by moving a number.
- **GQA's 1.37× margin stays open.**

### R13 — a result that confirms a prediction deserves more scrutiny

The KV curve confirms Morpheus's prediction, so it got the harder look, and two things came
out of it that a satisfied reading would have missed:

- The confirmation is **only for the KV cache**. "Flat at 1–3 across all 32 layers" is true of
  the layers and false of the model, because the model has a 65th output that is not a layer.
  A reader who takes "flat at 1–3" as the finding gets the wrong model.
- The exceedance predicate is **Morpheus's number, not mine**. My first version used "3× the
  observed baseline"; on real data it flagged outputs 63–64 and my next instinct was to widen
  the multiple until it did not. That instinct is the defect — a threshold fitted after the
  fact cannot contradict the person who set it. The predicate is now the prediction, recorded
  above before any ULP existed, and the two overshoots stand in the record.
