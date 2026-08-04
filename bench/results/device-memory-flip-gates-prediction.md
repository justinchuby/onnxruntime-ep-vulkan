# Predictions, written before any run this round — the four gates on `DEVICE_MEMORY`

Written 2026-08-03, Switch, **before** the first probe of this session executed. Recorded so
that if a gate turns out not to block the flip, the argument is what was wrong, not the
measurement explained afterwards.

The question this round is narrower than "is `DEVICE_MEMORY` safe". It is: **which of the four
gates named at the end of Session 46n is a reason the DEFAULT cannot change?** A gate that
produces the same behaviour in both lanes is a defect the project owns either way and is
therefore *not* a blocker to the flip, however serious it is on its own.

## The discriminator every gate is scored on

> Does the gate's failure **separate the two lanes**?
> `DEVICE_MEMORY` unset (shipping) vs `DEVICE_MEMORY=1` (resident), same inputs, same box.

A gate that does not separate cannot be a reason to keep the flag off, because turning the
flag off does not avoid it. This is the same discriminator that refuted my own five-form
causal story last round: withholding one form and withholding nine gave the identical refusal,
so the withheld forms were never what bound.

## Predictions

| # | gate | prediction | blocks the flip? |
|---|---|---|---|
| 1 | ctx-512 device loss | **does not separate.** Both lanes submit the same 355 dispatches per step and the same kernels; the flag moves where buffers live, not what is computed. Incident record already states the pre-fix binary and the default lane produced identical text. Also **will not reproduce at all** this session — it has not reproduced in 6 arms + 300 standalone GQA iterations since 2026-08-02. | **NO** — predicted |
| 2 | `MIXED` two-device frame | **separates, but in the reporting, not the computation.** Two `HandleRegistry`s at two device indices declare two frames; `tally::device_frame()` becomes `MIXED` and `device_authoritative_observable()` must go false (R12). Predicted: on this box the two indices stand up **two** providers (they are genuinely two physical devices), so `frames_declared() == 2` and the `MIXED` arm fires. No output byte changes. | **NO for correctness, YES for any published counter** — predicted |
| 3 | concurrent sessions on threads | **the one I cannot predict.** The handle registry, the tally and the provider map are process-global; the residency screen runs on `free`. Two sessions inferring simultaneously on two threads is the only one of the four where a lane-separating *correctness* failure is mechanically available. Predicted: outputs stay bit-identical (the spans are disjoint by allocation) but **at least one process-global counter is wrong**, because none of them is per-session. | **YES** — predicted the only true blocker |
| 4 | ctx 8192 | **separates in the direction that ARGUES FOR the flip.** The shipping lane fails at the first Compute with 0 dispatches; the resident lane completes at 5.51 GB. A gate whose failure mode is "the default lane cannot run at all" is not a reason to keep the default. | **NO** — predicted, and it is an argument the other way |

**Therefore the pre-registered headline prediction is: exactly one of the four blocks the
flip, and it is #3.** If that survives, "not yet" becomes one named task.

## What would falsify each prediction

1. A ctx-512 loss that occurs in the resident lane and not the shipping lane, on shared inputs.
   (A loss in *both* confirms the prediction. A loss in *neither* is `UNIFORM(n, NO_LOSS)` and
   is **not quotable** as a clearance — see the positive-control note below.)
2. `frames_declared() == 1` on this box (then the frame cannot be exhibited here at all, which
   is a fact about the box, not the mechanism), or an output byte differing across the frame.
3. Any output byte differing between a threaded run and a serial run of the same inferences.
4. The shipping lane completing ctx 8192.

## The positive-control obligation I am accepting in advance

Gate 1 will almost certainly return `NO_LOSS` in every arm. Per §8.9.21 part 4 (`UNIFORM`), a
classifier returning one verdict over its whole input set is evidence about the mechanism until
a positive control demonstrates the other polarity **through the same predicate**. The predicate
here is `device_losses > 0` / the `VK_ERROR_DEVICE_LOST` string, and it **has** been observed in
its positive state — `bench/results/ctx512_device_lost.txt`, replayed by `ci/check_device_loss.py`
on every run. That is the control, it is on disk, and it is cited rather than re-manufactured.
Gate 1's clearance is therefore permitted to rest on *non-separation*, never on *non-occurrence*.

## What I am NOT predicting

No timing. No ratio. Niobe owns the clock this round. Every number below is a count, a byte
figure or a run/does-not-run verdict.
