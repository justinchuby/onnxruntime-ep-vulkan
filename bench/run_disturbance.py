"""``run_disturbance`` — was this run disturbed, measured in-band, without a clock comparison.

READ THIS FIRST: what this is NOT
----------------------------------
**It is not a second, independent opinion about whether a run is usable.** An earlier version of
this docstring said the nine traces it flags and the nine carrying a large per-inference spread were
"two different statistics over two different frames agreeing on the same nine runs, neither derived
from the other." **That was false, and it was my error rather than an inherited one** — I had the
agreement in front of me and read it as independence, when agreement between two statistics is
evidence *for* redundancy, not against it.

Niobe computed the correlation over the whole census and falsified it. I recomputed it at my own
commit rather than quote hers (R13): over 29 device-0 traces, same-ordinal RSD against whole-series
per-inference RSD gives **Spearman rho = 0.919**, **log-log Pearson r = 0.970**, median ratio
**1.170**. (Hers: 0.903 / 0.964 / 1.128 over 28; the small difference is one extra trace.)

The mechanism is hers and this module now measures it directly: a disturbance that scales a whole
submission moves every dispatch inside it together, so the two statistics are nearly the same
quantity **by construction**. :func:`localise` divides out per-inference level and finds that doing
so explains **84–87%** of the same-ordinal dispersion on the disturbed traces (``contended``
137.35% → 18.70%; ``notile`` 55.84% → 7.25%; ``ab_p0_r1`` 109.23% → 17.39%).

So: **"the guard and the tail agree" is close to tautological and must not be reported as
corroboration.**

What survives, and it is worth having
---------------------------------------
Whole-series RSD says *that* something moved. :func:`localise` says *what* moved and *where*, by
dividing out per-inference level and reporting how much of the dispersion that removes. Two
device-0 traces that whole-series RSD ranks as neighbours are different conditions:

* ``ab_p1_long`` — same-ordinal 35.31%, level-normalised **3.90%** (88.9% explained).
  ``SUBMISSION_LEVEL``: the dispatches inside each inference agreed with each other; the whole
  submission moved. A clock/power excursion or a queueing delay ahead of the submit.
* ``contended3`` — same-ordinal 41.11%, level-normalised **50.43%** (-22.7% explained).
  ``PER_DISPATCH``: the dispatches disagree *more* than the totals do. Foreign work interleaved
  between dispatches.

Nothing else in this project distinguishes those, and after the rho = 0.919 correction it is the
only novelty this module claims. Note the classification is derived from the measurement, not
attached to a trace by name.

Two results from that decomposition worth carrying forward:

* my own ``notile`` — the "70% kernel-time spread" from session 43 — is **87.0% SUBMISSION_LEVEL**.
  The spread was whole-inference scaling, which is consistent with, and independent evidence for,
  the ``SAME_FRAME_ORDERED_SELECTION`` reconciliation.
* my ``packed`` A/B trace is the only badly disturbed trace classified ``MIXED`` (44.2%), so it
  carries genuine per-dispatch disagreement on top of submission scaling. A guard whose author is
  exempt from it is not a guard.

What this detects
------------------
**Non-stationarity across repetitions of identical work.** Nothing more. It is a dispersion
statistic, so it sees *variation* and is blind to *level*. Read "The hole every dispersion
instrument shares" below before quoting it, because the obvious misreading -- "the guard passed, so
the machine was quiet" -- is wrong and the module ships a control that proves it wrong.

The hole every dispersion instrument shares, and it is the one that has cost us most
--------------------------------------------------------------------------------------
**No statistic computed from inside a series can detect a bias that scales the whole series.**

This is not a limitation of this module's design; it is a property of dispersion. Any measure of
spread is invariant under multiplying every sample by a constant, so a run that is uniformly wrong
is indistinguishable from a run that is uniformly right.

The project has a worked example and it is the most instructive artifact we own.
``trace_gemv_baseline_certified_dev0.json`` is:

* the cleanest whole-series RSD of all 29 traces (**0.118%**),
* the cleanest same-ordinal RSD (**0.624%**),
* unchanged by level normalisation (**0.632%**, -1.2% "explained" — there was nothing to explain),
* ``STEADY`` at n=46, 100% coverage, zero discarded,

**and it is 21.4x wrong.** Every dispersion measure this project owns certifies it. Only
``ci/check_device_state.py`` refuses it, because its evidence comes from **outside** the series.

Consequences, which this module is built to respect rather than to work around:

1. ``PASS`` here means *"repetitions agreed"*. It never means the machine was quiet, the clock was
   correct, or the figure is right.
2. A dispersion guard cannot replace the §10.0 obligation 8 device-state record, and no future one
   can either. That is structural, not a gap to be filled by a better statistic.
3. :func:`synthetic_uniform_slowdown` constructs a 2x-inflated run with perfect repetition, and
   this module's own test asserts the guard **passes** it. The hole is demonstrated, not described.

Niobe's formulation of the neighbouring trap, which belongs next to this one: *the tightness of a
short tail is evidence about the length of the window, not the state of the device. Any series that
moves contains a stretch that does not, and the more disturbed the series, the flatter its flattest
stretch.*

The quantity
------------
An inference dispatches the same kernels in the same order every time. Take the ``k``-th dispatch
of every inference and compute the RSD of that one dispatch across inferences; do it for every
``k``; report the median. Call it **same-ordinal RSD**.

Because ordinal ``k`` is the same node with the same shape and the same input size on every
repetition, its duration is a repeated measurement of one fixed quantity. Dispersion in it is not
workload structure -- it is the machine failing to do the same thing twice.

Why this shape rather than a wall-clock threshold
---------------------------------------------------
Every load guard this project has considered was wall-clock-threshold shaped, and every one fails
R9 amendment 5: a slow machine and a broken build both make numbers go up, so the check moves with
the reader's confidence rather than with the condition. Same-ordinal RSD does not have that shape.
It is dimensionless, it is internal to a single run, and it does not move when the model gets
bigger, the machine gets slower, or the kernel gets faster -- only when repetitions stop agreeing.

**This is a real property and it is still not independence.** Whole-series RSD is dimensionless and
internal too. Being well-shaped and being new are different virtues and only the first was earned.

Why it is not the same statistic as the per-inference spread
--------------------------------------------------------------
Per-inference GPU-busy RSD (what ``gpu_steady_tail`` operates on) is a **sum** over ~355 kernels,
and it conflates two different conditions:

* **drift** — the whole run trending, e.g. a warm-up ramp. Legitimate, expected, and the reason the
  tail discards a prefix.
* **jitter** — repetitions of one dispatch disagreeing. Not legitimate.

Same-ordinal RSD partially isolates the second — **partially**, at rho = 0.919, which is the whole
correction above. Two committed traces show the residual is real: at essentially the same
per-inference spread (``baseline`` 10.36%, ``ab_p0_r2`` 10.97%) their same-ordinal RSD differs by
2.75x (10.51% vs 3.82%). One drifted; the other jittered. A guard built on the per-inference spread
could not tell those two apart. But it would rank the other 27 traces almost identically, and that
is the part I previously did not say.

What it cannot see, and this is the load-bearing caveat
--------------------------------------------------------
**A uniformly slow run passes.** If a competing process took exactly half the GPU for the entire
run, every dispatch would be equally inflated, repetitions would agree perfectly, and same-ordinal
RSD would be *pristine*. :func:`synthetic_uniform_slowdown` constructs exactly that case and this
module's own test asserts that the guard **passes** it. That is not a bug being tolerated; it is
the boundary of the claim, demonstrated rather than described.

So:

* ``PASS`` means **"repetitions agreed"**. It does **not** mean the machine was quiet, the clock was
  correct, or the figure is right.
* It therefore **complements and does not subsume** the §10.0 obligation 8 device-state record.
  Obligation 8 is the only bias-sensitive instrument we have: it catches a board pinned at 210 MHz,
  which is precisely the uniform inflation this statistic is blind to. The two failure modes are
  orthogonal and both records are required. A run can be disturbed at a perfectly steady clock, and
  a run can be perfectly stationary at an entirely wrong one -- we have observed the second, at
  21.4x.

Nor does it subsume ``gpu_steady_tail``'s coverage floor, which asks a third question again: how
much of the run the published suffix actually covers.

The frame, chosen after the obvious frame was tried and failed
----------------------------------------------------------------
This measures over the **whole run**, not over the tail's published suffix.

The suffix was the first thing tried, because a companion to a figure ought to cover the same
inferences that figure covers. **It has no discriminating power there and the census says so:**
restricted to its own steady suffix, the most disturbed trace in the census (``contended``, 137.4%
over the whole run) reads **3.07%** -- inside the range spanned by the twelve traces that publish
(0.62%–3.97%). The tail's suffix selection has already found a quiet window; measuring inside it
finds a quiet window. So the honest scope is:

    **this guard is a statement about the run, not about the suffix cut from it.**

Its value against a published figure is exactly that: it can say *"the suffix you are quoting was
carved out of a violent run"*, which no statistic computed inside the suffix can say.

Threshold, and how honestly it separates -- WHICH IS LESS WELL THAN FIRST CLAIMED
----------------------------------------------------------------------------------
Placed by census over the committed device-0 traces, not by fitting the two the investigation
started from. **But the census has since falsified its own headline, and running
``--sensitivity`` is what caught it.**

The original claim, over 28 traces, was that the statistic is sharply bimodal with a gap
containing no trace at all: 19 traces from 0.624% to 10.507%, then nothing, then 9 traces from
35.313% to 137.352%, with 20% sitting in an empty gap 1.90x above the highest clean trace and
1.77x below the lowest disturbed one.

**With 29 traces the gap is no longer empty.** ``switch_resid`` reads **20.787%** — inside the
former gap and barely 1.04x over the bar. So:

* the separation on the clean side is unchanged: 1.98x above the highest undisturbed trace;
* on the disturbed side it has collapsed from 1.77x to **1.04x**;
* the verdict on ``switch_resid`` is therefore a **judgement, not a separation**, and must be
  reported as one.

That is the outcome the brief explicitly asked for over a picked number: *if the two populations
turn out not to separate cleanly, say that*. One additional trace was enough. An "empty gap" is a
statement about the traces you happen to have, and it decays as you collect more — which makes it
exactly the kind of claim that should be recomputed rather than quoted. **Re-run
``--sensitivity`` before relying on any of these numbers.**

**Publish the sensitivity, do not defend the number.** Niobe set a 5% flag threshold, got 7 flagged
traces, noticed her own instinct was to move it until the count reached zero, named that *choosing
the answer*, and published the whole table instead. Over the 29-trace device-0 census:

===========  =======
 threshold    flagged
===========  =======
 5%             20
 10%            11
 11%            10
 15%            10
 20%            10   <- DISTURBANCE_RSD_MAX
 25%             9
 30%             9
 35%             9
 36%             8
 50%             7
 75%             3
===========  =======

The answer is unchanged only from **11% to 20%** — a band that used to run to 35% and no longer
does, because ``switch_resid`` drops out at 25%. The threshold has started doing some of the
deciding. It is left at 20% because moving it to keep the old flat band would be choosing the
answer, which is the failure being avoided; but the narrowing is the honest headline and it should
be re-read whenever the trace set grows.

**The empty gap is a device-0 property and does not reproduce on device 1.** Over the 12 committed
Intel traces the distribution is continuous through the boundary: the single flagged trace
(``tiled_atomic_store``) sits at 21.36%, only **1.07x** over the bar, with the next below it at
9.85%. So on Intel the threshold is an ordinary cut through a populated region, not a line in an
empty gap, and a verdict near it there is a judgement rather than a separation. Two reasons this
is unsurprising and one consequence:

* the iGPU shares its power budget with the loaded CPU cores, so its floor of self-disagreement is
  higher and the two populations are not expected to be as cleanly split;
* there are 12 Intel traces against 28 NVIDIA ones, and a gap is the kind of feature a small sample
  fails to show.

Consequence: **quote the 1.90x/1.77x separation for device 0 only.** On device 1 the check is still
a real measurement and still fires, but its margin near the threshold is thin and a borderline
Intel verdict should be treated as such rather than as the clean call the NVIDIA census supports.

**The sensitivity table settles this more sharply than the gap argument did.** On device 1 the
flagged count is 11 at 5%, 7 at 10%, 5 at 11%, 4 at 15%, **1 at 20%**, and **0 at 25%** — the band
over which the answer is unchanged is a single point. The threshold is doing all of the deciding
there. On device 0 the band is 11%–20%. So:

    **device 0: a judgement at the boundary. device 1: a judgement, full stop.**

An Intel verdict from this check should be read as "this is where we drew the line", not as a
separation the data supports.

**And here is where it does not separate, which matters more than where it does.** The two
populations above are *not* the STEADY/refused populations. Against the tail's verdict there is
substantial overlap: publishing traces run 0.624%–10.507% and refused traces run 3.694%–137.352%.
Same-ordinal RSD does not predict whether the tail will publish, and must not be read as doing so.
What it predicts is whether the run was stationary, and the nine it flags are exactly the nine
traces carrying a >=30% per-inference spread. **That agreement is redundancy, not corroboration**
(rho = 0.919 — see the correction at the top of this module). It was previously written here as
"neither derived from the other", which was exactly backwards.

What it adds today: nothing, and that is measured
---------------------------------------------------
``ci/check_run_disturbance.py --corroborate`` computes this rather than leaving it to a claim
here. Over the 28-trace census: **9 flagged, 9 already refused by the tail's own floors, 0 runs
that the tail would publish.** On this evidence the check **adds no refusals**. It corroborates.

That is worth stating plainly because the temptation is to sell a new guard on coverage it does
not have. Its actual value is three things, none of which is extra coverage:

1. It refuses for a reason that **does not depend on suffix selection**. The coverage floor
   refuses ``contended`` because the surviving window is small; this refuses it because the run was
   violent. Those can come apart — a run can hold 60% coverage and still have a violent prefix.
2. It is the check that **still holds if a ``MARGINAL_TAIL``'s withheld median is ever published**.
   That is not hypothetical: ``contended`` is the most disturbed run in the census at 137.4%, and
   its *tail* RSD is 0.1067% — the third tightest of all 28, tighter than the certified baseline's
   0.1163%. If that median were ever quoted, our dirtiest run would read as our cleanest. This
   statistic is 6.9x over its bar on that run.
3. It **localises**. Whole-series RSD says the run moved; :func:`localise` says which inferences
   moved and how much of the movement was whole-submission scaling rather than dispatches
   disagreeing. That is the one thing here that is genuinely not available elsewhere, and after the
   rho = 0.919 correction it is the only claim of novelty this module makes.

Point 3 previously read *"two statistics computed over two frames, neither derived from the other,
agreeing on the same nine runs is itself evidence about the nine runs."* It is struck. Agreement
between two statistics that correlate at rho = 0.919 is evidence about the statistics, not about
the runs.

It also refuses **my own** ``packed`` and ``packed2`` A/B traces at 53.4% and 53.2%, which is the
right outcome and a reminder that a guard whose author is exempt from it is not a guard.

R13
---
A trace this module cannot parse, or one with too few repetitions or an unstable dispatch count, is
``ERROR(instrument=...)``. It is never a pass and never a detection.
"""

from __future__ import annotations

import statistics
from pathlib import Path

# Placed in the empty gap between the two populations of the 28-trace census (10.507% .. 35.313%),
# near its geometric centre. Moving it anywhere inside that gap changes no verdict on the census.
DISTURBANCE_RSD_MAX = 0.20

# Below this many repetitions the RSD of a single ordinal is not a statistic.
MIN_REPETITIONS = 3

# The fraction of inferences that must carry the modal dispatch count. A run whose dispatch count
# wanders is not repeating identical work, and comparing ordinal k across it compares different
# nodes -- an instrument error, not a disturbance.
MIN_WIDTH_AGREEMENT = 0.75


class InstrumentError(RuntimeError):
    """R13: this module failed to reach its observation. Not a pass, not a detection."""


def _rsd(xs: "list[float]") -> "float | None":
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    mean = statistics.fmean(xs)
    return (statistics.stdev(xs) / mean) if mean else None


def per_inference_kernel_us(trace: Path) -> "list[list[float]]":
    """Every kernel duration, grouped by inference, in dispatch order.

    Uses :func:`phases.attribute_gpu_ordinally`'s own cursor walk -- spans are attached to
    submissions by order and count, never by timestamp, because a device-lane ``ts`` carries an
    anchor uncertainty that reaches 314 ms on the Intel part.
    """
    import phases  # local: keeps this module importable without the bench package on sys.path

    events = phases.load(trace)
    subs = phases.subgraph_spans(events)
    gpus = phases.gpu_spans(events)
    if not subs or not gpus:
        raise InstrumentError(
            f"{Path(trace).name}: {len(subs)} subgraph spans, {len(gpus)} gpu spans")
    out, cursor = [], 0
    for s in subs:
        n = s.get("nodes")
        if not isinstance(n, int) or n < 0:
            continue
        take = gpus[cursor:cursor + n]
        cursor += len(take)
        durations = [g["gpu_ns"] / 1000.0 for g in take if g.get("gpu_ns") is not None]
        if durations:
            out.append(durations)
    if not out:
        raise InstrumentError(f"{Path(trace).name}: no kernel spans attributed to any inference")
    return out


def measure(inferences: "list[list[float]]") -> dict:
    """Same-ordinal dispersion over a run. Takes durations, not a path, so it is testable.

    **This is correlated with whole-series per-inference RSD at rho = 0.919** and is not a second
    opinion about the run. See the module docstring, and use :func:`localise` for the part that is
    not redundant.
    """
    if len(inferences) < MIN_REPETITIONS:
        raise InstrumentError(
            f"{len(inferences)} inferences; need >= {MIN_REPETITIONS} repetitions before the "
            "dispersion of a single ordinal is a statistic")

    widths = [len(x) for x in inferences]
    modal = statistics.mode(widths)
    full = [x for x in inferences if len(x) == modal]
    agreement = len(full) / len(inferences)
    if agreement < MIN_WIDTH_AGREEMENT:
        raise InstrumentError(
            f"only {agreement:.0%} of inferences dispatch the modal {modal} kernels "
            f"(need >= {MIN_WIDTH_AGREEMENT:.0%}); ordinal k is not the same node across this "
            "run, so comparing it compares different work")
    if len(full) < MIN_REPETITIONS:
        raise InstrumentError(
            f"only {len(full)} inferences carry the modal dispatch count; "
            f"need >= {MIN_REPETITIONS}")

    ordinals = [r for r in (_rsd([f[k] for f in full]) for k in range(modal)) if r is not None]
    if not ordinals:
        raise InstrumentError("no ordinal yielded a dispersion value")
    ordinals_sorted = sorted(ordinals)
    return {
        "same_ordinal_rsd_median": statistics.median(ordinals),
        "same_ordinal_rsd_p90": ordinals_sorted[int(0.9 * (len(ordinals_sorted) - 1))],
        "same_ordinal_rsd_max": ordinals_sorted[-1],
        "dispatches_per_inference": modal,
        "repetitions": len(full),
        "inferences_seen": len(inferences),
        "width_agreement": agreement,
        "frame": ("whole run, all inferences. NOT the tail's published suffix -- the statistic has "
                  "no discriminating power there (see module docstring)."),
    }


def classify(measurement: dict, threshold: float = DISTURBANCE_RSD_MAX) -> dict:
    """``PASS`` / ``FAIL(condition=RUN_DISTURBED)``, in ``check_device_state``'s vocabulary."""
    med = measurement["same_ordinal_rsd_median"]
    common = {
        "same_ordinal_rsd_median": med,
        "threshold": threshold,
        "margin_x": (med / threshold) if threshold else None,
        "detects": "non-stationarity across repetitions of identical work",
        "does_not_detect": ("a uniformly slow run. This statistic is blind to level; a run in "
                            "which every dispatch is equally inflated passes. PASS means "
                            "'repetitions agreed', never 'the machine was quiet'. The device-state "
                            "record (obligation 8) is the bias-sensitive companion and is still "
                            "required."),
    }
    if med > threshold:
        return {
            "verdict": "FAIL",
            "condition": "RUN_DISTURBED",
            "detail": (
                f"same-ordinal RSD {med:.2%} exceeds {threshold:.0%}: repetitions of identical "
                f"work disagree by {med / threshold:.2f}x the bar. The run was not stationary, so "
                "any suffix cut from it is a quiet window inside a disturbed run rather than a "
                "steady state. No timing figure from this run is publishable."),
            **common,
        }
    return {
        "verdict": "PASS",
        "condition": None,
        "detail": (
            f"same-ordinal RSD {med:.2%} is within {threshold:.0%}: across "
            f"{measurement['repetitions']} repetitions, each of the "
            f"{measurement['dispatches_per_inference']} dispatches agreed with itself. The run was "
            "stationary. This says nothing about whether it was fast, correct, or alone on the "
            "device."),
        **common,
    }


# --------------------------------------------------------------------------------------------
# Localisation. The part that is NOT redundant with whole-series RSD.
# --------------------------------------------------------------------------------------------

def localise(inferences: "list[list[float]]") -> dict:
    """Decompose a run's dispersion into *between-inference* and *within-inference* parts.

    Whole-series RSD says the run moved. This says **what** moved:

    * ``inference_inflation`` — each inference's total against the run's median total, so a reader
      can see *which repetitions* were hit rather than only that some were.
    * ``level_normalised_ordinal_rsd`` — same-ordinal RSD recomputed after dividing each inference
      by its own mean dispatch duration, i.e. after removing whole-submission scaling.
    * ``explained_by_level`` — how much of the raw same-ordinal dispersion that removal accounts
      for. This is Niobe's mechanism as a number: on the disturbed traces it runs 84–87%.

    **This is a decomposition, not a second opinion, and I checked before saying otherwise.**
    ``level_normalised_ordinal_rsd`` still correlates with whole-series RSD at Spearman **0.710**
    (log-log r = 0.885) over the 29-trace device-0 census — lower than the raw statistic's 0.919,
    but nowhere near independent. It is reported because it *separates two mechanisms*, not because
    it is new evidence:

    * ``ab_p1_long``  — whole 37.8%, normalised **3.9%** (88.9% explained). Entirely
      submission-level: the dispatches inside each inference agreed. A power/clock or queueing
      event, not a per-kernel one.
    * ``contended3``  — whole 34.4%, normalised **50.4%** (-22.7% explained). The dispatches
      disagree *more* than the totals do. Foreign work interleaved between dispatches.

    Those two look almost identical to whole-series RSD and are not the same condition. Telling
    them apart is the whole remaining value of this module.

    It inherits the level-blindness hole in full: normalising by per-inference level *removes* level
    on purpose, so a uniformly slow run is if anything even more invisible here. Nothing in this
    function can see a bias that scales the series.
    """
    m = measure(inferences)  # reuse every instrument gate, so failures stay ERROR not silence
    modal = m["dispatches_per_inference"]
    full = [x for x in inferences if len(x) == modal]

    totals = [sum(f) for f in full]
    med_total = statistics.median(totals)
    inflation = [(t / med_total) if med_total else float("nan") for t in totals]

    normalised = []
    for f in full:
        mean_k = statistics.fmean(f)
        normalised.append([v / mean_k for v in f] if mean_k else list(f))

    norm_ord = [r for r in (_rsd([f[k] for f in normalised]) for k in range(modal))
                if r is not None]
    if not norm_ord:
        raise InstrumentError("no ordinal yielded a level-normalised dispersion value")
    norm_med = statistics.median(norm_ord)
    raw_med = m["same_ordinal_rsd_median"]
    explained = (1.0 - norm_med / raw_med) if raw_med else None

    worst = sorted(range(len(inflation)), key=lambda i: inflation[i], reverse=True)[:5]
    return {
        "same_ordinal_rsd_median": raw_med,
        "level_normalised_ordinal_rsd": norm_med,
        "explained_by_level": explained,
        "inference_inflation_max": max(inflation),
        "inference_inflation_min": min(inflation),
        "inferences_over_1_10x": sum(1 for r in inflation if r > 1.10),
        "worst_inferences": [{"index": i, "inflation_x": inflation[i]} for i in worst],
        "character": _character(explained),
        "caveat": ("A decomposition of the dispersion, not an independent opinion: the normalised "
                   "statistic still correlates with whole-series RSD at Spearman 0.710. And like "
                   "every dispersion measure it is blind to a bias that scales the whole series."),
    }


def _character(explained: "float | None") -> str:
    """Name the mechanism the decomposition points at, derived rather than asserted."""
    if explained is None:
        return "UNDETERMINED(no raw dispersion to decompose)"
    if explained >= 0.60:
        return ("SUBMISSION_LEVEL: most of the dispersion is whole-inference scaling — the "
                "dispatches inside an inference agreed with each other. Consistent with a "
                "clock/power excursion or queueing delay ahead of the submission.")
    if explained <= 0.20:
        return ("PER_DISPATCH: removing per-inference level barely reduces the dispersion — "
                "individual dispatches disagree on their own. Consistent with foreign work "
                "interleaved between dispatches.")
    return ("MIXED: both whole-inference scaling and per-dispatch disagreement are present in "
            "comparable measure.")


# --------------------------------------------------------------------------------------------
# Synthetic controls. These construct the conditions the guard claims to separate, so that both
# arms are demonstrated from known ground truth rather than from traces we merely believe in.
# --------------------------------------------------------------------------------------------

def synthetic_clean(n_inferences: int = 20, n_kernels: int = 100) -> "list[list[float]]":
    """Heterogeneous kernels, perfectly repeatable. The guard must PASS."""
    shapes = [50.0, 120.0, 300.0, 800.0, 1500.0]
    one = [shapes[k % len(shapes)] for k in range(n_kernels)]
    return [list(one) for _ in range(n_inferences)]


def synthetic_uniform_slowdown(factor: float = 2.0, **kw) -> "list[list[float]]":
    """Every dispatch equally inflated for the whole run -- the hole in this guard, made explicit.

    A competing process taking a fixed share of the device for the entire run produces this. The
    guard **passes** it, and the accompanying test asserts that it does. Anyone reading a PASS as
    "the machine was quiet" is reading it wrong, and this function is the counter-example.
    """
    return [[d * factor for d in inf] for inf in synthetic_clean(**kw)]


def synthetic_jittered(amplitude: float = 0.5, **kw) -> "list[list[float]]":
    """Alternate inferences inflated -- the condition the guard exists to catch. Must FAIL."""
    base = synthetic_clean(**kw)
    return [[d * (1.0 + amplitude) if i % 2 else d for d in inf] for i, inf in enumerate(base)]
