"""``run_disturbance`` — was this run disturbed, measured in-band, without a clock comparison.

What this detects, stated before anything else
-----------------------------------------------
**Non-stationarity across repetitions of identical work.** Nothing more. It is a dispersion
statistic, so it sees *variation* and is blind to *level*. Read the "What it cannot see" section
before quoting it, because the obvious misreading -- "the guard passed, so the machine was quiet"
-- is wrong and the module ships a control that proves it wrong.

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

Why it is not the same statistic as the per-inference spread
--------------------------------------------------------------
Per-inference GPU-busy RSD (what ``gpu_steady_tail`` operates on) is a **sum** over ~355 kernels,
and it conflates two different conditions:

* **drift** — the whole run trending, e.g. a warm-up ramp. Legitimate, expected, and the reason the
  tail discards a prefix.
* **jitter** — repetitions of one dispatch disagreeing. Not legitimate.

Same-ordinal RSD isolates the second. Two committed traces make the distinction concrete: at
essentially the same per-inference spread (``baseline`` 10.36%, ``ab_p0_r2`` 10.97%) their
same-ordinal RSD differs by 2.75x (10.51% vs 3.82%). One drifted; the other jittered. A guard built
on the per-inference spread could not tell them apart.

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

Threshold, and how honestly it separates
------------------------------------------
Placed by census over all 28 committed device-0 traces, not by fitting the two traces the
investigation started from. As a population the statistic is sharply bimodal, with a gap that
contains no trace at all:

* 19 traces from 0.624% to **10.507%**
* a gap
* 9 traces from **35.313%** to 137.352%

``DISTURBANCE_RSD_MAX = 0.20`` sits in that gap, near its geometric centre (19.3%), 1.90x above the
highest undisturbed trace and 1.77x below the lowest disturbed one.

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

**And here is where it does not separate, which matters more than where it does.** The two
populations above are *not* the STEADY/refused populations. Against the tail's verdict there is
substantial overlap: publishing traces run 0.624%–10.507% and refused traces run 3.694%–137.352%.
Same-ordinal RSD does not predict whether the tail will publish, and must not be read as doing so.
What it predicts is whether the run was stationary, and the nine it flags are exactly the nine
traces carrying a >=30% per-inference spread -- two different statistics over two different frames
agreeing on the same nine runs, neither derived from the other.

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
3. Two statistics computed over two frames, neither derived from the other, agreeing on the same
   nine runs is itself evidence about the nine runs.

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
    """Same-ordinal dispersion over a run. Takes durations, not a path, so it is testable."""
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
