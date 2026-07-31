"""Sample statistics for latency measurements — the "and a variance number" half of the charter.

A single number is not a measurement. Every timing this harness reports is a
:class:`Sample`: median, spread, sample count, and the raw samples when asked for.

**Why median and MAD, not mean and standard deviation.** Latency distributions on a real
machine are right-skewed and contaminated: a scheduler preemption, a driver page fault, or a
GPU clock ramp adds a long tail that the mean absorbs and the median rejects. The median
absolute deviation (MAD) is the matching robust spread — one outlier can move a standard
deviation arbitrarily far, but moves the MAD by nothing until it crosses the middle of the
sample. Both are reported because they answer different questions:

* ``median`` — the typical run, the number to compare across builds.
* ``mad`` and ``iqr`` — how repeatable this measurement is *on this machine right now*. If the
  spread is wide, a 10% difference between two builds means nothing and the comparison must
  say so rather than colour it red.
* ``p05``/``p95`` — the tail. On a GPU, the tail is often the interesting part (first-touch
  page faults, clock ramp, another process on the device).
* ``minimum`` — the closest thing to "the machine's best effort with no interference". Useful
  as a sanity bound; never reported alone, because a benchmark that reports the minimum is
  measuring the luckiest run rather than the expected one.

``rsd`` (robust relative spread, ``1.4826 * mad / median``) is the single number used to decide
whether two medians can be meaningfully compared at all. 1.4826 is the constant that makes MAD
a consistent estimator of the standard deviation for normally distributed data, so ``rsd`` is
readable on the same scale as a coefficient of variation without inheriting its fragility.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

#: MAD → σ consistency constant for a normal distribution.
MAD_TO_SIGMA = 1.4826

#: Above this robust relative spread, a run is too noisy for a build-to-build comparison to
#: mean anything. Chosen to be visible rather than tuned: it is a prompt to re-measure, and it
#: is reported, never silently applied.
NOISY_RSD = 0.10


@dataclass
class Sample:
    """A set of timing samples in milliseconds, summarised robustly."""

    name: str
    samples: "list[float]" = field(default_factory=list)
    #: Wall time of the whole loop that produced ``samples``, including warmup, from the same
    #: ``perf_counter``. Used as the independent whole a trace decomposition is checked against
    #: (R11); ``None`` when the sample was not produced by a timed loop.
    loop_wall_ms: "float | None" = None

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else float("nan")

    @property
    def minimum(self) -> float:
        return min(self.samples) if self.samples else float("nan")

    @property
    def mad(self) -> float:
        """Median absolute deviation."""
        if not self.samples:
            return float("nan")
        med = self.median
        return statistics.median([abs(x - med) for x in self.samples])

    @property
    def iqr(self) -> float:
        if len(self.samples) < 4:
            return float("nan")
        ordered = sorted(self.samples)
        return _quantile(ordered, 0.75) - _quantile(ordered, 0.25)

    @property
    def p05(self) -> float:
        return _quantile(sorted(self.samples), 0.05) if self.samples else float("nan")

    @property
    def p95(self) -> float:
        return _quantile(sorted(self.samples), 0.95) if self.samples else float("nan")

    @property
    def rsd(self) -> float:
        """Robust relative spread: ``1.4826 * mad / median``. Unitless."""
        med = self.median
        if not self.samples or med <= 0:
            return float("nan")
        return MAD_TO_SIGMA * self.mad / med

    @property
    def noisy(self) -> bool:
        rsd = self.rsd
        return rsd == rsd and rsd > NOISY_RSD  # NaN-safe: NaN != NaN

    def to_dict(self, *, raw: bool = False) -> dict:
        d = {
            "name": self.name,
            "n": self.n,
            "median_ms": _round(self.median),
            "min_ms": _round(self.minimum),
            "mad_ms": _round(self.mad),
            "iqr_ms": _round(self.iqr),
            "p05_ms": _round(self.p05),
            "p95_ms": _round(self.p95),
            "rsd": _round(self.rsd, 4),
            "noisy": self.noisy,
        }
        if self.loop_wall_ms is not None:
            d["loop_wall_ms"] = _round(self.loop_wall_ms)
        if raw:
            d["samples_ms"] = [_round(s) for s in self.samples]
        return d

    def summary(self) -> str:
        flag = "  ⚠ noisy" if self.noisy else ""
        return (
            f"{self.median:8.3f} ms  ±{self.mad:6.3f} (MAD, rsd {self.rsd * 100:4.1f}%)  "
            f"[p05 {self.p05:7.3f} · p95 {self.p95:7.3f}]  n={self.n}{flag}"
        )


#: Above this half-to-half ratio a run has drifted rather than jittered, and its median is not
#: describing a steady state. Deliberately generous: this fires on a 25% shift between the first
#: and second half of a run, which no amount of ordinary jitter produces.
DRIFT_RATIO = 1.25


def drift(samples: "list[float]") -> dict:
    """Detect a run that is trending rather than jittering.

    Spread does not distinguish "noisy around a stable value" from "moving steadily in one
    direction", and the two demand opposite responses: the first is measured by taking more
    samples, the second is *invalidated* by them. On this project's Intel part the first Phi-3.5
    run went 724 → 903 → 1447 → 2080 → 2669 ms before flattening near 2780 — a monotone
    **degradation** over the first six iterations, the opposite direction from a warmup ramp.
    A median over those samples reports a number the device never sustains, and a *minimum*
    reports one it only ever achieved while cold; both look reasonable and both are wrong.

    Returns the two half-medians, their ratio, and the fraction of consecutive steps that move
    in the same direction. ``steady`` is false when either the halves disagree by more than
    :data:`DRIFT_RATIO` or the run is near-monotone, and a caller that reports a single number
    from a non-steady run is reporting an artefact of when it stopped measuring.
    """
    n = len(samples)
    out: dict = {"n": n}
    if n < 4:
        out.update(steady=None, reason="too few samples to test for drift")
        return out
    half = n // 2
    a = statistics.median(samples[:half])
    b = statistics.median(samples[-half:])
    ratio = (b / a) if a else float("nan")
    ups = sum(1 for x, y in zip(samples, samples[1:]) if y > x)
    monotone_fraction = max(ups, (n - 1) - ups) / (n - 1)
    steady = (ratio == ratio and (1 / DRIFT_RATIO) <= ratio <= DRIFT_RATIO
              and monotone_fraction < 0.85)
    out.update(
        first_half_median_ms=_round(a),
        second_half_median_ms=_round(b),
        ratio=_round(ratio, 4),
        monotone_fraction=_round(monotone_fraction, 4),
        steady=steady,
        direction=("slower" if ratio > 1 else "faster") if ratio == ratio else None,
        reason=None if steady else (
            f"the run drifted: first half median {a:.1f} ms, second half {b:.1f} ms "
            f"({ratio:.2f}x), {monotone_fraction:.0%} of steps in one direction. The median is "
            f"not describing a steady state — extend the warmup until it is, rather than "
            f"reporting a number that depends on when the measurement stopped."
        ),
    )
    return out


def _quantile(ordered: "list[float]", q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _round(x: float, digits: int = 4) -> float:
    return round(x, digits) if x == x else float("nan")  # NaN-safe


def comparable(a: Sample, b: Sample) -> bool:
    """Whether a difference between two medians can be read as a real difference.

    True only when both samples are individually repeatable. Two noisy samples can differ by a
    large percentage for no reason at all, and reporting that as a regression trains everyone
    to ignore the report.
    """
    return not a.noisy and not b.noisy


def relative_delta(base: Sample, new: Sample) -> float:
    """``(new - base) / base`` on medians. Positive means slower."""
    b, n = base.median, new.median
    if b <= 0 or b != b or n != n:
        return float("nan")
    return (n - b) / b


def significant(base: Sample, new: Sample, threshold: float) -> bool:
    """Whether a delta exceeds ``threshold`` *and* exceeds the samples' own noise.

    The second condition is what stops this harness from crying wolf: a 12% shift between two
    samples that each wobble by 15% is not a finding.
    """
    delta = abs(relative_delta(base, new))
    if delta != delta:
        return False
    noise = max(
        base.rsd if base.rsd == base.rsd else 0.0,
        new.rsd if new.rsd == new.rsd else 0.0,
    )
    return delta > threshold and delta > 2.0 * noise
