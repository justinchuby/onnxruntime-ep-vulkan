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
        if raw:
            d["samples_ms"] = [_round(s) for s in self.samples]
        return d

    def summary(self) -> str:
        flag = "  ⚠ noisy" if self.noisy else ""
        return (
            f"{self.median:8.3f} ms  ±{self.mad:6.3f} (MAD, rsd {self.rsd * 100:4.1f}%)  "
            f"[p05 {self.p05:7.3f} · p95 {self.p95:7.3f}]  n={self.n}{flag}"
        )


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
