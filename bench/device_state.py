"""The device-state companion a GPU-clock figure must carry before it may be quoted.

WHY THIS FILE EXISTS — AND WHY IT IS NOT A DIAGNOSTIC
=====================================================

`phases.gpu_steady_tail` is a **variance test over a suffix, and it cannot see a bias.** Switch
measured both failure modes on this board, and they are reproducible from committed artifacts by
``python bench/results/probe_gputenancy.py`` with no GPU present:

===========================================  =========  ==========  =============
run                                          verdict    median      error
===========================================  =========  ==========  =============
``soloA``       SOLE_TENANT                  STEADY     11.525 ms   —
``contended3``  truncated, 3 foreign jobs    STEADY     126.647 ms  **10.99x**
``base_b``      board pinned at idle clock   STEADY     246.735 ms  **21.4x**
===========================================  =========  ==========  =============

**In both failures the wrong number carried the better RSD than the right one** — 0.79% and 0.12%
against the correct run's 0.81%. This board idles at **210 MHz against a 3105 MHz boost**, and a run
held the whole way at idle clock is *perfectly steady*, so it produces the gate's **most confident
possible verdict**. A low clock does not raise RSD; it lowers it.

That is DESIGN.md **R9 rule 5, the anti-correlated falsifier**: where a check's confidence measure
is computed from the same series as the quantity it certifies, and it moves the *same* way as the
reader's confidence when the quantity is wrong, **no threshold repairs it** — tightening the RSD bar
admits *more* of the failure, not less. Such a check is demoted from a gate to a **precondition**,
and the claim is ``UNMEASURED`` until a second quantity, **from outside the series**, records the
state of the thing being measured.

So this module is not a diagnostic to consult when a number looks odd. It is a **required
companion**: a tail with no device-state record beside it, taken over the same window, is
``UNCERTIFIED`` and is not a number. **Absence of a check is a refusal, not a default green.**

I ALSO HAVE TO SAY WHAT THIS INSTRUMENT CANNOT SEE
==================================================

R9's silence clause, and rule 5 exists because this project skipped it once already. The companion's
own silence set is written into every record it produces (``silence_set`` below), because a caveat
that lives in the docs and not in the artifact does not travel with the number.

The sampler is Switch's ``bench/results/probe_gpustate.py``, imported and not re-implemented: it
owns the ``nvidia-smi`` parsing, the ancestry test that stops our own worker being counted as a
stranger, and the R13 error states. A second sampler here would be a second description of one
channel — the mistake this project refused for the verdict vocabulary and again for the worker's
stderr decoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"

#: Fraction of the board's own advertised maximum SM clock that a run must reach **at some point**
#: in its window. Chosen from the specimens, which separate by an order of magnitude and not by a
#: hair: every correctly-clocked run peaked at 2280-2490 MHz (73-80% of 3105), and both wrong ones
#: never left 210 MHz (6.8%). The *median* clock does not discriminate — ``after_coldboard`` is a
#: correct 11.524 ms run whose median sample is also 210 MHz, because the board idles between
#: inferences and the sampler runs for the whole command. The peak is what separates a board that
#: boosted from a board that never did.
BOOST_FLOOR = 0.5

#: Terminal verdicts. Three states, R13, and the instrument's own failure is one of them.
QUOTABLE = "QUOTABLE"
WITHHELD = "WITHHELD"
UNCERTIFIED = "UNCERTIFIED"
ERROR = "ERROR"

SILENCE_SET = [
    "Samples the board at 4 Hz over the whole command, so it characterises the *regime* a run sat "
    "in and cannot resolve a single inference. A boost that happened while our kernel was not "
    "submitting would satisfy the peak-clock floor.",
    "Reads the driver's own account. A clock the driver misreports is invisible to it.",
    "`nvidia-smi` is NVIDIA-only. On any other vendor the tenancy and clock record is "
    "UNOBSERVABLE — which is not SOLE_TENANT and not a pass (R12).",
    "Says nothing about host contention. That is `bench/contention.py`'s subject and it gates a "
    "different quantity.",
]


def _probe_module():
    """Switch's ``bench/results/probe_gpustate.py``, imported rather than re-implemented."""
    if str(_RESULTS) not in sys.path:
        sys.path.insert(0, str(_RESULTS))
    try:
        import probe_gpustate  # type: ignore
    except Exception:  # pragma: no cover - environment dependent
        return None
    return probe_gpustate


class Companion:
    """Samples device state for the duration of a window, and reports what it saw.

    Used around the traced pass, so the record covers **the same window** as the tail it will
    certify. A record taken at another time is a record about another run; the whole point of the
    companion is that it observed *this* one.
    """

    def __init__(self, board_index: int = 0, vendor_is_nvidia: bool = True) -> None:
        self.board_index = board_index
        self.vendor_is_nvidia = vendor_is_nvidia
        self._sampler = None
        self._mod = None
        self._error: "str | None" = None
        self._unobservable: "str | None" = None

    def start(self) -> "Companion":
        if not self.vendor_is_nvidia:
            self._unobservable = ("caller declared this device non-NVIDIA, and `nvidia-smi` reads "
                                  "NVIDIA boards only")
            return self
        self._mod = _probe_module()
        if self._mod is None:
            self._error = ("bench/results/probe_gpustate.py is not importable, so the companion "
                           "instrument does not exist on this host")
            return self
        try:
            self._mod._sample_once(self.board_index)
        except Exception as exc:
            # Two very different failures wear the same exception, and R12/R13 classify them
            # differently. `nvidia-smi` absent = the instrument we meant to use is missing, which
            # is ERROR(instrument). `nvidia-smi` present but unable to report on this board =
            # the event this counter measures cannot occur in its frame, which is UNOBSERVABLE.
            # The Intel Iris Xe is the second case, and calling it ERROR would have filed a
            # permanent property of the device as a transient fault of the harness.
            msg = str(exc)
            if "not on PATH" in msg:
                self._error = f"the companion instrument is absent: {msg}"
            else:
                self._unobservable = (f"`nvidia-smi` is installed but cannot report on board "
                                      f"index {self.board_index} ({msg.strip()}), so this device "
                                      f"is outside the instrument's frame")
            return self
        self._sampler = self._mod.Sampler(self.board_index)
        self._sampler.start()
        return self

    def own_root(self, pid: int) -> None:
        """Tell the sampler which PID is ours, so our own worker is not counted as a stranger.

        Must be called **while the child is still alive**: ancestry is checked live, and a worker
        that has exited cannot be interrogated. See ``phi35._run_worker``'s ``on_start``.
        """
        if self._sampler is not None:
            self._sampler.own_root = pid

    def stop(self) -> dict:
        """Return the device-state record for the window, in every case a *record*, never None."""
        if self._unobservable is not None:
            return {
                "verdict": "UNOBSERVABLE",
                "reason": (f"{self._unobservable}. R12: a counter whose event cannot occur in its "
                           f"frame reports UNOBSERVABLE, never SOLE_TENANT and never a pass."),
                "silence_set": SILENCE_SET,
            }
        if self._sampler is None:
            return {"verdict": "ERROR(instrument)", "reason": self._error or "sampler never started",
                    "silence_set": SILENCE_SET}
        self._sampler.stop.set()
        self._sampler.join(timeout=10.0)
        if self._sampler.error:
            return {"verdict": "ERROR(instrument)",
                    "reason": f"sampling failed mid-run: {self._sampler.error}",
                    "silence_set": SILENCE_SET}
        if not self._sampler.samples:
            return {"verdict": "ERROR(instrument)",
                    "reason": "the sampler produced no samples over the window",
                    "silence_set": SILENCE_SET}
        rec = self._mod.summarise(self._sampler.samples)
        rec["silence_set"] = SILENCE_SET
        rec["sampler"] = "bench/results/probe_gpustate.py (Switch)"
        return rec


def certify(tail: dict, state: "dict | None") -> dict:
    """Decide whether a ``gpu_steady_tail`` result may be quoted, given the device state.

    Four terminal verdicts and only one of them releases a number:

    ``QUOTABLE``
        the tail is ``STEADY`` **and** a device-state record over the same window says the board
        was sole tenant and reached its boost clock.
    ``WITHHELD``
        a condition was found — foreign GPU work, or a board that never left its idle clock. This
        is a *detection*.
    ``UNCERTIFIED``
        no device-state record, or one from an instrument that cannot observe this vendor. **Not a
        detection and not a pass**: the check did not run, so the figure is `UNMEASURED`.
    ``ERROR``
        the companion itself failed. R13: never a finding about contention or clocks.
    """
    verdict = (tail or {}).get("verdict")
    out = {
        "quotable": False,
        "tail_verdict": verdict,
        "companion": (state or {}).get("verdict"),
        "silence_set": SILENCE_SET,
        "basis": ("DESIGN.md R9 rule 5: an RSD over a suffix is silent about the level of that "
                  "suffix, so a device-clock figure is UNMEASURED until a second quantity from "
                  "outside the series records the state of the device."),
    }
    if state is None:
        out.update(verdict=UNCERTIFIED,
                   detail=("no device-state record accompanies this tail. The tenancy verdict and "
                           "the SM-clock record are a required companion, not a diagnostic: "
                           "`STEADY` was measured at 10.99x and at 21.4x wrong, both times with a "
                           "better RSD than the correct run. No figure is quotable."))
        return out
    companion = str(state.get("verdict") or "")
    if companion.startswith("ERROR"):
        out.update(verdict=ERROR,
                   detail=(f"ERROR(instrument=device_state): {state.get('reason')}. R13: an "
                           "instrument error is never a detection — this says nothing about "
                           "whether the board was contended, and the figure stays unquotable."))
        return out
    if companion == "UNOBSERVABLE":
        out.update(verdict=UNCERTIFIED,
                   detail=(f"{state.get('reason')} The tail is therefore UNCERTIFIED on this "
                           "device; it is not refused for a condition, it is unmeasured."))
        return out
    peak = (state.get("sm_mhz") or {}).get("max")
    board_max = state.get("sm_max_mhz")
    reasons = []
    if companion == "FOREIGN_GPU_WORK":
        reasons.append(
            f"another process held the GPU in {state.get('foreign_sample_fraction', 0):.0%} of "
            f"samples (pids {list((state.get('foreign_pids') or {}).keys())}). A steady figure "
            f"taken then is a precise measurement of a contended device: `contended3` truncated "
            f"read STEADY at 126.647 ms, 10.99x wrong, RSD 0.79%.")
    if peak is not None and board_max:
        if peak < BOOST_FLOOR * board_max:
            reasons.append(
                f"the board never left its idle clock: peak SM {peak:.0f} MHz against a "
                f"{board_max:.0f} MHz maximum ({peak / board_max:.0%}, floor "
                f"{BOOST_FLOOR:.0%}). A run held at idle clock is perfectly steady and produces "
                f"the gate's most confident verdict: `base_b` read STEADY at 246.735 ms, 21.4x "
                f"wrong, RSD 0.12%.")
    else:
        out.update(verdict=ERROR,
                   detail=("ERROR(instrument=device_state): the record carries no SM-clock series, "
                           "so the idle-clock regime cannot be excluded. Not a detection."))
        return out
    out["peak_sm_mhz"] = peak
    out["sm_max_mhz"] = board_max
    out["clock_ramp_x"] = state.get("clock_ramp_x")
    if reasons:
        out.update(verdict=WITHHELD, detail=" ".join(reasons))
        return out
    if verdict != "STEADY":
        out.update(verdict=UNCERTIFIED,
                   detail=(f"the device state is clean (sole tenant, peak {peak:.0f} of "
                           f"{board_max:.0f} MHz) but the tail itself is {verdict}. The companion "
                           f"cannot make a number out of a run that produced none."))
        return out
    out.update(
        quotable=True,
        verdict=QUOTABLE,
        detail=(f"sole tenant over {state.get('n')} samples across {state.get('seconds')} s, and "
                f"the board reached {peak:.0f} MHz of its {board_max:.0f} MHz maximum "
                f"({peak / board_max:.0%}). The tail is STEADY at {tail.get('median_ms')} ms over "
                f"n={tail.get('n')} ({tail.get('coverage')} coverage). Quotable **with its "
                f"companion attached** — the two travel together or neither does."),
    )
    return out
