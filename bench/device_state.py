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

#: A **fifth** certification outcome, and it exists because a partial record is not the same thing
#: as a missing one.
#:
#: Obligation 8 names two contents — a tenancy verdict **and** a clock record. Windows' vendor-
#: neutral ``\GPU Engine`` counters produce the first on any WDDM adapter and the second on none
#: (``bench/win_gpu_counters.py``). Folding that into ``UNCERTIFIED`` would lose the fact that
#: half the obligation was met; folding it into ``SOLE_TENANT`` would let half the evidence
#: release a figure, which is the loophole amendment 2 was written against and a worse one than an
#: empty record, **because it looks like diligence**.
#:
#: This is Tank's five-state discipline applied to certification: *bypassed*, *all-rejected* and
#: *unobservable* were three different things sharing one ``0``, and "no companion", "half a
#: companion" and "a companion that could not run" are three different things that would otherwise
#: share one ``UNCERTIFIED``.
UNCERTIFIED_PARTIAL = "UNCERTIFIED(partial_companion)"

#: Companion-record verdicts for the two-axis form. A record is now *two* independent axes —
#: tenancy and clock — each with its own producer and its own verdict, because on this project
#: they already come from different places and on most platforms one of them comes from nowhere.
TENANCY_ONLY = "TENANCY_ONLY"
UNOBSERVABLE = "UNOBSERVABLE"
NO_PRODUCER = "NO_PRODUCER"

SILENCE_SET = [
    "Samples the board at 4 Hz over the whole command, so it characterises the *regime* a run sat "
    "in and cannot resolve a single inference. A boost that happened while our kernel was not "
    "submitting would satisfy the peak-clock floor.",
    "Reads the driver's own account. A clock the driver misreports is invisible to it.",
    "`nvidia-smi` is NVIDIA-only. On any other vendor the tenancy and clock record is "
    "UNOBSERVABLE — which is not SOLE_TENANT and not a pass (R12).",
    "Says nothing about host contention. That is `bench/contention.py`'s subject and it gates a "
    "different quantity.",
    "The clock axis has exactly one producer on this project and it is NVIDIA-only. No producer "
    "for it exists on Intel here at all (`docs/PERF.md` §16.3), so an Intel device-clock figure is "
    "structurally uncertifiable on this hardware — a permanent classification, not a pending one.",
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


def _windows_module():
    """The vendor-neutral tenancy producer, when this host has one."""
    try:
        import win_gpu_counters  # type: ignore
    except Exception:  # pragma: no cover - environment dependent
        return None
    return win_gpu_counters


# ---------------------------------------------------------------------------------------------
# The record is two axes, and it exists on every platform even where nothing can fill it.
#
# Morpheus's amendment 1 to obligation 8: the obligation names a **record, not a tool** — tenancy
# verdict, clock min/median/max, board maximum, over the statistic's own window — and any platform
# that can produce that content satisfies it. The corollary he did not need to write, and which
# this project now needs, is the other direction: **a platform with no producer still emits the
# record**, with `NO_PRODUCER` in the axes nothing can fill.
#
# That is what keeps the artifact portable. A reader on Linux, Android or macOS gets the same
# shape, the same field names and an explicit statement of which axes are empty and why — rather
# than a missing key, which is indistinguishable from a key nobody thought to write.
# ---------------------------------------------------------------------------------------------

def empty_axis(reason: str, *, producer: "str | None" = None,
               verdict: str = NO_PRODUCER) -> dict:
    """An axis of the record with nothing in it, saying so."""
    return {"verdict": verdict, "producer": producer, "reason": reason}


def compose(tenancy: "dict | None", clock: "dict | None", *, window: "dict | None" = None) -> dict:
    """Build a device-state record from its two axes, and give the pair a composite verdict.

    The composite is deliberately *not* the better of the two axes:

    ============================  ==========================  =========================
    tenancy axis                  clock axis                  composite
    ============================  ==========================  =========================
    ``SOLE_TENANT``               a clock series              ``SOLE_TENANT`` (full)
    ``SOLE_TENANT``               ``UNOBSERVABLE``/absent     ``TENANCY_ONLY``
    ``FOREIGN_GPU_WORK``          anything                    ``FOREIGN_GPU_WORK``
    ``UNOBSERVABLE(*)``           anything                    ``UNOBSERVABLE``
    ``ERROR(instrument)``         anything                    ``ERROR(instrument)``
    ============================  ==========================  =========================

    Note the third row, which is the asymmetry that makes a half record safe to have at all: a
    **detection** survives a missing clock axis, because foreign work observed is not unobserved
    by the absence of a clock reading. A **pass** does not. This instrument may subtract
    confidence and may never add it.
    """
    tenancy = tenancy or empty_axis("no tenancy producer ran for this window")
    clock = clock or empty_axis("no clock producer ran for this window")
    t_verdict = str(tenancy.get("verdict") or "")
    c_verdict = str(clock.get("verdict") or "")
    has_clock = bool((clock.get("sm_mhz") or {}).get("max")) and bool(clock.get("sm_max_mhz"))

    if t_verdict.startswith("ERROR"):
        composite = "ERROR(instrument)"
    elif t_verdict.startswith("UNOBSERVABLE") or t_verdict in (NO_PRODUCER, ""):
        composite = UNOBSERVABLE
    elif t_verdict.startswith("FOREIGN_GPU_WORK"):
        composite = "FOREIGN_GPU_WORK"
    elif has_clock:
        composite = "SOLE_TENANT"
    else:
        composite = TENANCY_ONLY

    record = {
        "verdict": composite,
        "tenancy": tenancy,
        "clock": clock,
        "window": window or {},
        "silence_set": SILENCE_SET + list(tenancy.get("silence_set") or []),
        "obligation": ("DESIGN.md §10.0 obligation 8 requires BOTH a tenancy verdict and a clock "
                       "record over the statistic's own window. This record states which of the "
                       "two it has."),
        "axes_present": {"tenancy": not t_verdict.startswith(("ERROR", "UNOBSERVABLE"))
                         and t_verdict != NO_PRODUCER,
                         "clock": has_clock},
    }
    if composite == TENANCY_ONLY:
        record["reason"] = (
            f"tenancy was observed ({t_verdict}) and the clock was not ({c_verdict or 'absent'}). "
            f"Half of obligation 8. It can refuse a figure and it cannot release one.")
    # Keep the flat fields the NVIDIA-era readers use, so one record serves both.
    for key in ("sm_mhz", "sm_max_mhz", "clock_ramp_x", "n", "seconds"):
        if key in clock:
            record[key] = clock[key]
    for key in ("foreign_sample_fraction", "foreign_pids", "foreign_gpu_seconds", "own_pids_on_gpu",
                "own_gpu_seconds", "process_names"):
        if key in tenancy:
            record[key] = tenancy[key]
    return record


def from_nvidia_record(state: dict) -> dict:
    """Recast a ``probe_gpustate.summarise`` record into the two-axis form.

    Both axes come from the same producer there — that is a property of `nvidia-smi`, not of the
    obligation, and writing it out this way is what makes an Intel record comparable in *shape*
    to an NVIDIA one without being comparable in *content*.
    """
    verdict = str(state.get("verdict") or "")
    tenancy = {"verdict": verdict, "producer": state.get("sampler") or "nvidia-smi",
               **{k: state[k] for k in ("foreign_sample_fraction", "foreign_pids",
                                        "own_pids_on_gpu") if k in state}}
    clock = {"verdict": "OBSERVED" if state.get("sm_mhz") else UNOBSERVABLE,
             "producer": state.get("sampler") or "nvidia-smi",
             **{k: state[k] for k in ("sm_mhz", "sm_max_mhz", "clock_ramp_x", "util_pct",
                                      "power_w") if k in state}}
    return compose(tenancy, clock,
                   window={"n": state.get("n"), "seconds": state.get("seconds")})


def _tenancy_agrees(a: "str | None", b: "str | None") -> "bool | None":
    """Whether two tenancy verdicts from different instruments say the same thing.

    ``None`` — not "they disagree" — when either instrument could not answer. Two instruments and
    one of them silent is not corroboration and is not a conflict; it is one instrument.
    """
    def cls(v: "str | None") -> "str | None":
        s = str(v or "")
        if s.startswith("FOREIGN_GPU_WORK"):
            return "foreign"
        if s == "SOLE_TENANT":
            return "sole"
        return None

    ca, cb = cls(a), cls(b)
    if ca is None or cb is None:
        return None
    return ca == cb


class Companion:
    """Samples device state for the duration of a window, and reports what it saw.

    Used around the traced pass, so the record covers **the same window** as the tail it will
    certify. A record taken at another time is a record about another run; the whole point of the
    companion is that it observed *this* one.
    """

    def __init__(self, board_index: int = 0, vendor_is_nvidia: bool = True,
                 device_name: "str | None" = None, allow_windows_tenancy: bool = True) -> None:
        self.board_index = board_index
        self.vendor_is_nvidia = vendor_is_nvidia
        #: Vulkan device name, used to join to a WDDM adapter LUID for the tenancy axis. Without
        #: it the vendor-neutral producer cannot be pointed at the right adapter, and pointing it
        #: at the wrong one produces a *clean* record — so no name means no fallback.
        self.device_name = device_name
        self.allow_windows_tenancy = allow_windows_tenancy
        self._sampler = None
        self._mod = None
        self._error: "str | None" = None
        self._unobservable: "str | None" = None
        self._win = None
        self._win_error: "str | None" = None

    def _start_windows_tenancy(self) -> None:
        """Start the vendor-neutral tenancy producer, if this host has one and we know the adapter.

        This is the half-companion. It runs when the clock producer cannot see this device — and
        also *alongside* it on NVIDIA when asked, because two independently-authored instruments
        agreeing on tenancy is the corroboration §10.0 obligation 7 asks for and the only kind of
        evidence that has caught anything on this project.
        """
        if not self.allow_windows_tenancy or self.device_name is None:
            return
        mod = _windows_module()
        if mod is None or not mod.available():
            self._win_error = ("no vendor-neutral tenancy producer on this host: "
                               "bench/win_gpu_counters.py needs Windows PDH and a WDDM adapter")
            return
        started = mod.observe(self.device_name)
        if isinstance(started, dict):
            self._win_error = str(started.get("reason") or "tenancy producer refused to start")
            return
        self._win = started

    def start(self) -> "Companion":
        self._start_windows_tenancy()
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
        """Tell the samplers which PID is ours, so our own worker is not counted as a stranger.

        Must be called **while the child is still alive**: ancestry is checked live, and a worker
        that has exited cannot be interrogated. See ``phi35._run_worker``'s ``on_start``.
        """
        if self._sampler is not None:
            self._sampler.own_root = pid
        if self._win is not None:
            self._win.own_root = pid

    def _stop_windows(self) -> "dict | None":
        """Stop the vendor-neutral tenancy producer and return its record, or ``None``."""
        mod = _windows_module()
        if self._win is None or mod is None:
            if self._win_error:
                return {"verdict": "ERROR(instrument)", "reason": self._win_error,
                        "producer": "bench/win_gpu_counters.py"}
            return None
        self._win.stop.set()
        self._win.join(timeout=20.0)
        return mod.summarise(self._win)

    def stop(self) -> dict:
        """Return the device-state record for the window, in every case a *record*, never None."""
        win = self._stop_windows()
        if self._unobservable is not None:
            if win is not None and not str(win.get("verdict", "")).startswith(("ERROR",
                                                                               "UNOBSERVABLE")):
                # The clock producer cannot see this device; the tenancy producer can. That is
                # exactly half of obligation 8, and `compose` gives it its own verdict rather
                # than letting it borrow either neighbour's.
                return compose(win, empty_axis(
                    f"{self._unobservable}. No clock producer exists for this device on this "
                    f"platform, so the clock axis has no producer at all — not a reading that "
                    f"failed, a producer that does not exist.",
                    producer=None, verdict=NO_PRODUCER))
            return {
                "verdict": "UNOBSERVABLE",
                "reason": (f"{self._unobservable}. R12: a counter whose event cannot occur in its "
                           f"frame reports UNOBSERVABLE, never SOLE_TENANT and never a pass."),
                "tenancy": win or empty_axis("no tenancy producer ran"),
                "clock": empty_axis(self._unobservable, verdict=NO_PRODUCER),
                "silence_set": SILENCE_SET,
            }
        if self._sampler is None:
            return {"verdict": "ERROR(instrument)", "reason": self._error or "sampler never started",
                    "tenancy": win or empty_axis("no tenancy producer ran"),
                    "silence_set": SILENCE_SET}
        self._sampler.stop.set()
        self._sampler.join(timeout=10.0)
        if self._sampler.error:
            return {"verdict": "ERROR(instrument)",
                    "reason": f"sampling failed mid-run: {self._sampler.error}",
                    "tenancy": win or empty_axis("no tenancy producer ran"),
                    "silence_set": SILENCE_SET}
        if not self._sampler.samples:
            return {"verdict": "ERROR(instrument)",
                    "reason": "the sampler produced no samples over the window",
                    "tenancy": win or empty_axis("no tenancy producer ran"),
                    "silence_set": SILENCE_SET}
        rec = self._mod.summarise(self._sampler.samples)
        rec["silence_set"] = SILENCE_SET
        rec["sampler"] = "bench/results/probe_gpustate.py (Switch)"
        if win is not None:
            # Obligation 7, the corroboration rule: two independently-authored instruments on the
            # same question, and whether they agreed is in the artifact rather than in someone's
            # memory of both.
            rec["corroboration"] = {
                "second_instrument": "bench/win_gpu_counters.py (WDDM \\GPU Engine)",
                "tenancy_here": rec.get("verdict"),
                "tenancy_there": win.get("verdict"),
                "agree": _tenancy_agrees(rec.get("verdict"), win.get("verdict")),
                "record": win,
            }
        return rec


def _axis_reason(state: dict) -> str:
    """The reason an axis is empty, wherever the record chose to put it."""
    for key in ("tenancy", "clock"):
        axis = state.get(key) or {}
        if axis.get("reason"):
            return str(axis["reason"])
    return "no reason recorded."


def _foreign_detail(state: dict) -> str:
    """Describe the foreign work, from whichever producer's fields are present."""
    frac = state.get("foreign_sample_fraction")
    pids = list((state.get("foreign_pids") or state.get("foreign_gpu_seconds") or {}).keys())
    names = state.get("process_names") or {}
    who = ", ".join(f"{p} ({names.get(str(p), '?')})" for p in pids) or "unidentified pids"
    seen = f"{frac:.0%} of samples" if isinstance(frac, (int, float)) else (
        (state.get("tenancy") or {}).get("verdict") or "the window")
    return (f"another process held the GPU in {seen} ({who}). A steady figure taken then is a "
            f"precise measurement of a contended device: `contended3` truncated read STEADY at "
            f"126.647 ms, 10.99x wrong, RSD 0.79%.")


def certify(tail: dict, state: "dict | None") -> dict:
    """Decide whether a ``gpu_steady_tail`` result may be quoted, given the device state.

    Five terminal verdicts and only one of them releases a number:

    ``QUOTABLE``
        the tail is ``STEADY`` **and** a device-state record over the same window says the board
        was sole tenant and reached its boost clock.
    ``WITHHELD``
        a condition was found — foreign GPU work, or a board that never left its idle clock. This
        is a *detection*, and it survives a missing clock axis.
    ``UNCERTIFIED``
        no device-state record, or one from an instrument that cannot observe this vendor. **Not a
        detection and not a pass**: the check did not run, so the figure is `UNMEASURED`.
    ``UNCERTIFIED(partial_companion)``
        a **half** companion: tenancy observed over the window, no clock record. Distinguished
        from `UNCERTIFIED` because "nobody looked" and "one of the two things was looked at" are
        different states, and from `QUOTABLE` because the missing half is the half that caught the
        21.4x error. It is never a pass.
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
    if companion.startswith("UNOBSERVABLE"):
        out.update(verdict=UNCERTIFIED,
                   detail=(f"{state.get('reason') or _axis_reason(state)} The tail is therefore "
                           "UNCERTIFIED on this device; it is not refused for a condition, it is "
                           "unmeasured."))
        return out
    peak = (state.get("sm_mhz") or {}).get("max")
    board_max = state.get("sm_max_mhz")
    reasons = []
    if companion.startswith("FOREIGN_GPU_WORK"):
        reasons.append(_foreign_detail(state))
    if companion == TENANCY_ONLY:
        # Half of obligation 8, and the half that cannot release a number. Note this is reached
        # only when the tenancy axis came back **clean** — `compose` routes an observed condition
        # to FOREIGN_GPU_WORK above, because a detection does not need the clock axis to stand.
        out.update(
            verdict=UNCERTIFIED_PARTIAL,
            half_companion=True,
            missing=["sm_clock"],
            tenancy_verdict=(state.get("tenancy") or {}).get("verdict"),
            detail=(
                f"a tenancy record covers this window and there is no clock record: "
                f"{(state.get('clock') or {}).get('reason') or 'no clock producer on this device'} "
                f"The tenancy half is real evidence and it is not the half that catches this "
                f"failure: the 21.4x-wrong run was **verified sole tenant** and was wrong because "
                f"the board never left 210 MHz, with the project's second-best RSD. Median clock "
                f"does not separate it either — 210 MHz is also the median of a correct run. "
                f"Half a companion may refuse a figure and may never release one, so this is "
                f"UNCERTIFIED and it is recorded as *partial* rather than *absent*, because "
                f"'nobody looked' and 'we looked at one of the two things' are different states "
                f"and only one of them is worth acting on."),
        )
        return out
    if peak is not None and board_max:
        if peak < BOOST_FLOOR * board_max:
            reasons.append(
                f"the board never left its idle clock: peak SM {peak:.0f} MHz against a "
                f"{board_max:.0f} MHz maximum ({peak / board_max:.0%}, floor "
                f"{BOOST_FLOOR:.0%}). A run held at idle clock is perfectly steady and produces "
                f"the gate's most confident verdict: `base_b` read STEADY at 246.735 ms, 21.4x "
                f"wrong, RSD 0.12%.")
    elif reasons:
        # A condition was found and the clock axis is missing. The detection stands: no clock
        # reading would un-observe the foreign work. Withheld, and the record says which axis
        # was empty so nobody mistakes this for a full companion.
        out.update(verdict=WITHHELD, half_companion=True, missing=["sm_clock"],
                   detail=" ".join(reasons) + " (No clock record accompanied this window; the "
                                              "detection does not need one — a missing clock "
                                              "reading cannot unobserve foreign work.)")
        return out
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
