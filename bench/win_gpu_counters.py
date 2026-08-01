"""A vendor-neutral **tenancy** producer for the device-state companion — and only tenancy.

WHAT THIS IS FOR
================

`DESIGN.md` §10.0 obligation 8 makes a device-clock figure quotable only when a device-state record
covers the statistic's own window, and it names two contents: **a tenancy verdict and a clock
record**. `bench/device_state.py`'s only producer is `nvidia-smi`, which supplies both — on NVIDIA.
On the Iris Xe the record is `UNOBSERVABLE`, so **no Intel device-clock figure has ever been
quotable**, and the open question on this project (the 4.39× of the Intel/NVIDIA kernel gap that
memory bandwidth does not explain) lives entirely on the device we cannot instrument.

Windows exposes two counters that are not vendor-locked to a GPU vendor:

    \\GPU Engine(pid_<pid>_luid_<hi>_<lo>_phys_<n>_eng_<n>_engtype_<name>)\\Running Time
    \\GPU Engine(...)\\Utilization Percentage

They are produced by the **WDDM scheduler**, not by a vendor tool, so both adapters on this desk
appear in them: `luid 0x00010AA0` = `Intel(R) Iris(R) Xe Graphics`, `luid 0x00010EE0` =
`NVIDIA GeForce RTX 4060 Laptop GPU`. That is the whole of the good news, and the next section is
the reason this module's name says `counters` and not `companion`.

WHAT IT WITNESSES, STATED AS A CAPABILITY AND NOT AS A HOPE
===========================================================

Measured on this box (``python bench/results/probe_wingpu.py``), not assumed:

* **It witnesses our own submissions.** A Vulkan compute run on the Iris Xe appears as engine
  Running Time accruing against *our worker's* PID on the Intel LUID, engine type ``3D`` — WDDM
  has no separate compute node on either adapter here, so compute is scheduled on the 3D node and
  ``engtype_Compute`` never appears. An instrument that cannot see our own work cannot see anyone
  else's either, so this is the precondition for the tenancy claim and it is checked, not assumed.
* **It witnesses other processes' submissions on the same adapter, per PID.** That is the tenancy
  question, and it is answered for **any** vendor whose driver is a WDDM driver.
* **It does not witness clock.** There is no frequency counter in any `GPU *` counter set on this
  system (`GPU Engine`, `GPU Adapter Memory`, `GPU Local/Non Local Adapter Memory`,
  `GPU Process Memory` — enumerated, none carries MHz), and no `root\\wmi` class exposes one.

**The third bullet is not a gap that a better parse would close, and this is the part that decides
what the module is allowed to do.** Running Time is a *duration*: the number of 100 ns ticks the
engine spent executing a process's work. When the board is held at its idle clock the same kernel
occupies the engine **longer**, so engine time moves the *same way* as the GPU-busy figure it would
be certifying. It is a second copy of the quantity under certification, taken through a different
API — **not a second quantity from outside the series**, which is exactly what §10.0.1 R9
amendment 5 requires. Feeding it into the clock half of obligation 8 would reproduce the
same-source falsifier one level up, which is the error that put 246.735 ms into the record with an
RSD of 0.12%.

So this module produces **one half of a companion, and says so in the record.** `device_state.py`
gives that half its own verdict (`TENANCY_ONLY`) and its own certification outcome
(`UNCERTIFIED(partial_companion)`); it never becomes `SOLE_TENANT` and never releases a figure.

WHICH WAY THIS CHECK MOVES WHEN ITS SUBJECT IS WRONG (R9 amendment 5)
=====================================================================

Amendment 5's question is not "is the check sound" but **"which way does it move when the thing it
watches is wrong?"** This one is safe for a reason worth stating explicitly, because it is the
reason a half companion may exist at all:

> A tenancy-only record can only ever move a figure **toward refusal**. `FOREIGN_GPU_WORK` found
> without a clock record is still a detection — the foreign work is observed, and no clock reading
> would unobserve it. `SOLE_TENANT` found without a clock record is **not** a pass, because the
> failure the clock record exists to catch is untouched by it. The asymmetry is the whole design:
> **this instrument may subtract confidence and may never add it.**

That is what stops a partial record from being a worse loophole than an empty one. A record that
could certify on half the evidence would *look* like diligence, and would be read as one.

WHAT IT CANNOT SEE
==================

Written into every record as ``silence_set``, because a caveat that lives in a docstring does not
travel with the number.
"""

from __future__ import annotations

import ctypes
import re
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

try:  # pragma: no cover - the import itself is the platform test
    import winreg
except ImportError:  # not Windows: the producer does not exist here, and that is a record, not a
    winreg = None  # crash. `available()` returns False and the tenancy axis is NO_PRODUCER.

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"

#: Instance names look like
#: ``pid_30232_luid_0x00000000_0x00010AA0_phys_0_eng_0_engtype_3D``.
_INSTANCE_RE = re.compile(
    r"pid_(?P<pid>\d+)_luid_(?P<luid_hi>0x[0-9A-Fa-f]+)_(?P<luid_lo>0x[0-9A-Fa-f]+)"
    r"_phys_(?P<phys>\d+)_eng_(?P<eng>\d+)_engtype_(?P<engtype>\w+)",
    re.IGNORECASE,
)

#: ``Running Time`` is reported in 100 ns units, the same unit WDDM keeps it in. Verified against
#: wall clock: a saturating run accrues engine ticks at very nearly ``1e7`` per second of engine
#: occupancy, and a fully idle adapter accrues none.
TICKS_PER_SECOND = 1e7

#: PID 4 is the Windows kernel (``System``). It accrues Copy-engine time doing paging on behalf of
#: *whichever* process caused a fault, so it is neither ours nor a stranger's in any useful sense.
#: Counting it as foreign would make the detector fire on every run, and a detector that always
#: fires is a constant, not a detector (Switch's ``_is_ours``, learned the same way). It is
#: reported in its own class and excluded from the tenancy verdict.
KERNEL_PIDS = frozenset({0, 4})

#: A foreign process must hold the adapter for at least this fraction of the window's wall time
#: before the window is called contended. Not a tuning knob for taste: below it sits the desktop's
#: own housekeeping, and above it sits work large enough to displace ours. Where the adapter drives
#: the display this threshold is *irrelevant* — see :data:`DISPLAY_TENANCY_IS_STRUCTURAL`.
FOREIGN_BUSY_FRACTION = 0.01

#: On a hybrid laptop the panel is wired to the integrated adapter, so the desktop compositor holds
#: it continuously. That is not a transient condition to wait out; it is a property of the machine.
#: The record says so rather than reporting a contended window every time.
DISPLAY_TENANCY_IS_STRUCTURAL = True

#: A window the sampler went blind in for longer than this multiple of its own interval is not a
#: window it can speak for. Set from the failure that produced it: ancestry resolution for every
#: instance on the adapter cost 21.2 s per round, so a 62 s window got **three** samples and the
#: record came back clean about a run it had not watched.
MAX_BLIND_GAP_FACTOR = 10.0

TENANCY_SOLE = "SOLE_TENANT"
TENANCY_FOREIGN = "FOREIGN_GPU_WORK"
TENANCY_STRUCTURAL = "FOREIGN_GPU_WORK(display)"
UNOBSERVABLE = "UNOBSERVABLE"
ERROR_INSTRUMENT = "ERROR(instrument)"

#: The window had an owner and **our own work never appeared on this adapter**. Measured on the
#: first run of ``probe_wingpu.py``: a mis-joined LUID (Vulkan *enumeration* index 1 is the NVIDIA
#: board; the EP's *selection* index 1 is the Iris Xe) pointed the sampler at an adapter nobody
#: touched, and the record came back a clean ``SOLE_TENANT``.
#:
#: That is R9 amendment 5's question answered the wrong way: when the join is wrong, the check
#: moves **with** the reader's confidence — the less related the adapter, the cleaner its record —
#: and no threshold on foreign time repairs it. So a tenancy record must carry positive evidence
#: that the sampler was watching the device our work ran on, and the absence of that evidence is
#: this verdict: not a detection, not a pass, and not repairable by tightening.
TENANCY_UNWITNESSED = "UNOBSERVABLE(self_not_witnessed)"

#: No owner was declared for the window, so no PID can be classified as ours. Survey-grade: fine
#: for asking who else is on an adapter, never admissible as a companion.
TENANCY_NO_OWNER = "UNOBSERVABLE(no_window_owner)"

SILENCE_SET = [
    "Produces NO clock record. Windows exposes no GPU frequency counter (all five `GPU *` counter "
    "sets enumerated; none carries MHz) and no root\\wmi class here does either. Engine Running "
    "Time is a duration and moves the same way as the figure it would certify, so it is not a "
    "second quantity from outside the series (R9 amendment 5) and is never used as one.",
    "Attributes work to a PID, not to a queue or a kernel. It cannot say whether foreign work "
    "overlapped our dispatches or merely shared the window.",
    "Engine time is scheduler bookkeeping, not occupancy: a process holding the engine with a "
    "small dispatch and a process saturating it are both `busy`.",
    "Sees only WDDM adapters on Windows. On Linux, Android, or macOS this producer does not "
    "exist and the tenancy axis is NO_PRODUCER — which is not SOLE_TENANT and not a pass (R12).",
    "Re-enumerates instances periodically; a foreign process that starts and exits entirely "
    "between two enumerations is invisible to it. PDH caches the instance list per process, so "
    "the enumeration is forced (PdhEnumObjects refresh) — without that, nothing that started "
    "after the sampler is ever seen, and the record reads clean.",
    "Sampling costs ~0.1-0.5 s of host CPU per enumeration, so the instrument is itself a small "
    "source of the host contention that `bench/contention.py` gates on.",
    "Joins Vulkan device name to adapter LUID through a registry description string. A wrong "
    "join yields a *clean* record, so the join is checked by requiring our own work to appear on "
    "the adapter (UNOBSERVABLE(self_not_witnessed)) rather than trusted.",
]


# ---------------------------------------------------------------------------------------------
# PDH — the counters are read in-process. `Get-Counter` costs ~5 s per wildcard call on this box,
# which would make the instrument a larger perturbation than the thing it watches.
# ---------------------------------------------------------------------------------------------

class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class _PDH_RAW_COUNTER(ctypes.Structure):
    _fields_ = [
        ("CStatus", wintypes.DWORD),
        ("TimeStamp", _FILETIME),
        ("FirstValue", ctypes.c_longlong),
        ("SecondValue", ctypes.c_longlong),
        ("MultiCount", wintypes.DWORD),
    ]


class CounterError(RuntimeError):
    """The instrument failed. R13: never a finding about tenancy."""


#: ``PERF_DETAIL_WIZARD`` — every counter, including the ones perfmon hides behind "advanced".
_PERF_DETAIL_WIZARD = 400

#: ``PDH_MORE_DATA``. Returned when the buffer is too small — including when the instance set grew
#: since the sizing call, which is the case this instrument is looking for.
_PDH_MORE_DATA = 0x800007D2


def _pdh():
    if sys.platform != "win32":
        raise CounterError("pdh.dll exists on Windows only")
    try:
        return ctypes.WinDLL("pdh.dll")
    except OSError as exc:  # pragma: no cover - environment dependent
        raise CounterError(f"pdh.dll could not be loaded: {exc}") from exc


def _refresh_object_cache() -> None:
    """Force PDH to re-enumerate the performance objects.

    **Without this the instrument cannot see any process that started after it did**, and this is
    not a subtle degradation: PDH caches the instance list per process, so a sampler that opened
    its query before a foreign job launched reports a clean adapter for the whole window. Measured
    — our own worker's ``\\GPU Engine`` instances were invisible to a non-refreshing sampler for
    the entire 60 s it ran, and appeared 14 s into the run once the refresh was added.

    That failure has the same shape as every other one this instrument has produced: **it reads
    clean.** A cache miss and a quiet adapter are the same bytes. R9 amendment 5 says a check that
    moves with the reader's confidence when its subject is wrong cannot be repaired by tightening
    — so it is repaired here, at the source, and the interlock that catches a residual failure is
    ``TENANCY_UNWITNESSED``.
    """
    pdh = _pdh()
    size = wintypes.DWORD(0)
    pdh.PdhEnumObjectsW(None, None, None, ctypes.byref(size), _PERF_DETAIL_WIZARD, True)
    if size.value:
        buf = ctypes.create_unicode_buffer(size.value + 2)
        pdh.PdhEnumObjectsW(None, None, buf, ctypes.byref(size), _PERF_DETAIL_WIZARD, False)


_EXPAND_LOCK = threading.Lock()
_EXPAND_CACHE: "dict" = {"t": 0.0, "paths": []}

#: How stale an enumeration may be before it is redone. Exists because the probe runs one sampler
#: per adapter and each enumeration costs ~0.5-1.0 s: four samplers each refreshing independently
#: starved the whole set, and the blind-gap interlock (correctly) refused the resulting record.
#: One enumeration serves every sampler within this window.
EXPAND_MAX_AGE_S = 2.0


def expand_engine_paths(refresh: bool = True, max_age: "float | None" = None) -> "list[str]":
    """Every ``\\GPU Engine(...)\\Running Time`` instance path currently known to PDH."""
    max_age = EXPAND_MAX_AGE_S if max_age is None else max_age
    with _EXPAND_LOCK:
        if max_age > 0 and time.time() - _EXPAND_CACHE["t"] < max_age and _EXPAND_CACHE["paths"]:
            return list(_EXPAND_CACHE["paths"])
        paths = _expand_engine_paths_uncached(refresh)
        _EXPAND_CACHE["t"] = time.time()
        _EXPAND_CACHE["paths"] = paths
        return list(paths)


def _expand_engine_paths_uncached(refresh: bool = True) -> "list[str]":
    pdh = _pdh()
    if refresh:
        _refresh_object_cache()
    path = "\\GPU Engine(*)\\Running Time"
    size = wintypes.DWORD(0)
    pdh.PdhExpandWildCardPathW(None, path, None, ctypes.byref(size), 0)
    if size.value == 0:
        raise CounterError("PDH expanded no instances for \\GPU Engine(*) — the counter set is "
                           "present but empty, or the provider is disabled")
    # PDH_MORE_DATA on the real call is normal: the instance set can grow between the sizing call
    # and the read, which is exactly what happens when the thing we are watching for (a new
    # process on the adapter) shows up. Retry with a bigger buffer rather than failing.
    buf = None
    for _ in range(6):
        size.value = int(size.value * 1.5) + 1024
        buf = ctypes.create_unicode_buffer(size.value + 2)
        status = pdh.PdhExpandWildCardPathW(None, path, buf, ctypes.byref(size), 0)
        if status == 0:
            break
        if status & 0xFFFFFFFF != _PDH_MORE_DATA:
            raise CounterError(f"PdhExpandWildCardPathW failed 0x{status & 0xFFFFFFFF:08x}")
    else:
        raise CounterError("PdhExpandWildCardPathW kept asking for more room; the instance set is "
                           "growing faster than it can be enumerated")
    out: "list[str]" = []
    cur: "list[str]" = []
    for i in range(size.value):
        ch = buf[i]
        if ch == "\x00":
            if not cur:
                break
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    return out


def parse_instance(path: str) -> "dict | None":
    m = _INSTANCE_RE.search(path)
    if not m:
        return None
    return {
        "pid": int(m.group("pid")),
        "luid": f"{m.group('luid_hi').lower()}_{m.group('luid_lo').lower()}",
        "phys": int(m.group("phys")),
        "eng": int(m.group("eng")),
        "engtype": m.group("engtype").lower(),
    }


# ---------------------------------------------------------------------------------------------
# Adapter identity. The counters key on LUID; every other instrument on this project keys on the
# Vulkan device name. The join has to be made somewhere, and it is made here, explicitly, with
# ambiguity classified as an instrument error rather than resolved by picking the first match.
# ---------------------------------------------------------------------------------------------

_DIRECTX_KEY = r"SOFTWARE\Microsoft\DirectX"


def adapters() -> "list[dict]":
    """Adapters the DirectX runtime has registered, with their LUIDs.

    ``HKLM\\SOFTWARE\\Microsoft\\DirectX`` carries one subkey per adapter the runtime has seen,
    each with a ``Description`` and an ``AdapterLuid``. Stale entries survive there (this box has
    two Intel entries and two NVIDIA ones, three of them with LUIDs that appear in no counter
    instance), which is why :func:`luid_for_adapter` requires the LUID to be *present in the live
    counter set* before it will use it.
    """
    out: "list[dict]" = []
    if winreg is None:
        raise CounterError("this host has no Windows registry, so no WDDM adapter table exists")
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DIRECTX_KEY)
    except OSError as exc:
        raise CounterError(f"{_DIRECTX_KEY} is not readable: {exc}") from exc
    with root:
        i = 0
        while True:
            try:
                name = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            try:
                with winreg.OpenKey(root, name) as sub:
                    desc, _ = winreg.QueryValueEx(sub, "Description")
                    luid, _ = winreg.QueryValueEx(sub, "AdapterLuid")
            except OSError:
                continue
            lo = luid & 0xFFFFFFFF
            hi = (luid >> 32) & 0xFFFFFFFF
            out.append({
                "key": name,
                "description": desc,
                "luid_int": luid,
                "luid": f"0x{hi:08x}_0x{lo:08x}",
            })
    return out


def live_luids() -> "set[str]":
    """LUIDs that actually appear in the counter instance set right now."""
    seen: "set[str]" = set()
    for p in expand_engine_paths():
        info = parse_instance(p)
        if info:
            seen.add(info["luid"])
    return seen


def luid_for_adapter(device_name: str) -> str:
    """LUID for the adapter whose registered description matches ``device_name``.

    Raises :class:`CounterError` — never returns a guess — when the name matches no live adapter
    or more than one. Two identical adapters in a machine is the case that makes name-matching
    unsound, and this project has already been bitten once by an instrument that resolved an
    ambiguity by picking a side (`devices.identify_by_timestamp`). ``ERROR(instrument)`` is the
    honest outcome and it is not a finding about tenancy.
    """
    live = live_luids()
    if not live:
        raise CounterError("no GPU Engine instances exist, so no adapter can be identified")
    want = (device_name or "").strip().lower()
    if not want:
        raise CounterError("no device name was supplied to join against an adapter LUID")
    matches = {a["luid"] for a in adapters()
               if a["luid"] in live and a["description"].strip().lower() == want}
    if not matches:
        loose = {a["luid"] for a in adapters()
                 if a["luid"] in live and (want in a["description"].lower()
                                           or a["description"].lower() in want)}
        matches = loose
    if len(matches) == 1:
        return next(iter(matches))
    if not matches:
        raise CounterError(
            f"no live adapter is registered under a description matching {device_name!r}; "
            f"live LUIDs are {sorted(live)}")
    raise CounterError(
        f"{device_name!r} matches {len(matches)} live adapters ({sorted(matches)}). A tenancy "
        f"record naming the wrong adapter is worse than none.")


# ---------------------------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------------------------

def _is_ours_fn():
    """Switch's ancestry test, imported rather than re-implemented.

    ``bench/results/probe_gpustate.py`` already owns "is this PID our own worker", including the
    live-interrogation requirement and the deliberate choice to call an unreadable process
    *foreign* so the detector keeps its ability to fire. A second implementation here would be a
    second description of one rule, which is the mistake this project refused for the verdict
    vocabulary and again for the worker's stderr decoder.
    """
    if str(_RESULTS) not in sys.path:
        sys.path.insert(0, str(_RESULTS))
    try:
        import probe_gpustate  # type: ignore
    except Exception:  # pragma: no cover - environment dependent
        return None
    return probe_gpustate._is_ours


class Sampler(threading.Thread):
    """Accumulates per-(pid, engine) GPU engine time on one adapter over a window.

    ``Running Time`` is cumulative, so the window's engine time for an instance is
    ``last - first`` over the samples in which it was seen. That makes the *quantity* a count of
    ticks rather than a sampled rate — §10.0.4's preference — and it makes the sample rate matter
    only for catching instances that come and go.
    """

    def __init__(self, luid: str, interval: float = 1.0, reexpand_every: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.luid = luid
        self.interval = interval
        self.reexpand_every = reexpand_every
        self.stop = threading.Event()
        self.error: "str | None" = None
        self.own_root: "int | None" = None
        #: instance-key -> {"first": ticks, "last": ticks, "t_first": s, "t_last": s, ...}
        self.tracks: "dict[str, dict]" = {}
        self.t_start: "float | None" = None
        self.t_end: "float | None" = None
        self.rounds = 0
        self._own_seen: "set[int]" = set()
        self._ancestry: "dict[int, bool]" = {}
        self._round_ms: "list[float]" = []
        self._read_times: "list[float]" = []
        self._pdh = None
        self._query = None
        self._counters: "dict[str, wintypes.HANDLE]" = {}

    # -- lifecycle ----------------------------------------------------------------------------

    def open(self) -> "Sampler":
        self._pdh = _pdh()
        q = wintypes.HANDLE()
        status = self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(q))
        if status != 0:
            raise CounterError(f"PdhOpenQueryW failed 0x{status & 0xFFFFFFFF:08x}")
        self._query = q
        self._add_new_paths()
        return self

    def close(self) -> None:
        if self._query is not None and self._pdh is not None:
            self._pdh.PdhCloseQuery(self._query)
            self._query = None

    # -- sampling -----------------------------------------------------------------------------

    def _add_new_paths(self) -> int:
        added = 0
        for path in expand_engine_paths():
            info = parse_instance(path)
            if not info or info["luid"] != self.luid or path in self._counters:
                continue
            handle = wintypes.HANDLE()
            if self._pdh.PdhAddCounterW(self._query, path, 0, ctypes.byref(handle)) == 0:
                self._counters[path] = handle
                added += 1
        return added

    def _read(self) -> None:
        now = time.time()
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            return
        self._read_times.append(now)
        ctype = wintypes.DWORD()
        raw = _PDH_RAW_COUNTER()
        for path, handle in list(self._counters.items()):
            if self._pdh.PdhGetRawCounterValue(handle, ctypes.byref(ctype), ctypes.byref(raw)) != 0:
                continue
            if raw.CStatus != 0:
                continue
            info = parse_instance(path)
            if info is None:
                continue
            track = self.tracks.get(path)
            if track is None:
                self.tracks[path] = {
                    **info,
                    "first": raw.FirstValue,
                    "last": raw.FirstValue,
                    "t_first": now,
                    "t_last": now,
                }
            else:
                track["last"] = raw.FirstValue
                track["t_last"] = now
        self.rounds += 1

    def run(self) -> None:  # pragma: no cover - thread body, exercised by probe_wingpu.py
        try:
            if self._query is None:
                self.open()
            self.t_start = time.time()
            self._read()
            last_expand = time.time()
            while not self.stop.is_set():
                if self.stop.wait(self.interval):
                    break
                t0 = time.perf_counter()
                if time.time() - last_expand >= self.reexpand_every:
                    self._add_new_paths()
                    last_expand = time.time()
                self._read()
                self._classify_pending()
                self._round_ms.append((time.perf_counter() - t0) * 1000.0)
            self._add_new_paths()
            self._read()
            self._classify_pending()
            self.t_end = time.time()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.close()

    def _classify_pending(self) -> None:
        """Decide ancestry **while the processes are alive**, not at summarise time.

        The companion's first certified run reported foreign work against our own worker because
        ancestry was resolved after the child had exited. The same trap is here and it is closed
        the same way: every PID that does work is classified on sight.

        Two economies, both of which change what the instrument can see and are therefore stated
        rather than buried. Ancestry is resolved **once per PID** and cached, and only for PIDs
        that have accrued engine time *inside this window* — measured, the naive version cost
        **21.2 s per round** walking ancestry for all 204 instances on this adapter, which is why
        the first run of this sampler completed 3 rounds in 62 s and never saw its own worker
        appear. An instrument too slow to sample is an instrument that reports
        ``UNOBSERVABLE(self_not_witnessed)`` about a run it simply missed.
        """
        is_ours = _is_ours_fn()
        if is_ours is None or self.own_root is None:
            return
        for track in self.tracks.values():
            pid = track["pid"]
            if "ours" in track or pid in KERNEL_PIDS:
                continue
            if track["last"] <= track["first"]:
                continue  # idle on this adapter in this window; nothing to attribute either way
            cached = self._ancestry.get(pid)
            if cached is None:
                cached = bool(pid == self.own_root or is_ours(pid, self.own_root))
                self._ancestry[pid] = cached
            track["ours"] = cached
            if cached:
                self._own_seen.add(pid)

    # -- reduction ----------------------------------------------------------------------------

    def summarise(self) -> dict:
        return summarise(self)


def summarise(sampler: "Sampler") -> dict:
    """Reduce a window to a **tenancy record**, and say plainly that the clock axis is empty."""
    base = {
        "producer": "windows/pdh \\GPU Engine — bench/win_gpu_counters.py",
        "luid": sampler.luid,
        "silence_set": list(SILENCE_SET),
    }
    if sampler.error:
        return {**base, "verdict": ERROR_INSTRUMENT, "reason": sampler.error}
    if not sampler.tracks or sampler.rounds < 2:
        return {**base, "verdict": ERROR_INSTRUMENT,
                "reason": (f"the sampler completed {sampler.rounds} enumeration(s) over the "
                           f"window, which is not enough to difference a cumulative counter")}
    gaps = [b - a for a, b in zip(sampler._read_times, sampler._read_times[1:])]
    worst_gap = max(gaps) if gaps else None
    if worst_gap is not None and worst_gap > MAX_BLIND_GAP_FACTOR * sampler.interval:
        return {
            **base, "verdict": ERROR_INSTRUMENT,
            "reason": (f"the sampler went blind for {worst_gap:.1f} s inside a window it is asked "
                       f"to characterise, against a requested interval of {sampler.interval:.1f} s "
                       f"(limit {MAX_BLIND_GAP_FACTOR}x). A cumulative counter still totals "
                       f"correctly across a gap, but an instance that appears and vanishes inside "
                       f"one is never enumerated at all — including our own worker. R13: this is "
                       f"the instrument failing, and it is not a finding of SOLE_TENANT."),
            "sampling": {"rounds": sampler.rounds, "worst_gap_s": round(worst_gap, 2),
                         "interval_s": sampler.interval},
        }
    seconds = (sampler.t_end or time.time()) - (sampler.t_start or time.time())
    ours: "dict[int, float]" = {}
    foreign: "dict[int, float]" = {}
    kernel: "dict[int, float]" = {}
    engtypes: "dict[str, float]" = {}
    for track in sampler.tracks.values():
        ticks = max(0, track["last"] - track["first"])
        if ticks == 0:
            continue
        secs = ticks / TICKS_PER_SECOND
        pid = track["pid"]
        engtypes[track["engtype"]] = engtypes.get(track["engtype"], 0.0) + secs
        if pid in KERNEL_PIDS:
            kernel[pid] = kernel.get(pid, 0.0) + secs
        elif track.get("ours"):
            ours[pid] = ours.get(pid, 0.0) + secs
        else:
            foreign[pid] = foreign.get(pid, 0.0) + secs
    foreign_busy = sum(foreign.values())
    frac = (foreign_busy / seconds) if seconds > 0 else 0.0
    names = _process_names(set(foreign) | set(ours))
    verdict = TENANCY_SOLE
    reason = None
    if frac >= FOREIGN_BUSY_FRACTION:
        verdict = TENANCY_FOREIGN
        if DISPLAY_TENANCY_IS_STRUCTURAL and _looks_like_display_only(foreign, names):
            verdict = TENANCY_STRUCTURAL
    # The interlocks come **after** the tenancy computation and override it, because a clean
    # tenancy verdict from an adapter we cannot show we were running on is the failure mode that
    # reads best. See TENANCY_UNWITNESSED.
    if sampler.own_root is None:
        verdict = TENANCY_NO_OWNER
        reason = ("no owning PID was declared for this window, so our own submissions cannot be "
                  "told from a stranger's. Survey-grade only.")
    elif not ours:
        verdict = TENANCY_UNWITNESSED
        reason = (f"a window owner was declared (pid {sampler.own_root}) but no engine time on "
                  f"LUID {sampler.luid} was attributed to it or its descendants. Either the work "
                  f"ran on a different adapter than the one sampled, or it never reached the GPU. "
                  f"A tenancy verdict from an adapter our work was not seen on is not evidence "
                  f"about our run — and it reads clean, which is why it is refused here.")
    return {
        **base,
        "verdict": verdict,
        "reason": reason,
        "seconds": round(seconds, 2),
        "rounds": sampler.rounds,
        "own_root": sampler.own_root,
        "own_gpu_seconds": {str(k): round(v, 4) for k, v in sorted(ours.items())},
        "foreign_gpu_seconds": {str(k): round(v, 4) for k, v in sorted(foreign.items())},
        "kernel_gpu_seconds": {str(k): round(v, 4) for k, v in sorted(kernel.items())},
        "process_names": names,
        "engine_types": {k: round(v, 4) for k, v in sorted(engtypes.items())},
        "foreign_busy_fraction": round(frac, 4),
        "we_were_seen_on_this_adapter": bool(ours),
        "sampling": {
            "rounds": sampler.rounds,
            "interval_s": sampler.interval,
            "worst_gap_s": round(worst_gap, 2) if worst_gap is not None else None,
            "round_ms_max": round(max(sampler._round_ms), 1) if sampler._round_ms else None,
        },
        "clock": {
            "verdict": UNOBSERVABLE,
            "reason": ("Windows exposes no GPU clock counter. Engine `Running Time` is a duration "
                       "and rises when the clock falls, so it is a second copy of the quantity "
                       "under certification and is not admissible as the clock half of "
                       "DESIGN.md §10.0 obligation 8 (R9 amendment 5)."),
        },
    }


def _process_names(pids: "set[int]") -> "dict[str, str]":
    try:
        import psutil
    except Exception:  # pragma: no cover
        return {}
    out: "dict[str, str]" = {}
    for pid in sorted(pids):
        try:
            out[str(pid)] = psutil.Process(pid).name()
        except Exception:
            out[str(pid)] = "<gone>"
    return out


#: Processes whose presence on an adapter is the desktop existing, not a workload arriving.
_DISPLAY_PROCESSES = frozenset({"dwm.exe", "explorer.exe", "csrss.exe", "searchhost.exe",
                                "shellexperiencehost.exe", "startmenuexperiencehost.exe",
                                "textinputhost.exe", "sihost.exe"})


def _looks_like_display_only(foreign: "dict[int, float]", names: "dict[str, str]") -> bool:
    """Whether every foreign holder is the desktop itself.

    Reported as its own verdict rather than waved through. The distinction matters because it is
    **permanent** on a machine whose panel hangs off this adapter: a run that waits for the
    compositor to go away waits forever, and a verdict that cannot ever be cleared should say that
    rather than look like bad luck.
    """
    if not foreign:
        return False
    return all(names.get(str(pid), "").lower() in _DISPLAY_PROCESSES for pid in foreign)


def observe(device_name: str, interval: float = 1.0) -> "Sampler | dict":
    """Start a sampler for ``device_name``, or return an ``ERROR(instrument)`` record.

    Never raises at the call site: the caller's job is to attach a record, and "the instrument
    failed" is one of the records it may have to attach.
    """
    try:
        luid = luid_for_adapter(device_name)
    except CounterError as exc:
        return {"producer": "windows/pdh \\GPU Engine — bench/win_gpu_counters.py",
                "verdict": ERROR_INSTRUMENT, "reason": str(exc), "silence_set": list(SILENCE_SET)}
    sampler = Sampler(luid, interval=interval)
    try:
        sampler.open()
    except CounterError as exc:
        return {"producer": "windows/pdh \\GPU Engine — bench/win_gpu_counters.py",
                "verdict": ERROR_INSTRUMENT, "reason": str(exc), "luid": luid,
                "silence_set": list(SILENCE_SET)}
    sampler.start()
    return sampler


def available() -> bool:
    """Whether this producer exists on this host at all."""
    if sys.platform != "win32":
        return False
    try:
        return bool(live_luids())
    except CounterError:
        return False


if __name__ == "__main__":  # pragma: no cover - manual use
    import json

    print(json.dumps({"adapters": adapters(), "live_luids": sorted(live_luids())}, indent=2))
