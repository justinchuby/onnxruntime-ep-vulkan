"""The count-based ceiling, with a frame, an extent, and a refusal state.

WHY THIS EXISTS AS A MODULE AND NOT A PROBE
===========================================
On this hardware a wall-clock figure is `STEADY_UNCERTIFIED` by default and will stay that
way: the box is shared with another team and 7.73 foreign busy cores against a 0.5 threshold
is the normal state, not an anomaly (`docs/PERF.md` §20). Performance work therefore runs on
counts, and Switch's bandwidth roofline stops being a diagnostic and becomes the primary
performance instrument.

An instrument that carries that weight needs what the certification apparatus already has:
a stated **frame**, a stated **extent**, and the ability to **refuse**. A bound that
extrapolates cheerfully past its inputs is the same failure as a variance test that cannot
see a bias -- it returns a confident number about a world it never observed.

WHAT IT REFUSES, AND WHAT CHANGED
=================================
The obvious refusal is a context length nobody has run. The important one was
different, and it has now been discharged -- by exactly the thing that named it.

**Previously: `GroupQueryAttention` was declined and executed on CPU.** Its proof
verdict was `DIVERGENT` (worst_rel 16.73), all 32 instances declined, and that is
why the graph carried 33 islands and 323 claimed nodes rather than one island and
355. GQA is *the op that reads the KV cache*, so on that build the KV-cache bytes
were not GPU DRAM traffic at all, and charging them to a GPU roofline was a bound
on a machine we were not running. The extent was `[0]`.

**Switch has since landed GQA.** Read off this binary's own counters, not off a
message: `subgraphs_live` 1, `claimed_nodes` 355 of 363, `model_output_equivalence`
MATCH, and the session's own claim log naming `com.microsoft::GroupQueryAttention
x32 proven`. The KV bytes are ours. That refusal is discharged.

DISCHARGING IT EXPOSED A SECOND CONDITION UNDERNEATH
====================================================
`island_bytes_phi35.json` computes `kv_bytes(past_len)` analytically. Nobody had
shown the modelled bytes were the moved bytes -- the claim Switch had to earn
*separately* for weights (amplification 1.000000, from 116,324,352 InB loads x 16 B,
with the two non-tautological factors measured apart). So it was measured:
`bench/results/probe_kv_bytes_earned.py`, counters only, slope of slopes.

    READBACK  MEASURED      393,216 B per past token, ratio 1.000000, on BOTH
                            segments (0->128 and 128->512), linearity spread
                            0.000000. The modelled KV magnitude is confirmed to
                            the byte.
    UPLOAD    UNOBSERVABLE  flat at 399,376 B/inference at past_len 0, 128 AND 512.
                            The past KV cache does not reach the device by the
                            staging path these counters watch. It reaches it
                            somehow. The counter is blind to the path; its silence
                            is not evidence that the read side is free.

That measurement earns the KV term's *magnitude* and, in doing so, turns up a term
the roofline never modelled at all: **the present KV cache crosses device->host in
full every inference**, (past_len + 1) * 393,216 B. At past_len 0 that is 457 KB
against a ~2.1 GB DRAM stream and is harmless. At past_len 512 it is 202 MB.

TWO EXTENTS, BECAUSE THERE ARE TWO QUESTIONS
============================================
"Is the DRAM bound admissible here?" and "is the DRAM bound the floor here?" are
different questions and after this round they have different answers:

    extent()          -- where the DRAM bound describes this build.  Now the FULL
                         GRID: GQA is claimed, so the KV bytes are the device's.
    binding_extent()  -- where the DRAM bound is also the floor of the inference.
                         Still `[0]`, and now for a MEASURED reason rather than a
                         structural one.

The second holds only where the host<->device transfer term provably cannot bind.
That is decided per context by a crossover: the link speed at which transfer time
equals DRAM time. At past_len 0 the crossover is 0.10 GB/s -- below the slowest
PCIe configuration ever shipped (x1 gen1, 0.25 GB/s), so DRAM binds without anyone
measuring this machine's link. At past_len 128 the crossover is 6.1 GB/s and at 512
it is 22.5 GB/s, both plainly within the range of real links, so which term binds
cannot be settled without a measured link bandwidth -- which is a *timing*, and
timings on this box are `STEADY_UNCERTIFIED` by standing policy (§20).

R12: a term whose event cannot occur in the frame reports **UNOBSERVABLE**, never a
number -- and notably never `0` either. `0` for the upload term would claim the read
side of the KV cache is free; `UNOBSERVABLE` says we did not see it.

    **At past_len = 0 the transfer term cannot bind, so the DRAM bound is the floor,
    and 12.1847 ms -- zero context, one token -- is the only quotable figure we hold.**

The one *binding* bound and the one quotable figure still sit at the same context.
That was true before for a structural reason and is true now for a measured one --
but it must be asserted per context and not assumed to travel: the DRAM bound is now
admissible at 128..8192, where we hold no quotable figure at all. The bound is
admissible in regimes where the comparison is not. `compare()` enforces that.

THE FRAME ASSERTS ITSELF
========================
The extent is read from a run record. While GQA was being fixed, this module went on
reporting extent `[0]` off a record from the previous binary while the DLL beside it
had already changed -- the artifact is `bench/_scratch/ceiling_stale_record_artifact.txt`.
It was right about a build nobody was running.

So `load()` now hashes the DLL and refuses a claim record that did not come from it
(`CeilingError`). A refusal that has just been satisfied is exactly when it is most
likely to become decoration, so `test_ceiling.py` carries a positive control: a
synthetic *declined* record, in frame, must still produce extent `[0]`. Discharging
the condition once did not wire it open.

Usage:
    from ceiling import Ceiling
    c = Ceiling.load()
    c.extent()                 -> admissible: the full grid
    c.binding_extent()         -> [0]
    c.floor_ms(0)              -> BOUND, 8.2180 ms, binding
    c.floor_ms(8192)           -> BOUND for DRAM, binding UNOBSERVABLE
    c.compare(12.1847, past_len=0)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"
_ROOT = _HERE.parent

#: The bound holds and may be quoted against a figure at the same context.
BOUND = "BOUND"
#: The bound's inputs do not describe this frame, or this context was never run.
#: Never `0`, never a number -- see the module docstring.
UNOBSERVABLE = "UNOBSERVABLE"
#: The module could not reach its observation. Never a detection (R13).
ERROR = "ERROR"

#: Records this module reads. Both are committed and neither needs a GPU or the 2 GB model.
ROOFLINE_RECORD = _RESULTS / "roofline_phi35.json"
ISLAND_BYTES_RECORD = _RESULTS / "island_bytes_phi35.json"
#: The measured host<->device transfer term. Counters only, so it is not a timing.
KV_BYTES_RECORD = _RESULTS / "kv_bytes_earned.json"

#: The run record the GQA claim status is read from. It must be in the same binary frame as
#: any figure this ceiling is used to judge, which is why `load()` refuses one that is not.
DEFAULT_CLAIM_RECORD = _RESULTS / "phi35-7c9d1b7-dev0.json"

#: The slowest PCIe configuration ever shipped (x1 gen1, 250 MB/s). Used only as a LOWER
#: bound on this machine's link, so that "transfer cannot bind here" can be established
#: without measuring the link -- a measurement that would be a timing, and timings on this
#: box are STEADY_UNCERTIFIED by standing policy (docs/PERF.md §20). Never used as an
#: estimate of the actual link, only to rule the transfer term OUT.
LINK_FLOOR_GB_PER_S = 0.25

#: The fastest consumer link ever shipped (PCIe 5.0 x16, 63 GB/s theoretical). The mirror
#: of the above and used the same way: only to rule the transfer term IN. This machine's
#: dGPU is nothing like this fast, so it is a generous bound and the conclusions it licenses
#: are conservative. Both constants are stated rather than tuned, and `binding()` publishes
#: the crossover at every context so a reader with a measured link can redo the call.
LINK_CEILING_GB_PER_S = 63.0

#: Total nodes probed in this graph. 355 of these in one island was the pre-ledger record,
#: is what a build with GQA claimed produces, and is what this build produces.
NODES_PROBED = 363

SILENCE_SET = [
    "It is a bound, not an estimate. A kernel that beats it is a refutation of the model; a "
    "kernel far above it is not thereby explained.",
    "Spec peak, not achieved bandwidth. Real GDDR6 sustains ~75-85% of spec on a pure stream, "
    "so the achievable floor is HIGHER than the number reported here.",
    "It is silent about latency, occupancy, launch overhead and synchronisation. The 32 extra "
    "pipeline drains per inference that island fragmentation costs are not in it (§19.6).",
    "It counts bytes named by the graph. A kernel that reads bytes the graph does not name is "
    "invisible to it.",
    "It is a DRAM bound and is silent about host<->device transfer. That silence is harmless "
    "at past_len 0 (857 KB against a ~2.1 GB stream) and is not harmless at past_len 512 "
    "(202 MB, measured) -- see binding_extent().",
    "The KV term's magnitude is earned on the readback axis only. How the past KV becomes "
    "device-resident is UNOBSERVABLE to the staging counters, and its DRAM amplification -- "
    "the factor Switch measured separately for weights as 1.000000 -- is unmeasured for KV.",
    "It says nothing about whether the run that produced the achieved figure was clocked at "
    "boost or sole-tenant. That is `bench/device_companion.py`'s subject and it gates a "
    "different quantity.",
]


def _sha256(path: Path) -> "str | None":
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class CeilingError(RuntimeError):
    """The module did not reach its observation. ERROR(instrument), never a detection."""


class Ceiling:
    """A bandwidth ceiling that knows what it is a ceiling *for*."""

    def __init__(self, roofline: dict, by_context: list, claim: dict, frame: dict,
                 transfer: dict):
        self._roofline = roofline
        self._by_context = {int(r["past_sequence_length"]): r for r in by_context}
        self._claim = claim
        self._frame = frame
        self._transfer = transfer

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, claim_record: "Path | None" = None,
             dll: "Path | None" = None) -> "Ceiling":
        missing = [p.name for p in (ROOFLINE_RECORD, ISLAND_BYTES_RECORD) if not p.is_file()]
        if missing:
            raise CeilingError(
                f"ERROR(instrument): required record(s) absent: {', '.join(missing)}. "
                "Reproduce with bench/results/probe_roofline.py and probe_island_bytes.py."
            )
        roofline = json.loads(ROOFLINE_RECORD.read_text(encoding="utf-8"))
        island = json.loads(ISLAND_BYTES_RECORD.read_text(encoding="utf-8"))
        by_context = island.get("by_context_length")
        if not by_context:
            raise CeilingError(
                "ERROR(instrument): island_bytes_phi35.json carries no by_context_length table."
            )

        dll_path = Path(dll) if dll else _ROOT / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
        dll_hash = _sha256(dll_path)

        claim_path = Path(claim_record) if claim_record else DEFAULT_CLAIM_RECORD
        claim = cls._read_claim(claim_path, dll_hash)

        frame = {
            "model": roofline.get("model"),
            "device": roofline.get("device"),
            "peak_GB_per_s": roofline["roofline_dram"]["peak_GB_per_s"],
            "peak_source": "spec: 128-bit GDDR6 at 16 Gbps",
            "byte_model_source": ROOFLINE_RECORD.name,
            "by_context_source": ISLAND_BYTES_RECORD.name,
            "claim_source": claim_path.name,
            "transfer_source": KV_BYTES_RECORD.name,
            "dll_sha256": dll_hash,
            "matmulnbits_nodes": roofline.get("matmulnbits_nodes"),
        }
        return cls(roofline, by_context, claim, frame, cls._read_transfer())

    @staticmethod
    def _read_transfer() -> dict:
        """The measured host<->device term. Absent is UNOBSERVABLE, not zero."""
        if not KV_BYTES_RECORD.is_file():
            return {
                "state": UNOBSERVABLE,
                "reason": f"no {KV_BYTES_RECORD.name}; the transfer term has not been measured, "
                          "so whether it binds is unknown at every context. Reproduce with "
                          "bench/results/probe_kv_bytes_earned.py.",
            }
        rec = json.loads(KV_BYTES_RECORD.read_text(encoding="utf-8"))
        measured = {int(r["past_len"]): r for r in rec["by_context"]}
        return {
            "state": "READ",
            "measured_at": sorted(measured),
            "by_past_len": measured,
            "readback_bytes_per_past_token": rec["readback"]["bytes_per_past_token"],
            "readback_factor": rec["readback"]["factor"],
            "upload_state": rec["upload"]["state"],
            "upload_bytes_per_inference": rec["upload"]["observed_bytes_per_inference"],
            "source": KV_BYTES_RECORD.name,
        }

    @staticmethod
    def _read_claim(path: Path, dll_hash: "str | None") -> dict:
        """Read the build's fusion/claim status. This is what decides the extent.

        The frame of a test result is the binary that ran it. This module once reported
        extent `[0]` off a record from the previous binary while the DLL beside it had
        already changed, and was confidently right about a build nobody was running. So the
        premise asserts itself here rather than being checked by a reader who remembers to.
        """
        if not path.is_file():
            raise CeilingError(
                f"ERROR(instrument): no run record at {path.name}; the build's claim status "
                "is unknown, so we cannot say which ops execute on the device. Produce one "
                "with: python bench/phi35.py --device 0 --iters 4 --warmup 1 --repeats 1 "
                "--no-phases --out bench/results/<name>.json"
            )
        doc = json.loads(path.read_text(encoding="utf-8"))
        record_hash = (doc.get("environment", {}).get("build", {}) or {}).get("sha256")
        if record_hash is None:
            raise CeilingError(
                f"ERROR(instrument): {path.name} names its binary only by size and mtime, so it "
                "cannot be checked against the DLL beside it. Re-record it with a bench/ that "
                "writes environment.build.sha256; a claim status that cannot be tied to a "
                "binary cannot decide an extent."
            )
        if dll_hash is None:
            raise CeilingError(
                "ERROR(instrument): the EP DLL could not be hashed, so the claim record cannot "
                "be shown to be in frame. Build it with: cargo build --release "
                "--manifest-path rust/Cargo.toml"
            )
        if record_hash.lower() != dll_hash.lower():
            raise CeilingError(
                f"ERROR(instrument): {path.name} is out of frame. It was recorded against DLL "
                f"{record_hash[:16]} and the DLL beside this module is {dll_hash[:16]}. The "
                "extent is read off the build, not off the record's age -- re-record against "
                "this binary rather than relaxing this check. This is never a detection (R13)."
            )
        rec = doc["results"][0]
        counters = rec.get("counters", {})
        islands = counters.get("subgraphs_live")
        claimed = rec.get("claimed_nodes")
        if islands is None:
            raise CeilingError(
                f"ERROR(instrument): {path.name} carries no subgraphs_live counter, which is "
                "the in-frame witness the extent is derived from."
            )
        # 32 GQA instances declining cuts a chain into 33 segments. The island count is the
        # in-frame, contention-independent witness that they declined.
        gqa_declined = islands > 1
        return {
            "state": "READ",
            "islands": islands,
            "claimed_nodes": claimed,
            "nodes_probed": NODES_PROBED,
            "dll_sha256": record_hash,
            "gqa_declined": gqa_declined,
            "reason": (
                f"{islands} islands: the 32 GroupQueryAttention instances declined (proof verdict "
                "DIVERGENT) and execute on CPU, so the KV cache is not read by the device."
                if gqa_declined else
                f"{islands} island, {claimed} of {NODES_PROBED} claimed: GroupQueryAttention is "
                "claimed, so the device does read the KV cache and the KV term of the bound is live."
            ),
        }

    # ----------------------------------------------------------------- frame

    def frame(self) -> dict:
        return dict(self._frame)

    def silence_set(self) -> list:
        return list(SILENCE_SET)

    def extent(self) -> dict:
        """Which context lengths this ceiling may be quoted at, and why."""
        grid = sorted(self._by_context)
        if self._claim.get("state") != "READ":
            return {
                "admissible": [],
                "grid": grid,
                "reason": self._claim.get("reason"),
            }
        if self._claim["gqa_declined"]:
            return {
                "admissible": [0],
                "grid": grid,
                "reason": (
                    "GroupQueryAttention is declined on this build and executes on CPU, so the "
                    "KV-cache bytes in the by-context table are not GPU DRAM traffic. At "
                    "past_len 0 the KV term is exactly zero and the question does not arise; at "
                    "every other context the bound would describe a machine we are not running."
                ),
            }
        return {
            "admissible": grid,
            "grid": grid,
            "reason": "GroupQueryAttention is claimed, so the device reads the KV cache and the "
                      "by-context table describes this build.",
        }

    # ------------------------------------------------------- the second extent

    def transfer(self, past_len: int) -> dict:
        """The measured host<->device staging term at `past_len`, or a refusal.

        This is not in the roofline. The roofline counts DRAM bytes named by the graph;
        this crosses the link. It was found by measuring whether the modelled KV bytes are
        the moved bytes (`probe_kv_bytes_earned.py`) and it is the reason `binding_extent()`
        is narrower than `extent()`.
        """
        t = self._transfer
        if t.get("state") != "READ":
            return {"state": UNOBSERVABLE, "reason": t.get("reason")}

        measured = t["by_past_len"]
        if past_len in measured:
            row = measured[past_len]
            return {
                "state": "MEASURED",
                "past_sequence_length": past_len,
                "upload_bytes_per_inference": row["upload_bytes_per_inference"],
                "readback_bytes_per_inference": row["readback_bytes_per_inference"],
                "total_bytes_per_inference": (row["upload_bytes_per_inference"]
                                              + row["readback_bytes_per_inference"]),
                "source": t["source"],
            }
        # The readback law is exact at every context that was run (ratio 1.000000, spread
        # 0.000000 between segments), but it was run over 0..512 and this is beyond it.
        # Saying so is the difference between a modelled term and a measured one.
        lo = max(measured)
        row = measured[lo]
        modelled = (row["readback_bytes_per_inference"]
                    + (past_len - lo) * t["readback_bytes_per_past_token"])
        return {
            "state": "MODELLED",
            "past_sequence_length": past_len,
            "upload_bytes_per_inference": row["upload_bytes_per_inference"],
            "readback_bytes_per_inference": modelled,
            "total_bytes_per_inference": row["upload_bytes_per_inference"] + modelled,
            "modelled_from": (
                f"readback measured exactly linear at past_len {t['measured_at']} "
                f"({t['readback_bytes_per_past_token']:.0f} B per past token, factor "
                f"{t['readback_factor']:.6f}); extended past {lo}, which is extrapolation."
            ),
            "source": t["source"],
        }

    def binding(self, past_len: int) -> dict:
        """Is the DRAM bound the FLOOR here, or merely a DRAM bound?

        Decided without measuring this machine's link, by asking at what link speed the
        transfer time would equal the DRAM time. If that crossover is below the slowest
        PCIe configuration ever shipped, transfer cannot bind and DRAM does. Otherwise the
        answer needs a link bandwidth we do not hold -- and cannot take, because measuring
        it is a timing (§20).
        """
        if past_len not in self._by_context:
            return {"state": UNOBSERVABLE, "reason": f"past_len {past_len} is not in the grid."}
        tr = self.transfer(past_len)
        if tr["state"] == UNOBSERVABLE:
            return {
                "state": UNOBSERVABLE,
                "reason": "the host<->device transfer term is unmeasured, so it cannot be "
                          "ruled out as the binding term. " + str(tr.get("reason")),
            }
        dram_s = self._by_context[past_len]["floor_ms_at_spec_peak"] / 1000.0
        crossover = tr["total_bytes_per_inference"] / dram_s / 1e9
        if crossover < LINK_FLOOR_GB_PER_S:
            state, binds_by = "BINDS", "DRAM"
            reason = (
                f"transfer would have to be slower than {crossover:.3f} GB/s to bind, which is "
                f"below the slowest PCIe configuration ever shipped ({LINK_FLOOR_GB_PER_S} GB/s). "
                "It cannot bind, so the DRAM bound is the floor."
            )
        elif crossover > LINK_CEILING_GB_PER_S:
            state, binds_by = UNOBSERVABLE, "TRANSFER"
            reason = (
                f"transfer binds unless the link is faster than {crossover:.2f} GB/s, which "
                f"exceeds the fastest consumer link ever shipped ({LINK_CEILING_GB_PER_S} GB/s "
                "PCIe 5.0 x16). On any link that exists this inference is bound by host<->device "
                "KV transfer, not by DRAM bandwidth, and the DRAM bound is not the floor -- it is "
                "far below it. This is a direction for work, not a figure."
            )
        else:
            state, binds_by = UNOBSERVABLE, "UNDECIDED"
            reason = (
                f"transfer binds if the link is slower than {crossover:.2f} GB/s. That is well "
                f"inside the range of real links, so which term binds cannot be settled without a "
                "measured link bandwidth -- and that measurement is a timing, which is "
                "STEADY_UNCERTIFIED on this box by standing policy (docs/PERF.md §20). The DRAM "
                "bound remains a valid DRAM bound here; it is not known to be the floor."
            )
        return {
            "state": state,
            "binds_by": binds_by,
            "past_sequence_length": past_len,
            "transfer_state": tr["state"],
            "transfer_bytes_per_inference": tr["total_bytes_per_inference"],
            "link_GB_per_s_at_which_transfer_binds": crossover,
            "link_floor_GB_per_s": LINK_FLOOR_GB_PER_S,
            "link_ceiling_GB_per_s": LINK_CEILING_GB_PER_S,
            "caveat": (
                None if tr["state"] == "MEASURED" else
                "the transfer term here is extrapolated past the largest context that was run "
                f"({tr.get('modelled_from')}), so this verdict is conditional on the readback law "
                "continuing to hold."
            ),
            "reason": reason,
        }

    def binding_extent(self) -> dict:
        """Where the DRAM bound is the floor of the inference, not merely a DRAM floor."""
        grid = sorted(self._by_context)
        adm = self.extent()["admissible"]
        binding = [p for p in adm if self.binding(p)["state"] == "BINDS"]
        return {
            "binding": binding,
            "admissible": adm,
            "grid": grid,
            "reason": (
                "The DRAM bound describes this build at every admissible context, but it is the "
                "floor only where the measured host<->device transfer term provably cannot bind. "
                "Elsewhere it is a DRAM bound quoted in a regime where a second, unmodelled term "
                "may dominate."
            ),
        }

    # ----------------------------------------------------------------- bound

    def floor_ms(self, past_len: int) -> dict:
        """The bandwidth floor at `past_len`, or a refusal."""
        ext = self.extent()
        base = {
            "past_sequence_length": past_len,
            "frame": self.frame(),
            "silence_set": self.silence_set(),
        }

        if past_len not in self._by_context:
            return {
                **base,
                "state": UNOBSERVABLE,
                "reason": (
                    f"past_len {past_len} is not in the byte model's grid {sorted(self._by_context)}. "
                    "The KV term is linear and could be interpolated, but a bound that "
                    "extrapolates its own inputs is not a bound anyone should sign; add the "
                    "point to probe_island_bytes.py instead."
                ),
            }

        if past_len not in ext["admissible"]:
            return {
                **base,
                "state": UNOBSERVABLE,
                "reason": ext["reason"],
                "claim_status": dict(self._claim),
                "would_have_reported_ms": self._by_context[past_len]["floor_ms_at_spec_peak"],
                "why_not_zero": (
                    "Reporting 0 for the KV term would claim the traffic is free. It is not "
                    "free, it is elsewhere -- on the CPU. UNOBSERVABLE says we are not entitled "
                    "to a GPU bandwidth bound here at all (R12)."
                ),
            }

        row = self._by_context[past_len]
        return {
            **base,
            "state": BOUND,
            "floor_ms_at_spec_peak": row["floor_ms_at_spec_peak"],
            "floor_ms_at_75pct_achievable": row["floor_ms_at_spec_peak"] / 0.75,
            "floor_ms_at_85pct_achievable": row["floor_ms_at_spec_peak"] / 0.85,
            "total_MiB": row["total_MiB"],
            "weight_MiB": row["weight_MiB"],
            "kv_cache_MiB": row["kv_cache_MiB"],
            "intermediates_MiB": row["intermediates_MiB"],
            "claim_status": dict(self._claim),
            "bound_is_for": "DRAM traffic named by the graph",
            "binding": self.binding(past_len),
            "kv_term_earned": {
                "magnitude": self._transfer.get("readback_factor"),
                "on_axis": "readback (device->host), 393,216 B per past token",
                "not_earned": "the read-side residency path (UNOBSERVABLE to the staging "
                              "counters) and the DRAM amplification factor",
            },
        }

    # --------------------------------------------------------------- compare

    def compare(self, figure_ms: float, past_len: int, *, figure_source: str = "") -> dict:
        """Judge an achieved figure against the bound at its own context.

        Refuses a comparison whose contexts differ. The roofline is not a constant -- 8.22 ms
        at zero context against 20.80 ms at 8192 -- so quoting a figure against a bound taken
        at another context is the same class of error as quoting a timing without its device
        state.
        """
        bound = self.floor_ms(past_len)
        out = {
            "figure_ms": figure_ms,
            "figure_source": figure_source,
            "past_sequence_length": past_len,
            "bound_state": bound["state"],
        }
        if bound["state"] != BOUND:
            return {
                **out,
                "state": UNOBSERVABLE,
                "quotable": False,
                "reason": bound.get("reason"),
            }
        floor = bound["floor_ms_at_spec_peak"]
        binding = bound["binding"]
        return {
            **out,
            "state": BOUND,
            "quotable": True,
            "floor_ms_at_spec_peak": floor,
            "fraction_of_roofline": floor / figure_ms,
            "headroom_x": figure_ms / floor,
            "floor_is_binding": binding["state"] == "BINDS",
            "binding": binding,
            "pairing": (
                "The bound is binding at this context and a quotable figure exists at this "
                "context. Both halves are needed and neither travels: the DRAM bound is "
                "admissible at 128..8192, where this project holds no quotable figure at all."
                if binding["state"] == "BINDS" else
                "The bound is admissible here but is NOT known to be the floor -- an unmodelled "
                "host<->device transfer term may dominate (see `binding`). A percentage-of-"
                "roofline read here would be a percentage of the wrong roofline."
            ),
            "must_be_quoted_with": (
                f"past_sequence_length={past_len}; the roofline is not a constant "
                f"({self._by_context[min(self._by_context)]['floor_ms_at_spec_peak']:.2f} ms at "
                f"{min(self._by_context)} against "
                f"{self._by_context[max(self._by_context)]['floor_ms_at_spec_peak']:.2f} ms at "
                f"{max(self._by_context)})."
            ),
            "silence_set": self.silence_set(),
        }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="the count-based ceiling, with its extent")
    ap.add_argument("--figure-ms", type=float, default=12.1847)
    ap.add_argument("--past-len", type=int, default=0)
    args = ap.parse_args()

    try:
        c = Ceiling.load()
    except CeilingError as exc:
        print(str(exc))
        return 4

    ext = c.extent()
    bext = c.binding_extent()
    print("=" * 78)
    print("COUNT-BASED CEILING -- frame, extent, refusal")
    print("=" * 78)
    print(f"  device      {c.frame()['device']}")
    print(f"  peak        {c.frame()['peak_GB_per_s']:.1f} GB/s ({c.frame()['peak_source']})")
    print(f"  dll         {str(c.frame()['dll_sha256'])[:16]}")
    print(f"  claim       {c.frame()['claim_source']}  ({c._claim['islands']} island(s), "
          f"{c._claim['claimed_nodes']} of {NODES_PROBED} claimed)")
    print(f"  grid        {ext['grid']}")
    print(f"  ADMISSIBLE  {ext['admissible']}   (the DRAM bound describes this build)")
    print(f"  BINDING     {bext['binding']}   (and is the floor of the inference)")
    print(f"  because     {ext['reason']}")
    print()
    for past in ext["grid"]:
        r = c.floor_ms(past)
        if r["state"] == BOUND:
            b = r["binding"]
            mark = {"DRAM": "FLOOR", "TRANSFER": "transfer-bound", "UNDECIDED": "undecided"}[b["binds_by"]]
            print(f"  past_len {past:>5}  {BOUND:<6} {r['floor_ms_at_spec_peak']:8.4f} ms"
                  f"   KV {r['kv_cache_MiB']:>7.0f} MiB"
                  f"   transfer {b['transfer_bytes_per_inference']/1024/1024:>8.2f} MiB "
                  f"[{b['transfer_state'][:4]}]"
                  f"   binds<{b['link_GB_per_s_at_which_transfer_binds']:>7.2f} GB/s   {mark}")
        else:
            print(f"  past_len {past:>5}  {r['state']:<13} would have said "
                  f"{r.get('would_have_reported_ms', float('nan')):8.4f} ms")
    print()
    cmp = c.compare(args.figure_ms, args.past_len, figure_source="phi35-certified-dev0.json")
    print(f"  {args.figure_ms} ms at past_len {args.past_len} -> {cmp['state']}")
    if cmp["quotable"]:
        print(f"    {cmp['fraction_of_roofline']:.1%} of roofline, headroom {cmp['headroom_x']:.2f}x"
              f"   floor_is_binding={cmp['floor_is_binding']}")
        print(f"    quote with: {cmp['must_be_quoted_with']}")
        print(f"    pairing:    {cmp['pairing']}")
    else:
        print(f"    {cmp['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
