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

WHAT IT REFUSES, AND THE ONE THAT MATTERS
=========================================
The obvious refusal is a context length nobody has run. The important one is different and
was found while wiring this up:

**`GroupQueryAttention` is declined on this build and executes on CPU.** Its proof verdict is
`DIVERGENT` (worst_rel 16.73), all 32 instances decline, and that is why the graph carries 33
islands and 323 claimed nodes rather than one island and 355. GQA is *the op that reads the KV
cache*. So on this build the KV-cache bytes are not GPU DRAM traffic at all.

`island_bytes_phi35.json` charges them to the GPU roofline anyway -- 48 MiB at past_len 128,
3072 MiB at 8192, where they become 60.5% of the modelled stream. Against this build that is a
bound on a machine we are not running, and the achieved figure it would be compared against
came from a GPU that never read those bytes.

R12: a term whose event cannot occur in the frame reports **UNOBSERVABLE**, never a number --
and here, notably, never `0` either. `0` would claim the traffic is free; `UNOBSERVABLE`
says we are not entitled to a GPU bandwidth bound at that context at all.

The consequence is worth stating plainly, because it cuts the right way:

    **At past_len = 0 the KV term is exactly zero, so the question does not arise, and the
    ceiling is admissible. That is the only context this project has ever run, and
    12.1847 ms -- zero context, one token -- is the only quotable figure we hold.**

The one admissible bound and the one quotable figure sit at the same context. That is not a
coincidence and it is why the comparison survives.

FALSIFIER
=========
The KV claim above is structural: it follows from GQA declining, which is a count read off
this binary's own counters (`subgraphs_live` 33, 323 of 363 claimed). It would be falsified by
a staging measurement at past_len > 0 showing KV-scale bytes crossing to the device. We run at
zero context, so no such measurement exists; if GQA is ever claimed, this module's extent
widens and that is a code change, not a judgement call.

Usage:
    from ceiling import Ceiling
    c = Ceiling.load()
    c.floor_ms(0)              -> BOUND, 8.2180 ms
    c.floor_ms(8192)           -> UNOBSERVABLE
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

#: The run record the GQA claim status is read from. It must be in the same binary frame as
#: any figure this ceiling is used to judge, which is why the DLL hash travels in the frame.
DEFAULT_CLAIM_RECORD = _RESULTS / "phi35-0baf660-dev0.json"

#: Total nodes probed in this graph. 355 of these in one island was the pre-ledger record;
#: 323 in 33 islands is what a build with GQA declined produces.
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

    def __init__(self, roofline: dict, by_context: list, claim: dict, frame: dict):
        self._roofline = roofline
        self._by_context = {int(r["past_sequence_length"]): r for r in by_context}
        self._claim = claim
        self._frame = frame

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, claim_record: "Path | None" = None) -> "Ceiling":
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

        claim_path = Path(claim_record) if claim_record else DEFAULT_CLAIM_RECORD
        claim = cls._read_claim(claim_path)

        dll = _ROOT / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
        frame = {
            "model": roofline.get("model"),
            "device": roofline.get("device"),
            "peak_GB_per_s": roofline["roofline_dram"]["peak_GB_per_s"],
            "peak_source": "spec: 128-bit GDDR6 at 16 Gbps",
            "byte_model_source": ROOFLINE_RECORD.name,
            "by_context_source": ISLAND_BYTES_RECORD.name,
            "claim_source": claim_path.name,
            "dll_sha256": _sha256(dll),
            "matmulnbits_nodes": roofline.get("matmulnbits_nodes"),
        }
        return cls(roofline, by_context, claim, frame)

    @staticmethod
    def _read_claim(path: Path) -> dict:
        """Read the build's fusion/claim status. This is what decides the extent."""
        if not path.is_file():
            return {
                "state": UNOBSERVABLE,
                "reason": f"no run record at {path.name}; the build's claim status is unknown, "
                          "so we cannot say which ops execute on the device.",
            }
        rec = json.loads(path.read_text(encoding="utf-8"))["results"][0]
        counters = rec.get("counters", {})
        islands = counters.get("subgraphs_live")
        claimed = rec.get("claimed_nodes")
        # 32 GQA instances declining cuts a chain into 33 segments. The island count is the
        # in-frame, contention-independent witness that they declined.
        gqa_declined = islands is not None and islands > 1
        return {
            "state": "READ",
            "islands": islands,
            "claimed_nodes": claimed,
            "nodes_probed": NODES_PROBED,
            "gqa_declined": gqa_declined,
            "reason": (
                f"{islands} islands: the 32 GroupQueryAttention instances declined (proof verdict "
                "DIVERGENT) and execute on CPU, so the KV cache is not read by the device."
                if gqa_declined else
                f"{islands} island: GroupQueryAttention is claimed, so the device does read the "
                "KV cache and the KV term of the bound is live."
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
        return {
            **out,
            "state": BOUND,
            "quotable": True,
            "floor_ms_at_spec_peak": floor,
            "fraction_of_roofline": floor / figure_ms,
            "headroom_x": figure_ms / floor,
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
    print("=" * 78)
    print("COUNT-BASED CEILING -- frame, extent, refusal")
    print("=" * 78)
    print(f"  device      {c.frame()['device']}")
    print(f"  peak        {c.frame()['peak_GB_per_s']:.1f} GB/s ({c.frame()['peak_source']})")
    print(f"  dll         {str(c.frame()['dll_sha256'])[:16]}")
    print(f"  grid        {ext['grid']}")
    print(f"  ADMISSIBLE  {ext['admissible']}")
    print(f"  because     {ext['reason']}")
    print()
    for past in ext["grid"]:
        r = c.floor_ms(past)
        if r["state"] == BOUND:
            print(f"  past_len {past:>5}  {BOUND:<13} floor {r['floor_ms_at_spec_peak']:8.4f} ms"
                  f"   (KV {r['kv_cache_MiB']:.0f} MiB)")
        else:
            print(f"  past_len {past:>5}  {r['state']:<13} would have said "
                  f"{r.get('would_have_reported_ms', float('nan')):8.4f} ms")
    print()
    cmp = c.compare(args.figure_ms, args.past_len, figure_source="phi35-certified-dev0.json")
    print(f"  {args.figure_ms} ms at past_len {args.past_len} -> {cmp['state']}")
    if cmp["quotable"]:
        print(f"    {cmp['fraction_of_roofline']:.1%} of roofline, headroom {cmp['headroom_x']:.2f}x")
        print(f"    quote with: {cmp['must_be_quoted_with']}")
    else:
        print(f"    {cmp['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
