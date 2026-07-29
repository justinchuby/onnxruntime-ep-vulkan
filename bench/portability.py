"""Portability envelope — is this configuration selectable on every device we admit?

Justin's standing directive: *要时刻注意跨平台通用性* — cross-platform generality, at all times.
A Vulkan EP that is really a desktop-NVIDIA EP has no reason to exist.

The performance version of that directive is narrow and checkable. **A configuration measured on
this desk is only a statement about the EP if that configuration is selectable on the devices the
EP admits.** A tile that needs 48 KiB of shared memory is a fine thing to measure on an RTX 4060,
but the number it produces describes a code path that a device admitted by `DESIGN.md` §7.2 may
never take — and quoting it as "the EP's throughput" is the same class of error as quoting a CPU
run as a Vulkan number (`PERF.md` §5.1). It is wrong in a way that looks entirely reasonable.

So this module encodes the **admission floor** — not a guess about mobile hardware, which we do
not own and must not invent numbers for, but the limits this project has already *decided* it will
admit:

* `DESIGN.md` §7.2 R3: ``maxComputeWorkGroupInvocations >= 256``.
* `DESIGN.md` §7.2 R4: ``maxComputeSharedMemorySize >= 16384`` (16 KiB — the Vulkan 1.0 required
  minimum, and our floor because R1 admits Vulkan 1.1).

Those two numbers are the whole envelope. A device that meets them is advertised to ORT; §7.0 says
capability shortfalls degrade **op coverage**, not **device availability**. So a device sitting
exactly on the floor is a device we promised to run on, and it has **16 KiB**, not 32, and not 48.

Note what this rules out that is easy to get wrong: the Iris Xe on this desk has 32 KiB, twice the
floor. It is our closest available proxy for the mobile memory model (it is UMA, as Adreno and Mali
are) but it is **not** a proxy for the mobile *shared-memory* budget. Passing on the Iris Xe is not
evidence that a 32 KiB tile is portable.

Subgroup size is treated the same way: Vulkan 1.1 guarantees the ``BASIC`` subgroup feature in the
compute stage and *nothing about the size*. Both GPUs here report 32, which is exactly the
coincidence most likely to bake a 32 into something. A configuration that names a subgroup size is
recorded as depending on it; it is never assumed.

This module states no performance number and never will. It answers one question about a
configuration: *could a device we admit even select this?*
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: `DESIGN.md` §7.2 R4 — the shared-memory budget every admitted device is guaranteed to have.
FLOOR_SHARED_MEMORY_BYTES = 16384

#: `DESIGN.md` §7.2 R3 — the workgroup size every admitted device is guaranteed to support.
FLOOR_WORKGROUP_INVOCATIONS = 256

#: Vulkan 1.1 guarantees subgroup ``BASIC`` in compute and says nothing about ``subgroupSize``.
#: There is no floor to check against; a config that depends on a size must say so.
SUBGROUP_SIZE_IS_GUARANTEED = False

#: Verdicts, in increasing order of "do not quote this as a property of the EP".
PORTABLE = "portable"
NEEDS_FALLBACK = "needs-fallback"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Configuration:
    """The knobs that decide *which kernel ran*, as opposed to how fast it ran.

    Every field is optional because the engine does not report them yet (`bench.py` writes
    ``tile_config: None``). Missing is recorded as :data:`UNKNOWN` and never as "fine" — the same
    rule as ``tile_config`` in ``compare.py``: two unknowns are unknown, never equal.
    """

    name: "str | None" = None
    shared_memory_bytes: "int | None" = None
    workgroup_invocations: "int | None" = None
    #: True when the kernel's correctness or shape depends on a particular subgroup size.
    depends_on_subgroup_size: bool = False
    subgroup_size: "int | None" = None
    #: True when the kernel assumes device-local memory is host-visible (a UMA shortcut).
    assumes_unified_memory: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shared_memory_bytes": self.shared_memory_bytes,
            "workgroup_invocations": self.workgroup_invocations,
            "depends_on_subgroup_size": self.depends_on_subgroup_size,
            "subgroup_size": self.subgroup_size,
            "assumes_unified_memory": self.assumes_unified_memory,
            "notes": self.notes,
        }


@dataclass
class Verdict:
    """Whether a configuration is selectable on every device the EP admits."""

    verdict: str
    reasons: "list[str]" = field(default_factory=list)

    @property
    def quotable_as_ep_behaviour(self) -> bool:
        """Whether a number from this configuration describes the EP rather than this desk.

        ``UNKNOWN`` is deliberately not quotable. A configuration nobody recorded is not a
        configuration anybody can reproduce.
        """
        return self.verdict == PORTABLE

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "quotable_as_ep_behaviour": self.quotable_as_ep_behaviour,
        }


def evaluate(config: Configuration) -> Verdict:
    """Judge ``config`` against the §7.2 admission floor.

    Returns :data:`NEEDS_FALLBACK` — not "fails" — when the configuration exceeds the floor,
    because exceeding it is legitimate and expected: §7.0's rule is that shortfalls degrade op
    coverage, not device availability, which means a bigger tile is allowed to exist *provided a
    floor-compliant path exists too*. The verdict's job is to stop the fast path's number from
    being reported as the EP's number without the fallback being named.
    """
    reasons: "list[str]" = []
    unknown = False

    if config.shared_memory_bytes is None:
        unknown = True
        reasons.append("shared memory budget not recorded")
    elif config.shared_memory_bytes > FLOOR_SHARED_MEMORY_BYTES:
        reasons.append(
            f"uses {config.shared_memory_bytes} B of shared memory, above the §7.2 R4 floor of "
            f"{FLOOR_SHARED_MEMORY_BYTES} B that every admitted device is guaranteed to have. "
            f"Selectable here (Iris Xe 32 KiB, RTX 4060 48 KiB) and not selectable on a device "
            f"sitting on the floor — a floor-compliant fallback must exist and be named."
        )

    if config.workgroup_invocations is None:
        unknown = True
        reasons.append("workgroup size not recorded")
    elif config.workgroup_invocations > FLOOR_WORKGROUP_INVOCATIONS:
        reasons.append(
            f"dispatches {config.workgroup_invocations} invocations per workgroup, above the "
            f"§7.2 R3 floor of {FLOOR_WORKGROUP_INVOCATIONS}."
        )

    if config.depends_on_subgroup_size:
        if config.subgroup_size is None:
            unknown = True
            reasons.append("depends on subgroup size but does not record which")
        else:
            reasons.append(
                f"depends on subgroupSize == {config.subgroup_size}. Vulkan 1.1 guarantees "
                f"subgroup BASIC in compute and nothing about the size; both GPUs on this desk "
                f"report 32, which is the coincidence most likely to bake a 32 into a kernel."
            )

    if config.assumes_unified_memory:
        reasons.append(
            "assumes device-local memory is host-visible. True on the Iris Xe and on Adreno/Mali; "
            "false on the RTX 4060 except through the resizable-BAR window, which is not unified "
            "memory (see devices.py). A staging path must still exist."
        )

    if unknown:
        return Verdict(UNKNOWN, reasons)
    if reasons:
        return Verdict(NEEDS_FALLBACK, reasons)
    return Verdict(
        PORTABLE,
        [
            f"within the §7.2 admission floor ({FLOOR_SHARED_MEMORY_BYTES} B shared, "
            f"{FLOOR_WORKGROUP_INVOCATIONS} invocations); selectable on every device the EP "
            f"advertises."
        ],
    )


def fits_device(config: Configuration, shared_memory_bytes: int,
                workgroup_invocations: int) -> bool:
    """Whether ``config`` is selectable on a device with these reported limits.

    Used to answer the concrete local question — "is the tile I measured on the 4060 even
    selectable on the Iris Xe?" — from the device's *reported* limits rather than from a constant.
    An unrecorded requirement returns ``False``: a configuration whose needs are unknown cannot be
    shown to fit.
    """
    if config.shared_memory_bytes is None or config.workgroup_invocations is None:
        return False
    return (
        config.shared_memory_bytes <= shared_memory_bytes
        and config.workgroup_invocations <= workgroup_invocations
    )


def transfer_model_merge_refusal(fits: "list[dict]") -> "str | None":
    """Refuse to combine transfer-cost fits taken across different transfer classes.

    A UMA part and a discrete part are not the same measurement with different constants: on the
    Iris Xe the "upload" may be a mapped write with no copy at all, while on the RTX 4060 it is a
    staging buffer and a PCIe DMA. An affine model fitted across both describes neither, and the
    blended constant would be plausible — between the two — which is exactly the failure mode
    worth refusing rather than warning about.

    ``transfer_calibration.py`` makes this hard to reach (one device per run, and the emitted Rust
    literal is stamped with the transfer class), but a human combining two JSON files by hand is
    the obvious way in.
    """
    classes = sorted({f.get("transfer_class") for f in fits}, key=lambda c: (c is None, c or ""))
    if len(classes) <= 1 and None not in classes:
        return None
    if None in classes:
        return (
            "one or more transfer fits do not record a transfer class. A fit that does not say "
            "whether it came from a UMA or a discrete part cannot be combined with anything, "
            "because the two are not the same measurement."
        )
    return (
        f"refusing to combine transfer fits across transfer classes {classes}. On a UMA part an "
        f"upload may be a mapped write with no copy; on a discrete part it is a staging buffer "
        f"and a bus transfer. A single affine model fitted across both describes neither, and its "
        f"constants would land plausibly between the two. Fit them separately (PERF.md §1.4)."
    )


def describe(config: Configuration, verdict: Verdict) -> str:
    lines = [f"config  : {config.name or 'unnamed'} — {verdict.verdict}"]
    lines += [f"          · {r}" for r in verdict.reasons]
    if not verdict.quotable_as_ep_behaviour:
        lines.append(
            "          ⚠ a number from this configuration describes this desk, not the EP."
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual use
    for c in [
        Configuration(name="floor-compliant", shared_memory_bytes=16384,
                      workgroup_invocations=256),
        Configuration(name="4060-tuned 48KiB", shared_memory_bytes=49152,
                      workgroup_invocations=256),
        Configuration(name="xe-tuned 32KiB", shared_memory_bytes=32768, workgroup_invocations=256),
        Configuration(name="unrecorded"),
    ]:
        print(describe(c, evaluate(c)))
        print()
