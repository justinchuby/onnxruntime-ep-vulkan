"""Per-device facts that decide whether two numbers may be compared.

`environment.py` records *the machine*. This module records *the device*, at the level of detail
that determines whether a measurement means anything:

* **`timestamp_period_ns`** — nanoseconds per GPU tick. On this machine it is `1.0` on the
  RTX 4060 and **`52.0833` on the Iris Xe**. Code that assumes 1.0 under-reports Intel GPU time
  by 52x, which does not look absurd — it looks like a triumph. Measured, not assumed.
* **`timestamp_valid_bits`** — 64 on the 4060, **36** on the Iris Xe. Neither is 32, and one of
  them is the "never shift by 64" case. Determines the tick wrap period, which is reported.
* **`uma`** — whether device-local memory is also host-visible. On a UMA part an "upload" may be
  a pointer handoff; on a discrete part it is a PCIe copy. A transfer cost model fitted across
  both is meaningless, so `transfer_calibration.py` refuses to do it.
* **`max_compute_shared_memory` / `subgroup_size`** — 32 KiB vs 48 KiB here. A tile
  configuration tuned on the 4060 may not even be *selectable* on the Iris Xe, so a speedup that
  does not name its tile config is comparing two different kernels.
* **`has_calibrated_timestamps`** — whether `VK_EXT_calibrated_timestamps` is available, i.e.
  whether host<->device correlation gets `maxDeviation` or the bracketing fallback's much larger
  error bar (`docs/PERF.md` §3.7).

Source is `vulkaninfo` from the Vulkan SDK, deliberately *not* our own EP: these facts are the
ones we would use to check our EP's arithmetic, and a self-report cannot check itself. When
`vulkaninfo` is unavailable every field is `None` and a reason is recorded; nothing is guessed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

VULKANINFO_ENV = "ONNXRUNTIME_EP_VULKAN_VULKANINFO"

_CANDIDATES = ("vulkaninfoSDK", "vulkaninfo")


def _fmt_version(packed: "int | str | None") -> "str | None":
    """`VK_MAKE_API_VERSION` decode, so a fingerprint reads `1.4.325` and not `4210501`."""
    if packed is None:
        return None
    if isinstance(packed, str):
        return packed
    return f"{(packed >> 22) & 0x7F}.{(packed >> 12) & 0x3FF}.{packed & 0xFFF}"


@dataclass
class DeviceFacts:
    """The device-level facts a performance number is only meaningful alongside."""

    index: int
    name: str = "unknown"
    device_type: "str | None" = None
    api_version: "str | None" = None
    driver_version: "str | None" = None
    driver_name: "str | None" = None
    timestamp_period_ns: "float | None" = None
    timestamp_valid_bits: "int | None" = None
    compute_queue_family: "int | None" = None
    compute_queue_count: "int | None" = None
    max_compute_shared_memory: "int | None" = None
    max_workgroup_invocations: "int | None" = None
    subgroup_size: "int | None" = None
    min_subgroup_size: "int | None" = None
    max_subgroup_size: "int | None" = None
    uma: "bool | None" = None
    device_local_host_visible: "bool | None" = None
    device_local_bytes: "int | None" = None
    has_calibrated_timestamps: "bool | None" = None
    has_host_query_reset: "bool | None" = None
    # Stable identity (issue #18 — onnxruntime-ep-vulkan/rust's device selector). Sourced from
    # `VkPhysicalDeviceIDProperties`/`VkPhysicalDevicePCIBusInfoPropertiesEXT` via vulkaninfo's
    # JSON profile, formatted identically to the Rust EP's `query_device_identity`
    # (`rust/src/vk/instance.rs`) so a value copied from one side matches the other byte-for-byte:
    # `uuid` is 32 lowercase hex chars with no separators, `pci` is `domain:bus:device.function`
    # with domain/bus/device zero-padded hex and function a single hex digit.
    uuid: "str | None" = None
    luid: "str | None" = None
    pci: "str | None" = None
    notes: "list[str]" = field(default_factory=list)

    # -- derived ---------------------------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        """Stable short identity used to key results and to refuse cross-device comparisons.

        Deliberately includes the driver version: the same silicon on a different driver is a
        different device for performance purposes, and treating it otherwise is how a driver
        regression gets attributed to a code change.
        """
        parts = [
            re.sub(r"\s+", "-", self.name.strip()).lower(),
            (self.device_type or "unknown-type").lower(),
            f"api{self.api_version or '?'}",
            f"drv{self.driver_version or '?'}",
        ]
        return "/".join(parts)

    @property
    def timestamp_wrap_seconds(self) -> "float | None":
        """How long before the GPU tick counter wraps. Below ~an hour, wraps are routine."""
        if self.timestamp_valid_bits is None or not self.timestamp_period_ns:
            return None
        if self.timestamp_valid_bits <= 0:
            return None
        return (2.0**self.timestamp_valid_bits) * self.timestamp_period_ns / 1e9

    @property
    def timestamps_usable(self) -> "bool | None":
        """`timestampValidBits == 0` means this queue reports no GPU time at all — not bad time."""
        if self.timestamp_valid_bits is None:
            return None
        return self.timestamp_valid_bits > 0

    @property
    def assume_one_error_factor(self) -> "float | None":
        """How wrong a hardcoded ``timestampPeriod = 1.0`` would be on this device."""
        return self.timestamp_period_ns

    @property
    def transfer_class(self) -> str:
        """``uma`` or ``discrete`` — transfer models may not be shared across the two."""
        if self.uma is True:
            return "uma"
        if self.uma is False:
            return "discrete"
        return "unknown"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint
        d["timestamp_wrap_seconds"] = self.timestamp_wrap_seconds
        d["timestamps_usable"] = self.timestamps_usable
        d["transfer_class"] = self.transfer_class
        return d

    def summary(self) -> str:
        period = (
            f"{self.timestamp_period_ns:g} ns/tick"
            if self.timestamp_period_ns is not None
            else "period=?"
        )
        bits = (
            f"{self.timestamp_valid_bits} valid bits"
            if self.timestamp_valid_bits is not None
            else "validBits=?"
        )
        wrap = self.timestamp_wrap_seconds
        wrap_s = f", wraps every {wrap:,.0f}s" if wrap else ""
        shared = (
            f"{self.max_compute_shared_memory // 1024} KiB shared"
            if self.max_compute_shared_memory
            else "shared=?"
        )
        return (
            f"device {self.index}: {self.name} [{self.device_type or '?'}] "
            f"driver {self.driver_version or '?'}\n"
            f"    timing   : {period}, {bits}{wrap_s}\n"
            f"    compute  : {shared}, subgroup {self.subgroup_size or '?'}, "
            f"queue family {self.compute_queue_family if self.compute_queue_family is not None else '?'}\n"
            f"    transfer : {self.transfer_class}"
            f"{'  (single device-local heap: no separate system memory to copy from)' if self.uma else ''}"
            f"{'  (+ device-local host-visible BAR window — NOT unified memory)' if self.device_local_host_visible and not self.uma else ''}\n"
            f"    calibrated_timestamps: {self.has_calibrated_timestamps}"
        )


# -------------------------------------------------------------------------------------------
# vulkaninfo discovery and parsing
# -------------------------------------------------------------------------------------------


def find_vulkaninfo() -> "Path | None":
    explicit = os.environ.get(VULKANINFO_ENV)
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for name in _CANDIDATES:
            exe = Path(sdk) / "Bin" / (name + (".exe" if sys.platform == "win32" else ""))
            if exe.is_file():
                return exe
            exe = Path(sdk) / "bin" / name
            if exe.is_file():
                return exe
    for name in _CANDIDATES:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _run(exe: Path, args: "list[str]", timeout: int = 180) -> "str | None":
    try:
        out = subprocess.run(
            [str(exe), *args], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 and not out.stdout:
        return None
    return out.stdout


def _device_count(exe: Path) -> int:
    """How many physical devices vulkaninfo can see. ``--json=N`` is per device."""
    text = _run(exe, ["--summary"]) or ""
    return max(len(re.findall(r"^GPU(\d+)", text, re.M)), 0)


def _bytes_to_hex(arr: object) -> "str | None":
    """`[170, 223, ...]` (vulkaninfo's JSON byte-array encoding) -> `"aadf..."`, lowercase, no
    separators — the same encoding `query_device_identity` (`rust/src/vk/instance.rs`) produces.
    """
    if not isinstance(arr, list) or not arr:
        return None
    try:
        return "".join(f"{int(b) & 0xFF:02x}" for b in arr)
    except (TypeError, ValueError):
        return None


def _parse_json_profile(payload: dict, index: int) -> DeviceFacts:
    dev = payload["capabilities"]["device"]
    props = dev.get("properties", {})
    core = props.get("VkPhysicalDeviceProperties", {})
    limits = core.get("limits", {})
    facts = DeviceFacts(index=index, name=core.get("deviceName", "unknown"))
    facts.device_type = core.get("deviceType")
    facts.api_version = _fmt_version(core.get("apiVersion"))

    v11 = props.get("VkPhysicalDeviceVulkan11Properties", {})
    facts.subgroup_size = v11.get("subgroupSize")
    # Stable identity (issue #18): `deviceUUID` is core Vulkan 1.1 and always present here;
    # `deviceLUID` is only meaningful when the driver set `deviceLUIDValid` (mostly Windows).
    facts.uuid = _bytes_to_hex(v11.get("deviceUUID"))
    if v11.get("deviceLUIDValid"):
        facts.luid = _bytes_to_hex(v11.get("deviceLUID"))
    v12 = props.get("VkPhysicalDeviceVulkan12Properties", {})
    facts.driver_name = v12.get("driverName") or v12.get("driverID")
    # `driverInfo` is the version a human recognises ("591.55"); the packed `driverVersion`
    # integer is encoded differently by every vendor, so it is kept only as a fallback.
    facts.driver_version = v12.get("driverInfo") or (
        str(core.get("driverVersion")) if core.get("driverVersion") is not None else None
    )
    v13 = props.get("VkPhysicalDeviceVulkan13Properties", {})
    facts.min_subgroup_size = v13.get("minSubgroupSize")
    facts.max_subgroup_size = v13.get("maxSubgroupSize")

    # Stable identity (issue #18): only present when `VK_EXT_pci_bus_info` is supported —
    # absent on MoltenVK and some mobile ICDs, so `None` here is a fact, not a parsing failure.
    pci_info = props.get("VkPhysicalDevicePCIBusInfoPropertiesEXT")
    if pci_info:
        facts.pci = (
            f"{pci_info.get('pciDomain', 0):04x}:{pci_info.get('pciBus', 0):02x}:"
            f"{pci_info.get('pciDevice', 0):02x}.{pci_info.get('pciFunction', 0):x}"
        )

    facts.timestamp_period_ns = limits.get("timestampPeriod")
    facts.max_compute_shared_memory = limits.get("maxComputeSharedMemorySize")
    facts.max_workgroup_invocations = limits.get("maxComputeWorkGroupInvocations")

    # The compute queue family we would actually submit on: the first with COMPUTE.
    for qi, entry in enumerate(dev.get("queueFamiliesProperties", [])):
        qf = entry.get("VkQueueFamilyProperties", {})
        if "VK_QUEUE_COMPUTE_BIT" in qf.get("queueFlags", []):
            facts.compute_queue_family = qi
            facts.compute_queue_count = qf.get("queueCount")
            facts.timestamp_valid_bits = qf.get("timestampValidBits")
            break

    exts = set(dev.get("extensions", {}) or ())
    facts.has_calibrated_timestamps = bool(
        {"VK_EXT_calibrated_timestamps", "VK_KHR_calibrated_timestamps"} & exts
    )
    facts.has_host_query_reset = "VK_EXT_host_query_reset" in exts
    return facts


_GPU_HEADER = re.compile(r"^GPU(?P<i>\d+):?\s*$", re.M)


def _parse_memory_from_text(text: str, count: int) -> "dict[int, dict]":
    """Per device: is it UMA, and is there a device-local *host-visible* window?

    **These are two different questions and conflating them is a trap with a plausible-looking
    answer.** A discrete NVIDIA part exposes a DEVICE_LOCAL|HOST_VISIBLE memory type — the
    resizable-BAR window — so "some memory type is both" reports the RTX 4060 as unified, which
    is wrong, and wrong in the direction that would make us skip the staging copy we actually
    need to measure.

    The operational definition of UMA used here is: **no memory heap lacks DEVICE_LOCAL.** That
    is precisely the question "is there separate system memory the GPU has to copy across?".
    Verified against this machine: Iris Xe has one DEVICE_LOCAL heap (UMA); the RTX 4060 has a
    8.3 GB DEVICE_LOCAL heap plus a 34 GB non-device-local host heap (discrete), despite
    exposing a BAR window.

    Text parsing because vulkaninfo's JSON output is a profile document and omits memory
    properties entirely.
    """
    result: "dict[int, dict]" = {}
    blocks = re.split(r"\nVkPhysicalDeviceMemoryProperties:", text)[1:]
    for i, block in enumerate(blocks[:count] if count else blocks):
        heaps = re.findall(
            r"memoryHeaps\[(\d+)\]:(.*?)(?=memoryHeaps\[|memoryTypes:)", block, re.S
        )
        if not heaps:
            continue
        device_local = []
        host_only = []
        for hi, body in heaps:
            size_m = re.search(r"size\s+=\s+(\d+)", body)
            size = int(size_m.group(1)) if size_m else 0
            if "MEMORY_HEAP_DEVICE_LOCAL_BIT" in body:
                device_local.append((int(hi), size))
            else:
                host_only.append((int(hi), size))

        bar = False
        for chunk in re.split(r"memoryTypes\[\d+\]:", block)[1:]:
            head = chunk.split("usable for:", 1)[0]
            if (
                "MEMORY_PROPERTY_DEVICE_LOCAL_BIT" in head
                and "MEMORY_PROPERTY_HOST_VISIBLE_BIT" in head
            ):
                bar = True
                break

        result[i] = {
            "uma": not host_only,
            "device_local_host_visible": bar,
            "device_local_bytes": max((s for _, s in device_local), default=0),
            "host_heap_bytes": max((s for _, s in host_only), default=0),
        }
    return result


def _sweep_stale_scratch() -> None:
    """Remove ``vkfacts-*`` directories left behind by an interrupted probe.

    The scratch directory has to live under ``bench/`` rather than the system temp directory.
    That means an interrupted run leaves a directory git can see, and one of them was committed
    before this sweep existed. `TemporaryDirectory` already cleans up on the normal path; this
    covers the abnormal one, at the start of the next probe.
    """
    here = Path(__file__).resolve().parent
    for stale in here.glob("vkfacts-*"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)


def probe(exe: "Path | None" = None) -> "tuple[list[DeviceFacts], str]":
    """Return ``(devices, note)``. A non-empty note means the facts could not be obtained."""
    _sweep_stale_scratch()
    exe = exe or find_vulkaninfo()
    if exe is None:
        return [], (
            f"vulkaninfo not found (set {VULKANINFO_ENV} or VULKAN_SDK) — per-device timing "
            f"facts unavailable; GPU timings from this run cannot be validated"
        )
    count = _device_count(exe)
    if count == 0:
        return [], "vulkaninfo reported no GPUs"

    devices: "list[DeviceFacts]" = []
    with tempfile.TemporaryDirectory(prefix="vkfacts-", dir=str(Path(__file__).parent)) as td:
        for i in range(count):
            out = Path(td) / f"dev{i}.json"
            if _run(exe, [f"--json={i}", "-o", str(out)]) is None and not out.is_file():
                devices.append(
                    DeviceFacts(index=i, notes=[f"vulkaninfo --json={i} produced no output"])
                )
                continue
            try:
                payload = json.loads(out.read_text("utf-8", "replace"))
                devices.append(_parse_json_profile(payload, i))
            except (OSError, ValueError, KeyError) as exc:
                devices.append(DeviceFacts(index=i, notes=[f"unparseable vulkaninfo JSON: {exc}"]))

        text = _run(exe, ["-o", str(Path(td) / "all.txt")])
        try:
            text = (Path(td) / "all.txt").read_text("utf-8", "replace")
        except OSError:
            text = text or ""

    if text:
        mem = _parse_memory_from_text(text, count)
        for d in devices:
            if d.index in mem:
                d.uma = mem[d.index]["uma"]
                d.device_local_host_visible = mem[d.index]["device_local_host_visible"]
                d.device_local_bytes = mem[d.index]["device_local_bytes"] or None
            else:
                d.notes.append("memory properties unavailable — UMA/discrete class unknown")
    else:
        for d in devices:
            d.notes.append("memory properties unavailable — UMA/discrete class unknown")

    return devices, ""


def capture() -> dict:
    """JSON-serialisable per-device record for embedding in a result file."""
    devices, note = probe()
    return {
        "devices": [d.to_dict() for d in devices],
        "note": note,
        "source": "vulkaninfo (Vulkan SDK)",
    }


def by_index(record: dict, index: int) -> "dict | None":
    for d in record.get("devices", []):
        if d.get("index") == index:
            return d
    return None


# ---------------------------------------------------------------------------------------------
# Which index is `ep.device_index`?
#
# There are TWO orderings on this machine and they are not the same one:
#
#   * `vkEnumeratePhysicalDevices` order — what `vulkaninfo` prints, what `probe()` returns,
#     and what `epctl --probe-loader` labels "Device N". On this laptop: 0 = Intel, 1 = NVIDIA.
#   * **best-first** order — `engine.rs::probe_devices` is documented "Sorted best-first by
#     DeviceKind::score, so index 0 is the default device", and `ep.device_index` indexes into
#     *that* list. Discrete (4) outranks Integrated (3), so on this laptop 0 = NVIDIA, 1 = Intel.
#
# A benchmark that passes `ep.device_index = 0` and then labels the row with `probe()[0]` prints
# the Intel name over NVIDIA numbers. That is not a rounding error, it is a mislabelled device,
# and it is exactly the class of defect that makes a results table worse than no table. So the
# index is never trusted: the row is labelled from the *trace's own* timestamp fingerprint and
# `device_identity_check` goes red if the two disagree.
# ---------------------------------------------------------------------------------------------

_KIND_SCORE = {
    "discrete": 4,
    "integrated": 3,
    "virtual": 2,
    "cpu": 1,
}


def _kind_score(device_type: "str | None") -> int:
    t = (device_type or "").lower()
    for key, score in _KIND_SCORE.items():
        if key in t:
            return score
    return 0


def ep_selection_order(devices: "list[DeviceFacts]") -> "list[DeviceFacts]":
    """The devices in the order ``ep.device_index`` selects them: best-first, ties by enum order.

    Mirrors ``engine.rs::probe_devices``' documented contract. Returned in order; element *i* is
    the device an ``ep.device_index = i`` session binds.
    """
    return sorted(devices, key=lambda d: (-_kind_score(d.device_type), d.index))


def ep_index_of(devices: "list[DeviceFacts]", enumeration_index: int) -> "int | None":
    """``ep.device_index`` value that selects the device at ``vkEnumeratePhysicalDevices`` index."""
    for pos, d in enumerate(ep_selection_order(devices)):
        if d.index == enumeration_index:
            return pos
    return None


def by_ep_index(devices: "list[DeviceFacts]", ep_index: int) -> "DeviceFacts | None":
    order = ep_selection_order(devices)
    if 0 <= ep_index < len(order):
        return order[ep_index]
    return None


def identify_by_timestamp(
    devices: "list[DeviceFacts]",
    period_ns: "float | None",
    valid_bits: "int | None",
) -> "tuple[DeviceFacts | None, str]":
    """Name the device a trace came from, using only facts carried *in the trace*.

    ``timestampPeriod``/``timestampValidBits`` are a strong discriminator here: 52.0833/36 on the
    Intel part, 1.0/64 on the NVIDIA part. Returns ``(device, reason)``; ``device`` is ``None``
    whenever the fingerprint is absent or fails to pick out exactly one device, because a guess is
    not an identification.
    """
    if period_ns is None and valid_bits is None:
        return None, "trace carries no timestamp fingerprint (was TRACE_GPU set?)"

    matches = []
    for d in devices:
        if period_ns is not None and d.timestamp_period_ns is not None:
            if abs(d.timestamp_period_ns - period_ns) > 1e-3:
                continue
        if valid_bits is not None and d.timestamp_valid_bits is not None:
            if d.timestamp_valid_bits != valid_bits:
                continue
        matches.append(d)

    if len(matches) == 1:
        return matches[0], (
            f"timestamp fingerprint period={period_ns} bits={valid_bits} matches exactly one device"
        )
    if not matches:
        return None, (
            f"timestamp fingerprint period={period_ns} bits={valid_bits} matches NO probed device"
        )
    names = ", ".join(m.name for m in matches)
    return None, (
        f"timestamp fingerprint period={period_ns} bits={valid_bits} is ambiguous between: {names}"
    )


def identify_by_uuid(devices: "list[DeviceFacts]", uuid: "str | None") -> "tuple[DeviceFacts | None, str]":
    """Name the device an evidence/proof frame came from using the stable Vulkan device UUID.

    This is a strictly stronger discriminator than :func:`identify_by_timestamp`: the UUID (issue
    #18) is an exact identity, not a fingerprint that can coincide between two identical GPUs.
    Prefer this whenever the evidence carries a ``uuid`` (e.g. the modelrunner's
    ``devices_seen``/``ep_device`` JSON fields, or the EP's ``vulkan.device_uuid`` metadata); fall
    back to :func:`identify_by_timestamp` only when no UUID was recorded (older evidence, or an
    ICD that never populated one). Returns ``(device, reason)``; ``device`` is ``None`` whenever
    the UUID is absent or does not name exactly one currently-probed device, because a guess is
    not an identification.
    """
    if not uuid:
        return None, "evidence carries no device uuid (issue #18 identity was not recorded)"

    normalized = uuid.lower()
    matches = [d for d in devices if d.uuid and d.uuid.lower() == normalized]

    if len(matches) == 1:
        return matches[0], f"uuid {uuid} matches exactly one probed device"
    if not matches:
        return None, f"uuid {uuid} matches NO probed device (stale evidence or device removed?)"
    # A UUID is defined to be globally unique per physical device; more than one probed device
    # reporting the same UUID means `probe()` itself double-counted a device, not a real ambiguity.
    names = ", ".join(m.name for m in matches)
    return None, f"uuid {uuid} matches more than one probed device (probe() bug?): {names}"


def device_identity_check(
    devices: "list[DeviceFacts]",
    ep_index: int,
    period_ns: "float | None",
    valid_bits: "int | None",
    uuid: "str | None" = None,
) -> dict:
    """Falsifier: does the device we *labelled* the row with match the device that actually ran?

    Goes red when the fingerprint in the trace names a different device than the one
    ``ep.device_index = ep_index`` was assumed to select. On red the caller must withhold the
    device name entirely rather than print a plausible wrong one.

    ``uuid`` (issue #18), when the evidence carries one, is checked in preference to the
    timestamp fingerprint: it is an exact identity rather than a fingerprint that can coincide
    between two identical GPUs. Callers with no recorded uuid keep the prior timestamp-only
    behaviour unchanged — this parameter is additive and optional.
    """
    assumed = by_ep_index(devices, ep_index)
    observed, why = (
        identify_by_uuid(devices, uuid)
        if uuid
        else identify_by_timestamp(devices, period_ns, valid_bits)
    )

    out = {
        "check": "device_identity",
        "ep_device_index": ep_index,
        "assumed_from_ep_order": assumed.name if assumed else None,
        "observed_from_trace": observed.name if observed else None,
        "reason": why,
        "asserts": (
            "the device named on a results row is the device whose timestamp fingerprint appears "
            "in that row's trace"
        ),
    }

    if observed is None:
        out["ok"] = None
        out["verdict"] = "UNVERIFIED"
        out["detail"] = f"device identity not established from the trace: {why}"
        out["name_may_be_quoted"] = False
        return out

    if assumed is None:
        out["ok"] = True
        out["verdict"] = "TRACE_ONLY"
        out["detail"] = f"no assumption to check; trace says {observed.name}"
        out["name_may_be_quoted"] = True
        out["device"] = observed
        return out

    same = observed.index == assumed.index
    out["ok"] = same
    out["verdict"] = "MATCH" if same else "MISLABELLED"
    out["name_may_be_quoted"] = True
    out["device"] = observed
    out["detail"] = (
        f"trace and ep-order agree: {observed.name}"
        if same
        else (
            f"ep.device_index={ep_index} was assumed to be {assumed.name!r} but the trace's "
            f"timestamp fingerprint is {observed.name!r} — row relabelled from the trace"
        )
    )
    return out


if __name__ == "__main__":  # pragma: no cover - manual use
    found, why = probe()
    if why:
        print(why)
    for dev in found:
        print(dev.summary())
        print()
    if len(found) > 1:
        periods = {d.timestamp_period_ns for d in found if d.timestamp_period_ns is not None}
        if len(periods) > 1:
            print(
                "NOTE: these devices do not share a timestampPeriod "
                f"({sorted(periods)}). A hardcoded 1.0 would be wrong by up to "
                f"{max(periods):g}x on this machine alone."
            )
        classes = {d.transfer_class for d in found}
        if len(classes - {"unknown"}) > 1:
            print(
                "NOTE: these devices do not share a transfer class "
                f"({sorted(classes)}). Fit transfer models separately; a single model is "
                "meaningless across UMA and discrete parts."
            )
