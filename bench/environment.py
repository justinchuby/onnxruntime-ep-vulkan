"""Environment capture — every number this harness prints is stamped with this.

A timing without the machine it was taken on is a rumour. `capture()` collects, in one
JSON-serialisable dict:

* **host** — OS, release, arch, CPU model and core count, Python and ORT versions.
* **build** — which EP artifact was benchmarked: path, size, mtime, cargo profile as inferred
  from the path, and whether the build had shaders at all (an
  ``ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1`` artifact advertises zero devices and can
  produce no GPU number, which must be visible in the record rather than inferred from an
  odd-looking result).
* **devices** — every Vulkan device the EP's own loader probe reports. Parsed from
  ``epctl --probe-loader``: the EP's own view, not a second opinion from another tool that
  might disagree with it.
* **env** — the ``ONNXRUNTIME_EP_VULKAN_*`` variables in effect, because several of them change
  what is measured (validation layers on is a different machine, for benchmarking purposes).
* **producers** — who built the graphs. A benchmark artefact is relative to its producer in
  exactly the way op coverage is (``OP_COVERAGE.md`` §4.18): two exporters emit different op sets
  for the same architecture, so the graph's origin belongs next to the device and the driver.
  See ``producers.py``.
* **contention** — how busy the machine was. Added after the same device, build and test were
  measured 9.5× apart on host recording time depending only on what else was running. A stored
  number whose environment record omits the machine's load cannot be re-checked later, because
  the largest single influence on it left no trace. See ``contention.py``.

Nothing here fails a run. A missing ``epctl`` yields ``devices: []`` and a recorded reason.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

EP_ENV_PREFIX = "ONNXRUNTIME_EP_VULKAN_"
EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
EPCTL_ENV = "ONNXRUNTIME_EP_VULKAN_EPCTL"

#: ``Device 0 [Vulkan enum index 0]: Intel(R) Iris(R) Xe Graphics [Vulkan 1.4.309]`` and friends.
#: The ``[Vulkan enum index N]`` annotation was added to make the two index spaces explicit;
#: both the old (``Device N: Name``) and new format are accepted.
#: The call site strips trailing ``— gate PASS/FAIL`` before matching.
_DEVICE_LINE = re.compile(
    r"Device\s+(?P<index>\d+)(?:\s*\[Vulkan enum index \d+\])?\s*:\s*(?P<name>.+?)\s*(?:\[(?P<detail>[^\]]*)\])?\s*$"
)


def _cpu_model() -> str:
    """Best-effort CPU model string, without adding a dependency for it."""
    if sys.platform == "win32":
        return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor() or "unknown")
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text("utf-8", "replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _find_epctl() -> "Path | None":
    """Locate ``epctl``: explicit env var, then next to the EP artifact, then PATH."""
    explicit = os.environ.get(EPCTL_ENV)
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    lib = os.environ.get(EP_LIB_ENV)
    if lib:
        exe = "epctl.exe" if sys.platform == "win32" else "epctl"
        candidate = Path(lib).resolve().parent / exe
        if candidate.is_file():
            return candidate
    found = shutil.which("epctl")
    return Path(found) if found else None


def probe_devices() -> "tuple[list[dict], str]":
    """Return ``(devices, note)`` from ``epctl --probe-loader``.

    The note is non-empty exactly when the device list could not be obtained, and it is
    recorded in the result rather than printed and forgotten — a benchmark run with no device
    record is not one you can cite later.
    """
    epctl = _find_epctl()
    if epctl is None:
        return [], (
            f"epctl not found (set {EPCTL_ENV}, or build it: "
            f"cargo build --release --bin epctl) — no device record for this run"
        )
    try:
        out = subprocess.run(
            [str(epctl), "--probe-loader"], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        return [], f"epctl --probe-loader failed: {exc}"

    text = (out.stdout or "") + (out.stderr or "")
    devices: "list[dict]" = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("Device"):
            continue
        # Strip trailing "— gate PASS/FAIL" annotation added by probe_loader_report so the
        # end-anchored regex continues to work.
        for sep in ("  —", " —", "—"):
            if sep in line:
                line = line[:line.index(sep)].rstrip()
                break
        m = _DEVICE_LINE.match(line)
        if not m:
            continue
        devices.append(
            {
                "index": int(m.group("index")),
                "name": m.group("name").strip(),
                "detail": (m.group("detail") or "").strip(),
                "line": line,
            }
        )
    note = "" if devices else "epctl reported no Vulkan devices (no ICD, or all gated out)"
    return devices, note


def _sha256(path: Path) -> "str | None":
    """Hash the artifact. The frame of a test result is the binary that ran it, so a record
    that names its binary only by size and mtime cannot be checked against a later build."""
    try:
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def build_info() -> dict:
    """Describe the EP artifact under test."""
    lib = os.environ.get(EP_LIB_ENV)
    info: dict = {"lib": lib, "present": False}
    if not lib:
        info["note"] = f"{EP_LIB_ENV} is unset — the EP was not benchmarked"
        return info
    p = Path(lib)
    if not p.is_file():
        info["note"] = f"{EP_LIB_ENV} points at a file that does not exist: {lib}"
        return info
    stat = p.stat()
    parts = {part.lower() for part in p.resolve().parts}
    profile = "release" if "release" in parts else "debug" if "debug" in parts else "unknown"
    info.update(
        {
            "present": True,
            "lib": str(p.resolve()),
            "bytes": stat.st_size,
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
            "profile": profile,
            "sha256": _sha256(p),
        }
    )
    if os.environ.get("ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC") == "1":
        info["shaders"] = "MISSING — built with ALLOW_MISSING_GLSLC=1; advertises zero devices"
    return info


def source_commit() -> dict:
    """Which commit the code under test came from, and whether the tree was clean.

    A stored number's frame is the *source* as much as it is the binary: ``build.sha256``
    identifies the artifact that ran, but only a commit says what it was built from, and only
    the dirty flag says whether that commit describes it. A record naming a commit taken from a
    dirty tree names a state that exists on one disk and nowhere else.

    ``dirty`` is ``None``, never ``False``, when git could not be consulted. "Unknown" and
    "clean" must not collapse: ``bench/attribution_gate.py::environment_witnesses`` refuses
    anything that is not exactly ``False``, and it can only do that if this reports the
    difference.
    """
    repo = Path(__file__).resolve().parents[1]
    out: dict = {"commit": None, "dirty": None, "note": None}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=30
        )
        if head.returncode != 0:
            out["note"] = f"git rev-parse HEAD failed: {(head.stderr or '').strip()[:200]}"
            return out
        out["commit"] = head.stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, timeout=60
        )
        if status.returncode != 0:
            out["note"] = "git status failed; the tree's cleanliness is unknown, not clean"
            return out
        out["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        out["note"] = f"git unavailable: {exc!r}; the commit is unknown, not absent"
    return out


def capture(load_seconds: float = 0.0) -> dict:
    """Collect the full environment record.

    ``load_seconds > 0`` adds a machine-load spot-check. It is off by default because ``capture``
    is called from several places that must stay instant, and it is *not* a substitute for the
    monitor that runs across a measurement: a spot-check taken before a forty-minute run says
    nothing about minute twenty. Its purpose is to make a stored artifact re-checkable — see
    ``contention_note``.
    """
    try:
        import onnxruntime as ort

        ort_version = ort.__version__
    except Exception:  # pragma: no cover - ORT optional here, required by bench.py
        ort_version = None

    devices, device_note = probe_devices()
    try:
        import devices as device_mod

        device_facts = device_mod.capture()
    except Exception:  # pragma: no cover - never fail a run over the environment record
        device_facts = {"devices": [], "note": "device fact probe unavailable"}
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {
            "os": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "onnxruntime": ort_version,
        },
        "build": build_info(),
        "source_commit": source_commit(),
        "devices": devices,
        "device_note": device_note,
        "device_facts": device_facts,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(EP_ENV_PREFIX)},
        "contention": _contention_spot_check(load_seconds),
        "contention_note": (
            "How busy the machine was. This belongs next to the driver version and the CPU model "
            "for the same reason those do: it changes the answer. The same device, build and test "
            "measured 9.5x apart on host command-buffer recording depending only on what else was "
            "running (docs/PERF.md §10). Without this field a stored number cannot be re-checked "
            "later, because the one thing that most affected it left no trace. The authoritative "
            "record is the per-pass `machine_quiescence` in the result artifact, which covers the "
            "whole measurement; this is a spot-check of the moment the environment was captured."
        ),
    }


def _contention_spot_check(seconds: float) -> dict:
    if seconds <= 0:
        return {"verdict": "UNMEASURED",
                "reason": "no load sample was requested at capture time; see the per-pass "
                          "machine_quiescence records, which are the authoritative ones"}
    try:
        import contention as contention_mod

        return contention_mod.quiescence(
            contention_mod.sample_now(seconds), contention_mod.occupancy_check())
    except Exception as exc:  # pragma: no cover - never fail a run over the environment record
        return {"verdict": "UNMEASURED", "reason": f"load probe unavailable: {exc!r}"}


def describe(record: dict) -> str:
    """One-screen human summary of a captured environment."""
    host = record.get("host", {})
    build = record.get("build", {})
    lines = [
        f"host    : {host.get('os')} {host.get('release')} ({host.get('machine')})",
        f"cpu     : {host.get('cpu')} x{host.get('cpu_count')}",
        f"runtime : onnxruntime {host.get('onnxruntime')} / python {host.get('python')}",
        f"build   : {build.get('lib')} [{build.get('profile')}]",
    ]
    if build.get("note"):
        lines.append(f"          note: {build['note']}")
    if build.get("shaders"):
        lines.append(f"          shaders: {build['shaders']}")
    for d in record.get("devices", []):
        lines.append(f"device  : {d['index']}: {d['name']} {d['detail']}".rstrip())
    for d in record.get("device_facts", {}).get("devices", []):
        period = d.get("timestamp_period_ns")
        lines.append(
            f"          [{d.get('index')}] {d.get('transfer_class')}, "
            f"{period if period is not None else '?'} ns/tick, "
            f"{d.get('timestamp_valid_bits')} valid bits, "
            f"{(d.get('max_compute_shared_memory') or 0) // 1024} KiB shared, "
            f"driver {d.get('driver_version')}"
        )
    if record.get("device_note"):
        lines.append(f"          note: {record['device_note']}")
    c = record.get("contention") or {}
    if c:
        s = c.get("survey") or {}
        detail = (f" — foreign load {s['mean_foreign_busy_cores']:.2f} of {s['cores']} cores"
                  if s.get("available") else f" — {c.get('reason', '')}")
        lines.append(f"load    : {c.get('verdict')}{detail}")
    for p in record.get("producers", []):
        lines.append(f"producer: {p.get('fingerprint')}")
    if record.get("env"):
        lines.append(f"env     : {json.dumps(record['env'])}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual use
    print(describe(capture()))
