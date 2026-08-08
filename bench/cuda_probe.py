"""CUDA-arm feasibility probe: does ORT's CUDA EP actually execute here?

Issue #69 asks whether the Vulkan EP can beat ORT's CUDA EP.  Before any
comparison is meaningful, three questions have to be answered separately, and a
"yes" to the first two does not imply a "yes" to the third:

1. Is ``CUDAExecutionProvider`` *listed* by ``get_available_providers()``?
   Listing is a build-time fact.  ORT lists every provider its binary was built
   with, whether or not the provider's shared library can be loaded.
2. Does a session *created* with the CUDA EP still hold it after creation?
   ORT silently drops a provider whose library fails to load and falls through
   to the next entry in the provider list, so ``get_providers()`` on the live
   session is the only honest answer.
3. Do the graph's *nodes* actually run on CUDA?  A session that reports
   ``CUDAExecutionProvider`` in ``get_providers()`` may still have assigned
   every node to the CPU EP.  Only the ORT profile answers this.

This module answers all three and refuses to conflate them.  A CUDA arm that
loses question 2 or 3 is not a slower CUDA arm; it is *not a CUDA arm*, and any
number taken from it is a CPU number wearing a CUDA label.

Run directly for a human-readable verdict::

    python -m bench.cuda_probe
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_paths import dump_public_json  # noqa: E402

# --- verdict vocabulary -----------------------------------------------------
#
# Deliberately not booleans.  "CUDA is unavailable" and "we could not tell
# whether CUDA is available" are different findings with different consequences
# for issue #69, and a bool cannot carry the difference.

CUDA_USABLE = "CUDA_USABLE"
CUDA_NOT_LISTED = "CUDA_NOT_LISTED"
CUDA_LISTED_NOT_LOADABLE = "CUDA_LISTED_NOT_LOADABLE"
CUDA_LOADED_NO_NODES = "CUDA_LOADED_NO_NODES"
CUDA_PROBE_ERROR = "CUDA_PROBE_ERROR"

VERDICTS = (
    CUDA_USABLE,
    CUDA_NOT_LISTED,
    CUDA_LISTED_NOT_LOADABLE,
    CUDA_LOADED_NO_NODES,
    CUDA_PROBE_ERROR,
)


@dataclass
class DriverFacts:
    """Host NVIDIA driver / device identity, read from ``nvidia-smi``.

    Absent ``nvidia-smi`` is recorded as absent rather than as a zero: a device
    we cannot interrogate is not a device we know to be missing.
    """

    producer: str = "none_available"
    driver_version: "str | None" = None
    cuda_driver_version: "str | None" = None
    device_names: "list[str]" = field(default_factory=list)
    device_uuids: "list[str]" = field(default_factory=list)
    error: "str | None" = None


def nvidia_smi_facts() -> DriverFacts:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return DriverFacts(producer="none_available", error="nvidia-smi not on PATH")
    try:
        query = subprocess.run(
            [exe, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - host dependent
        return DriverFacts(producer="none_available", error=f"nvidia-smi failed: {exc}")
    if query.returncode != 0:
        return DriverFacts(producer="none_available",
                           error=f"nvidia-smi exit {query.returncode}: {query.stderr.strip()[:200]}")

    names: "list[str]" = []
    uuids: "list[str]" = []
    driver: "str | None" = None
    for line in query.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        names.append(parts[0])
        uuids.append(parts[1])
        driver = parts[2]

    cuda_driver = None
    try:
        banner = subprocess.run([exe], capture_output=True, text=True, timeout=60, check=False)
        m = re.search(r"CUDA Version:\s*([0-9.]+)", banner.stdout)
        if m:
            cuda_driver = m.group(1)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - host dependent
        pass

    return DriverFacts(
        producer="nvidia-smi",
        driver_version=driver,
        cuda_driver_version=cuda_driver,
        device_names=names,
        device_uuids=uuids,
    )


def _build_cuda_version(ort) -> "str | None":
    """The CUDA toolkit version the *wheel* was built against.

    Distinct from the driver's reported CUDA version.  ORT's CUDA EP requires a
    driver at least as new as its build toolkit's major line, so recording only
    one of the two makes an impossible pairing look possible.
    """
    try:
        info = ort.get_build_info()
    except Exception:  # pragma: no cover - older ORT
        return None
    m = re.search(r"CUDA_VERSION\s*=\s*([0-9.]+)", info)
    return m.group(1) if m else None


def _preload_dlls(ort) -> dict:
    """Ask ORT to add its NVIDIA pip-wheel dependencies to the DLL search path.

    On Windows the CUDA/cuDNN DLLs from ``nvidia-*-cu12`` wheels are not on
    ``PATH``; without this the provider library fails to load for a reason that
    has nothing to do with the GPU.  Recorded as a fact of the arm, because a
    reader must be able to tell "loaded after preload" from "loaded natively".
    """
    if not hasattr(ort, "preload_dlls"):
        return {"attempted": False, "reason": "onnxruntime.preload_dlls absent"}
    try:
        ort.preload_dlls()
        return {"attempted": True, "ok": True}
    except Exception as exc:
        return {"attempted": True, "ok": False, "error": str(exc)[:400]}


def ort_build_facts() -> dict:
    """ORT version, build config and *listed* providers.

    ``listed`` is explicitly named ``providers_listed`` and never ``providers``,
    so no caller can mistake a build-time list for a runtime capability.
    """
    import onnxruntime as ort

    facts = {
        "version": ort.__version__,
        "cuda_version_used_in_build": _build_cuda_version(ort),
        "providers_listed": list(ort.get_available_providers()),
        "device_string": ort.get_device(),
        "package_root": str(Path(ort.__file__).parent),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        facts["build_info"] = ort.get_build_info()
    except Exception as exc:  # pragma: no cover - older ORT
        facts["build_info"] = f"unavailable: {exc}"
    return facts


def _tiny_matmul_model(path: Path, k: int = 256) -> Path:
    """Smallest graph that a real CUDA EP would certainly claim.

    A single ``MatMul`` on float32.  If the CUDA EP declines *this*, the
    provider is not executing anything, and no larger model will change that.
    Written with raw protobuf construction so the probe does not require the
    ``onnx`` package (the CUDA venv is deliberately minimal).
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    w = np.random.default_rng(0).standard_normal((k, k)).astype(np.float32)
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="probe_matmul")
    graph = helper.make_graph(
        [node], "cuda_probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, k])],
        [numpy_helper.from_array(w, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, str(path))
    return path


def node_provider_partition(profile_path: Path) -> dict:
    """Read an ORT profile and count graph nodes per execution provider.

    This is the only mechanical proof of *where the work ran*.  ORT emits one
    ``*_kernel_time`` event per executed node carrying a
    ``provider`` arg; anything else in the file is session or model-level
    bookkeeping and is not counted.
    """
    with profile_path.open("r", encoding="utf-8") as fh:
        events = json.load(fh)

    partition: "dict[str, int]" = {}
    node_names: "dict[str, list[str]]" = {}
    total_dur_us: "dict[str, float]" = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("cat") != "Node":
            continue
        name = ev.get("name", "")
        if not name.endswith("_kernel_time"):
            continue
        args = ev.get("args") or {}
        provider = args.get("provider") or "UNATTRIBUTED"
        partition[provider] = partition.get(provider, 0) + 1
        node_names.setdefault(provider, []).append(name[: -len("_kernel_time")])
        total_dur_us[provider] = total_dur_us.get(provider, 0.0) + float(ev.get("dur", 0) or 0)

    return {
        "partition": partition,
        "node_names": node_names,
        "kernel_time_us": total_dur_us,
        "kernel_events": sum(partition.values()),
    }


def probe(scratch: Path) -> dict:
    """Answer all three questions and return a single record."""
    scratch.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "schema": "cuda_probe/1",
        "driver": asdict(nvidia_smi_facts()),
    }

    try:
        import onnxruntime as ort
    except Exception as exc:
        record["verdict"] = CUDA_PROBE_ERROR
        record["error"] = f"onnxruntime import failed: {exc}"
        return record

    record["ort"] = ort_build_facts()
    record["preload_dlls"] = _preload_dlls(ort)

    if "CUDAExecutionProvider" not in record["ort"]["providers_listed"]:
        record["verdict"] = CUDA_NOT_LISTED
        record["reason"] = ("the installed onnxruntime build does not list a CUDA provider; "
                            "no CUDA number can be produced from this install")
        return record

    model = scratch / "cuda_probe_matmul.onnx"
    try:
        _tiny_matmul_model(model)
    except Exception as exc:
        record["verdict"] = CUDA_PROBE_ERROR
        record["error"] = f"could not build probe model: {exc}"
        return record

    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.profile_file_prefix = str(scratch / "cuda_probe_profile")
    opts.log_severity_level = 3

    try:
        sess = ort.InferenceSession(
            str(model), sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception as exc:
        record["verdict"] = CUDA_LISTED_NOT_LOADABLE
        record["reason"] = ("session creation with the CUDA EP raised; the provider library "
                            "or its CUDA/cuDNN dependencies could not be loaded")
        record["error"] = str(exc)[:2000]
        return record

    held = list(sess.get_providers())
    record["providers_held_by_session"] = held
    if "CUDAExecutionProvider" not in held:
        sess.end_profiling()
        record["verdict"] = CUDA_LISTED_NOT_LOADABLE
        record["reason"] = ("ORT silently dropped the CUDA EP during session creation and fell "
                            "through to the next provider; the session is not a CUDA session")
        return record

    import numpy as np
    k = sess.get_inputs()[0].shape[1]
    feed = {"x": np.zeros((1, int(k)), dtype=np.float32)}
    try:
        for _ in range(3):
            sess.run(None, feed)
    except Exception as exc:
        sess.end_profiling()
        record["verdict"] = CUDA_PROBE_ERROR
        record["error"] = f"probe inference failed: {exc}"[:2000]
        return record

    profile = Path(sess.end_profiling())
    try:
        attrib = node_provider_partition(profile)
    except Exception as exc:
        record["verdict"] = CUDA_PROBE_ERROR
        record["error"] = f"profile unreadable: {exc}"
        return record

    record["attribution"] = attrib
    record["profile_path"] = str(profile)

    cuda_nodes = attrib["partition"].get("CUDAExecutionProvider", 0)
    if cuda_nodes == 0:
        record["verdict"] = CUDA_LOADED_NO_NODES
        record["reason"] = ("the CUDA EP loaded but claimed zero graph nodes; every node ran "
                            "elsewhere, so this session's timings are not CUDA timings")
        return record

    record["verdict"] = CUDA_USABLE
    record["reason"] = f"CUDA EP claimed {cuda_nodes} node(s) on a single-MatMul probe graph"
    return record


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    scratch = Path(argv[0]) if argv else Path(__file__).resolve().parent / "results" / "_cuda_probe"
    rec = probe(scratch)
    out = scratch / "cuda_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_public_json(rec, out, sort_keys=True)

    print(f"verdict : {rec['verdict']}")
    if rec.get("reason"):
        print(f"reason  : {rec['reason']}")
    if rec.get("error"):
        print(f"error   : {rec['error'][:600]}")
    ort_facts = rec.get("ort") or {}
    if ort_facts:
        print(f"ort     : {ort_facts.get('version')} listed={ort_facts.get('providers_listed')}")
    drv = rec.get("driver") or {}
    print(f"driver  : producer={drv.get('producer')} version={drv.get('driver_version')} "
          f"cuda={drv.get('cuda_driver_version')} devices={drv.get('device_names')}")
    if rec.get("attribution"):
        print(f"partition: {rec['attribution']['partition']}")
    print(f"record  : {out}")
    return 0 if rec["verdict"] == CUDA_USABLE else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
