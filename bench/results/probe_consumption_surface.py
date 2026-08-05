"""What ORT's plugin-EP API actually requires of a consumer — measured, not read.

`README.md` documents a consumption path (`import onnxruntime_ep_vulkan`) and the 26 test
files use a different one (`ort.register_execution_provider_library(name, abs_path)`). Before
deciding whether the fix is a package or a documentation correction, this probe establishes
what the ORT API *does*, because the argument "a wrapper around one call is not worth a
package" is only decidable once you know how many ways that one call can go wrong.

**Every case runs in its own subprocess.** Plugin-EP registration is process-global and
partly irreversible: once a name is registered, a later case that registers the same name
raises, and a case that asks whether an *unregistered* name falls back cannot be run after
one that registered it. Measuring these in one interpreter would make each reading
conditional on the order of the ones before it, which is the cumulative-counter defect
(`docs/PERF.md` §20.6) in a different channel. Six subprocesses cost a few seconds and buy
six independent readings.

Each case declares its `expect` before it runs, and the record stores `observed`, `expect`
and `match`. A case whose observation cannot be taken at all reports `UNOBSERVABLE` — never
a default that reads like a measurement.

Usage
-----
    ONNXRUNTIME_VULKAN_EP_LIB=<abs path to cdylib> python bench/results/probe_consumption_surface.py

Writes ``bench/results/consumption_surface_dev0.json``.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
EP_NAME = "VulkanExecutionProvider"

# --------------------------------------------------------------------------------------
# Case bodies. Each is a standalone program run in a fresh interpreter. It must print
# exactly one line of JSON to stdout as its last line; anything else it emits (the EP's own
# log lines go to stderr, but ORT warnings can land on stdout) is ignored by the parser,
# which reads the last parseable JSON line.
# --------------------------------------------------------------------------------------

_TRIVIAL_MODEL = '''
def _trivial_add_model():
    from onnx import helper, TensorProto
    g = helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["c"])], "g",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [4]),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, [4])],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    return m.SerializeToString()
'''

_EMIT = '''
import json as _json
def emit(**kw):
    print("@@" + _json.dumps(kw))
'''

CASES: dict[str, dict[str, object]] = {
    # 1 ------------------------------------------------------------------------------
    "relative_path_anchor": {
        "question": "A relative library path — is it resolved against the caller's CWD?",
        "expect": "not_cwd",
        "why": (
            "If a relative path does not resolve against the CWD, then the README's one-liner "
            "is not merely inconvenient, it is a documented call a user cannot make portably: "
            "the absolute path is mandatory and nothing tells them so."
        ),
        "body": _EMIT + '''
import os, onnxruntime as ort
rel = os.path.join("definitely", "not", "here.dll")
try:
    ort.register_execution_provider_library("RelProbe", rel)
    emit(observed="loaded", anchor=None)
except Exception as exc:
    msg = str(exc)
    # The error text names the path ORT actually tried, which is the reading we want.
    import re
    m = re.search(r'Error loading "([^"]+)"', msg)
    tried = m.group(1) if m else None
    cwd = os.getcwd()
    if tried is None:
        emit(observed="UNOBSERVABLE", reason="error text did not name a path", msg=msg[:400])
    else:
        anchored_at_cwd = os.path.normcase(os.path.abspath(tried)).startswith(
            os.path.normcase(cwd) + os.sep)
        emit(observed="cwd" if anchored_at_cwd else "not_cwd",
             tried=tried, cwd=cwd)
''',
    },
    # 2 ------------------------------------------------------------------------------
    "registration_name_is_arbitrary": {
        "question": "Does ORT check the registration name against anything in the library?",
        "expect": "arbitrary",
        "why": (
            "If the name is a label chosen by the caller, then the string in "
            "providers=[...] and the string passed to register must agree, and nothing in "
            "either the library or ORT enforces that agreement. That is a coupling only the "
            "caller can get wrong, and only a shim can own on the caller's behalf."
        ),
        "body": _EMIT + '''
import os, onnxruntime as ort
lib = os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
try:
    ort.register_execution_provider_library("NotOurNameAtAll", lib)
except Exception as exc:
    emit(observed="rejected", error=str(exc)[:400])
else:
    avail = list(ort.get_available_providers())
    devs = []
    try:
        devs = [d.ep_name for d in ort.get_ep_devices()]
    except Exception:
        pass
    emit(observed="arbitrary",
         registered_under="NotOurNameAtAll",
         in_available_providers="NotOurNameAtAll" in avail,
         ep_device_entries=sum(1 for n in devs if n == "NotOurNameAtAll"))
''',
    },
    # 3 ------------------------------------------------------------------------------
    "double_registration": {
        "question": "Is registering the same name twice idempotent, or does it raise?",
        "expect": "raises",
        "why": (
            "A raise means the documented call is not safe to run twice in one process — "
            "a notebook re-run, a pytest session that imports two modules, or any library "
            "that registers on import. A shim that is not idempotent reproduces this; a "
            "one-line README instruction cannot fix it at all."
        ),
        "body": _EMIT + '''
import os, onnxruntime as ort
lib = os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
ort.register_execution_provider_library("VulkanExecutionProvider", lib)
try:
    ort.register_execution_provider_library("VulkanExecutionProvider", lib)
except Exception as exc:
    emit(observed="raises", error=str(exc)[:300])
else:
    emit(observed="idempotent")
''',
    },
    # 4 ------------------------------------------------------------------------------
    "unregistered_name_failure_mode": {
        "question": (
            "A user who never registers, but asks for the EP by name in providers=[...] — "
            "do they get an error, or a working session that silently never touches the GPU?"
        ),
        "expect": "silent_cpu_fallback",
        "why": (
            "This is the failure the current README manufactures. A reader hits "
            "ModuleNotFoundError on the import line, deletes the import (the obvious fix), "
            "and keeps the providers list. If that path returns correct numbers with no "
            "error, the project's own 'silently wrong is worse than loudly broken' rule is "
            "being broken at the very first thing a user does."
        ),
        "body": _EMIT + _TRIVIAL_MODEL + '''
import warnings, numpy as np, onnxruntime as ort
blob = _trivial_add_model()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        sess = ort.InferenceSession(
            blob, providers=["VulkanExecutionProvider", "CPUExecutionProvider"])
    except Exception as exc:
        emit(observed="raises", error=str(exc)[:300])
    else:
        got = list(sess.get_providers())
        out = sess.run(None, {"a": np.ones(4, np.float32),
                              "b": np.full(4, 2.0, np.float32)})[0]
        emit(observed="silent_cpu_fallback" if got == ["CPUExecutionProvider"] else "other",
             providers=got,
             numerically_correct=bool(np.allclose(out, 3.0)),
             warnings=[str(w.message)[:160] for w in caught])
''',
    },
    # 5 ------------------------------------------------------------------------------
    "artifact_link_dependencies": {
        "question": "What must be resolvable on the loader path for the artifact to load at all?",
        "expect": "no_onnxruntime_import",
        "why": (
            "tests/ops/conftest.py's diagnostic protocol tells a failing user the cause is "
            "'a missing DLL dependency (onnxruntime.dll, Vulkan loader, or MSVC runtime)'. "
            "Whether onnxruntime is in the import table decides how much a wheel has to "
            "carry: an import-table dependency on the host runtime means a wheel must "
            "locate ORT's own libraries; no such dependency means the wheel ships one file."
        ),
        "body": _EMIT + '''
import os, re, sys
lib = os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
blob = open(lib, "rb").read()
if sys.platform == "win32":
    pat = rb"[A-Za-z0-9_.\\-]+\\.dll"
elif sys.platform == "darwin":
    pat = rb"[A-Za-z0-9_./@\\-]+\\.dylib"
else:
    pat = rb"lib[A-Za-z0-9_.\\-]+\\.so[0-9.]*"
names = sorted({m.group(0).decode("ascii", "replace") for m in re.finditer(pat, blob)})
ort_named = [n for n in names if "onnxruntime" in n.lower() and "vulkan" not in n.lower()]
emit(observed="no_onnxruntime_import" if not ort_named else "onnxruntime_import",
     shared_library_strings=names, onnxruntime_referencing=ort_named)
''',
    },
    # 6 ------------------------------------------------------------------------------
    "verified_registration_positive": {
        "question": "The positive state: register correctly and actually execute on the EP.",
        "expect": "ep_selected",
        "why": (
            "A detector never seen in its positive state has no demonstrated positive state "
            "(docs/PERF.md §22). Cases 1-4 are all negative readings; without this one the "
            "probe has only shown ways to fail and has not shown that the path it recommends "
            "works on this box."
        ),
        "body": _EMIT + _TRIVIAL_MODEL + '''
import os, numpy as np, onnxruntime as ort
lib = os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
ort.register_execution_provider_library("VulkanExecutionProvider", lib)
blob = _trivial_add_model()
sess = ort.InferenceSession(
    blob, providers=["VulkanExecutionProvider", "CPUExecutionProvider"])
got = list(sess.get_providers())
out = sess.run(None, {"a": np.ones(4, np.float32),
                      "b": np.full(4, 2.0, np.float32)})[0]
emit(observed="ep_selected" if "VulkanExecutionProvider" in got else "ep_absent",
     providers=got, numerically_correct=bool(np.allclose(out, 3.0)),
     output=[float(v) for v in out])
''',
    },
}


def _run_case(name: str, spec: dict, lib: str | None) -> dict:
    env = dict(os.environ)
    if lib:
        env[EP_LIB_ENV] = lib
    needs_lib = EP_LIB_ENV in spec["body"]  # type: ignore[operator]
    if needs_lib and not lib:
        return {
            "case": name,
            "question": spec["question"],
            "expect": spec["expect"],
            "why": spec["why"],
            "verdict": "UNOBSERVABLE",
            "reason": f"{EP_LIB_ENV} is unset; this case needs the built artifact",
        }
    proc = subprocess.run(
        [sys.executable, "-c", spec["body"]],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            try:
                payload = json.loads(line[2:])
            except json.JSONDecodeError:
                pass
    rec: dict = {
        "case": name,
        "question": spec["question"],
        "expect": spec["expect"],
        "why": spec["why"],
        "exit_code": proc.returncode,
    }
    if payload is None:
        rec["verdict"] = "UNOBSERVABLE"
        rec["reason"] = "case emitted no parseable reading"
        rec["stderr_tail"] = proc.stderr[-600:]
        return rec
    rec.update(payload)
    rec["match"] = payload.get("observed") == spec["expect"]
    rec["verdict"] = "MATCH" if rec["match"] else "MISMATCH"
    return rec


def main() -> int:
    lib = os.environ.get(EP_LIB_ENV)
    if lib:
        p = Path(lib)
        lib = str(p.resolve()) if p.exists() else None
    results = [_run_case(n, s, lib) for n, s in CASES.items()]

    verdicts = [r["verdict"] for r in results]
    if "MISMATCH" in verdicts:
        overall = "MISMATCH"
    elif "UNOBSERVABLE" in verdicts:
        overall = "PARTIAL"
    else:
        overall = "MATCH"

    record = {
        "probe": "consumption_surface",
        "question": (
            "What does ORT's plugin-EP API require of a consumer, and how many ways can the "
            "single documented call go wrong?"
        ),
        "method": "one fresh subprocess per case; registration is process-global state",
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "ort_version": _ort_version(),
        "artifact": _artifact_frame(lib),
        "cases": results,
        "verdict": overall,
        "matched": sum(1 for v in verdicts if v == "MATCH"),
        "total": len(verdicts),
    }
    out = HERE / "consumption_surface_dev0.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    for r in results:
        print(f"  {r['verdict']:<12} {r['case']}  observed={r.get('observed')!r}")
    print(f"\n{record['matched']}/{record['total']} matched -> {overall}")
    print(f"record: {out}")
    return 0 if overall != "MISMATCH" else 1


def _ort_version() -> str:
    try:
        import onnxruntime  # noqa: PLC0415

        return onnxruntime.__version__
    except Exception:  # pragma: no cover - onnxruntime is a hard dependency of the suite
        return "UNOBSERVABLE"


def _artifact_frame(lib: str | None) -> dict:
    if not lib:
        return {"path": None, "reason": f"{EP_LIB_ENV} unset or path missing"}
    import hashlib

    p = Path(lib)
    return {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
