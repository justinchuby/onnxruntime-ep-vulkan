"""End-to-end demonstration: a user who has never built this repository gets a session.

    python python/verify_cleanroom.py

Creates a **fresh virtual environment outside the repository**, installs only the wheel and
its dependencies into it, and then — in that interpreter, with the repository unreachable —
imports the package, registers the EP, runs a trivial model, and asserts the EP was
selected. Writes ``bench/results/cleanroom_install_dev0.json``.

Why outside the repository
--------------------------
:func:`onnxruntime_ep_vulkan.library_path` falls back to a source checkout's
``rust/target/release`` when it finds a ``rust/`` directory above itself. A venv created
inside this tree would have that fallback available, so a passing run would not
discriminate between "the wheel carried the artifact" and "the package found the build I
already had". The venv therefore goes in a sibling directory of the repository, and the
run additionally asserts that the resolved library path lies **inside the venv's
site-packages** — a positional check, not a trust exercise.

What this demonstrates and what it does not
-------------------------------------------
It demonstrates: from a wheel and nothing else, on this platform, `import` → `register` →
`InferenceSession` → the Vulkan EP is selected → correct output. That is the positive state
of the consumption path, observed rather than reasoned about.

It does not demonstrate: that this holds on any other OS, driver or architecture; that
`pip install onnxruntime-ep-vulkan` from an index works (nothing is published, deliberately
— no release process is in scope); or that a wheel built here runs on a machine that never
had the Vulkan SDK, since this box has it and a Vulkan *loader* is a runtime requirement
the wheel does not and cannot carry.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY_DIR = Path(__file__).resolve().parent
REPO = PY_DIR.parent
DEFAULT_ENV = REPO.parent / "onnxruntime-ep-vulkan-cleanroom"

# Runs inside the fresh interpreter. Prints one JSON line prefixed with @@.
_CONSUMER = r'''
import json, sys, sysconfig, os
out = {}
try:
    import numpy as np
    import onnxruntime as ort
    from onnx import TensorProto, helper
    import onnxruntime_ep_vulkan as vk

    out["package_file"] = vk.__file__
    site = sysconfig.get_paths()["purelib"]
    out["site_packages"] = site

    path = vk.register_execution_provider_library()
    out["registered_path"] = path
    out["artifact_inside_site_packages"] = os.path.normcase(
        os.path.abspath(path)).startswith(os.path.normcase(os.path.abspath(site)) + os.sep)
    out["provenance"] = vk.verify_provenance()

    g = helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["c"])], "g",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [4]),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, [4])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10

    sess = ort.InferenceSession(m.SerializeToString(), providers=vk.providers())
    out["session_providers"] = list(sess.get_providers())
    res = sess.run(None, {"a": np.ones(4, np.float32),
                          "b": np.full(4, 2.0, np.float32)})[0]
    out["output"] = [float(v) for v in res]
    out["numerically_correct"] = bool(np.allclose(res, 3.0))
    vk.assert_ep_selected(sess)
    out["ep_selected"] = True
    out["onnxruntime_version"] = ort.__version__
    out["verdict"] = "PASS" if out["numerically_correct"] and \
        out["artifact_inside_site_packages"] else "FAIL"
except Exception as exc:
    out["verdict"] = "FAIL"
    out["error"] = f"{type(exc).__name__}: {exc}"
print("@@" + json.dumps(out))
'''


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("$", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], **kw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wheel", type=Path, default=None,
                    help="wheel to install (default: newest in python/dist)")
    ap.add_argument("--env", type=Path, default=DEFAULT_ENV)
    ap.add_argument("--keep", action="store_true", help="do not delete the venv afterwards")
    args = ap.parse_args(argv)

    wheel = args.wheel
    if wheel is None:
        wheels = sorted((PY_DIR / "dist").glob("*.whl"),
                        key=lambda p: p.stat().st_mtime)
        if not wheels:
            raise SystemExit("no wheel in python/dist -- run python/build_wheel.py first")
        wheel = wheels[-1]
    wheel = wheel.resolve()

    env_dir = args.env.resolve()
    if env_dir.is_relative_to(REPO):
        raise SystemExit(
            f"refusing to build the clean-room venv inside the repository ({env_dir}): "
            f"the source-checkout fallback would be reachable and the run would not "
            f"discriminate between the wheel and the local build."
        )
    if env_dir.exists():
        shutil.rmtree(env_dir)

    record: dict = {
        "probe": "cleanroom_install",
        "question": (
            "Does a user who has never built this repository get a working Vulkan EP "
            "session from the wheel alone?"
        ),
        "wheel": wheel.name,
        "wheel_sha256": __import__("hashlib").sha256(wheel.read_bytes()).hexdigest(),
        "venv": str(env_dir),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    try:
        if _run([sys.executable, "-m", "venv", env_dir]).returncode != 0:
            raise SystemExit("venv creation failed")
        py = env_dir / ("Scripts" if os.name == "nt" else "bin") / (
            "python.exe" if os.name == "nt" else "python")

        install = _run([py, "-m", "pip", "install", "--disable-pip-version-check",
                        "-q", wheel, "onnx", "numpy"], capture_output=True, text=True)
        record["pip_returncode"] = install.returncode
        if install.returncode != 0:
            record["verdict"] = "UNOBSERVABLE"
            record["reason"] = "pip install failed in the clean venv"
            record["pip_stderr"] = install.stderr[-1500:]
        else:
            frozen = _run([py, "-m", "pip", "freeze"], capture_output=True, text=True)
            record["installed"] = sorted(frozen.stdout.split())
            # cwd is deliberately outside the repository too, so nothing on sys.path[0]
            # can reach it either.
            proc = _run([py, "-c", _CONSUMER], capture_output=True, text=True,
                        cwd=str(env_dir))
            payload = None
            for line in proc.stdout.splitlines():
                if line.startswith("@@"):
                    payload = json.loads(line[2:])
            if payload is None:
                record["verdict"] = "UNOBSERVABLE"
                record["reason"] = "the clean interpreter emitted no reading"
                record["stderr_tail"] = proc.stderr[-1500:]
            else:
                record.update(payload)
    finally:
        if not args.keep and env_dir.exists():
            shutil.rmtree(env_dir, ignore_errors=True)

    out = REPO / "bench" / "results" / "cleanroom_install_dev0.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: record[k] for k in record
                             if k in ("verdict", "session_providers", "output",
                                      "artifact_inside_site_packages", "error",
                                      "reason")}, indent=2))
    print(f"record: {out}")
    return 0 if record.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
