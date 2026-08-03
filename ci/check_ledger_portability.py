#!/usr/bin/env python3
"""``check_ledger_portability`` — a run that proves nothing must not be a run that passes.

WHAT HAPPENED
=============

Asked whether the EP is cross-platform, I built it on Linux (WSL Ubuntu 24.04) at
``d375a4d``, pointed it at lavapipe, and ran the harness's own device probe. The EP
loaded, registered, enumerated the device, and then declined every single form::

    [VulkanEP] proof ledger fault: ledger entry for
      "ai.onnx::Tanh/6+/f32>f32/ew_unary_tanh_f32/static/n1" was proven against shader
      digest 16a64dbeb2dbf63d but this build's modules ["ew_unary_tanh_f32"] hash to
      8f87214a7ca41ca9. The entry describes a kernel that has been replaced; re-prove
      it with `gen_proof_ledger.py --reprove`.

    [§8.9.7] this session claims 0/1 nodes; all work runs on the CPU EP.

    EXIT = 0

The same probe on Windows/NVIDIA at the same commit reports ``session claims 1 proven
form(s)`` and no faults at all. **Nothing about the kernel changed between those two
runs.** The GLSL is the same tree at the same commit. What differs is the ``glslc`` that
compiled it — Vulkan SDK 1.4.350.0 on Windows, Ubuntu ``shaderc 2023.8`` / ``glslang
14.0.0`` in WSL — and ``registry::shader_digest_for`` hashes the *embedded SPIR-V bytes*.
Different compiler, different bytes, different digest, every entry stale.

WHY THIS IS A REPORTING DEFECT AND NOT A LEDGER BUG
====================================================

The ledger did exactly what it says it does, and its docstring is honest about it. The
defect is downstream and it is mine: **a ledger fault degrades to the CPU EP and the
process exits 0.** That is the silent-CPU-fallback class again — the eighth time — and
it arrives through a mechanism that no existing screen watches:

* ``check_device_loss`` keys on Vulkan device-loss text. There is no lost device here.
* ``disable_cpu_ep_fallback`` (Trinity's) makes ORT refuse at session creation when
  nodes are assigned to CPU. It would fire here — but only for a caller that sets it,
  and the op harness does not; it *skips* instead.
* The op-correctness lane would go **green having asserted nothing**: every device test
  is guarded by ``tests/ops/conftest.py::_probe_vulkan_device``, which returns ``False``
  when the EP claims no node, and the resulting skip reason is *"No Vulkan device
  available — either no ICD is installed or all devices failed the capability gate."*
  On lavapipe both halves of that sentence are false. A Vulkan device is available and
  it passed the capability gate: ``epctl --probe-loader`` reports gate PASS,
  ``llvmpipe (LLVM 20.1.2)``, Vulkan 1.4.318, ``subgroup_size = 8``, on the same box with
  the same ICD, seconds earlier. The harness reports an EP *decision* as a *device
  absence*, and that is the misname that would have made the Linux lane look clean.

So the failure mode this screen exists for is not "the ledger is stale". It is **a lane
that claims nothing, proves nothing, and reports success** — where the honest report is
that the run did not happen.

WHAT THIS CHECKS, AND WHAT IT DOES NOT
=======================================

Three conditions, each keyed on stable failure *text* rather than a count (R13):

1. ``ledger_fault`` — the run emitted at least one proof-ledger fault. Quoted verbatim,
   one specimen, with the entry key and both digests.
2. ``claimed_nothing`` — the run announced ``claims 0/N nodes`` on a lane declared to be
   a device lane. A device lane whose EP claimed nothing did not exercise the EP.
3. ``device_absence_misnamed`` — the run reported "No Vulkan device available" while the
   same run (or a caller-named loader artifact) recorded a loader gate PASS. Those two
   cannot both be true; one of them is a misname, and the harness's is the one that
   silences tests.

It DELIBERATELY DOES NOT:

* **Take the exit status as an input.** The defect *is* an exit status of 0. A screen
  that filtered on exit status would accept the defect as its filter.
* **Judge whether re-proving is the right fix.** It is not; see §7.19. Re-proving on
  every platform makes the digest a build fingerprint rather than a kernel identity, and
  ``gen_proof_ledger.py --reprove`` without ``--append`` is currently destructive
  (Mouse, unfixed at time of writing). This screen reports; it does not remediate.
* **Cover a run it was not given.** With no ``--run-log`` it reports UNOBSERVABLE for
  every condition rather than PASS. R12: "I was not pointed at a run" is not "the run
  was clean". This is the whole reason it takes a named artifact and does not go
  hunting the tree — unlike ``check_device_loss``'s tier-1, there is no text here that
  no negative control would ever emit on purpose. ``bench/results`` holds artifacts that
  quote ledger faults deliberately.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TAG = "LEDGER-PORTABILITY"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

#: The fault the EP prints when an entry's recorded digest disagrees with this build's
#: modules. Stable: it is assembled in ``registry.rs`` and asserted by
#: ``tests/ops/test_proof_ledger.py``.
LEDGER_FAULT_RE = re.compile(r"proof ledger fault: ledger entry for[^\n]*", re.I)

#: §8.9.7's disclosure line when the session claimed no node.
CLAIMED_NOTHING_RE = re.compile(r"this session claims (0)/(\d+) nodes[^\n]*", re.I)

#: The harness's skip reason. Quoted from ``tests/ops/conftest.py``.
DEVICE_ABSENT_RE = re.compile(r"No Vulkan device available[^\n]*", re.I)

#: What ``epctl --probe-loader`` prints when the capability gate passed.
LOADER_GATE_PASS_RE = re.compile(r"\bGATE\b[^\n]*\bPASS\b|gate:\s*PASS", re.I)


def report_pass(detail: str) -> int:
    print(f"{TAG}: PASS — {detail}", flush=True)
    return EXIT_PASS


def report_fail(condition: str, detail: str) -> int:
    print(f"{TAG}: FAIL(condition={condition})", flush=True)
    print(detail, flush=True)
    print(
        f"{TAG}: this is a finding about a run, not about this check. The producing "
        "process very probably exited 0 — that is the defect, not a reason to discount "
        "the finding. A run whose EP claimed nothing is not a run of the EP.",
        flush=True,
    )
    return EXIT_FAIL_CONDITION


def report_instrument_error(instrument: str, detail: str) -> int:
    print(f"{TAG}: ERROR(instrument={instrument})", flush=True)
    print(detail, flush=True)
    print(
        f"{TAG}: the check did not reach its observation, so this is NOT a detection "
        "(§10.0.1 R13). Do not route it as a portability fault and do not read it as a "
        "clean run.",
        flush=True,
    )
    return EXIT_ERROR_INSTRUMENT


def read_artifact(path: Path) -> tuple[str | None, str | None]:
    """Return ``(text, error)``. Never raises for a caller-supplied path."""
    if not path.exists():
        return None, f"named run artifact does not exist: {path}"
    if path.is_dir():
        return None, f"named run artifact is a directory, not a file: {path}"
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except OSError as exc:
        return None, f"could not read named run artifact {path}: {exc}"


def scan(text: str) -> dict[str, list[str]]:
    """Extract one specimen per condition. Specimens, not counts (R13)."""
    found: dict[str, list[str]] = {}
    faults = LEDGER_FAULT_RE.findall(text)
    if faults:
        found["ledger_fault"] = [faults[0].strip()]
    nothing = CLAIMED_NOTHING_RE.search(text)
    if nothing:
        found["claimed_nothing"] = [nothing.group(0).strip()]
    absent = DEVICE_ABSENT_RE.search(text)
    if absent:
        found["device_absent"] = [absent.group(0).strip()]
    if LOADER_GATE_PASS_RE.search(text):
        found["loader_gate_pass"] = ["loader gate PASS recorded in this artifact"]
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--run-log",
        action="append",
        default=[],
        metavar="PATH",
        help="an artifact the caller names as the evidence of ONE run. Required: this "
        "screen does not scan the tree, because the tree holds artifacts that quote "
        "ledger faults on purpose.",
    )
    ap.add_argument(
        "--device-lane",
        action="store_true",
        help="declare that the named run was expected to exercise the EP on a device. "
        "Without it, `claimed_nothing` is UNOBSERVABLE rather than a failure — a "
        "CPU-only or build-only run legitimately claims nothing.",
    )
    ap.add_argument(
        "--loader-artifact",
        metavar="PATH",
        help="an `epctl --probe-loader` artifact from the same box and ICD. Supplying "
        "it is what makes `device_absence_misnamed` observable; without it the screen "
        "cannot tell a real absent device from a misnamed EP decision.",
    )
    args = ap.parse_args(argv)

    if not args.run_log:
        return report_instrument_error(
            "no_run_named",
            "This screen requires --run-log. It reports UNOBSERVABLE rather than PASS\n"
            "when pointed at nothing, because 'I was not given a run' and 'the run was\n"
            "clean' are different facts (§10.0.1 R12) and only one of them is evidence.",
        )

    texts: list[tuple[Path, str]] = []
    for raw in args.run_log:
        path = Path(raw)
        text, err = read_artifact(path)
        if err is not None:
            return report_instrument_error("artifact_unreadable", err)
        texts.append((path, text or ""))

    loader_gate_passed = False
    loader_where = ""
    if args.loader_artifact:
        lpath = Path(args.loader_artifact)
        ltext, lerr = read_artifact(lpath)
        if lerr is not None:
            return report_instrument_error("loader_artifact_unreadable", lerr)
        if LOADER_GATE_PASS_RE.search(ltext or ""):
            loader_gate_passed = True
            loader_where = str(lpath)

    print(
        f"{TAG}: frame — {len(texts)} named run artifact(s); "
        f"device-lane={'yes' if args.device_lane else 'no'}; "
        f"loader gate evidence={'yes (' + loader_where + ')' if loader_gate_passed else 'ABSENT'}.",
        flush=True,
    )
    if not args.device_lane:
        print(
            f"{TAG}: condition `claimed_nothing` is UNOBSERVABLE in this frame — the "
            "caller did not declare a device lane, and a run that was never meant to "
            "reach a device claims nothing correctly. Not zero findings.",
            flush=True,
        )
    if not loader_gate_passed:
        print(
            f"{TAG}: condition `device_absence_misnamed` is UNOBSERVABLE in this frame "
            "— no loader-gate artifact was supplied, so a 'No Vulkan device available' "
            "line cannot be distinguished from a device that is genuinely absent. Not "
            "zero findings.",
            flush=True,
        )

    for path, text in texts:
        found = scan(text)
        if "ledger_fault" in found:
            return report_fail(
                "ledger_fault",
                f"{path}:\n\n    {found['ledger_fault'][0]}\n\n"
                "An entry's recorded shader digest disagrees with this build's modules,\n"
                "so the form is unproven HERE and the work went to the CPU EP. The\n"
                "digest is over the embedded SPIR-V bytes, which are a property of the\n"
                "glslc that compiled them, not of the kernel and not of the device — so\n"
                "this fires on a build that changed nothing about the kernel at all.",
            )
        if args.device_lane and "claimed_nothing" in found:
            return report_fail(
                "claimed_nothing",
                f"{path}:\n\n    {found['claimed_nothing'][0]}\n\n"
                "The caller declared this a device lane. The EP claimed no node, so\n"
                "every assertion downstream of it was made against the CPU EP or was\n"
                "skipped. Whatever this run reported, it did not report on the EP.",
            )
        if loader_gate_passed and "device_absent" in found:
            return report_fail(
                "device_absence_misnamed",
                f"{path}:\n\n    {found['device_absent'][0]}\n\n"
                f"but {loader_where} records a loader capability gate PASS on the same\n"
                "box and ICD. Both cannot be true. The EP found a device and declined\n"
                "the work; the harness reported that decision as an absent device, which\n"
                "converts a finding into a skip and a skip into a green lane.",
            )

    return report_pass(
        f"{len(texts)} named run artifact(s) carry no ledger fault"
        + (", claimed at least one node" if args.device_lane else "")
        + (", and no misnamed device absence" if loader_gate_passed else "")
        + ". Conditions marked UNOBSERVABLE above were not checked and are not claimed."
    )


def main_guarded(argv=None) -> int:
    try:
        return main(argv)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - an instrument outage is never a detection
        return report_instrument_error(
            "unhandled_exception", f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    sys.exit(main_guarded())
