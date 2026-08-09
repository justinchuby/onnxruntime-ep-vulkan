"""Static SPIR-V comparison of the two `gqa_f16` modules issue #96 is about.

The compiled delta between `c96e7d9` and `85fbda2` is two files, and only one of them reaches the
device: `rust/shaders/glsl/gqa_f16.comp`. This tool disassembles both builds' `gqa_f16.spv`,
diffs them with the IDs normalised, counts opcodes on each side, and re-runs the comparison with
the candidate's specialisation constant **frozen to the value decode actually resolves** (1). It
writes one JSON artifact that carries the evidence rather than a description of it.

What it can and cannot settle, stated on the artifact itself:

  * it CAN settle what is in the **module** — the SPIR-V handed to `vkCreateComputePipelines`;
  * it CANNOT settle what the **driver** generates from that module. NVIDIA compiles SPIR-V to
    SASS inside the pipeline call with the specialisation constant already resolved, and no
    SPIR-V-level tool observes that output. A module-level identity is therefore evidence about
    the input to the driver's compiler, not a proof about its output.

Usage (needs `spirv-dis`, `spirv-val`, `spirv-opt`, `spirv-diff` from the Vulkan SDK on PATH):

    python bench/results/spirv_gqa_crossbuild.py \
        --baseline-spv <baseline build>/out/spv/gqa_f16.spv \
        --candidate-spv <candidate build>/out/spv/gqa_f16.spv \
        --out bench/results/spirv_gqa_crossbuild.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

OPCODE = re.compile(r"^\s*(?:%\S+\s*=\s*)?(Op[A-Za-z0-9]+)")
# The decoration and the definition are the only places a module may name the workgroup size
# without an instruction depending on it.
WORKGROUP_SIZE = re.compile(r"gl_WorkGroupSize|WorkgroupSize|LocalSize")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def run(*cmd: str) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed: {out.stderr.strip()[:400]}")
    return out.stdout


def tool_versions() -> dict:
    v = {}
    for t in ("spirv-dis", "spirv-val", "spirv-opt", "spirv-diff"):
        exe = shutil.which(t)
        if not exe:
            raise SystemExit(f"{t} not on PATH — add the Vulkan SDK's Bin directory")
        try:
            v[t] = subprocess.run([t, "--version"], capture_output=True,
                                  text=True).stdout.strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            v[t] = "unknown"
    return v


def histogram(dis: str) -> dict:
    c: collections.Counter = collections.Counter()
    for line in dis.splitlines():
        m = OPCODE.match(line)
        if m:
            c[m.group(1)] += 1
    return dict(sorted(c.items()))


def workgroup_size_references(dis: str) -> list:
    """Lines that mention the workgroup size at all.

    The claim this supports is narrow and checkable: if the only mentions are the decoration and
    the constant's own definition, then **no instruction in the body consumes it**, so there is
    no loop bound and no address computation for a specialisation constant to stop folding. That
    is the module-level half of hypothesis H1, and it is the half SPIR-V can actually answer.
    """
    return [ln.strip() for ln in dis.splitlines() if WORKGROUP_SIZE.search(ln)]


def storage_class_counts(dis: str) -> dict:
    c: collections.Counter = collections.Counter()
    for line in dis.splitlines():
        m = re.search(r"OpVariable\s+\S+\s+(\w+)", line)
        if m:
            c[m.group(1)] += 1
        m2 = re.search(r"OpTypePointer\s+(\w+)", line)
        if m2:
            c[f"ptr:{m2.group(1)}"] += 1
    return dict(sorted(c.items()))


def diff_lines(raw: str) -> list:
    """Only the changed lines of a `spirv-diff` unified output."""
    return [ln.rstrip() for ln in raw.splitlines()
            if (ln.startswith("+") or ln.startswith("-"))
            and not ln.startswith(("+++", "---"))]


# Changes that are *declarations of the workgroup size* or module bookkeeping, not executable
# body. `; Bound: N` is a header comment — the id bound necessarily moves when two ids are added
# — and counting it as a body change reported `body_instructions_differ: true` for a module whose
# body is byte-identical. That was this tool's own first answer and it was wrong.
DECLARATION_CHANGE = re.compile(
    r"^\s*[+-]\s*;\s*Bound:|SpecId|WorkgroupSize|WorkGroupSize|OpConstantComposite"
    r"|OpSpecConstant")


def body_instruction_changes(changed: list) -> list:
    """Changed lines that are neither module bookkeeping nor a workgroup-size declaration."""
    return [ln for ln in changed if not DECLARATION_CHANGE.search(ln)]


def analyse(spv: Path, tmp: Path, tag: str) -> dict:
    dis = run("spirv-dis", "--no-header", str(spv))
    (tmp / f"{tag}.dis").write_text(dis, encoding="utf-8")
    val = subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(spv)],
                         capture_output=True, text=True)
    return {
        "spv_sha256": sha256(spv),
        "spv_bytes": spv.stat().st_size,
        "disassembly_sha256": hashlib.sha256(dis.encode()).hexdigest(),
        "spirv_val": "PASS" if val.returncode == 0 else val.stderr.strip()[:400],
        "instruction_count": sum(histogram(dis).values()),
        "opcode_histogram": histogram(dis),
        "workgroup_size_references": workgroup_size_references(dis),
        "storage_classes": storage_class_counts(dis),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-spv", required=True)
    ap.add_argument("--candidate-spv", required=True)
    ap.add_argument("--baseline-commit", default="")
    ap.add_argument("--candidate-commit", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    versions = tool_versions()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        base_spv, cand_spv = Path(a.baseline_spv), Path(a.candidate_spv)
        base = analyse(base_spv, tmp, "base")
        cand = analyse(cand_spv, tmp, "cand")

        frozen = tmp / "cand_frozen1.spv"
        run("spirv-opt", "--set-spec-const-default-value", "0:1", "--freeze-spec-const",
            str(cand_spv), "-o", str(frozen))
        froz = analyse(frozen, tmp, "frozen")

        raw = subprocess.run(["spirv-diff", str(base_spv), str(cand_spv)],
                             capture_output=True, text=True).stdout
        raw_frozen = subprocess.run(["spirv-diff", str(base_spv), str(frozen)],
                                    capture_output=True, text=True).stdout

    keys = set(base["opcode_histogram"]) | set(cand["opcode_histogram"])
    hist_delta = {k: cand["opcode_histogram"].get(k, 0) - base["opcode_histogram"].get(k, 0)
                  for k in sorted(keys)
                  if cand["opcode_histogram"].get(k, 0) != base["opcode_histogram"].get(k, 0)}

    changed = diff_lines(raw)
    changed_frozen = diff_lines(raw_frozen)

    art = {
        "instrument": Path(__file__).name,
        "issue": 96,
        "question": ("What, exactly, differs between the two compiled gqa_f16 SPIR-V modules, "
                     "and does anything in the body depend on the workgroup size?"),
        "tools": versions,
        "arms": {
            "baseline": {"commit": a.baseline_commit, "shader_source":
                         "rust/shaders/glsl/gqa_f16.comp @ layout(local_size_x = 1, ...)", **base},
            "candidate": {"commit": a.candidate_commit, "shader_source":
                          "rust/shaders/glsl/gqa_f16.comp @ layout(local_size_x_id = 0, "
                          "local_size_x = 1, ...)", **cand},
        },
        "candidate_frozen_to_1": {
            "how": "spirv-opt --set-spec-const-default-value 0:1 --freeze-spec-const",
            "why": ("Phi-3.5 decode has B*Nq*S = 32 invocations, so `gqa_local_size` returns 1 "
                    "and SpecId 0 resolves to 1. Freezing it is the closest a static tool can "
                    "get to the module the driver is actually asked to compile at decode."),
            **froz,
        },
        "module_diff": {
            "spirv_diff_changed_lines": changed,
            "spirv_diff_changed_line_count": len(changed),
            "opcode_histogram_delta": hist_delta,
            "instruction_count_delta": cand["instruction_count"] - base["instruction_count"],
        },
        "module_diff_after_freezing": {
            "spirv_diff_changed_lines": changed_frozen,
            "spirv_diff_changed_line_count": len(changed_frozen),
        },
        "findings": {
            "body_instructions_differ": bool(body_instruction_changes(changed)),
            "body_instruction_changed_lines": body_instruction_changes(changed),
            "workgroup_size_consumed_by_body": None,   # filled below
            "shared_memory_present": any(k.startswith("Workgroup") or k == "ptr:Workgroup"
                                         for k in cand["storage_classes"]),
        },
        "what_this_cannot_show": (
            "The driver's final machine code. NVIDIA compiles SPIR-V to SASS inside "
            "vkCreateComputePipelines with SpecId 0 already resolved to 1, and no SPIR-V tool "
            "observes that output. Module-level identity is evidence about the INPUT to the "
            "driver's compiler; register allocation, scheduling and instruction selection "
            "downstream of it are not observable here and are not claimed either way. "
            "VK_KHR_pipeline_executable_properties is the instrument that would answer it and "
            "this repository does not have one."),
    }
    # A body that consumes the size would show it as an operand of a real instruction. The
    # decoration, the definition, the debug name and `OpExecutionMode ... LocalSize` are all
    # *declarations* of the size, not uses of it — counting them as consumers is how this
    # instrument first reported `True` for a module whose body never reads the value.
    DECLARATION = re.compile(r"OpDecorate|OpExecutionMode|OpName|OpMemberName"
                             r"|=\s*OpSpecConstant|=\s*OpConstant")
    consumers = [ln for ln in cand["workgroup_size_references"] if not DECLARATION.search(ln)]
    art["findings"]["workgroup_size_consumed_by_body"] = bool(consumers)
    art["findings"]["workgroup_size_consumer_lines"] = consumers
    art["findings"]["workgroup_size_reference_lines_candidate"] = cand["workgroup_size_references"]
    art["findings"]["declaration_note"] = (
        "Both modules also carry `OpExecutionMode ... LocalSize 1 1 1`. Where a "
        "`WorkgroupSize`-decorated constant is present the SPIR-V specification says it "
        "supersedes that execution mode, so the two are not in conflict; both are declarations "
        "of the geometry and neither is an instruction that reads it.")

    art["module_diff_after_freezing"]["body_instruction_changed_lines"] = \
        body_instruction_changes(changed_frozen)
    Path(a.out).write_text(json.dumps(art, indent=2), encoding="utf-8")
    print(f"  changed lines (raw)    : {len(changed)}")
    print(f"  changed lines (frozen) : {len(changed_frozen)}")
    print(f"  body instructions differ: {art['findings']['body_instructions_differ']}")
    print(f"  opcode histogram delta : {hist_delta}")
    print(f"  body consumes size     : {art['findings']['workgroup_size_consumed_by_body']}")
    print(f"  shared memory present  : {art['findings']['shared_memory_present']}")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
