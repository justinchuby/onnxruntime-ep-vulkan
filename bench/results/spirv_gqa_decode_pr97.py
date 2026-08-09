"""Issue #96: what PR #97's decode GQA module costs statically, at each KV-parallel factor.

`spirv_gqa_crossbuild.py` diffs two compilations of *the same* shader. That instrument does not
apply here: PR #97 adds a new file, `gqa_decode_f16.comp`, so there is nothing to diff against --
a `spirv-diff` between two unrelated shaders is noise dressed as evidence. What is answerable
statically is the shape of the new module, and one question in particular that the timing cannot
reach:

    PR #97 selects a KV-parallel factor `W` at pipeline creation via specialisation constant 0.
    The shader sizes its `shared` arrays with `#define MAX_KV_PARALLEL 16` -- a preprocessor
    constant, not the specialisation constant. So does a `W=1` pipeline still declare the full
    16-lane workgroup allocation?

That matters because workgroup memory is an occupancy input: a 1-lane workgroup that still
reserves the 16-lane footprint costs residency it cannot use, and `past=32` (where the host rule
picks `W=1`) is exactly the length where PR #97 has the least to gain and the most to lose. It is
also the question that decides whether a "literal-1 decode variant" is worth proposing to #90.

Method
------
Compile the shader with the production flags (`-fshader-stage=compute --target-env=vulkan1.1 -O`,
read from `rust/build.rs` rather than guessed), then use `spirv-opt --freeze-spec-const` to
produce the module a driver is asked to compile at each `W`, and measure:

  * `OpVariable ... Workgroup` sizes, summed -- the declared workgroup memory footprint;
  * private/function-scope array sizes -- the per-lane register/scratch pressure;
  * barrier and control-flow counts;
  * total instruction count.

Hard limit, stated because it is the same one the Phase 1 SPIR-V section carries
-------------------------------------------------------------------------------
This is a **static module** comparison. SPIR-V is an intermediate form: the driver compiles it
again to machine code, and register allocation, scratch spilling and the final occupancy are
decided there. Nothing in this file can prove what the GPU executed. Where a claim needs the
machine code, it is not made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

#: Read off `rust/build.rs` (`-fshader-stage=compute --target-env=vulkan1.1 -O`), so this module
#: compiles the shader the same way the shipped library does.
GLSLC_FLAGS = ("-fshader-stage=compute", "--target-env=vulkan1.1", "-O")

#: PR #97's `GQA_DECODE_MAX_KV_PARALLEL`; every power of two the host rule can select.
W_VALUES = (1, 2, 4, 8, 16)


def run(*cmd: str) -> str:
    p = subprocess.run(list(cmd), capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"FAILED {' '.join(cmd)}\n{p.stderr}")
    return p.stdout


def tool_versions() -> dict:
    out = {}
    for t in ("glslc", "spirv-dis", "spirv-val", "spirv-opt"):
        p = subprocess.run([t, "--version"], capture_output=True, text=True)
        line = (p.stdout or p.stderr).strip().splitlines()
        out[t] = line[0] if line else "UNKNOWN"
    return out


def _types(dis: str) -> dict:
    """Map result-id -> (kind, detail) for the type instructions we need to size arrays."""
    ints, types = {}, {}
    for m in re.finditer(r"^\s*(%\w+) = OpConstant %\w+ (\d+)$", dis, re.M):
        ints[m.group(1)] = int(m.group(2))
    for line in dis.splitlines():
        m = re.match(r"\s*(%\w+) = OpTypeFloat (\d+)", line)
        if m:
            types[m.group(1)] = ("scalar", int(m.group(2)) // 8)
            continue
        m = re.match(r"\s*(%\w+) = OpTypeInt (\d+)", line)
        if m:
            types[m.group(1)] = ("scalar", int(m.group(2)) // 8)
            continue
        m = re.match(r"\s*(%\w+) = OpTypeVector (%\w+) (\d+)", line)
        if m:
            types[m.group(1)] = ("vector", (m.group(2), int(m.group(3))))
            continue
        m = re.match(r"\s*(%\w+) = OpTypeArray (%\w+) (%\w+)", line)
        if m:
            types[m.group(1)] = ("array", (m.group(2), ints.get(m.group(3))))
            continue
        m = re.match(r"\s*(%\w+) = OpTypePointer (\w+) (%\w+)", line)
        if m:
            types[m.group(1)] = ("pointer", (m.group(2), m.group(3)))
    return types


def _size_of(tid: str, types: dict, depth: int = 0) -> "int | None":
    if depth > 8 or tid not in types:
        return None
    kind, det = types[tid]
    if kind == "scalar":
        return det
    if kind == "vector":
        inner = _size_of(det[0], types, depth + 1)
        return inner * det[1] if inner else None
    if kind == "array":
        inner = _size_of(det[0], types, depth + 1)
        return inner * det[1] if (inner and det[1]) else None
    return None


def storage_bytes(dis: str, storage_class: str) -> dict:
    """Declared bytes per variable in one storage class, and their sum.

    Reads the *declared* size. A driver may lay `shared` out with padding, and may spill
    function-scope arrays to scratch rather than registers; neither is visible here.
    """
    types = _types(dis)
    per_var, total = {}, 0
    for m in re.finditer(r"^\s*(%\w+) = OpVariable (%\w+) " + storage_class + r"\s*$", dis, re.M):
        var, ptr = m.group(1), m.group(2)
        if ptr not in types or types[ptr][0] != "pointer":
            continue
        nbytes = _size_of(types[ptr][1][1], types)
        if nbytes:
            per_var[var] = nbytes
            total += nbytes
    return {"per_variable_bytes": per_var, "total_bytes": total,
            "variable_count": len(per_var)}


def histogram(dis: str) -> dict:
    h = {}
    for line in dis.splitlines():
        m = re.search(r"(?:^\s*%\w+ = )?(Op\w+)", line)
        if m:
            h[m.group(1)] = h.get(m.group(1), 0) + 1
    return dict(sorted(h.items()))


def analyse(spv: Path, tag: str, tmp: Path) -> dict:
    dis = run("spirv-dis", "--no-header", str(spv))
    (tmp / f"{tag}.dis").write_text(dis, encoding="utf-8")
    val = subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(spv)],
                         capture_output=True, text=True)
    h = histogram(dis)
    return {
        "spv_sha256": hashlib.sha256(spv.read_bytes()).hexdigest(),
        "spv_bytes": spv.stat().st_size,
        "disassembly_sha256": hashlib.sha256(dis.encode()).hexdigest(),
        "spirv_val": "PASS" if val.returncode == 0 else val.stderr.strip()[:400],
        "instruction_count": sum(h.values()),
        "workgroup_memory": storage_bytes(dis, "Workgroup"),
        "function_scope_memory": storage_bytes(dis, "Function"),
        "barriers": h.get("OpControlBarrier", 0) + h.get("OpMemoryBarrier", 0),
        "branches": h.get("OpBranch", 0) + h.get("OpBranchConditional", 0),
        "loop_merges": h.get("OpLoopMerge", 0),
        "spec_constants": h.get("OpSpecConstant", 0),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-shader", required=True, help="gqa_decode_f16.comp from PR #97")
    ap.add_argument("--legacy-shader", required=True, help="gqa_f16.comp, for context only")
    ap.add_argument("--include-dir", required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        def compile_(src: str, tag: str) -> Path:
            spv = tmp / f"{tag}.spv"
            run("glslc", *GLSLC_FLAGS, f"-I{a.include_dir}", "-o", str(spv), src)
            return spv

        decode_spv = compile_(a.decode_shader, "decode")
        legacy_spv = compile_(a.legacy_shader, "legacy")

        decode = analyse(decode_spv, "decode", tmp)
        legacy = analyse(legacy_spv, "legacy", tmp)

        per_w = {}
        for w in W_VALUES:
            frozen = tmp / f"decode_w{w}.spv"
            run("spirv-opt", "--set-spec-const-default-value", f"0:{w}", "--freeze-spec-const",
                str(decode_spv), "-o", str(frozen))
            per_w[str(w)] = analyse(frozen, f"decode_w{w}", tmp)

    shared_sizes = {w: v["workgroup_memory"]["total_bytes"] for w, v in per_w.items()}
    shared_constant = len(set(shared_sizes.values())) == 1
    w1, w16 = per_w["1"], per_w["16"]

    art = {
        "instrument": Path(__file__).name,
        "issue": 96,
        "question": ("What does PR #97's gqa_decode_f16 module declare statically, and does "
                     "freezing the KV-parallel spec constant to 1 shrink it?"),
        "commit": a.commit,
        "tools": tool_versions(),
        "glslc_flags": list(GLSLC_FLAGS),
        "glslc_flags_provenance": "rust/build.rs, the same flags the shipped library is built with",
        "legacy_gqa_f16_for_context": {
            "note": ("Included only to size the two modules against each other. These are "
                     "DIFFERENT shaders; no line-level diff between them would mean anything."),
            **legacy,
        },
        "decode_unspecialised": decode,
        "decode_per_kv_parallel": per_w,
        "findings": {
            "workgroup_bytes_by_W": shared_sizes,
            "workgroup_allocation_is_independent_of_W": shared_constant,
            "workgroup_bytes_note": (
                "The shader sizes its shared arrays with `#define MAX_KV_PARALLEL 16`, a "
                "preprocessor constant, not with specialisation constant 0. If the figure above "
                "is flat across W, a W=1 pipeline declares the same workgroup memory as a W=16 "
                "pipeline and pays the same occupancy input for lanes it will not use."
                if shared_constant else
                "The declared workgroup footprint varies with W; the compiler folded the "
                "specialisation constant into the allocation."),
            "instruction_count_by_W": {w: v["instruction_count"] for w, v in per_w.items()},
            "w1_vs_w16_instruction_delta": w1["instruction_count"] - w16["instruction_count"],
            "barriers_by_W": {w: v["barriers"] for w, v in per_w.items()},
            "function_scope_bytes_by_W": {w: v["function_scope_memory"]["total_bytes"]
                                          for w, v in per_w.items()},
            "decode_vs_legacy_instruction_ratio": (
                round(decode["instruction_count"] / legacy["instruction_count"], 3)
                if legacy["instruction_count"] else None),
        },
        "limits": [
            "Static module only. The driver recompiles SPIR-V to machine code; register "
            "allocation, scratch spilling and final occupancy are decided there and are not "
            "observable here.",
            "Declared workgroup bytes are not necessarily allocated bytes; a driver may pad.",
            "`gqa_f16` and `gqa_decode_f16` are different shaders. Nothing here is a diff.",
        ],
    }
    Path(a.out).write_text(json.dumps(art, indent=2), encoding="utf-8")
    f = art["findings"]
    print(f"workgroup bytes by W: {f['workgroup_bytes_by_W']}")
    print(f"independent of W:     {f['workgroup_allocation_is_independent_of_W']}")
    print(f"instructions by W:    {f['instruction_count_by_W']}")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
