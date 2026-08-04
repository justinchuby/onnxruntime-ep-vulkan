"""Blast radius of the `_GLSL450` opcode-table defect: which kernels, and silent or loud.

WHAT THIS ANSWERS
=================
If a trace was taken over kernel K with the OLD table, could the error have changed what the
interpreter computed for K -- and would anybody have noticed?

    old table: 30:Fma, 37:FMax, 40:FClamp, 43:FMin      (50 absent)
    real:      30:Log2, 37:FMin, 40:FMax, 43:FClamp, 50:Fma

A CORRECTION TO MY OWN CLAIM (2026-08-03, second pass)
=======================================================
I wrote in §23.3 and in `switch-spirv-interpreter-glsl450-table-wrong.md` that **a relu would
have been computed as a minimum, returning a plausible float, silently**. That is false, and the
discriminator is the OPERAND COUNT, which I did not check before saying it:

  * A wrong name that takes MORE operands than the real one indexes past the end of the operand
    list and raises `IndexError`. **Loud.** `relu` is ext-inst 40; the old table called it
    `FClamp`, which reads `args[2]`; `max(x, 0.0)` supplies two. It would have raised on the
    first invocation.
  * A wrong name that takes FEWER operands silently ignores the extra and returns a plausible
    float. **Silent.** That is ext-inst 37 (real `FMin`, old `FMax`) and ext-inst 43 (real
    `FClamp`, old `FMin`, which drops the upper bound entirely).

So the silent set is `{37, 43}` -- `celu`, `hardsigmoid`, `hardswish` -- and not `relu`. The
general lesson is unchanged and the vivid example was wrong. Verified by executing all three
forms under both tables rather than by reading the dispatch code; see `classify()`.
"""

import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "bench" / "results"))

import spirv_simt  # noqa: E402
from spirv_simt import Dispatch, InstrumentError, SpirvModule, _GLSL450  # noqa: E402

import probe_kv_write_redundancy as W  # noqa: E402

OLD = {30: "Fma", 37: "FMax", 40: "FClamp", 43: "FMin"}
SUSPECT = set(OLD) | {50}

SRC = """#version 450
layout(local_size_x = 8) in;
layout(std430, binding = 0) readonly buffer A {{ float a[]; }};
layout(std430, binding = 2) writeonly buffer O {{ float o[]; }};
void main() {{ uint i = gl_GlobalInvocationID.x; float x = a[i]; {body} }}
"""

FORMS = {
    40: ("max(x, 0.0)", "o[i] = max(x, 0.0);"),
    37: ("min(x, 0.0)", "o[i] = min(x, 0.0);"),
    43: ("clamp(x, 0.0, 1.0)", "o[i] = clamp(x, 0.0, 1.0);"),
    50: ("fma(x, x, 1.0)", "o[i] = fma(x, x, 1.0);"),
}


def _run(body: str) -> np.ndarray:
    mod = SpirvModule(W.compile_glsl(SRC.format(body=body), "blast"))
    n = 32
    a = np.linspace(-2.0, 2.0, n).astype(np.float32)
    d = Dispatch(groups=(n // 8, 1, 1), local_size=(8, 1, 1), spec={}, push_constants=[],
                 buffers={0: a.copy(), 2: np.zeros(n, np.float32)})
    mod.run_traced(d)
    return d.buffers[2].copy()


def classify() -> dict:
    """Execute each suspect opcode under both tables. Silence is measured, not argued."""
    good_table = dict(_GLSL450)
    out = {}
    for num, (label, body) in FORMS.items():
        spirv_simt._GLSL450.clear()
        spirv_simt._GLSL450.update(good_table)
        try:
            correct = _run(body)
        except InstrumentError as e:
            out[num] = {"form": label, "verdict": "UNCOMPILABLE", "detail": str(e)}
            continue
        spirv_simt._GLSL450.clear()
        spirv_simt._GLSL450.update(good_table)
        spirv_simt._GLSL450.update(OLD)
        spirv_simt._GLSL450.pop(50, None)
        try:
            under_old = _run(body)
        except Exception as e:  # noqa: BLE001 - the point is which exception, if any
            out[num] = {"form": label, "real_name": good_table.get(num),
                        "old_name": OLD.get(num, "(absent)"),
                        "verdict": "LOUD",
                        "detail": f"{type(e).__name__}: {e}"}
            continue
        finally:
            spirv_simt._GLSL450.clear()
            spirv_simt._GLSL450.update(good_table)
        same = bool(np.array_equal(correct, under_old))
        out[num] = {
            "form": label, "real_name": good_table.get(num),
            "old_name": OLD.get(num, "(absent)"),
            "verdict": "HARMLESS" if same else "SILENT",
            "correct_head": [float(v) for v in correct[:4]],
            "under_old_head": [float(v) for v in under_old[:4]],
        }
    spirv_simt._GLSL450.clear()
    spirv_simt._GLSL450.update(good_table)
    return out


def survey() -> tuple[list, list, list]:
    root = REPO / "rust" / "target" / "release"
    touched, clean, noext, seen = [], [], [], set()
    for p in sorted(root.rglob("*.spv")):
        if p.stem in seen:
            continue
        try:
            mod = SpirvModule(p.read_bytes())
        except InstrumentError:
            continue
        seen.add(p.stem)
        counts = collections.Counter(
            i.operands[1] for i in mod.body if i.op == "OpExtInst")
        if not counts:
            noext.append(p.stem)
            continue
        hit = SUSPECT & set(counts)
        (touched if hit else clean).append((p.stem, dict(counts), sorted(hit)))
    return touched, clean, noext


def main() -> int:
    verdicts = classify()
    print("IS THE ERROR SILENT? — executed under both tables, not reasoned about")
    print("=" * 84)
    for num in sorted(verdicts):
        v = verdicts[num]
        print(f"  ext-inst {num:>2}  real {v.get('real_name')!s:<8} old said "
              f"{v.get('old_name')!s:<10} {v['form']:<20} -> {v['verdict']}")
        if v["verdict"] == "SILENT":
            print(f"      correct {v['correct_head']}")
            print(f"      old     {v['under_old_head']}")
        elif v["verdict"] == "LOUD":
            print(f"      {v['detail']}")
    silent = {n for n, v in verdicts.items() if v["verdict"] == "SILENT"}
    print(f"\n  SILENT set: {sorted(silent)}  (these returned plausible floats)")
    print(f"  LOUD set:   {sorted(n for n, v in verdicts.items() if v['verdict'] == 'LOUD')}"
          f"  (these raised on the first invocation)")

    touched, clean, noext = survey()
    print()
    print("KERNELS IN THIS BUILD THAT ISSUE A SUSPECT OPCODE")
    print("=" * 84)
    at_risk = []
    for stem, counts, hit in touched:
        marks = []
        for n in hit:
            v = verdicts.get(n, {})
            marks.append(f"{n}={_GLSL450.get(n)}[{v.get('verdict', '?')}]")
            if v.get("verdict") == "SILENT":
                at_risk.append(stem)
        print(f"  {stem:<34} " + ", ".join(marks))
    at_risk = sorted(set(at_risk))
    print(f"\n  SILENTLY MISCOMPUTED: {at_risk or '(none)'}")
    print(f"  {len(clean)} kernels issue ext-inst but no suspect opcode; "
          f"{len(noext)} issue none at all.")

    print()
    print("THE KERNELS THE STANDING MEASUREMENTS RUN OVER")
    print("=" * 84)
    for stem, counts, _ in clean:
        if stem in ("q_gemv_matmul_nbits_f16", "gqa_f16"):
            print(f"  {stem:<34} "
                  + ", ".join(f"{n}={_GLSL450.get(n)}x{k}" for n, k in sorted(counts.items()))
                  + "   UNTOUCHED")
    for stem in noext:
        if stem == "q_gemv_matmul_nbits_f32":
            print(f"  {stem:<34} (no ext-inst)                          UNTOUCHED")

    rec = {
        "what": "blast radius of the _GLSL450 opcode-table defect",
        "old_table": OLD,
        "verdicts": {str(k): v for k, v in verdicts.items()},
        "silently_miscomputed_kernels": at_risk,
        "kernels_issuing_a_suspect_opcode": [t[0] for t in touched],
        "correction": (
            "my earlier claim that a relu would have been silently miscomputed is FALSE: "
            "ext-inst 40 under the old table mapped to FClamp, which reads a third operand "
            "max() does not supply, so it raises IndexError. The silent set is {37, 43}."),
    }
    out = REPO / "bench" / "results" / "glsl450_blast_radius.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nrecord: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
