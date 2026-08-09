"""Held-out mutations for the decode-time GQA KV-parallel change (issue #90).

A green suite is not evidence that its assertions work. It is evidence that the world is
currently the way they expect. The only way to tell an assertion from a decoration is to break
the thing it claims to pin and watch it go red — and to do that against the REAL artifact, not a
model of it: each mutation below edits a tracked source file, REBUILDS the release EP, runs a
named selection of the production suite, and restores the file. Nothing here simulates a defect
in Python and calls it a mutation.

WHAT EACH MUTATION IS FOR
=========================
Every one breaks exactly one property this change claims, chosen to cover the axes a reviewer
would ask about:

  STRIDE      the strided KV partition `t = lane; t < total; t += W` covers every position
              exactly once. Mutated to `t += W + 1`, which skips positions without changing the
              shape of anything.
  MERGE       the online-softmax merge rescales BOTH sides by `exp(m - m')`. Mutated to drop the
              incoming side's weight, which is the classic "looks plausible, is wrong" merge bug.
  BARRIER     the reduction's `barrier()` sits at loop-body scope, outside the `if`, so every
              invocation reaches it. Mutated to move it inside the `if` — a non-uniform barrier,
              which is undefined behaviour and the single most likely portability defect in a
              cooperative kernel.
  SELECTOR    the optimized shader is selected only at `seq_len == 1`. Mutated to `<= 8`, which
              is EXACTLY the held-out mutation the task brief names: it must be caught by the
              prefill guards, not by luck.
  W_CAP       W is capped at 16 because 32 lanes x 128 floats of accumulator is 16,384 B, the
              entire Vulkan 1.1 guaranteed shared-memory floor. Mutated to 32.
  PIPELINE    the two kernels must not share a pipeline cache key. Mutated so the KV lane count
              is dropped from the key, which is how a stale alias binds the wrong SPIR-V.
  LEDGER      the new shader claims under its own `@kvpar` subject. Mutated so the suffix is
              never appended, which is the B1/B2 defect: the new kernel would inherit `gqa_f16`'s
              proof.
  REFUSAL     the evidence probe removes speed fields when equivalence fails. Mutated so the
              refusal path attaches them anyway, which is the B-series defect the rejected
              artifact shipped.

The last one needs no rebuild — it is a Python-level mutation of the probe's own decision
function — and it is run in-process.

A mutation that leaves the suite green names an assertion that is decoration, and this probe
exits non-zero for it.

Output: bench/results/gqa_decode_kv_parallel_mutations.json

Usage:
    python tests/ops/probe_gqa_decode_kv_parallel_mutations.py
    python tests/ops/probe_gqa_decode_kv_parallel_mutations.py --only STRIDE,MERGE
    python tests/ops/probe_gqa_decode_kv_parallel_mutations.py --list
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_OUT = _ROOT / "bench" / "results" / "gqa_decode_kv_parallel_mutations.json"

SHADER = _ROOT / "rust" / "shaders" / "glsl" / "gqa_decode_f16.comp"
ATTENTION = _ROOT / "rust" / "src" / "ops" / "attention.rs"
REGISTRY = _ROOT / "rust" / "src" / "registry.rs"
SUITE = "tests/ops/test_gqa_decode_kv_parallel.py"


class Mutation:
    """One held-out mutation: a file, a text swap, and the selection that must go red."""

    def __init__(self, name, breaks, path, old, new, select, needs_rebuild=True):
        self.name = name
        self.breaks = breaks
        self.path = path
        self.old = old
        self.new = new
        self.select = select
        self.needs_rebuild = needs_rebuild


MUTATIONS = [
    Mutation(
        "STRIDE",
        "the strided KV partition no longer covers every position exactly once",
        SHADER,
        "for (uint t = lane; t < total_len; t += W) {",
        "for (uint t = lane; t < total_len; t += W + 1u) {",
        f"{SUITE}::test_lane_matrix_matches_cpu",
    ),
    Mutation(
        "MERGE",
        "the online-softmax merge stops rescaling the incoming partial",
        SHADER,
        "            float w2 = exp(m2 - mm);",
        "            float w2 = 1.0;",
        f"{SUITE}::test_lane_matrix_matches_cpu",
    ),
    Mutation(
        "BARRIER",
        "the reduction barrier becomes non-uniform (undefined behaviour)",
        SHADER,
        """        }
        memoryBarrierShared();
        barrier();
    }""",
        """            memoryBarrierShared();
            barrier();
        }
    }""",
        f"{SUITE}::test_lane_matrix_matches_cpu",
    ),
    Mutation(
        "SELECTOR",
        "the decode-only selector boundary widens from seq_len == 1 to seq_len <= 8",
        ATTENTION,
        "    if seq_len != 1 {",
        "    if seq_len > 8 {",
        f"{SUITE}::test_prefill_sequence_lengths_refuse_the_parallel_kernel",
    ),
    Mutation(
        "W_CAP",
        "the W <= 16 shared-memory cap is raised past the Vulkan 1.1 16 KiB floor",
        ATTENTION,
        "pub const GQA_DECODE_MAX_KV_LANES: u32 = 16;",
        "pub const GQA_DECODE_MAX_KV_LANES: u32 = 32;",
        f"{SUITE}::test_override_clamps_and_floors",
    ),
    Mutation(
        "PIPELINE",
        "the KV lane count is dropped from the pipeline cache key, aliasing the two variants",
        ATTENTION,
        "            spec_constants: vec![kv_lanes],",
        "            spec_constants: vec![],",
        f"{SUITE}::test_lane_counts_are_distinct_pipelines_within_one_process",
    ),
    Mutation(
        "LEDGER",
        "the new shader stops claiming under its own @kvpar subject and inherits gqa_f16's proof",
        REGISTRY,
        'format!("{stem}@kvpar")',
        'format!("{stem}")',
        f"{SUITE}::test_the_decode_form_claims_under_its_own_proof_key",
    ),
]

PY_MUTATIONS = ["REFUSAL"]


# ---------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _patched(path: Path, old: str, new: str):
    original = path.read_text(encoding="utf-8")
    if old not in original:
        raise LookupError(f"anchor not found in {path.name}: {old!r}")
    if original.count(old) != 1:
        raise LookupError(
            f"anchor is not unique in {path.name} ({original.count(old)} hits): {old!r}"
        )
    try:
        path.write_text(original.replace(old, new, 1), encoding="utf-8")
        yield
    finally:
        path.write_text(original, encoding="utf-8")


def _build(env) -> "tuple[bool, str]":
    p = subprocess.run(
        ["cargo", "build", "--release", "--lib"],
        cwd=str(_ROOT / "rust"),
        env=env,
        capture_output=True,
        text=True,
    )
    return p.returncode == 0, (p.stderr or "")[-1500:]


def _pytest(select: str, env) -> "tuple[int, str]":
    p = subprocess.run(
        [env.get("KVPAR_PY", sys.executable), "-m", "pytest", select, "-q", "--no-header",
         "-x"],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    return p.returncode, (p.stdout or "")[-1200:]


def _run_source_mutation(mut: Mutation, env) -> dict:
    """Apply, rebuild, run, restore. A rebuild failure is a CAUGHT outcome of its own kind.

    A mutation that will not compile has still been caught — by the compiler rather than by a
    test — and this probe says so explicitly instead of scoring it as a detection by the suite.
    That distinction matters: `BARRIER` is undefined behaviour that compiles fine, and if it were
    silently lumped in with a compile error nobody would notice the suite never saw it.
    """
    rec = {"mutation": mut.name, "breaks": mut.breaks,
           "file": mut.path.relative_to(_ROOT).as_posix(), "selection": mut.select}
    t0 = time.perf_counter()
    try:
        with _patched(mut.path, mut.old, mut.new):
            built, err = _build(env)
            if not built:
                rec.update({"outcome": "CAUGHT(compiler)", "detail": err[-400:]})
                return rec
            code, tail = _pytest(mut.select, env)
            rec.update(
                {
                    "outcome": "CAUGHT" if code != 0 else "MISSED",
                    "pytest_exit": code,
                    "detail": tail[-500:],
                }
            )
    except LookupError as exc:
        rec.update({"outcome": "ERROR(instrument)", "detail": str(exc)})
    finally:
        rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def _run_refusal_mutation(env) -> dict:
    """Break the evidence probe's structural removal and prove a speed field appears.

    This is the only mutation with no rebuild, because the property it breaks lives in Python:
    `_publishable` is the single gate between "the arms were not shown equivalent" and "this
    record carries a speedup". The mutation makes it return True unconditionally, feeds it the
    exact shape of the rejected artifact's own p128 record — DIVERGENT, arms present — and
    asserts that the mutated code DOES emit `kernel_speedup`. If it does not, the structural rule
    is not load-bearing and the refusal in the real probe is decoration.
    """
    rec = {
        "mutation": "REFUSAL",
        "breaks": "the evidence probe's structural removal of speed fields on non-equivalence",
        "file": "bench/results/probe_gqa_decode_kv_parallel.py",
        "selection": "in-process",
    }
    sys.path.insert(0, str(_ROOT / "bench" / "results"))
    try:
        import probe_gqa_decode_kv_parallel as probe
    except Exception as exc:  # noqa: BLE001
        rec.update({"outcome": "ERROR(instrument)", "detail": f"import failed: {exc}"})
        return rec

    divergent = [{
        "past": 128,
        "equivalence": {"equivalent": False, "outputs_total": 65, "outputs_compared": 65,
                        "worst": {"name": "logits", "max_abs": 0.523438},
                        "reason": "past-128 DIVERGENT"},
    }]
    timing = [{
        "case": "decode/M1/past128",
        "past": 128,
        "arms": {
            "parallel": {"median_us": 23610.0, "witness_ok": True, "repeatable": True,
                         "expected_kernel": probe.PARALLEL_EVENT, "graph_total_us": 41600.0},
            "serial": {"median_us": 35780.0, "witness_ok": True, "repeatable": True,
                       "expected_kernel": probe.SERIAL_EVENT, "graph_total_us": 54290.0},
        },
    }]

    control = probe._apply_structural_removal(
        copy.deepcopy(divergent), copy.deepcopy(timing), "control"
    )
    if "kernel_speedup" in control[0] or "verdict" in control[0]:
        rec.update({
            "outcome": "ERROR(instrument)",
            "detail": "the UNMUTATED probe already publishes a speed field for a DIVERGENT "
                      "case; the structural rule is absent, not merely unproven",
        })
        return rec
    if "refusal" not in control[0]:
        rec.update({"outcome": "ERROR(instrument)",
                    "detail": "the unmutated probe produced neither a speed field nor a refusal"})
        return rec

    real = probe._publishable
    try:
        probe._publishable = lambda *_a, **_k: (True, [])
        mutated = probe._apply_structural_removal(
            copy.deepcopy(divergent), copy.deepcopy(timing), "control"
        )
    finally:
        probe._publishable = real

    published = "kernel_speedup" in mutated[0]
    rec.update({
        "outcome": "CAUGHT" if published else "MISSED",
        "detail": (
            f"unmutated: refusal with reasons {control[0]['refusal']['reasons'][0][:120]!r}; "
            f"mutated: kernel_speedup="
            f"{mutated[0].get('kernel_speedup')} verdict={mutated[0].get('verdict')}"
        ),
        "note": "CAUGHT here means the mutation DID produce the forbidden field, which proves "
                "the gate that normally removes it is load-bearing rather than decorative.",
    })
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated mutation names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args(argv)

    if args.list:
        for m in MUTATIONS:
            print(f"{m.name:10s} {m.breaks}")
        for n in PY_MUTATIONS:
            print(f"{n:10s} the evidence probe's structural removal")
        return 0

    wanted = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    env = dict(os.environ)
    lib = env.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib or not Path(lib).is_file():
        print("refusing: ONNXRUNTIME_VULKAN_EP_LIB unset or missing", file=sys.stderr)
        return 2

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "rust/src", "rust/shaders"],
        cwd=str(_ROOT), capture_output=True, text=True,
    ).stdout
    baseline_dirty = bool(dirty.strip())

    rows = []
    for mut in MUTATIONS:
        if wanted and mut.name not in wanted:
            continue
        print(f"[mut] {mut.name}: {mut.breaks}", flush=True)
        rec = _run_source_mutation(mut, env)
        print(f"[mut] {mut.name}: {rec['outcome']} ({rec.get('seconds')}s)", flush=True)
        rows.append(rec)
    if not wanted or "REFUSAL" in wanted:
        print("[mut] REFUSAL: the evidence probe's structural removal", flush=True)
        rec = _run_refusal_mutation(env)
        print(f"[mut] REFUSAL: {rec['outcome']}", flush=True)
        rows.append(rec)

    print("[mut] restoring and rebuilding the unmutated tree ...", flush=True)
    built, err = _build(env)

    doc = {
        "schema": "gqa_decode_kv_parallel_mutations/1",
        "issue": 90,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "each mutation edits a tracked source file, rebuilds the release EP, runs the "
                  "named selection, and restores the file; REFUSAL is an in-process mutation of "
                  "the evidence probe's own decision function",
        "suite": SUITE,
        "tree_was_dirty_in_rust_before_run": baseline_dirty,
        "restored_build_ok": built,
        "restored_build_error": "" if built else err[-400:],
        "mutations": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    missed = [r for r in rows if r["outcome"] == "MISSED"]
    errors = [r for r in rows if r["outcome"].startswith("ERROR")]
    print(f"\nwitness: {Path(args.out).relative_to(_ROOT).as_posix()}")
    print(f"  CAUGHT  {len([r for r in rows if r['outcome'].startswith('CAUGHT')])}")
    print(f"  MISSED  {len(missed)}")
    print(f"  ERROR   {len(errors)}")
    if not built:
        print("MUTATION PROBE: ERROR(instrument=restore_build_failed)", file=sys.stderr)
        return 4
    if errors:
        print(f"MUTATION PROBE: ERROR(instrument) — {[r['mutation'] for r in errors]}",
              file=sys.stderr)
        return 4
    if missed:
        print(f"MUTATION PROBE: FAIL(condition=mutations_missed) — "
              f"{[r['mutation'] for r in missed]}", file=sys.stderr)
        return 1
    print("MUTATION PROBE: PASS — every mutation was caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
