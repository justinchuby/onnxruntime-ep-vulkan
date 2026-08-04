"""Gates over the KV write-redundancy probe and the KV lever ledger.

WHY THIS FILE EXISTS
====================
Two things landed this round that a later edit could silently undo:

1. `gqa_f16.comp` writes `present_key`/`present_value` from **one** invocation per KV group.
   The G writers it used to have were bit-identical, so nothing that compares outputs can tell
   the two versions apart -- the redundancy is invisible to every correctness test in the tree
   and would come straight back the next time somebody touches the write.

2. The SIMT interpreter's GLSL.std.450 opcode table was **wrong** for `FMin`/`FMax`/`FClamp`/
   `Fma`. Two of the four (37 `FMin` read as `FMax`, 43 `FClamp` read as `FMin`) return plausible
   floats silently; the other two raise. The interpreter's only correctness control was a
   quantised GEMV that calls none of them.

Both are gated here by executing the compiled SPIR-V, not by reading the source.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "bench" / "results"))

from spirv_simt import Dispatch, InstrumentError, SpirvModule  # noqa: E402

import probe_kv_write_redundancy as W  # noqa: E402


# ----------------------------------------------------------------------------------
# The interpreter's own instruction table, checked against the disassembler's names.
# ----------------------------------------------------------------------------------


def _spv_paths() -> list[pathlib.Path]:
    return sorted((ROOT / "rust" / "target").rglob("*.spv"))


@pytest.mark.parametrize("num,name", [
    (27, "Exp"), (28, "Log"), (31, "Sqrt"), (32, "InverseSqrt"),
    (37, "FMin"), (40, "FMax"), (42, "SMax"), (43, "FClamp"), (45, "SClamp"),
    (50, "Fma"), (58, "PackHalf2x16"), (62, "UnpackHalf2x16"),
])
def test_glsl450_numbering_matches_the_specification(num: int, name: str) -> None:
    """The numbers `_GLSL450` maps are the GLSL.std.450 numbers, not a remembered set.

    The four that were wrong before this round -- 37/40/43 and Fma -- are all in this list.
    """
    from spirv_simt import _GLSL450
    assert _GLSL450.get(num) == name, (
        f"GLSL.std.450 {num} is {name}; the table says {_GLSL450.get(num)!r}. A wrong entry "
        f"here makes the interpreter compute a different function; whether it does so silently "
        f"or raises depends on the operand count, and 37 and 43 were the silent ones."
    )


def test_the_shaders_in_this_tree_only_use_understood_ext_instructions() -> None:
    """Every ext-inst any compiled module in this tree issues is one the table names.

    An unnamed one raises at execution rather than being skipped, so this is a coverage
    check, not a correctness check -- but it is how a new shader's new opcode gets noticed
    before somebody trusts a trace taken over it.
    """
    from spirv_simt import _GLSL450
    paths = _spv_paths()
    if not paths:
        pytest.skip("no compiled SPIR-V under rust/target; build the EP first")
    unknown: dict[int, str] = {}
    for p in paths:
        try:
            m = SpirvModule(p.read_bytes())
        except InstrumentError:
            continue
        for ins in m.body:
            if ins.op == "OpExtInst" and ins.operands[1] not in _GLSL450:
                unknown.setdefault(ins.operands[1], p.name)
    assert not unknown, f"ext-inst numbers with no name in the table: {unknown}"


_EXT_CONTROL_GLSL = """#version 450
layout(local_size_x = 8) in;
layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 1) readonly buffer B { float b[]; };
layout(std430, binding = 2) writeonly buffer O { float o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    float x = a[i], y = b[i];
    o[i * 4u + 0u] = min(x, y);
    o[i * 4u + 1u] = max(x, y);
    o[i * 4u + 2u] = clamp(x, -0.5, 0.5);
    o[i * 4u + 3u] = fma(x, y, 1.0);
}
"""


def test_min_max_clamp_and_fma_are_executed_as_themselves() -> None:
    """The correctness control the interpreter never had for the four opcodes it got wrong.

    `_GLSL450` said 30:Fma, 37:FMax, 40:FClamp, 43:FMin; the real numbering is 37:FMin, 40:FMax,
    43:FClamp, 50:Fma. Two of those return plausible floats silently (37 and 43, whose wrong names
    take FEWER operands than the real ones and drop the extra); two raise. The error stayed
    invisible because the interpreter's only correctness control was a quantised GEMV that issues
    none of them.

    This compiles a kernel that calls all four through glslc, so the opcode NUMBERS come from the
    real toolchain rather than from anybody's memory, executes it in the interpreter, and
    compares against numpy. Verified to fail under the old table before being trusted under the
    new one; see `bench/results/probe_glsl450_blast_radius.py` for the silent/loud split.
    """
    try:
        spv = W.compile_glsl(_EXT_CONTROL_GLSL, "ext-control")
    except InstrumentError as e:
        pytest.skip(f"glslc unavailable: {e}")

    mod = SpirvModule(spv)
    issued = {i.operands[1] for i in mod.body if i.op == "OpExtInst"}
    assert {37, 40, 43, 50} <= issued, (
        f"the control kernel did not issue the four opcodes under test; it issued {issued}. "
        f"A control that does not exercise the thing is not a control.")

    n = 64
    rng = np.random.default_rng(4242)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    d = Dispatch(
        groups=(n // 8, 1, 1),
        local_size=(8, 1, 1),
        spec={},
        push_constants=[],
        buffers={0: a.copy(), 1: b.copy(), 2: np.zeros(n * 4, np.float32)},
    )
    mod.run_traced(d)
    got = d.buffers[2].reshape(n, 4)

    np.testing.assert_allclose(got[:, 0], np.minimum(a, b), rtol=0, atol=0)
    np.testing.assert_allclose(got[:, 1], np.maximum(a, b), rtol=0, atol=0)
    np.testing.assert_allclose(got[:, 2], np.clip(a, -0.5, 0.5), rtol=0, atol=0)
    np.testing.assert_allclose(got[:, 3], a * b + 1.0, rtol=1e-6, atol=1e-6)


# ----------------------------------------------------------------------------------
# The group write leader.
# ----------------------------------------------------------------------------------


def _compiled_gqa() -> SpirvModule:
    try:
        return SpirvModule(W.compile_glsl(W.SHADER_SRC.read_text(encoding="utf-8"), "worktree"))
    except InstrumentError as e:
        pytest.skip(f"cannot compile gqa_f16.comp here: {e}")


@pytest.mark.parametrize("G", [1, 2, 4, 8])
def test_exactly_one_invocation_writes_each_present_word(G: int) -> None:
    """At every grouping, each new-token word of `present_key` has exactly one writer.

    This is the regression gate. It executes the compiled module over the whole grid, so it
    cannot be satisfied by a comment or by a source pattern.
    """
    nkv = 2
    c = W.Case("gate", 1, 1, nkv * G, nkv, 32, past_len=4, growing=False, seed=100 + G)
    assert c.G == G
    mod = _compiled_gqa()
    d = c.dispatch()
    _, wk = mod.run_traced(d, store_binding=W.PRESENT_K_BINDING)
    assert wk.touched_words > 0, "nothing was written; the probe is measuring an empty grid"
    assert wk.max_writers_per_word == 1, (
        f"at G={G} some word of `present_key` has {wk.max_writers_per_word} writers. The G "
        f"query heads of a KV group index `present` by kv_h and write identical values, so "
        f"this is invisible to every output comparison and costs G x the KV write bandwidth."
    )


def test_the_deduplication_is_bit_exact_against_the_redundant_version() -> None:
    """Removing the redundant writers changes no output bit, at G=4, arena and growing.

    The safety of the masked half-word writes used to rest on the redundancy (identical values,
    complementary masks). This is the check that it now rests on disjointness instead.
    """
    mod = _compiled_gqa()
    base_text = W.baseline_source("main")
    have_baseline = base_text != (ROOT / W.SHADER_REL).read_text(encoding="utf-8")
    base_mod = SpirvModule(W.compile_glsl(base_text, "baseline@main")) if have_baseline else None

    for growing in (False, True):
        lane = "growing" if growing else "arena"
        c = W.Case("gate", 1, 3, 8, 2, 32, past_len=5, growing=growing, seed=200)
        d = c.dispatch()
        mod.run_traced(d)
        ref = W.numpy_reference(c)
        got = W.unpack_f16(d.buffers[W.ATTN_OUT_BINDING], c.n_out).reshape(ref.shape)
        rel = float(np.max(np.abs(got - ref)) / max(float(np.max(np.abs(ref))), 1e-9))
        assert rel < 5e-3, (
            f"{lane}: the deduplicated kernel disagrees with a "
            f"numpy GQA reference by {rel:.3e}; a word may have lost its only writer")

        if base_mod is None:
            continue
        db = c.dispatch()
        base_mod.run_traced(db)
        for binding, what in ((W.ATTN_OUT_BINDING, "attn_output"),
                              (W.PRESENT_K_BINDING, "present_key"),
                              (W.PRESENT_V_BINDING, "present_value")):
            assert np.array_equal(d.buffers[binding], db.buffers[binding]), (
                f"{lane}: {what} differs bit-for-bit from the redundant kernel at main. The "
                f"dedup is only sound if the surviving writer covers every bit the G wrote.")


def test_present_words_are_fully_covered_after_deduplication() -> None:
    """Every new-token word still gets written. One writer, not zero.

    The failure mode the dedup could introduce is a word whose only writer was removed -- which
    would read back as whatever was there before, i.e. plausible stale KV rather than a crash.
    """
    c = W.Case("gate", 2, 2, 8, 2, 64, past_len=3, growing=False, seed=300)
    mod = _compiled_gqa()
    d = c.dispatch()
    _, wk = mod.run_traced(d, store_binding=W.PRESENT_K_BINDING)
    row = c.D // 2
    expected = 0
    for b in range(c.B):
        for kv in range(c.Nkv):
            for s in range(c.S):
                base = ((b * c.Nkv + kv) * c.present_len + c.past_len + s) * row
                seg = wk.writer_invocations[base:base + row]
                assert seg.min() == 1 and seg.max() == 1, (
                    f"new-token word run at {base} has writer counts "
                    f"{seg.min()}..{seg.max()}, expected exactly 1")
                expected += row
    assert int((wk.writer_invocations == 1).sum()) >= expected


# ----------------------------------------------------------------------------------
# The lever ledger.
# ----------------------------------------------------------------------------------


def test_the_lever_ledger_regenerates_and_every_figure_carries_a_class() -> None:
    """The ledger is a generator, not a table. It must run, and it must class every lever."""
    script = ROOT / "bench" / "results" / "probe_kv_lever_ledger.py"
    p = subprocess.run([sys.executable, str(script)], cwd=str(ROOT),
                       capture_output=True, text=True)
    assert p.returncode == 0, f"the ledger generator failed:\n{p.stdout}\n{p.stderr}"
    rec = json.loads((ROOT / "bench" / "results" / "kv_lever_ledger.json")
                     .read_text(encoding="utf-8"))
    assert {L["id"] for L in rec["levers"]} == {"L1", "L2", "L3", "L4", "L5"}
    for L in rec["levers"]:
        assert L["class"] in ("SPECIFICATION", "MEASUREMENT", "MODEL"), L["id"]
        assert L["baseline"], f"{L['id']} has no named baseline; a ratio without one is not one"
        assert L["axis"].split(" ")[0] in ("LINK", "DRAM", "FOOTPRINT"), L["id"]


def test_the_retracted_three_are_recorded_as_retracted_not_as_figures() -> None:
    """2.21 / 3.17 / 4.06 must survive in the record only as targets that did not reproduce."""
    rec = json.loads((ROOT / "bench" / "results" / "kv_lever_ledger.json")
                     .read_text(encoding="utf-8"))
    r = rec["retracted"]
    assert set(r["targets"].values()) == {2.21, 3.17, 4.06}
    assert "RETRACTED" in r["conclusion"]
    for L in rec["levers"]:
        for v in (L.get("ratio"), L.get("ratio_int8_best"), L.get("ratio_int4_best")):
            if isinstance(v, (int, float)):
                assert round(v, 2) not in (2.21, 3.17, 4.06), (
                    f"{L['id']} publishes {v}, one of the retracted figures")


def test_int8_footprint_figures_are_derived_not_transcribed() -> None:
    """L5's bytes/token must follow from the graph's geometry, not from a remembered table."""
    rec = json.loads((ROOT / "bench" / "results" / "kv_lever_ledger.json")
                     .read_text(encoding="utf-8"))
    L5 = [L for L in rec["levers"] if L["id"] == "L5"][0]
    assert L5["class"] == "MODEL", "no int8 kernel exists; no int8 byte can be MEASUREMENT"
    elems = L5["derivation"]["elements_per_token"]
    for lane in L5["lanes"]:
        payload = elems * lane["bits"] // 8
        scales = lane["kv_bytes_per_token"] - payload
        assert scales > 0, f"{lane['lane']} carries no scale bytes; it cannot be dequantised"
        assert lane["kv_bytes_per_token"] < L5["derivation"]["fp16_kv_bytes_per_token"]


# ----------------------------------------------------------------------------------
# The gate that was missing, and whose absence cost a rollback.
# ----------------------------------------------------------------------------------


def test_the_ledger_on_disk_describes_the_shaders_in_this_build() -> None:
    """A shader edit that leaves its proof-ledger entry stale must fail HERE, not at merge.

    This is the gate whose absence cost `2b38528` a rollback. That commit's `--check` was green
    in its own worktree because the entry had been re-proved against *that* build; after the
    merge the entry described a kernel the merged binary did not contain, and the only symptom
    was a silent `SUBJECT-CHANGED` decline. The user-visible cost was 355 claimed nodes falling
    to 323 and one island shattering into 33 -- a performance regression wearing the clothes of
    a correctness mechanism working correctly.

    `--check` alone is not sufficient and that is the point: the file is internally consistent
    either way. The question is whether it agrees with the DLL, so the DLL has to be read.
    """
    sys.path.insert(0, str(ROOT / "rust" / "tools"))
    try:
        from gen_proof_ledger import _find_lib, check_against_build  # noqa: PLC0415
    except Exception as e:  # pragma: no cover - tool absent
        pytest.skip(f"gen_proof_ledger not importable: {e}")

    lib = _find_lib(os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB", ""))
    if lib is None or not pathlib.Path(lib).is_file():
        pytest.skip(
            "no EP binary found; set ONNXRUNTIME_VULKAN_EP_LIB to "
            "rust/target/release/onnxruntime_vulkan_ep.dll. Note this skip is the reason the "
            "check must never be read as a PASS when the variable is unset.")

    problems, _ = check_against_build(ROOT / "evidence" / "proof_ledger.jsonl",
                                      pathlib.Path(lib))
    assert not problems, (
        "the ledger does not describe this build:\n  " + "\n  ".join(problems) +
        "\n\nRepair, and BOTH steps are required:\n"
        "  1. python rust/tools/gen_proof_ledger.py --model <single-form case> --reprove --append\n"
        "  2. rebuild -- proof_ledger.jsonl is include_str!'d, so the on-disk file takes "
        "effect only at the next build.\n"
        "Use a graph from evidence/cases/, not Phi-3.5: the prove pass sets "
        "`disable_cpu_ep_fallback`, and Phi-3.5 contains Shape/ReduceSum/If, which this EP "
        "registers no handler for, so the session can never build.")
