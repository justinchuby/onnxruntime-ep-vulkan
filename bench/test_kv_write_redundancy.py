"""Gates over the KV write-redundancy probe and the KV lever ledger.

WHY THIS FILE EXISTS
====================
Two things landed this round that a later edit could silently undo:

1. `gqa_f16.comp` writes `present_key`/`present_value` from **one** invocation per KV group.
   The G writers it used to have were bit-identical, so nothing that compares outputs can tell
   the two versions apart -- the redundancy is invisible to every correctness test in the tree
   and would come straight back the next time somebody touches the write.

2. The SIMT interpreter's GLSL.std.450 opcode table was **wrong** for `FMin`/`FMax`/`FClamp`/
   `Fma`. Every one of those returns a plausible float, and the interpreter's only correctness
   control was a quantised GEMV that calls none of them.

Both are gated here by executing the compiled SPIR-V, not by reading the source.
"""

from __future__ import annotations

import json
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
        f"here makes the interpreter compute a different function and return a plausible float."
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
