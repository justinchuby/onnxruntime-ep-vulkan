#!/usr/bin/env python3
"""How many invocations write each word of `present_key`? Executed, not argued. No clock.

WHY THIS EXISTS
===============
`gqa_f16.comp` indexes `present_key`/`present_value` by `kv_h = h / G`, but the dispatch is one
invocation per `(b, h, s_local)` -- one per **query** head. At `G = Nq/Nkv > 1` the G query heads
of one KV group therefore name the *same* `present` words and write them with the *same* values.
Correctness is untouched (identical values, complementary `And`/`Or` masks, the last operation is
always a correct `Or`), which is precisely why no test could see it: it is G x the KV write
traffic and zero x the error.

Phi-3.5-mini has `Nq/Nkv = 32/32 = 1`. It is structurally incapable of exhibiting this. Every
number this project has published about KV write bandwidth came from a model where G = 1.

WHAT THIS DOES
==============
Runs the *compiled* SPIR-V (`bench/spirv_simt.py`) over the whole dispatch grid and records, for
every 32-bit word of binding 7 (`present_key`) and binding 8 (`present_value`):

  * `word_writes`          -- write instructions naming the word (what the memory system sees)
  * `writer_invocations`   -- distinct invocations that wrote it (what the claim is about)

Two variants are compiled from source with the build's own `glslc` flags, and the baseline is
taken from **git**, not from memory:

  * BASELINE -- `git show <base-ref>:rust/shaders/glsl/gqa_f16.comp`
  * FIXED    -- the working tree

Four arms per variant: `G = 1` (Phi-3.5's grouping) and `G = 4` (Llama-3 / Mistral / Qwen2), each
at decode (`S = 1`) and prefill (`S = 8`).

CORRECTNESS IS READ BEFORE ANY BYTE COUNT
=========================================
For every arm, all three outputs of the two variants -- `attn_output`, `present_key`,
`present_value` -- are compared **bit-exact** (`uint32` word equality on the raw buffers). If any
arm disagrees the probe prints the disagreement and exits non-zero without publishing a single
byte figure. A saving that came with a changed output is not a saving.

A second control guards against both variants being trivially wrong together: `attn_output` is
compared against a numpy GQA reference (fp32 over fp16 inputs, plain softmax rather than online,
so this one carries a tolerance and is a sanity check, not the correctness claim).

WHAT IT DOES NOT ESTABLISH
==========================
The trace counts **words named by write instructions**, not DRAM transactions. For a KV cache
larger than L2 and written once per token these coincide, by the same argument
`probe_roofline.py` makes for the weight stream -- and that is an argument, not this measurement.
Per-dispatch figures below are MEASUREMENT; anything multiplied out to a whole inference is
MODEL and is labelled so in the record.

Run:  python bench/results/probe_kv_write_redundancy.py [--base-ref main]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))

from spirv_simt import Dispatch, InstrumentError, SpirvModule  # noqa: E402

SHADER_REL = "rust/shaders/glsl/gqa_f16.comp"
SHADER_SRC = ROOT / SHADER_REL
SHADER_INC = ROOT / "rust" / "shaders" / "include"

PRESENT_K_BINDING = 7
PRESENT_V_BINDING = 8
ATTN_OUT_BINDING = 6


def _tool(name: str) -> str | None:
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        cand = pathlib.Path(sdk) / "Bin" / (name + (".exe" if os.name == "nt" else ""))
        if cand.is_file():
            return str(cand)
    return shutil.which(name)


def compile_glsl(text: str, label: str) -> bytes:
    """Compile GLSL with the flags `rust/build.rs` uses. Any other flags are a different kernel."""
    glslc = _tool("glslc")
    if glslc is None:
        raise InstrumentError(
            "glslc not found (set VULKAN_SDK or put it on PATH); this probe will not read a "
            "module it did not compile"
        )
    with tempfile.TemporaryDirectory(dir=str(ROOT / "rust" / "target")) as td:
        d = pathlib.Path(td)
        comp = d / "gqa_f16.comp"
        comp.write_text(text, encoding="utf-8")
        spv = d / "gqa_f16.spv"
        p = subprocess.run(
            [glslc, "-fshader-stage=compute", "--target-env=vulkan1.1", "-O",
             f"-I{SHADER_INC}", str(comp), "-o", str(spv)],
            capture_output=True, text=True,
        )
        if p.returncode != 0:
            raise InstrumentError(f"glslc failed for {label}:\n{p.stdout}\n{p.stderr}")
        return spv.read_bytes()


def baseline_source(ref: str) -> str:
    p = subprocess.run(["git", "show", f"{ref}:{SHADER_REL}"],
                       cwd=str(ROOT), capture_output=True, text=True)
    if p.returncode != 0:
        raise InstrumentError(f"`git show {ref}:{SHADER_REL}` failed:\n{p.stderr}")
    return p.stdout


# -- dispatch construction ---------------------------------------------------------------------


def pack_f16(a: np.ndarray) -> np.ndarray:
    """Flat f16 array -> the uint32 word array the shaders' packing convention names."""
    h = np.asarray(a, dtype=np.float16).ravel()
    if h.size % 2:
        h = np.concatenate([h, np.zeros(1, np.float16)])
    return h.view(np.uint32).copy()


def unpack_f16(w: np.ndarray, n: int) -> np.ndarray:
    return w.view(np.float16)[:n].astype(np.float32)


class Case:
    """One GQA shape, with buffers built once and cloned per variant."""

    def __init__(self, name: str, B: int, S: int, Nq: int, Nkv: int, D: int,
                 past_len: int, growing: bool, seed: int = 0):
        self.name, self.B, self.S = name, B, S
        self.Nq, self.Nkv, self.D = Nq, Nkv, D
        self.G = Nq // Nkv
        self.past_len, self.growing = past_len, growing
        self.past_stride = past_len
        self.present_len = past_len + S if growing else max(past_len + S, past_len)
        if not growing:
            self.past_stride = self.present_len
        self.rot = D
        rh = D // 2
        rng = np.random.default_rng(seed)

        def rnd(n):
            return rng.standard_normal(n).astype(np.float16)

        self.n_qkv = B * S * (Nq + 2 * Nkv) * D
        self.n_past = B * Nkv * self.past_stride * D
        self.n_pres = B * Nkv * self.present_len * D
        self.n_out = B * S * Nq * D
        max_pos = self.present_len + S + 8
        self.src = {
            0: pack_f16(rnd(self.n_qkv)),
            1: pack_f16(rnd(self.n_past)),
            2: pack_f16(rnd(self.n_past)),
            3: np.full(B, past_len + S - 1, dtype=np.uint32),
            4: pack_f16(np.cos(np.arange(max_pos * rh) * 0.017).astype(np.float16)),
            5: pack_f16(np.sin(np.arange(max_pos * rh) * 0.017).astype(np.float16)),
            6: np.zeros((self.n_out + 1) // 2, np.uint32),
            7: np.zeros((self.n_pres + 1) // 2, np.uint32),
            8: np.zeros((self.n_pres + 1) // 2, np.uint32),
        }
        if not growing:
            # Arena: `present` IS `past`. One allocation, seeded with the past tokens.
            self.src[7] = self.src[1].copy()
            self.src[8] = self.src[2].copy()
        self.push = [B, S, Nq, Nkv, D, self.rot, self.present_len, self.past_stride,
                     struct.unpack("<I", struct.pack("<f", float(1.0 / np.sqrt(D))))[0]]

    def dispatch(self) -> Dispatch:
        bufs = {k: v.copy() for k, v in self.src.items()}
        if not self.growing:
            # Same allocation on both sides of the aliasing argument, as the arena lane runs it.
            bufs[1] = bufs[7]
            bufs[2] = bufs[8]
        return Dispatch(groups=(self.B * self.Nq * self.S, 1, 1), local_size=(1, 1, 1),
                        spec={}, push_constants=self.push, buffers=bufs)

    def label(self) -> str:
        return (f"{self.name} B{self.B} S{self.S} Nq{self.Nq} Nkv{self.Nkv} D{self.D} "
                f"G{self.G} past{self.past_len} "
                f"{'growing' if self.growing else 'arena'}")


def numpy_reference(c: Case) -> np.ndarray:
    """Plain (not online) softmax GQA in fp32 over the same fp16 inputs. Tolerance control."""
    B, S, Nq, Nkv, D = c.B, c.S, c.Nq, c.Nkv, c.D
    rh = c.rot // 2
    qkv = unpack_f16(c.src[0], c.n_qkv).reshape(B, S, (Nq + 2 * Nkv) * D)
    pk = unpack_f16(c.src[1], c.n_past).reshape(B, Nkv, c.past_stride, D)
    pv = unpack_f16(c.src[2], c.n_past).reshape(B, Nkv, c.past_stride, D)
    cosc = unpack_f16(c.src[4], c.src[4].size * 2).reshape(-1, rh)
    sinc = unpack_f16(c.src[5], c.src[5].size * 2).reshape(-1, rh)
    out = np.zeros((B, S, Nq * D), np.float32)

    def rope(vec, pos):
        v = vec.copy()
        x, y = vec[:rh], vec[rh:2 * rh]
        cs, sn = cosc[pos], sinc[pos]
        v[:rh] = x * cs - y * sn
        v[rh:2 * rh] = y * cs + x * sn
        return v

    for b in range(B):
        for h in range(Nq):
            kv_h = h // c.G
            for s in range(S):
                pos = c.past_len + s
                q = rope(qkv[b, s, h * D:(h + 1) * D], pos)
                keys, vals = [], []
                for t in range(c.past_len):
                    keys.append(pk[b, kv_h, t])
                    vals.append(pv[b, kv_h, t])
                for t2 in range(s + 1):
                    ko = Nq * D + kv_h * D
                    vo = (Nq + Nkv) * D + kv_h * D
                    keys.append(rope(qkv[b, t2, ko:ko + D], c.past_len + t2))
                    vals.append(qkv[b, t2, vo:vo + D])
                sc = np.array([float(q @ k) for k in keys], np.float32) / np.sqrt(D)
                e = np.exp(sc - sc.max())
                w = e / e.sum()
                out[b, s, h * D:(h + 1) * D] = (w[:, None] * np.array(vals)).sum(0)
    return out


# -- one measurement ---------------------------------------------------------------------------


def measure(mod: SpirvModule, c: Case) -> dict:
    d = c.dispatch()
    _, wk = mod.run_traced(d, store_binding=PRESENT_K_BINDING)
    d2 = c.dispatch()
    _, wv = mod.run_traced(d2, store_binding=PRESENT_V_BINDING)
    # `d2`'s buffers are the ones to report: both runs are deterministic and identical.
    new_words = np.zeros(wk.words, bool)
    row = c.D // 2
    for b in range(c.B):
        for kv in range(c.Nkv):
            for s in range(c.S):
                base = ((b * c.Nkv + kv) * c.present_len + c.past_len + s) * row
                new_words[base:base + row] = True
    wi = wk.writer_invocations
    return {
        "buffers": d2.buffers,
        "present_key": {
            "store_instructions": wk.store_instructions,
            "named_bytes": wk.named_bytes,
            "touched_words": wk.touched_words,
            "max_writers_per_word": wk.max_writers_per_word,
            "writers_per_new_token_word_min": int(wi[new_words].min()) if new_words.any() else 0,
            "writers_per_new_token_word_max": int(wi[new_words].max()) if new_words.any() else 0,
            "new_token_words": int(new_words.sum()),
        },
        "present_value": {
            "store_instructions": wv.store_instructions,
            "named_bytes": wv.named_bytes,
            "touched_words": wv.touched_words,
            "max_writers_per_word": wv.max_writers_per_word,
        },
    }


CASES = [
    # Phi-3.5-mini: Nq == Nkv, G == 1. The grouping this project has always run.
    Case("phi35-like", 1, 1, 8, 8, 32, past_len=6, growing=False, seed=1),
    Case("phi35-like", 1, 4, 8, 8, 32, past_len=6, growing=True, seed=2),
    # Llama-3-8B / Mistral-7B / Qwen2-7B: Nq/Nkv == 4.
    Case("gqa4", 1, 1, 8, 2, 32, past_len=6, growing=False, seed=3),
    Case("gqa4", 1, 1, 8, 2, 32, past_len=6, growing=True, seed=7),
    Case("gqa4", 1, 4, 8, 2, 32, past_len=6, growing=True, seed=4),
    Case("gqa4", 2, 3, 8, 2, 64, past_len=5, growing=True, seed=5),
    # Llama-2-70B / Gemma-2-27B style: Nq/Nkv == 8.
    Case("gqa8", 1, 2, 16, 2, 32, past_len=4, growing=True, seed=6),
    # Llama-3-8B's real per-layer attention shape, decode, both conventions.
    Case("llama3-8b-decode", 1, 1, 32, 8, 128, past_len=8, growing=False, seed=8),
    Case("llama3-8b-decode", 1, 1, 32, 8, 128, past_len=8, growing=True, seed=9),
    # Phi-3.5-mini's real per-layer attention shape, decode. The control that must not move.
    Case("phi35-decode", 1, 1, 32, 32, 96, past_len=8, growing=False, seed=10),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ref", default="main",
                    help="git ref whose gqa_f16.comp is the baseline (default: main)")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "kv_write_redundancy.json"))
    args = ap.parse_args()

    base_text = baseline_source(args.base_ref)
    fixed_text = SHADER_SRC.read_text(encoding="utf-8")
    if base_text == fixed_text:
        print(f"NOTE: working tree matches {args.base_ref}; both arms are the same kernel.")

    base_spv = compile_glsl(base_text, f"baseline@{args.base_ref}")
    fixed_spv = compile_glsl(fixed_text, "fixed@worktree")
    base_mod = SpirvModule(base_spv)
    fixed_mod = SpirvModule(fixed_spv)

    rows = []
    correctness_failures = []
    ref_max_err = 0.0
    for c in CASES:
        b = measure(base_mod, c)
        f = measure(fixed_mod, c)

        # --- correctness, read before any byte count ---------------------------------
        exact = {}
        for binding, nm in ((ATTN_OUT_BINDING, "attn_output"),
                            (PRESENT_K_BINDING, "present_key"),
                            (PRESENT_V_BINDING, "present_value")):
            same = bool(np.array_equal(b["buffers"][binding], f["buffers"][binding]))
            exact[nm] = same
            if not same:
                diff = int(np.count_nonzero(b["buffers"][binding] != f["buffers"][binding]))
                correctness_failures.append(
                    f"{c.label()}: {nm} differs in {diff} of "
                    f"{b['buffers'][binding].size} words")
        ref = numpy_reference(c)
        got = unpack_f16(f["buffers"][ATTN_OUT_BINDING], c.n_out).reshape(ref.shape)
        err = float(np.max(np.abs(got - ref)) / max(float(np.max(np.abs(ref))), 1e-9))
        ref_max_err = max(ref_max_err, err)

        rows.append({
            "case": c.label(), "name": c.name, "B": c.B, "S": c.S, "Nq": c.Nq,
            "Nkv": c.Nkv, "D": c.D, "G": c.G, "past_len": c.past_len,
            "convention": "growing" if c.growing else "arena",
            "bit_exact_vs_baseline": exact,
            "numpy_reference_rel_err": err,
            "baseline": {k: v for k, v in b.items() if k != "buffers"},
            "fixed": {k: v for k, v in f.items() if k != "buffers"},
            "present_key_write_bytes_baseline": b["present_key"]["named_bytes"],
            "present_key_write_bytes_fixed": f["present_key"]["named_bytes"],
            "reduction": (b["present_key"]["named_bytes"]
                          / max(f["present_key"]["named_bytes"], 1)),
        })

    print("KV PRESENT-WRITE REDUNDANCY — executed over the compiled grid, no clock")
    print(f"baseline: git {args.base_ref}:{SHADER_REL}")
    print()
    if correctness_failures:
        print("CORRECTNESS CONTROL FAILED — no byte figure is published.")
        for m in correctness_failures:
            print("  ", m)
        return 2
    print(f"correctness: all {len(CASES)} arms bit-exact vs baseline on attn_output, "
          f"present_key, present_value")
    print(f"numpy GQA reference: max relative error {ref_max_err:.3e} (tolerance control)")
    print()
    hdr = f"{'case':<46} {'G':>2} {'writers/word':>13} {'writers/word':>13} {'K bytes':>10} {'K bytes':>10} {'x':>6}"
    print(hdr)
    print(f"{'':<46} {'':>2} {'base':>13} {'fixed':>13} {'base':>10} {'fixed':>10} {'':>6}")
    for r in rows:
        print(f"{r['case']:<46} {r['G']:>2} "
              f"{r['baseline']['present_key']['writers_per_new_token_word_max']:>13} "
              f"{r['fixed']['present_key']['writers_per_new_token_word_max']:>13} "
              f"{r['present_key_write_bytes_baseline']:>10} "
              f"{r['present_key_write_bytes_fixed']:>10} "
              f"{r['reduction']:>6.2f}")
    print()

    # The detector must be seen in BOTH states or it is not a detector.
    pos = [r for r in rows if r["G"] > 1]
    neg = [r for r in rows if r["G"] == 1]
    pos_fired = all(
        r["baseline"]["present_key"]["writers_per_new_token_word_max"] == r["G"] for r in pos)
    pos_fixed = all(
        r["fixed"]["present_key"]["writers_per_new_token_word_max"] == 1 for r in pos)
    neg_flat = all(r["reduction"] == 1.0 for r in neg)
    print(f"positive control (G>1, baseline writers/word == G):        "
          f"{'PASS' if pos_fired else 'FAIL'}  ({len(pos)} arms)")
    print(f"repaired state   (G>1, fixed    writers/word == 1):        "
          f"{'PASS' if pos_fixed else 'FAIL'}  ({len(pos)} arms)")
    print(f"negative control (G==1, nothing changes at all):           "
          f"{'PASS' if neg_flat else 'FAIL'}  ({len(neg)} arms)")

    record = {
        "probe": "kv_write_redundancy",
        "base_ref": args.base_ref,
        "shader": SHADER_REL,
        "PROVENANCE": {
            "writers_per_new_token_word": "MEASUREMENT",
            "store_instructions": "MEASUREMENT",
            "named_bytes": "MEASUREMENT",
            "reduction": "MEASUREMENT",
            "bit_exact_vs_baseline": "MEASUREMENT",
            "numpy_reference_rel_err": "MEASUREMENT",
        },
        "controls": {
            "positive_G_gt_1_baseline_writers_equals_G": pos_fired,
            "repaired_G_gt_1_fixed_writers_equals_1": pos_fixed,
            "negative_G_eq_1_unchanged": neg_flat,
            "all_arms_bit_exact": True,
            "numpy_reference_max_rel_err": ref_max_err,
        },
        "arms": rows,
    }
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nrecord: {args.out}")
    ok = pos_fired and pos_fixed and neg_flat and ref_max_err < 5e-3
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InstrumentError as e:
        print(f"ERROR(instrument): {e}")
        sys.exit(3)
