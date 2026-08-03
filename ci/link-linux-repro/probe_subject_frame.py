"""Scratch falsifier: is the Linux 103/103 SUBJECT-CHANGED a *source* change or a *toolchain* change?

`--check` classifies on spirv_digest first and unconditionally: bytes differ -> SUBJECT-CHANGED,
which asserts "the ledger describes a kernel this binary does not contain". If `source_digest`
agrees for the same entries, that assertion is false and the correct reading is
PROVEN-ELSEWHERE{toolchain} -- the state Mouse added for exactly this and which this ordering
makes unreachable.

Prints, per entry: spirv agree?, source agree?, recorded vs build toolchain.
"""
import collections
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "rust" / "tools"))

import gen_proof_ledger as G  # noqa: E402

lib = pathlib.Path(os.environ["ONNXRUNTIME_VULKAN_EP_LIB"])
led = REPO / "evidence" / "proof_ledger.jsonl"
lines = [l for l in (x.strip() for x in led.read_text(encoding="utf-8").splitlines())
         if l and not l.startswith("#")]
entries = [json.loads(l) for l in lines[1:]]

tally = collections.Counter()
build_tc = set()
rec_tc = set()
examples = {}
for e in entries:
    stems = sorted(e.get("shaders") or [])
    subj = G._shader_subject(lib, stems)
    build_tc.add(subj.get("toolchain", ""))
    rec_tc.add(e.get("toolchain", ""))
    spirv_same = subj.get("spirv_digest") == (e.get("shader_digest") or "")
    src_rec = e.get("source_digest") or ""
    src_cur = subj.get("source_digest")
    if not src_rec or src_cur is None:
        src_state = "UNRECORDED"
    else:
        src_state = "same" if src_cur == src_rec else "MOVED"
    tally[(spirv_same, src_state)] += 1
    examples.setdefault((spirv_same, src_state), e.get("key", ""))

print(f"lib            : {lib}")
print(f"entries        : {len(entries)}")
print(f"build toolchain: {sorted(build_tc)}")
print(f"ledger toolchn : {sorted(rec_tc)}")
print("")
print("spirv_same  source_digest   count  example")
for (spirv_same, src_state), n in sorted(tally.items(), key=lambda kv: -kv[1]):
    print(f"  {str(spirv_same):<9} {src_state:<14} {n:>5}  {examples[(spirv_same, src_state)][:70]}")
print("")
same_src_diff_spirv = sum(n for (s, st), n in tally.items() if not s and st == "same")
if same_src_diff_spirv:
    print(f"VERDICT: {same_src_diff_spirv} entr(ies) have IDENTICAL source_digest and DIFFERENT "
          f"spirv_digest.")
    print("         --check calls every one of them SUBJECT-CHANGED ('a kernel this binary does "
          "not contain').")
    print("         The source did not move. This is a toolchain frame difference and "
          "PROVEN-ELSEWHERE{toolchain} is the state for it.")
else:
    print("VERDICT: no entry has same-source/different-SPIR-V; SUBJECT-CHANGED is not "
          "misclassifying here.")
