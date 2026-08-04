#!/usr/bin/env python3
"""Execute a compiled SPIR-V compute module and record every address it loads.

WHY THIS EXISTS
===============
`bench/results/probe_island_bytes.py` claimed an amplification of exactly 1.000000 for the
weight stream, and every number in that claim was a **literal**.  The reasoning behind it --
a def-use walk establishing one load per blob, and a workgroup partition establishing one
workgroup per blob -- happened once, in a head, and what landed in the tree was its
conclusion.  A kernel change that re-read every blob eight times would not have moved it.

This module is the walk, executable.  It parses a compiled SPIR-V module (the *binary*, the
one the pipeline was created from, located by the digest the proof ledger already records),
substitutes the specialization constants the pipeline was created with, and **runs it** over
the whole dispatch grid, recording the byte range of every storage-buffer load.

WHAT IT IS AND IS NOT
=====================
* It is an interpreter, not a pattern match.  It has no idea what a "blob" is.  It reports
  the multiset of words the module loaded from a binding and lets the caller divide.
* It is **not** a model of the memory system.  It counts what *load instructions name*, which
  is the same distinction `probe_roofline.py` draws between the weight stream (where named
  bytes are DRAM traffic) and the activation row (where they are cache hits).  For the weight
  stream those coincide; the caller is responsible for saying so.
* Execution is SIMT-in-lockstep over numpy: one array lane per invocation, a boolean mask per
  basic block, and the structured control flow SPIR-V guarantees.  Barriers are no-ops
  because regions execute to completion before their merge, which is stronger than a barrier.

CORRECTNESS OF THE INTERPRETER ITSELF
=====================================
An address trace from a broken interpreter is worth nothing, so the interpreter is checked by
**running a real quantised GEMV and comparing the output it computes against a numpy
reference** (`bench/test_weight_reread.py::test_interpreter_reproduces_the_gemv`).  If the
arithmetic is right the control flow is right, and the control flow is what the address trace
is made of.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

__all__ = ["InstrumentError", "SpirvModule", "LoadTrace", "StoreTrace", "Dispatch"]

MAX_LOOP_ITERATIONS = 1 << 22


class InstrumentError(RuntimeError):
    """This instrument failed. Never a finding about the shader."""


# -- SPIR-V binary parsing ---------------------------------------------------------------------

SPIRV_MAGIC = 0x07230203

#: The subset this interpreter understands. An opcode outside it raises rather than being
#: skipped: a silently ignored instruction is how an address trace becomes plausible and wrong.
_OPNAMES = {
    0: "OpNop", 1: "OpUndef", 5: "OpName", 6: "OpMemberName", 7: "OpString",
    8: "OpLine", 317: "OpNoLine",
    11: "OpExtInstImport", 12: "OpExtInst",
    14: "OpMemoryModel", 15: "OpEntryPoint", 16: "OpExecutionMode", 17: "OpCapability",
    19: "OpTypeVoid", 20: "OpTypeBool", 21: "OpTypeInt", 22: "OpTypeFloat",
    23: "OpTypeVector", 24: "OpTypeMatrix", 28: "OpTypeArray", 29: "OpTypeRuntimeArray",
    30: "OpTypeStruct", 32: "OpTypePointer", 33: "OpTypeFunction",
    41: "OpConstantTrue", 42: "OpConstantFalse", 43: "OpConstant",
    44: "OpConstantComposite", 46: "OpConstantNull",
    48: "OpSpecConstantTrue", 49: "OpSpecConstantFalse", 50: "OpSpecConstant",
    51: "OpSpecConstantComposite", 52: "OpSpecConstantOp",
    54: "OpFunction", 55: "OpFunctionParameter", 56: "OpFunctionEnd", 57: "OpFunctionCall",
    59: "OpVariable", 61: "OpLoad", 62: "OpStore", 63: "OpCopyMemory",
    65: "OpAccessChain", 66: "OpInBoundsAccessChain",
    71: "OpDecorate", 72: "OpMemberDecorate", 73: "OpDecorationGroup",
    77: "OpVectorExtractDynamic", 78: "OpVectorInsertDynamic", 79: "OpVectorShuffle",
    80: "OpCompositeConstruct", 81: "OpCompositeExtract", 82: "OpCompositeInsert",
    83: "OpCopyObject", 84: "OpTranspose",
    109: "OpConvertFToU", 110: "OpConvertFToS", 111: "OpConvertSToF", 112: "OpConvertUToF",
    113: "OpUConvert", 114: "OpSConvert", 115: "OpFConvert", 124: "OpBitcast",
    126: "OpSNegate", 127: "OpFNegate",
    128: "OpIAdd", 129: "OpFAdd", 130: "OpISub", 131: "OpFSub",
    132: "OpIMul", 133: "OpFMul", 134: "OpUDiv", 135: "OpSDiv", 136: "OpFDiv",
    137: "OpUMod", 138: "OpSRem", 139: "OpSMod", 140: "OpFRem", 141: "OpFMod",
    148: "OpDot",
    154: "OpAny", 155: "OpAll", 156: "OpIsNan", 157: "OpIsInf",
    164: "OpLogicalEqual", 165: "OpLogicalNotEqual",
    166: "OpLogicalOr", 167: "OpLogicalAnd", 168: "OpLogicalNot",
    169: "OpSelect", 170: "OpIEqual", 171: "OpINotEqual",
    172: "OpUGreaterThan", 173: "OpSGreaterThan",
    174: "OpUGreaterThanEqual", 175: "OpSGreaterThanEqual",
    176: "OpULessThan", 177: "OpSLessThan",
    178: "OpULessThanEqual", 179: "OpSLessThanEqual",
    180: "OpFOrdEqual", 181: "OpFUnordEqual", 182: "OpFOrdNotEqual", 183: "OpFUnordNotEqual",
    184: "OpFOrdLessThan", 185: "OpFUnordLessThan",
    186: "OpFOrdGreaterThan", 187: "OpFUnordGreaterThan",
    188: "OpFOrdLessThanEqual", 189: "OpFUnordLessThanEqual",
    190: "OpFOrdGreaterThanEqual", 191: "OpFUnordGreaterThanEqual",
    194: "OpShiftRightLogical", 195: "OpShiftRightArithmetic", 196: "OpShiftLeftLogical",
    197: "OpBitwiseOr", 198: "OpBitwiseXor", 199: "OpBitwiseAnd", 200: "OpNot",
    227: "OpAtomicLoad", 228: "OpAtomicStore",
    234: "OpAtomicIAdd", 235: "OpAtomicISub",
    240: "OpAtomicAnd", 241: "OpAtomicOr", 242: "OpAtomicXor",
    245: "OpPhi", 246: "OpLoopMerge", 247: "OpSelectionMerge", 248: "OpLabel",
    249: "OpBranch", 250: "OpBranchConditional", 251: "OpSwitch",
    252: "OpKill", 253: "OpReturn", 254: "OpReturnValue", 255: "OpUnreachable",
    224: "OpControlBarrier", 225: "OpMemoryBarrier",
    331: "OpModuleProcessed",
}

#: GLSL.std.450 instruction numbers, from the extended-instruction-set specification.
#:
#: The four entries that used to live here -- `30: "Fma", 43: "FMin", 37: "FMax", 40: "FClamp"`
#: -- were **wrong**. The numbers were checked by disassembling every `.spv` in the tree and
#: matching the opcode histogram against the ops the sources actually call (40 appears in
#: `ew_unary_relu`/`mish`/`softplus`, 43 in `hardsigmoid`/`hardswish`, 37 in `celu`, 45 in
#: `gather`'s index clamp, 42 in `gqa_f16`'s `max(int, 0)`).
#:
#: HOW BADLY WRONG, MEASURED RATHER THAN ASSERTED. The first account of this said every one of
#: them would have returned a plausible float. That is false; the discriminator is the operand
#: count. A wrong name taking MORE operands than the real one indexes past the end of the operand
#: list and raises -- 40 (`FMax`) read as `FClamp` reads `args[2]`, and `max(x, 0.0)` supplies
#: two, so a `relu` would have raised `IndexError` on its first invocation. A wrong name taking
#: FEWER silently drops the extra: 37 (`FMin`) read as `FMax` returns a maximum, and 43
#: (`FClamp`) read as `FMin` drops the upper bound entirely. So the SILENT set is `{37, 43}` and
#: the silently-miscomputed kernels are exactly `celu`, `hardsigmoid` and `hardswish`.
#: `bench/results/probe_glsl450_blast_radius.py` executes all four under both tables and prints
#: the split; `bench/test_kv_write_redundancy.py` gates the numbering against glslc's own output.
#:
#: Nothing caught it because the interpreter's only correctness control was a quantised GEMV,
#: which issues none of them. That control now exists.
_GLSL450 = {
    1: "Round", 2: "RoundEven", 3: "Trunc", 4: "FAbs", 5: "SAbs", 6: "FSign", 7: "SSign",
    8: "Floor", 9: "Ceil", 10: "Fract",
    13: "Sin", 14: "Cos", 15: "Tan", 16: "Asin", 17: "Acos", 18: "Atan",
    19: "Sinh", 20: "Cosh", 21: "Tanh", 22: "Asinh", 23: "Acosh", 24: "Atanh", 25: "Atan2",
    26: "Pow", 27: "Exp", 28: "Log", 29: "Exp2", 30: "Log2",
    31: "Sqrt", 32: "InverseSqrt",
    37: "FMin", 38: "UMin", 39: "SMin", 40: "FMax", 41: "UMax", 42: "SMax",
    43: "FClamp", 44: "UClamp", 45: "SClamp", 46: "FMix",
    50: "Fma",
    58: "PackHalf2x16", 62: "UnpackHalf2x16",
}


@dataclass
class Instr:
    op: str
    result: int | None
    result_type: int | None
    operands: list[int]
    #: Literal words, for instructions that carry them (OpConstant, OpDecorate, OpSwitch).
    words: list[int] = field(default_factory=list)


def _parse(binary: bytes) -> list[Instr]:
    if len(binary) < 20:
        raise InstrumentError("SPIR-V module shorter than a header")
    magic = struct.unpack("<I", binary[:4])[0]
    if magic != SPIRV_MAGIC:
        raise InstrumentError(f"not a SPIR-V module: magic {magic:#x}")
    words = np.frombuffer(binary, dtype="<u4")
    out: list[Instr] = []
    i = 5
    n = len(words)
    while i < n:
        w = int(words[i])
        count, opcode = w >> 16, w & 0xFFFF
        if count == 0:
            raise InstrumentError(f"zero-length instruction at word {i}")
        body = [int(x) for x in words[i + 1: i + count]]
        name = _OPNAMES.get(opcode)
        out.append(Instr(name or f"Op#{opcode}", None, None, body, body))
        i += count
    return out


#: Instructions whose first operand is a result type and second a result id.
_TYPED_RESULT = {
    "OpExtInst", "OpUndef", "OpVariable", "OpLoad", "OpAccessChain", "OpInBoundsAccessChain",
    "OpVectorExtractDynamic", "OpVectorInsertDynamic", "OpVectorShuffle",
    "OpCompositeConstruct", "OpCompositeExtract", "OpCompositeInsert", "OpCopyObject",
    "OpConvertFToU", "OpConvertFToS", "OpConvertSToF", "OpConvertUToF", "OpUConvert",
    "OpSConvert", "OpFConvert", "OpBitcast", "OpSNegate", "OpFNegate",
    "OpIAdd", "OpFAdd", "OpISub", "OpFSub", "OpIMul", "OpFMul", "OpUDiv", "OpSDiv",
    "OpFDiv", "OpUMod", "OpSRem", "OpSMod", "OpFRem", "OpFMod", "OpDot",
    "OpAny", "OpAll", "OpIsNan", "OpIsInf",
    "OpLogicalEqual", "OpLogicalNotEqual", "OpLogicalOr", "OpLogicalAnd", "OpLogicalNot",
    "OpSelect", "OpIEqual", "OpINotEqual", "OpUGreaterThan", "OpSGreaterThan",
    "OpUGreaterThanEqual", "OpSGreaterThanEqual", "OpULessThan", "OpSLessThan",
    "OpULessThanEqual", "OpSLessThanEqual",
    "OpFOrdEqual", "OpFUnordEqual", "OpFOrdNotEqual", "OpFUnordNotEqual",
    "OpFOrdLessThan", "OpFUnordLessThan", "OpFOrdGreaterThan", "OpFUnordGreaterThan",
    "OpFOrdLessThanEqual", "OpFUnordLessThanEqual",
    "OpFOrdGreaterThanEqual", "OpFUnordGreaterThanEqual",
    "OpShiftRightLogical", "OpShiftRightArithmetic", "OpShiftLeftLogical",
    "OpBitwiseOr", "OpBitwiseXor", "OpBitwiseAnd", "OpNot",
    "OpAtomicLoad", "OpAtomicIAdd", "OpAtomicISub", "OpAtomicAnd", "OpAtomicOr",
    "OpAtomicXor", "OpPhi",
    "OpConstant", "OpConstantTrue", "OpConstantFalse", "OpConstantComposite",
    "OpConstantNull", "OpSpecConstant", "OpSpecConstantTrue", "OpSpecConstantFalse",
    "OpSpecConstantComposite", "OpSpecConstantOp", "OpFunction", "OpFunctionCall",
    "OpFunctionParameter",
}


@dataclass
class LoadTrace:
    """What one binding's loads named, in words, over one dispatch.

    `word_reads[i]` is the number of load instructions that named 32-bit word `i` of the
    binding. Everything the caller wants -- amplification, coverage, per-blob multiplicity --
    is a reduction of this array, which is why the array is the product rather than a summary.
    """

    binding: int
    words: int
    word_reads: np.ndarray
    #: Distinct workgroup ids that named each word, capped: stores min and max seen.
    word_min_wg: np.ndarray
    word_max_wg: np.ndarray
    #: `result-type id -> (load instruction count, width in 32-bit words)` per load *site*.
    sites: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def load_instructions(self) -> int:
        return sum(c for c, _ in self.sites.values())

    @property
    def named_bytes(self) -> int:
        return int(sum(c * w for c, w in self.sites.values()) * 4)

    @property
    def load_widths_bytes(self) -> list[int]:
        """The widths actually executed, from each load's SPIR-V result type.

        Read rather than assumed: the packed and general paths of `q_gemv` differ only in this
        and in the instruction count -- their byte totals are identical -- so an
        amplification-only reading cannot tell them apart.
        """
        return sorted({w * 4 for c, w in self.sites.values() if c})

    @property
    def touched_words(self) -> int:
        return int(np.count_nonzero(self.word_reads))

    @property
    def max_reads_per_word(self) -> int:
        return int(self.word_reads.max()) if self.word_reads.size else 0

    @property
    def words_read_by_more_than_one_workgroup(self) -> int:
        touched = self.word_reads > 0
        return int(np.count_nonzero(self.word_min_wg[touched] != self.word_max_wg[touched]))


@dataclass
class StoreTrace:
    """What one binding's *writes* named, in words, over one dispatch.

    The mirror of `LoadTrace`, and it exists for the same reason: a claim that a kernel writes
    each destination word once is a claim about the executed grid, not about the source text.
    Every instruction that names a word for writing is counted -- `OpStore` and the read-modify-
    write atomics alike -- because the redundancy this measures is redundancy of *memory
    traffic*, and an `atomicAnd`/`atomicOr` pair moves the cache line exactly as a store does.

    `writer_invocations[i]` counts *invocations* that wrote word `i` at least once, which is the
    quantity a "G invocations write the same word" claim is about; `word_writes[i]` counts
    instructions, which is what the memory system sees. They differ by the fixed per-write
    instruction multiplicity of the kernel (2 for this project's masked half-word writes), so
    both are published and neither is derived from the other.
    """

    binding: int
    words: int
    word_writes: np.ndarray
    writer_invocations: np.ndarray
    word_min_wg: np.ndarray
    word_max_wg: np.ndarray
    sites: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def store_instructions(self) -> int:
        return sum(c for c, _ in self.sites.values())

    @property
    def named_bytes(self) -> int:
        return int(sum(c * w for c, w in self.sites.values()) * 4)

    @property
    def touched_words(self) -> int:
        return int(np.count_nonzero(self.word_writes))

    @property
    def max_writers_per_word(self) -> int:
        return int(self.writer_invocations.max()) if self.writer_invocations.size else 0

    @property
    def words_written_by_more_than_one_workgroup(self) -> int:
        touched = self.word_writes > 0
        return int(np.count_nonzero(self.word_min_wg[touched] != self.word_max_wg[touched]))

    def write_amplification(self) -> float:
        """Instructions naming a word, per word actually reached. 1.0 only if nothing repeats."""
        t = self.touched_words
        return float(self.word_writes.sum() / t) if t else 0.0


@dataclass
class Dispatch:
    """Everything `vkCmdDispatch` and `vkCreateComputePipelines` fixed for one node."""

    groups: tuple[int, int, int]
    local_size: tuple[int, int, int]
    spec: dict[int, int]
    push_constants: list[int]
    #: `binding -> flat numpy array of 32-bit words` (uint32/float32 views of the same memory).
    buffers: dict[int, np.ndarray]


class SpirvModule:
    """A parsed SPIR-V compute module that can be executed over a dispatch grid."""

    def __init__(self, binary: bytes):
        self.binary = binary
        self.instrs = _parse(binary)
        self.types: dict[int, Instr] = {}
        self.consts: dict[int, Instr] = {}
        self.names: dict[int, str] = {}
        self.decor: dict[int, dict[str, int]] = {}
        self.member_offsets: dict[int, dict[int, int]] = {}
        self.variables: dict[int, Instr] = {}
        self.ext_imports: dict[int, str] = {}
        self.entry_local_size = (1, 1, 1)
        self.body: list[Instr] = []
        self._scan()

    # -- module scan ---------------------------------------------------------------------
    def _scan(self) -> None:
        in_function = False
        for ins in self.instrs:
            ops = ins.operands
            op = ins.op
            if op.startswith("Op#"):
                if in_function:
                    raise InstrumentError(f"unknown opcode in function body: {op}")
                continue
            if op == "OpExtInstImport":
                self.ext_imports[ops[0]] = _decode_string(ops[1:])
            elif op == "OpName":
                self.names[ops[0]] = _decode_string(ops[1:])
            elif op == "OpEntryPoint":
                pass
            elif op == "OpExecutionMode":
                if len(ops) >= 5 and ops[1] == 17:  # LocalSize
                    self.entry_local_size = (ops[2], ops[3], ops[4])
            elif op == "OpDecorate":
                d = self.decor.setdefault(ops[0], {})
                kind = ops[1]
                val = ops[2] if len(ops) > 2 else 1
                d[_DECOR.get(kind, str(kind))] = val
            elif op == "OpMemberDecorate":
                if len(ops) >= 4 and ops[2] == 35:  # Offset
                    self.member_offsets.setdefault(ops[0], {})[ops[1]] = ops[3]
            elif op.startswith("OpType"):
                self.types[ops[0]] = Instr(op, ops[0], None, ops[1:], ops[1:])
            elif op in ("OpConstant", "OpConstantTrue", "OpConstantFalse",
                        "OpConstantComposite", "OpConstantNull", "OpSpecConstant",
                        "OpSpecConstantTrue", "OpSpecConstantFalse",
                        "OpSpecConstantComposite", "OpSpecConstantOp", "OpUndef"):
                self.consts[ops[1]] = Instr(op, ops[1], ops[0], ops[2:], ops[2:])
            elif op == "OpVariable":
                if in_function:
                    self.body.append(_typed(ins))
                else:
                    self.variables[ops[1]] = Instr(op, ops[1], ops[0], ops[2:], ops[2:])
            elif op == "OpFunction":
                in_function = True
            elif op == "OpFunctionEnd":
                in_function = False
            elif in_function:
                self.body.append(_typed(ins))
        if not self.body:
            raise InstrumentError("module has no function body to execute")
        self._blocks()

    def _blocks(self) -> None:
        self.blocks: dict[int, dict] = {}
        self.block_order: list[int] = []
        cur = None
        for ins in self.body:
            if ins.op == "OpLabel":
                cur = ins.result
                self.blocks[cur] = {"body": [], "term": None, "loop": None, "sel": None,
                                    "phis": []}
                self.block_order.append(cur)
                continue
            if cur is None:
                if ins.op == "OpVariable":
                    continue
                raise InstrumentError(f"instruction {ins.op} outside any basic block")
            b = self.blocks[cur]
            if ins.op == "OpPhi":
                b["phis"].append(ins)
            elif ins.op == "OpLoopMerge":
                b["loop"] = (ins.operands[0], ins.operands[1])
            elif ins.op == "OpSelectionMerge":
                b["sel"] = ins.operands[0]
            elif ins.op in ("OpBranch", "OpBranchConditional", "OpSwitch", "OpReturn",
                            "OpReturnValue", "OpKill", "OpUnreachable"):
                b["term"] = ins
            else:
                b["body"].append(ins)
        # Function-scope OpVariable instructions live in the entry block but are allocated once.
        self.func_vars = [i for i in self.body if i.op == "OpVariable"]

    # -- type helpers --------------------------------------------------------------------
    def scalar_count(self, tid: int) -> int:
        t = self.types[tid]
        if t.op in ("OpTypeInt", "OpTypeFloat", "OpTypeBool"):
            return 1
        if t.op == "OpTypeVector":
            return self.scalar_count(t.operands[0]) * t.operands[1]
        if t.op == "OpTypeArray":
            return self.scalar_count(t.operands[0]) * self.const_int(t.operands[1])
        if t.op == "OpTypeRuntimeArray":
            return self.scalar_count(t.operands[0])
        if t.op == "OpTypeStruct":
            return sum(self.scalar_count(m) for m in t.operands)
        raise InstrumentError(f"scalar_count of {t.op}")

    def leaf_scalar(self, tid: int) -> Instr:
        t = self.types[tid]
        while t.op in ("OpTypeVector", "OpTypeArray", "OpTypeRuntimeArray", "OpTypeStruct",
                       "OpTypePointer"):
            nxt = t.operands[1] if t.op == "OpTypePointer" else t.operands[0]
            t = self.types[nxt]
        return t

    def const_int(self, cid: int) -> int:
        return int(self.const_scalar(cid))

    def const_scalar(self, cid: int):
        c = self.consts[cid]
        if c.op in ("OpConstantTrue", "OpSpecConstantTrue"):
            return True
        if c.op in ("OpConstantFalse", "OpSpecConstantFalse"):
            return False
        if c.op in ("OpConstant", "OpSpecConstant"):
            t = self.types[c.result_type]
            raw = c.operands[0]
            if t.op == "OpTypeFloat":
                return float(np.frombuffer(struct.pack("<I", raw), dtype="<f4")[0])
            if t.op == "OpTypeInt" and t.operands[1] == 1:
                return int(np.int32(np.uint32(raw)))
            return int(raw)
        if c.op == "OpConstantNull":
            return 0
        raise InstrumentError(f"const_scalar of {c.op}")

    # -- execution -----------------------------------------------------------------------
    def run(self, d: Dispatch, trace_binding: int | None = None,
            wg_batch: int | None = None) -> LoadTrace | None:
        """Execute the whole grid. Returns the load trace for `trace_binding`, if asked."""
        load, _ = self.run_traced(d, load_binding=trace_binding, wg_batch=wg_batch)
        return load

    def run_traced(self, d: Dispatch, load_binding: int | None = None,
                   store_binding: int | None = None,
                   wg_batch: int | None = None) -> tuple[LoadTrace | None, StoreTrace | None]:
        """Execute the whole grid, tracing reads of one binding and writes of another."""
        trace = None
        if load_binding is not None:
            var = self._binding_var(load_binding)
            words = d.buffers[load_binding].size
            trace = LoadTrace(
                binding=load_binding, words=words,
                word_reads=np.zeros(words, dtype=np.uint32),
                word_min_wg=np.full(words, np.iinfo(np.uint32).max, dtype=np.uint32),
                word_max_wg=np.zeros(words, dtype=np.uint32),
            )
            self._trace_var = var
        else:
            self._trace_var = None
        self._trace = trace

        wtrace = None
        if store_binding is not None:
            wvar = self._binding_var(store_binding)
            wwords = d.buffers[store_binding].size
            wtrace = StoreTrace(
                binding=store_binding, words=wwords,
                word_writes=np.zeros(wwords, dtype=np.uint32),
                writer_invocations=np.zeros(wwords, dtype=np.uint32),
                word_min_wg=np.full(wwords, np.iinfo(np.uint32).max, dtype=np.uint32),
                word_max_wg=np.zeros(wwords, dtype=np.uint32),
            )
            self._wtrace_var = wvar
        else:
            self._wtrace_var = None
        self._wtrace = wtrace
        # `writer_invocations` counts invocations, not instructions, so a word an invocation
        # writes twice (the And then the Or of one masked half-word write) must count once.
        # The set of (word, global invocation index) pairs already seen is kept per batch;
        # invocation indices are batch-local but a workgroup never spans two batches, so no
        # invocation is split across them.
        self._wseen: set | None = set() if store_binding is not None else None

        gx, gy, gz = d.groups
        total_groups = gx * gy * gz
        lsz = d.local_size[0] * d.local_size[1] * d.local_size[2]
        batch = wg_batch or max(1, min(total_groups, max(1, (1 << 21) // max(lsz, 1))))
        for start in range(0, total_groups, batch):
            stop = min(total_groups, start + batch)
            if self._wseen is not None:
                self._wseen = set()
            self._run_batch(d, start, stop)
        return trace, wtrace

    def _binding_var(self, binding: int) -> int:
        for vid, dec in self.decor.items():
            if dec.get("Binding") == binding and vid in self.variables:
                return vid
        raise InstrumentError(f"no storage buffer decorated Binding {binding} in this module")

    def _run_batch(self, d: Dispatch, g0: int, g1: int) -> None:
        gx, gy, _gz = d.groups
        lx, ly, lz = d.local_size
        lsz = lx * ly * lz
        ngroups = g1 - g0
        T = ngroups * lsz
        self.T = T
        self.ngroups = ngroups

        gid = np.repeat(np.arange(g0, g1, dtype=np.uint32), lsz)
        self.wg_index = np.repeat(np.arange(ngroups, dtype=np.uint32), lsz)
        wg_x = gid % gx
        wg_y = (gid // gx) % gy
        wg_z = gid // (gx * gy)
        lid = np.tile(np.arange(lsz, dtype=np.uint32), ngroups)
        l_x = lid % lx
        l_y = (lid // lx) % ly
        l_z = lid // (lx * ly)

        self.vals: dict[int, np.ndarray] = {}
        self.phi_pending: dict[int, dict[int, np.ndarray]] = {}
        self.spec = dict(d.spec)
        self._const_cache: dict[int, np.ndarray] = {}
        self.storage: dict[int, dict] = {}
        self.builtins: dict[int, np.ndarray] = {}

        for vid, var in self.variables.items():
            sc = var.operands[0]
            ptr_t = self.types[var.result_type]
            pointee = ptr_t.operands[1]
            dec = self.decor.get(vid, {})
            if "BuiltIn" in dec:
                b = dec["BuiltIn"]
                if b == 26:  # WorkgroupId
                    arr = np.stack([wg_x, wg_y, wg_z], axis=1)
                elif b == 27:  # LocalInvocationId
                    arr = np.stack([l_x, l_y, l_z], axis=1)
                elif b == 24:  # NumWorkgroups
                    arr = np.tile(np.array(d.groups, dtype=np.uint32), (T, 1))
                elif b == 25:  # WorkgroupSize (normally a constant, not a variable)
                    arr = np.tile(np.array(d.local_size, dtype=np.uint32), (T, 1))
                elif b == 28:  # GlobalInvocationId
                    arr = np.stack([wg_x * lx + l_x, wg_y * ly + l_y, wg_z * lz + l_z],
                                   axis=1)
                else:
                    raise InstrumentError(f"unsupported BuiltIn {b}")
                # Registered both as a value (a whole-variable `OpLoad`) and as storage (an
                # `OpAccessChain` to one component); glslang emits both spellings.
                self.builtins[vid] = arr
                self.storage[vid] = {"kind": "thread", "data": arr, "type": pointee}
                continue
            if sc == 12:  # StorageBuffer
                binding = dec.get("Binding")
                if binding is None or binding not in d.buffers:
                    raise InstrumentError(f"no buffer supplied for binding {binding}")
                self.storage[vid] = {"kind": "global", "data": d.buffers[binding],
                                     "type": pointee}
            elif sc == 9:  # PushConstant
                arr = np.array(d.push_constants, dtype=np.uint32)
                self.storage[vid] = {"kind": "global", "data": arr, "type": pointee}
            elif sc == 4:  # Workgroup
                n = self.scalar_count(pointee)
                self.storage[vid] = {
                    "kind": "wg", "type": pointee,
                    "data": np.zeros((ngroups, n), dtype=self._np_dtype(pointee))}
            elif sc in (6, 7):  # Private / Function
                n = self.scalar_count(pointee)
                self.storage[vid] = {
                    "kind": "thread", "type": pointee,
                    "data": np.zeros((T, n), dtype=self._np_dtype(pointee))}
            else:
                raise InstrumentError(f"unsupported storage class {sc}")

        for var in self.func_vars:
            pointee = self.types[var.result_type].operands[1]
            n = self.scalar_count(pointee)
            self.storage[var.result] = {
                "kind": "thread", "type": pointee,
                "data": np.zeros((T, n), dtype=self._np_dtype(pointee))}

        entry = self.block_order[0]
        mask = np.ones(T, dtype=bool)
        self._run_region(entry, frozenset(), mask)

    def _np_dtype(self, tid: int):
        t = self.leaf_scalar(tid)
        if t.op == "OpTypeFloat":
            return np.float32
        if t.op == "OpTypeBool":
            return np.uint32
        if t.op == "OpTypeInt":
            return np.int32 if t.operands[1] == 1 else np.uint32
        raise InstrumentError(f"no numpy dtype for {t.op}")

    # -- structured control flow ---------------------------------------------------------
    def _goto(self, label: int, mask: np.ndarray, src: int, out: dict) -> None:
        for phi in self.blocks[label]["phis"]:
            ops = phi.operands
            val = None
            for i in range(0, len(ops), 2):
                if ops[i + 1] == src:
                    val = self.value(ops[i])
                    break
            if val is None:
                continue
            buf = self.phi_pending.setdefault(label, {})
            prev = buf.get(phi.result)
            if prev is None:
                prev = np.zeros_like(val)
            m = mask if val.ndim == 1 else mask[:, None]
            buf[phi.result] = np.where(m, val, prev)
        prev = out.get(label)
        out[label] = mask if prev is None else (prev | mask)

    def _run_region(self, entry: int, stops: frozenset, mask: np.ndarray) -> dict:
        """Execute from `entry` until every lane has reached a label in `stops` or returned."""
        out: dict[int, np.ndarray] = {}
        blk = entry
        skip_loop = False
        while True:
            if blk in stops:
                raise InstrumentError("internal: entered a stop label")
            b = self.blocks[blk]
            if b["loop"] is not None and not skip_loop:
                merge, cont = b["loop"]
                mask = self._run_loop(blk, merge, cont, mask, out, stops)
                if merge in stops:
                    self._merge_out(out, merge, mask)
                    return out
                if not mask.any():
                    return out
                blk = merge
                skip_loop = False
                continue
            skip_loop = False
            self._exec_block(blk, mask)
            t = b["term"]
            if t is None:
                raise InstrumentError(f"block {blk} has no terminator")
            if t.op in ("OpReturn", "OpReturnValue", "OpKill", "OpUnreachable"):
                return out
            targets = self._targets(t, mask)
            if len(targets) == 1:
                tgt, m = targets[0]
                if tgt in stops:
                    self._goto(tgt, m, blk, out)
                    return out
                self._goto(tgt, m, blk, {})
                mask = m
                blk = tgt
                continue
            sel = b["sel"]
            if sel is not None:
                inner_stops = stops | {sel}
                acc: dict[int, np.ndarray] = {}
                for tgt, m in targets:
                    if not m.any():
                        continue
                    if tgt in inner_stops:
                        self._goto(tgt, m, blk, acc)
                        continue
                    self._goto(tgt, m, blk, {})
                    r = self._run_region(tgt, frozenset(inner_stops), m)
                    for lbl, mm in r.items():
                        self._merge_out(acc, lbl, mm)
                for lbl, mm in acc.items():
                    if lbl != sel:
                        self._merge_out(out, lbl, mm)
                mask = acc.get(sel)
                if mask is None or not mask.any():
                    return out
                if sel in stops:
                    self._merge_out(out, sel, mask)
                    return out
                blk = sel
                continue
            # An unstructured-looking two-way branch: a loop's condition or its back edge.
            for tgt, m in targets:
                if not m.any():
                    continue
                if tgt in stops:
                    self._goto(tgt, m, blk, out)
                    continue
                self._goto(tgt, m, blk, {})
                r = self._run_region(tgt, stops, m)
                for lbl, mm in r.items():
                    self._merge_out(out, lbl, mm)
            return out

    @staticmethod
    def _merge_out(out: dict, label: int, mask: np.ndarray) -> None:
        prev = out.get(label)
        out[label] = mask if prev is None else (prev | mask)

    def _run_loop(self, header: int, merge: int, cont: int, mask: np.ndarray,
                  outer_out: dict, outer_stops: frozenset) -> np.ndarray:
        cur = mask
        exited = np.zeros_like(mask)
        stops = frozenset({cont, merge})
        for _ in range(MAX_LOOP_ITERATIONS):
            if not cur.any():
                return exited
            r = self._run_region_from_header(header, stops, cur)
            for lbl, mm in r.items():
                if lbl == merge:
                    exited = exited | mm
                elif lbl != cont:
                    self._merge_out(outer_out, lbl, mm)
            nxt = r.get(cont)
            if nxt is None or not nxt.any():
                return exited
            r2 = self._run_region(cont, frozenset({header, merge}), nxt)
            for lbl, mm in r2.items():
                if lbl == merge:
                    exited = exited | mm
                elif lbl != header:
                    self._merge_out(outer_out, lbl, mm)
            cur = r2.get(header)
            if cur is None:
                return exited
        raise InstrumentError(
            f"loop at block {header} ran {MAX_LOOP_ITERATIONS} iterations without draining; "
            "this is the interpreter failing to terminate, not a finding about the shader"
        )

    def _run_region_from_header(self, header: int, stops: frozenset,
                                mask: np.ndarray) -> dict:
        """`_run_region` starting at a loop header, without re-entering the loop construct."""
        out: dict[int, np.ndarray] = {}
        b = self.blocks[header]
        self._exec_block(header, mask)
        t = b["term"]
        targets = self._targets(t, mask)
        if len(targets) == 1:
            tgt, m = targets[0]
            if tgt in stops:
                self._goto(tgt, m, header, out)
                return out
            self._goto(tgt, m, header, {})
            r = self._run_region(tgt, stops, m)
            for lbl, mm in r.items():
                self._merge_out(out, lbl, mm)
            return out
        for tgt, m in targets:
            if not m.any():
                continue
            if tgt in stops:
                self._goto(tgt, m, header, out)
                continue
            self._goto(tgt, m, header, {})
            r = self._run_region(tgt, stops, m)
            for lbl, mm in r.items():
                self._merge_out(out, lbl, mm)
        return out

    def _targets(self, t: Instr, mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
        if t.op == "OpBranch":
            return [(t.operands[0], mask)]
        if t.op == "OpBranchConditional":
            c = self.value(t.operands[0]).astype(bool)
            return [(t.operands[1], mask & c), (t.operands[2], mask & ~c)]
        if t.op == "OpSwitch":
            sel = self.value(t.operands[0])
            default = t.operands[1]
            rest = t.operands[2:]
            taken = np.zeros_like(mask)
            outs: list[tuple[int, np.ndarray]] = []
            for i in range(0, len(rest), 2):
                lit, lbl = rest[i], rest[i + 1]
                m = mask & (sel == np.array(lit, dtype=sel.dtype))
                taken |= m
                outs.append((lbl, m))
            outs.append((default, mask & ~taken))
            return outs
        raise InstrumentError(f"not a branch: {t.op}")

    # -- values --------------------------------------------------------------------------
    def value(self, vid: int) -> np.ndarray:
        v = self.vals.get(vid)
        if v is not None:
            return v
        c = self._const_cache.get(vid)
        if c is not None:
            return c
        arr = self._eval_const(vid)
        self._const_cache[vid] = arr
        return arr

    def _eval_const(self, vid: int) -> np.ndarray:
        if vid in self.builtins:
            return self.builtins[vid]
        if vid in self.storage or vid in self.variables:
            raise InstrumentError(f"variable {vid} used as a value")
        c = self.consts.get(vid)
        if c is None:
            raise InstrumentError(f"undefined value %{vid}")
        spec_id = self.decor.get(vid, {}).get("SpecId")
        if c.op in ("OpSpecConstant",) and spec_id is not None and spec_id in self.spec:
            return self._bcast(np.array(self.spec[spec_id], dtype=self._np_dtype(c.result_type)))
        if c.op in ("OpConstant", "OpSpecConstant"):
            return self._bcast(np.array(self.const_scalar(vid),
                                        dtype=self._np_dtype(c.result_type)))
        if c.op in ("OpConstantTrue", "OpSpecConstantTrue"):
            return self._bcast(np.array(True))
        if c.op in ("OpConstantFalse", "OpSpecConstantFalse"):
            return self._bcast(np.array(False))
        if c.op in ("OpConstantComposite", "OpSpecConstantComposite"):
            parts = [self.value(o) for o in c.operands]
            return np.stack(parts, axis=1)
        if c.op == "OpConstantNull":
            t = self.types[c.result_type]
            n = t.operands[1] if t.op == "OpTypeVector" else 1
            z = np.zeros(self.T if n == 1 else (self.T, n),
                         dtype=self._np_dtype(c.result_type))
            return z
        if c.op == "OpUndef":
            return np.zeros(self.T, dtype=self._np_dtype(c.result_type))
        if c.op == "OpSpecConstantOp":
            sub = Instr(_OPNAMES[c.operands[0]], vid, c.result_type, c.operands[1:],
                        c.operands[1:])
            return self._alu(sub)
        raise InstrumentError(f"cannot evaluate constant {c.op}")

    def _bcast(self, scalar: np.ndarray) -> np.ndarray:
        return np.broadcast_to(scalar, (self.T,)).copy()

    # -- pointers ------------------------------------------------------------------------
    def _access_chain(self, ins: Instr) -> dict:
        base = ins.operands[0]
        idxs = ins.operands[1:]
        if base in self.storage:
            ptr = {"var": base, "off": np.zeros(self.T, dtype=np.int64),
                   "type": self.storage[base]["type"]}
        else:
            ptr = dict(self.vals[base])
            ptr["off"] = ptr["off"].copy()
        cur = ptr["type"]
        for idx in idxs:
            t = self.types[cur]
            if t.op == "OpTypeStruct":
                m = self.const_int(idx)
                off = self.member_offsets.get(cur, {}).get(m)
                if off is None:
                    off = sum(self.scalar_count(x) for x in t.operands[:m])
                else:
                    off //= 4
                ptr["off"] = ptr["off"] + off
                cur = t.operands[m]
            elif t.op in ("OpTypeArray", "OpTypeRuntimeArray"):
                elem = t.operands[0]
                stride = self.scalar_count(elem)
                ptr["off"] = ptr["off"] + self.value(idx).astype(np.int64) * stride
                cur = elem
            elif t.op == "OpTypeVector":
                ptr["off"] = ptr["off"] + self.value(idx).astype(np.int64)
                cur = t.operands[0]
            else:
                raise InstrumentError(f"access chain into {t.op}")
        ptr["type"] = cur
        return ptr

    def _mem(self, ptr: dict):
        st = self.storage[ptr["var"]]
        return st

    def _load(self, ptr: dict, mask: np.ndarray) -> np.ndarray:
        st = self._mem(ptr)
        width = self.scalar_count(ptr["type"])
        off = np.where(mask, ptr["off"], 0)
        data = st["data"]
        if st["kind"] == "global":
            idx = off[:, None] + np.arange(width) if width > 1 else off
            idx = np.clip(idx, 0, data.size - 1)
            want = self._np_dtype(ptr["type"])
            src = data.view(want) if data.dtype != want else data
            out = src[idx]
        elif st["kind"] == "thread":
            rows = np.arange(self.T)
            if width > 1:
                out = data[rows[:, None], off[:, None] + np.arange(width)]
            else:
                out = data[rows, off]
        else:  # workgroup
            rows = self.wg_index.astype(np.int64)
            if width > 1:
                out = data[rows[:, None], off[:, None] + np.arange(width)]
            else:
                out = data[rows, off]
        return out

    def _store(self, ptr: dict, val: np.ndarray, mask: np.ndarray) -> None:
        st = self._mem(ptr)
        width = self.scalar_count(ptr["type"])
        data = st["data"]
        val = np.asarray(val)
        # Only the active lanes are indexed. Parking inactive lanes on a "harmless" slot is not
        # harmless: a masked-off lane and a live one can land on the same index, and numpy's
        # scatter keeps the last write, so the live store disappears. That is how the first run
        # of this interpreter produced a GEMV whose column 0 was wrong in every tile.
        sel = np.nonzero(mask)[0]
        if sel.size == 0:
            return
        off = ptr["off"][sel]
        want = data.dtype
        src = val[sel]
        src = src.astype(want, copy=False) if src.dtype != want else src
        if st["kind"] == "global":
            if width > 1:
                for c in range(width):
                    data[off + c] = src[:, c]
            else:
                data[off] = src
            return
        rows = (sel if st["kind"] == "thread"
                else self.wg_index[sel].astype(np.int64))
        if width > 1:
            for c in range(width):
                data[rows, off + c] = src[:, c]
        else:
            data[rows, off] = src

    def _record_load(self, ptr: dict, mask: np.ndarray) -> None:
        tr = self._trace
        if tr is None or ptr["var"] != self._trace_var:
            return
        width = self.scalar_count(ptr["type"])
        key = ptr["type"]
        cnt, w = tr.sites.get(key, (0, width))
        active = int(np.count_nonzero(mask))
        tr.sites[key] = (cnt + active, width)
        if active == 0:
            return
        off = ptr["off"][mask].astype(np.int64)
        wgs = self.wg_index[mask].astype(np.uint32)
        if width > 1:
            off = (off[:, None] + np.arange(width)).ravel()
            wgs = np.repeat(wgs, width)
        off = np.clip(off, 0, tr.words - 1)
        tr.word_reads += np.bincount(off, minlength=tr.words).astype(np.uint32)
        np.minimum.at(tr.word_min_wg, off, wgs)
        np.maximum.at(tr.word_max_wg, off, wgs)

    def _record_store(self, ptr: dict, mask: np.ndarray) -> None:
        tr = self._wtrace
        if tr is None or ptr["var"] != self._wtrace_var:
            return
        width = self.scalar_count(ptr["type"])
        key = ptr["type"]
        cnt, w = tr.sites.get(key, (0, width))
        active = int(np.count_nonzero(mask))
        tr.sites[key] = (cnt + active, width)
        if active == 0:
            return
        lanes = np.nonzero(mask)[0]
        off = ptr["off"][lanes].astype(np.int64)
        wgs = self.wg_index[lanes].astype(np.uint32)
        if width > 1:
            off = (off[:, None] + np.arange(width)).ravel()
            wgs = np.repeat(wgs, width)
            lanes = np.repeat(lanes, width)
        off = np.clip(off, 0, tr.words - 1)
        tr.word_writes += np.bincount(off, minlength=tr.words).astype(np.uint32)
        np.minimum.at(tr.word_min_wg, off, wgs)
        np.maximum.at(tr.word_max_wg, off, wgs)
        seen = self._wseen
        first = np.fromiter(
            ((int(o), int(l)) not in seen for o, l in zip(off, lanes)),
            dtype=bool, count=off.size,
        )
        seen.update((int(o), int(l)) for o, l in zip(off[first], lanes[first]))
        if first.any():
            tr.writer_invocations += np.bincount(
                off[first], minlength=tr.words).astype(np.uint32)

    # -- instruction execution ----------------------------------------------------------
    def _exec_block(self, label: int, mask: np.ndarray) -> None:
        b = self.blocks[label]
        pending = self.phi_pending.get(label, {})
        for phi in b["phis"]:
            v = pending.get(phi.result)
            if v is None:
                raise InstrumentError(f"phi %{phi.result} reached with no incoming value")
            self.vals[phi.result] = v
        for ins in b["body"]:
            self._exec(ins, mask)

    def _exec(self, ins: Instr, mask: np.ndarray) -> None:
        op = ins.op
        if op in ("OpAccessChain", "OpInBoundsAccessChain"):
            self.vals[ins.result] = self._access_chain(ins)
            return
        if op == "OpLoad":
            src = ins.operands[0]
            if src in self.builtins:
                self.vals[ins.result] = self.builtins[src]
                return
            ptr = self.vals[src] if src not in self.storage else {
                "var": src, "off": np.zeros(self.T, dtype=np.int64),
                "type": self.storage[src]["type"]}
            self._record_load(ptr, mask)
            self.vals[ins.result] = self._load(ptr, mask)
            return
        if op == "OpStore":
            dst = ins.operands[0]
            ptr = self.vals[dst] if dst not in self.storage else {
                "var": dst, "off": np.zeros(self.T, dtype=np.int64),
                "type": self.storage[dst]["type"]}
            self._record_store(ptr, mask)
            self._store(ptr, self.value(ins.operands[1]), mask)
            return
        if op in ("OpControlBarrier", "OpMemoryBarrier"):
            return
        if op in ("OpAtomicAnd", "OpAtomicOr", "OpAtomicXor", "OpAtomicIAdd"):
            ptr = self.vals[ins.operands[0]]
            self._record_store(ptr, mask)
            val = self.value(ins.operands[3])
            st = self._mem(ptr)
            data = st["data"]
            off = np.where(mask, ptr["off"], data.size - 1)
            self.vals[ins.result] = data[off].copy()
            fn = {"OpAtomicAnd": np.bitwise_and, "OpAtomicOr": np.bitwise_or,
                  "OpAtomicXor": np.bitwise_xor, "OpAtomicIAdd": np.add}[op]
            neutral = {"OpAtomicAnd": np.uint32(0xFFFFFFFF)}.get(op, np.uint32(0))
            fn.at(data, off, np.where(mask, val.astype(data.dtype), neutral))
            return
        if op == "OpVariable":
            return
        self.vals[ins.result] = self._alu(ins)

    def _alu(self, ins: Instr) -> np.ndarray:
        op = ins.op
        o = ins.operands
        v = self.value
        if op == "OpExtInst":
            setname = self.ext_imports.get(o[0], "")
            if setname != "GLSL.std.450":
                raise InstrumentError(f"unsupported ext inst set {setname!r}")
            name = _GLSL450.get(o[1])
            args = [v(x) for x in o[2:]]
            if name == "UnpackHalf2x16":
                u = args[0].astype(np.uint32)
                return u.view(np.uint16).reshape(-1, 2).view(np.float16).astype(np.float32)
            if name == "PackHalf2x16":
                h = args[0].astype(np.float16)
                return np.ascontiguousarray(h).view(np.uint32).reshape(-1)
            if name == "Fma":
                return args[0] * args[1] + args[2]
            if name in ("FMin", "UMin", "SMin"):
                return np.minimum(args[0], args[1])
            if name in ("FMax", "UMax", "SMax"):
                return np.maximum(args[0], args[1])
            if name in ("FClamp", "UClamp", "SClamp"):
                return np.clip(args[0], args[1], args[2])
            if name == "Exp":
                return np.exp(args[0])
            if name == "Log":
                return np.log(args[0])
            if name == "Exp2":
                return np.exp2(args[0])
            if name == "Log2":
                return np.log2(args[0])
            if name == "Sqrt":
                return np.sqrt(args[0])
            if name == "InverseSqrt":
                return 1.0 / np.sqrt(args[0])
            if name == "FAbs":
                return np.abs(args[0])
            if name == "SAbs":
                return np.abs(args[0].astype(np.int32))
            if name == "Floor":
                return np.floor(args[0])
            if name == "Ceil":
                return np.ceil(args[0])
            if name == "Pow":
                return np.power(args[0], args[1])
            raise InstrumentError(f"unsupported GLSL.std.450 instruction {o[1]}")
        if op == "OpCompositeExtract":
            base = v(o[0])
            for i in o[1:]:
                base = base[:, i]
            return base
        if op == "OpCompositeConstruct":
            parts = [v(x) for x in o]
            cols = []
            for p in parts:
                cols.append(p[:, None] if p.ndim == 1 else p)
            return np.concatenate(cols, axis=1)
        if op == "OpVectorExtractDynamic":
            base, idx = v(o[0]), v(o[1]).astype(np.int64)
            return base[np.arange(base.shape[0]), idx]
        if op == "OpVectorShuffle":
            a, b = v(o[0]), v(o[1])
            n = a.shape[1]
            cols = [a[:, c] if c < n else b[:, c - n] for c in o[2:]]
            return np.stack(cols, axis=1)
        if op == "OpCopyObject":
            return v(o[0])
        if op == "OpSelect":
            c, a, b = v(o[0]), v(o[1]), v(o[2])
            cc = c.astype(bool)
            if a.ndim == 2 and cc.ndim == 1:
                cc = cc[:, None]
            return np.where(cc, a, b)
        if op == "OpBitcast":
            return v(o[0]).view(self._np_dtype(ins.result_type))
        if op in ("OpConvertUToF", "OpConvertSToF", "OpFConvert"):
            return v(o[0]).astype(np.float32)
        if op in ("OpConvertFToU", "OpConvertFToS", "OpUConvert", "OpSConvert"):
            return v(o[0]).astype(self._np_dtype(ins.result_type))
        if op == "OpPhi":
            raise InstrumentError("phi executed outside block prologue")

        unary = {"OpNot": np.bitwise_not, "OpLogicalNot": np.logical_not,
                 "OpSNegate": np.negative, "OpFNegate": np.negative,
                 "OpIsNan": np.isnan, "OpIsInf": np.isinf}
        if op in unary:
            return unary[op](v(o[0]))

        binary = {
            "OpIAdd": np.add, "OpFAdd": np.add, "OpISub": np.subtract, "OpFSub": np.subtract,
            "OpIMul": np.multiply, "OpFMul": np.multiply, "OpFDiv": np.divide,
            "OpUDiv": np.floor_divide, "OpSDiv": np.floor_divide,
            "OpUMod": np.mod, "OpSMod": np.mod, "OpSRem": np.fmod,
            "OpFRem": np.fmod, "OpFMod": np.mod,
            "OpShiftLeftLogical": np.left_shift, "OpShiftRightLogical": np.right_shift,
            "OpShiftRightArithmetic": np.right_shift,
            "OpBitwiseAnd": np.bitwise_and, "OpBitwiseOr": np.bitwise_or,
            "OpBitwiseXor": np.bitwise_xor,
            "OpLogicalAnd": np.logical_and, "OpLogicalOr": np.logical_or,
            "OpIEqual": np.equal, "OpINotEqual": np.not_equal,
            "OpLogicalEqual": np.equal, "OpLogicalNotEqual": np.not_equal,
            "OpUGreaterThan": np.greater, "OpSGreaterThan": np.greater,
            "OpUGreaterThanEqual": np.greater_equal, "OpSGreaterThanEqual": np.greater_equal,
            "OpULessThan": np.less, "OpSLessThan": np.less,
            "OpULessThanEqual": np.less_equal, "OpSLessThanEqual": np.less_equal,
            "OpFOrdEqual": np.equal, "OpFOrdNotEqual": np.not_equal,
            "OpFOrdLessThan": np.less, "OpFOrdGreaterThan": np.greater,
            "OpFOrdLessThanEqual": np.less_equal, "OpFOrdGreaterThanEqual": np.greater_equal,
        }
        if op in binary:
            a, b = v(o[0]), v(o[1])
            if op == "OpShiftRightLogical":
                a = a.astype(np.uint32, copy=False)
            res = binary[op](a, b)
            want = self._np_dtype(ins.result_type)
            if res.dtype != np.bool_ and res.dtype != want:
                res = res.astype(want, copy=False)
            return res
        if op == "OpDot":
            return np.sum(v(o[0]) * v(o[1]), axis=1)
        raise InstrumentError(f"unsupported instruction {op}")


_DECOR = {
    1: "SpecId", 6: "ArrayStride", 11: "BuiltIn", 24: "NonWritable",
    30: "Location", 33: "Binding", 34: "DescriptorSet", 35: "Offset",
}


def _typed(ins: Instr) -> Instr:
    """Split an instruction's operand list into (result type, result id, operands)."""
    op, ops = ins.op, ins.operands
    if op in _TYPED_RESULT:
        return Instr(op, ops[1], ops[0], ops[2:], ops[2:])
    if op in ("OpLabel", "OpExtInstImport", "OpString", "OpDecorationGroup"):
        return Instr(op, ops[0], None, ops[1:], ops[1:])
    return Instr(op, None, None, ops, ops)


def _decode_string(words: list[int]) -> str:
    raw = b"".join(struct.pack("<I", w) for w in words)
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
