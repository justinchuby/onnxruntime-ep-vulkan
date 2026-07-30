// The shared indexing header — broadcasting is written once, here, and nowhere else.
//
// `OP_COVERAGE.md` §5.1 is the argument: numpy broadcasting is the most repeated and most
// error-prone logic in an op library, and a backend that re-derives it per op pays for it per op,
// in code and in bugs that look like kernel bugs but are indexing bugs. So no template in this
// directory contains any broadcasting logic at all. The host (`ops::common::shape_plan`) resolves
// shapes once and hands the shader **element strides**, where a broadcast axis is simply a stride
// of zero. Everything below is a generic `linear output index -> per-input element offset` walk.
//
// # Contract with the host
//
// Push-constant block, little-endian, matching `ShapePlan::push_constants`:
//
//     offset  size  field
//     0       4     rank            (u32, <= 6)
//     4       4     elem_count      (u32, output element count)
//     8       24    out_shape[6]    (u32, right-aligned, left-padded with 1)
//     32      24    strides[0][6]   (u32, element strides; 0 == broadcast)
//     56      24    strides[1][6]   (arity >= 2)
//     80      24    strides[2][6]   (arity >= 3)
//     <after the last stride array>
//             16    params[4]       (f32, op attributes; all zero when the op has none)
//
// The params tail sits at an arity-dependent offset because `strides` is declared as
// `uint strides[EW_ARITY * EW_MAX_RANK]`. Nothing may be inserted between the two.
//
// Worst case 120 bytes, inside the 128-byte `maxPushConstantsSize` floor guaranteed by Vulkan 1.1.
//
// Specialisation constants, in the order `KernelRequest::spec_constants` supplies them:
//
//     id 0  local_size_x     (workgroup size; 256 unless a per-device tuner overrides it)
//     id 1  EW_IDENTICAL     (1 when every input already has the output shape)
//
// Descriptor set 0 holds the inputs in order followed by the single output, per `ENGINE.md` §5.2.
//
// # Byte-typed tensors
//
// ONNX stores `bool` and `uint8` as one byte per element and the baseline capability set does not
// include `storageBuffer8BitAccess` (`ENGINE.md` §4.1), so those tensors are addressed as packed
// `uint` words. Loads shift and mask. Stores use `atomicAnd` + `atomicOr` on **disjoint bit lanes**
// of the shared word, which is race-free in any interleaving precisely because the lanes are
// disjoint: each invocation only ever clears and sets the eight bits belonging to its own element.
//
// This imposes one requirement on the allocator, stated here because it is invisible from the Rust
// side: **a buffer holding a byte-typed tensor must be allocated and bound rounded up to a
// multiple of 4 bytes**, since the last word is written whole. Any sane allocator alignment
// satisfies this; it is written down so it cannot be broken silently.

#ifndef INDEXING_GLSL
#define INDEXING_GLSL

#ifndef EW_ARITY
#error "include indexing.glsl only after defining EW_ARITY"
#endif

// -- element type mapping -------------------------------------------------------------------
//
// `SCALAR_T` arrives as a -D define and is the *storage* type. `COMPUTE_T` is what the op
// arithmetic runs in: f16 tensors are stored as `float16_t` but computed in `float`, which costs
// nothing on any GPU that supports 16-bit storage and avoids depending on f16 overloads of the
// transcendental builtins.

#if defined(DTYPE_F16)
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define COMPUTE_T float
#define EW_FLOAT 1
#elif defined(DTYPE_F32)
#define COMPUTE_T float
#define EW_FLOAT 1
#elif defined(DTYPE_I64)
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#define COMPUTE_T int64_t
#define EW_SIGNED_INT 1
#elif defined(DTYPE_I32)
#define COMPUTE_T int
#define EW_SIGNED_INT 1
#elif defined(DTYPE_U8) || defined(DTYPE_BOOL)
#define COMPUTE_T uint
#define EW_PACKED_BYTES 1
#else
#error "no DTYPE_* define: the variant table must pass exactly one"
#endif

#define EW_MAX_RANK 6

// Attribute slots carried in the push-constant tail. Must equal `EW_PARAM_COUNT` in
// `ops/common/shape_plan.rs`; the host always pushes this many floats, zeroed for ops that have
// no attributes, so that one pipeline layout serves every variant of this template.
#define EW_PARAM_COUNT 4

layout(constant_id = 1) const uint EW_IDENTICAL = 0;

layout(push_constant, std430) uniform EwParams {
    uint rank;
    uint elem_count;
    uint out_shape[EW_MAX_RANK];
    uint strides[EW_ARITY * EW_MAX_RANK];
    float params[EW_PARAM_COUNT];
} pc;

// The whole of broadcasting: walk the output coordinate from the innermost axis outwards,
// accumulating `coord * stride`. Padding axes have extent 1, so their coordinate is 0 and they
// contribute nothing; broadcast axes have stride 0, so they contribute nothing either. There is
// no branch and no special case.
uint ew_offset(uint input_index, uint linear) {
    uint base = input_index * uint(EW_MAX_RANK);
    uint rem = linear;
    uint off = 0u;
    for (uint axis = uint(EW_MAX_RANK); axis > 0u; --axis) {
        uint dim = pc.out_shape[axis - 1u];
        uint coord = rem % dim;
        rem /= dim;
        off += coord * pc.strides[base + axis - 1u];
    }
    return off;
}

// The fast path: when every input already has the output shape, the offset *is* the linear index.
// `EW_IDENTICAL` is a specialisation constant, so the branch is resolved when the pipeline is
// created rather than per invocation.
uint ew_index(uint input_index, uint linear) {
    return EW_IDENTICAL != 0u ? linear : ew_offset(input_index, linear);
}

// -- accessor generators ---------------------------------------------------------------------
//
// Written as macros taking both the function name and the buffer name, because the GLSL
// preprocessor has no `##` token-pasting operator. Each template instantiates the ones it needs.

// Load an element of a `SCALAR_T`-typed buffer, promoted to COMPUTE_T.
#define EW_DEFINE_LOAD(FN, BUF)                                                                   \
    COMPUTE_T FN(uint i) { return COMPUTE_T(BUF.data[i]); }

// Load an element of a byte-packed (`bool`/`uint8`) buffer.
#define EW_DEFINE_LOAD_BYTE(FN, BUF)                                                              \
    uint FN(uint i) { return (BUF.data[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu; }

// Store a COMPUTE_T value into a `SCALAR_T`-typed buffer.
#define EW_DEFINE_STORE(FN, BUF)                                                                  \
    void FN(uint i, COMPUTE_T v) { BUF.data[i] = SCALAR_T(v); }

// Store one byte into a packed buffer. See the header note on why the two atomics are safe.
#define EW_DEFINE_STORE_BYTE(FN, BUF)                                                             \
    void FN(uint i, uint v) {                                                                     \
        uint word = i >> 2u;                                                                      \
        uint shift = (i & 3u) * 8u;                                                               \
        atomicAnd(BUF.data[word], ~(0xFFu << shift));                                             \
        atomicOr(BUF.data[word], (v & 0xFFu) << shift);                                           \
    }

#endif // INDEXING_GLSL
