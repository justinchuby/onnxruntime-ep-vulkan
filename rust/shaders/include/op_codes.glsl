// Op selector codes for the elementwise templates.
//
// `Kernel::defines` (src/ops/common/variants.rs) emits `-DEW_OP=OP_<NAME>` for every variant row,
// where `<NAME>` is the uppercased `op` field of the row's `kernel!(...)`. The numbers below are
// what those names expand to, so a template can switch on `#if EW_OP == OP_ADD`.
//
// The values themselves are arbitrary and never leave the compiler: nothing at runtime sees them.
// They must only be unique, and every `op` string in the registry must have one — a Rust test
// (`variants::tests::every_op_selector_has_a_glsl_code`) checks that this file covers the table.

#ifndef OP_CODES_GLSL
#define OP_CODES_GLSL

// -- binary ---------------------------------------------------------------------------------
#define OP_ADD          1
#define OP_SUB          2
#define OP_MUL          3
#define OP_DIV          4
#define OP_POW          5
#define OP_MOD          6
#define OP_AND          7
#define OP_OR           8
#define OP_XOR          9
#define OP_BITAND      10
#define OP_BITOR       11
#define OP_BITXOR      12
#define OP_BITSHIFT    13
#define OP_EQ          14
#define OP_GT          15
#define OP_GE          16
#define OP_LT          17
#define OP_LE          18
#define OP_PRELU       19
#define OP_MAX         20
#define OP_MIN         21
#define OP_MEAN        22

// -- unary maths ----------------------------------------------------------------------------
#define OP_ABS         30
#define OP_NEG         31
#define OP_RECIP       32
#define OP_SQRT        33
#define OP_EXP         34
#define OP_LOG         35
#define OP_SIN         36
#define OP_COS         37
#define OP_TAN         38
#define OP_ASIN        39
#define OP_ACOS        40
#define OP_ATAN        41
#define OP_SINH        42
#define OP_COSH        43
#define OP_TANH        44
#define OP_ASINH       45
#define OP_ACOSH       46
#define OP_ATANH       47
#define OP_CEIL        48
#define OP_FLOOR       49
#define OP_ROUND       50
#define OP_SIGN        51
#define OP_ERF         52
#define OP_NOT         53
#define OP_BITNOT      54
#define OP_ISNAN       55
#define OP_ISINF       56
#define OP_IDENTITY    57

// -- activations ----------------------------------------------------------------------------
#define OP_RELU        70
#define OP_SIGMOID     71
#define OP_HARDSWISH   72
#define OP_SOFTPLUS    73
#define OP_SOFTSIGN    74
#define OP_MISH        75
#define OP_HARDSIGMOID 76
#define OP_LEAKYRELU   77
#define OP_ELU         78
#define OP_SELU        79
#define OP_CELU        80
#define OP_TRELU       81
#define OP_SHRINK      82
#define OP_GELU        83
#define OP_SWISH       84

// -- select ---------------------------------------------------------------------------------
#define OP_WHERE       90
#define OP_CLIP        91

// Op codes whose result is a boolean tensor regardless of the input element type. ONNX stores
// `bool` as one byte per element, so these write through the packed-byte store path.
#if EW_OP == OP_EQ || EW_OP == OP_GT || EW_OP == OP_GE || EW_OP == OP_LT || EW_OP == OP_LE \
    || EW_OP == OP_ISNAN || EW_OP == OP_ISINF || EW_OP == OP_NOT || EW_OP == OP_AND \
    || EW_OP == OP_OR || EW_OP == OP_XOR
#define EW_BOOL_RESULT 1
#endif

#endif // OP_CODES_GLSL
