# `rust/ci/linux-stub-include` — three headers that make a Linux compile check possible on Windows

## What these are

Declarations for the *only* C library headers that
`third_party/onnxruntime/include/onnxruntime_c_api.h` includes. They exist so that `clang`, driven
by `bindgen`, can **parse** the ORT headers while targeting `x86_64-unknown-linux-gnu` from a
machine that has no Linux sysroot.

They are not a libc. Nothing is ever linked against them, no program is ever produced from them,
and they are used by exactly one command: `cargo ci --cross` (see `rust/xtask/src/main.rs`).

## Why they exist

`rust/tests/portability.rs` records why this repository has a text-based portability lint at all:
on 2026-07-29 a Linux-only compile error blocked the Linux lane for a full CI cycle. Its module
docs also state that cross-compiling locally was tried and rejected —

> Clang's own builtin headers can be supplied with `BINDGEN_EXTRA_CLANG_ARGS` (that fixes
> `stdbool.h`), but `stdlib.h` belongs to glibc, and getting it means vendoring or downloading a
> Linux sysroot. That is infrastructure we would have to maintain, on every dev box, for one
> lint's worth of value. Rejected — see D-T20.

That premise was wrong, and it cost a red `main` on 2026-08-06 when the model runner merged with
two Linux-only type errors in `modelrunner/src/ortapi.rs` (issue #39). The ORT headers do not *use*
glibc — they include `<stdbool.h>`, `<stdlib.h>` and `<string.h>` and then declare their own types.
A real sysroot was never needed; a handful of declarations were. Vendoring "a Linux sysroot" is
indeed infrastructure nobody wants to maintain. Vendoring `size_t` is not.

## Why this is a check and not a hope

A text lint reads source and guesses. This does not guess: it runs the real front end at the real
target and reports the real errors, which for issue #39's two defects were exactly

```text
error[E0277]: the trait bound `u32: Neg` is not satisfied      (modelrunner/src/ortapi.rs:133)
error[E0308]: mismatched types: expected `u32`, found `i32`    (modelrunner/src/ortapi.rs:158)
```

Neither was visible to the text lint, and neither could have been: the defective lines never name
the alias whose width they assume. `Api::check` matched on `GetErrorCode`'s result without ever
writing `ort::OrtErrorCode`. That is the class of bug a scanner cannot see and a compiler cannot
miss.

## What it does not check

Only that the crates **compile** for Linux. `cargo check` does not link, does not run tests, and
does not touch a device. CI's Linux lane remains the thing that runs them. This closes the gap
between "compiles on the machine it was written on" and "compiles at all on the other platform",
which is the gap that has now cost two red lanes.

## If a header needs another declaration

Add the narrowest declaration that lets clang parse, and say in a comment which ORT header needed
it. If this directory ever starts to look like a libc, that is the signal to stop and use a real
sysroot instead.
