/* Not a libc. See README.md in this directory.
 *
 * Needed by: third_party/onnxruntime/include/onnxruntime_c_api.h (`#include <stdlib.h>`), which
 * wants `size_t` for the many `size_t` counts in the C API. The allocator declarations are here
 * because ORT's headers declare function pointers with `malloc`/`free` compatible signatures and
 * a parse error in one of them would look like a bindgen bug rather than a missing stub.
 *
 * `__SIZE_TYPE__` and friends are clang builtins, so these are exactly the target's own types --
 * `unsigned long` for x86_64-unknown-linux-gnu -- and not a guess made on the host.
 */
#pragma once

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

void *malloc(size_t);
void *calloc(size_t, size_t);
void *realloc(void *, size_t);
void free(void *);
void abort(void);
