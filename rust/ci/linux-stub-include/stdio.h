/* Not a libc. See README.md in this directory.
 *
 * Needed by: third_party/onnxruntime/include/onnxruntime_ep_c_api.h, which mentions `FILE` in a
 * declaration. Only the opaque tag is required -- nothing here ever performs I/O.
 */
#pragma once

typedef struct _IO_FILE FILE;
