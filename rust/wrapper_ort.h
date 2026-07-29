// Single translation unit handed to bindgen for the ONNX Runtime plugin-EP C ABI.
//
// Only `onnxruntime_c_api.h` is included: it `#include`s `onnxruntime_ep_c_api.h` itself, and the
// EP header has NO include guard (verified in ORT v1.28.0), so including it a second time here is
// a hard clang error ("redefinition of 'OrtDataTransferImpl'", etc.). If a future ORT ever stops
// pulling the EP header in from the main one, the plugin-EP types disappear from the generated
// bindings and `src/factory.rs` stops compiling -- a loud failure, which is the point.
#include "onnxruntime_c_api.h"
