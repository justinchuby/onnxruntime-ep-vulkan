//! Drive the mock ONNX Runtime host against the plugin **linked as an rlib**.
//!
//! This is the fast path: no library loading, and because the driver shares the plugin's `log`
//! crate it can push a record straight through the logging bridge. See `tests/mock_ort/mod.rs`
//! for what the host checks and why it exists, and `tests/cdylib_load.rs` for the variant that
//! loads the real shared library the way ORT does.

mod mock_ort;

use mock_ort::{LogProbe, run_registration_scenario};
use onnxruntime_vulkan_ep::{CreateEpFactories, ReleaseEpFactory};

#[test]
fn a_mock_ort_host_can_register_use_and_release_the_plugin() {
    // SAFETY: these are the crate's own exported entry points, statically linked into this test
    // binary, so they are live for the whole process.
    unsafe { run_registration_scenario(CreateEpFactories, ReleaseEpFactory, LogProbe::Shared) };
}
