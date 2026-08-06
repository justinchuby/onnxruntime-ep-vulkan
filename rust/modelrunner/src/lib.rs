//! `ort-model-runner` -- a Rust-native, PyPI-free real-model validation runner.
//!
//! WHY THIS CRATE EXISTS
//! ---------------------
//! Every existing whole-model check in this repository (`rust/tools/probe_model_output_agreement.py`,
//! `rust/tools/probe_model_op_census.py`) proves its claim through the `onnxruntime` Python wheel.
//! That makes real-model evidence unobtainable wherever PyPI is unreachable, which is exactly the
//! environment the CI and several developer machines are in. This crate reproduces those claims
//! with nothing but the ONNX Runtime *shared library* -- no Python, no pip, no wheel, and no
//! third-party Rust crate other than `libloading` to open the library.
//!
//! WHAT IT PROVES, AND WHAT IT REFUSES TO PROVE
//! --------------------------------------------
//! A pass means all five of these held, each recorded separately in the evidence:
//!
//! 1. the model file matched a pinned SHA-256 from `bench/results/model_provenance.json`;
//! 2. an `OrtEpDevice` named `VulkanExecutionProvider` existed after registering the plugin, so
//!    the EP was *selected*, not merely requested;
//! 3. ORT's own profile attributed at least one `Node` to that provider -- the primary witness,
//!    because it comes from outside the frame under question;
//! 4. the EP's own `dispatches_executed` counter was non-zero -- corroborating evidence from
//!    inside the frame, which cannot substitute for (3) but must not contradict it;
//! 5. the Vulkan outputs agreed with the CPU outputs under a tolerance chosen *per model, in
//!    advance*, never widened to fit a result.
//!
//! Anything else is a failure with a reason, or an explicit `UNSUPPORTED` -- never a quiet pass.
//! In particular, a model that silently fell back to CPU satisfies (1) and (5) and is exactly the
//! vacuous pass guards (2)-(4) exist to prevent.

pub mod compare;
pub mod error;
pub mod evidence;
pub mod feeds;
pub mod foundry;
pub mod json;
pub mod ortapi;
pub mod ortlib;
pub mod provenance;
pub mod repo;
pub mod run;
pub mod sha256;

pub use error::{Failure, Result, Severity};
