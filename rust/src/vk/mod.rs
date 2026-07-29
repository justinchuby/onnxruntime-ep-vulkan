//! The Vulkan engine layer. **Only** this module tree may touch `ash` handles or `gpu-allocator`
//! allocations. Nothing outside `vk/` reaches a `vk::Buffer`, a `vk::CommandBuffer`, a
//! `vk::Pipeline`, or any other raw Vulkan handle — that boundary is the only thing that keeps
//! per-op shader changes from rippling into synchronization, memory, and descriptor code.
//!
//! The boundary is enforced mechanically by `tests/layering.rs`, which fails CI if a forbidden
//! token appears outside this tree.
//!
//! # Module layout (DESIGN.md §3)
//!
//! | Module | Contents |
//! |---|---|
//! | [`caps`] | Device capability discovery — the single capability oracle |
//! | [`barrier`] | Buffer memory barriers — the ONLY module that names Vulkan barrier types |
//! | [`device`] | Logical device wrapper — the ONLY call site for [`barrier::Barriers::select`] |

// Items are built out ahead of engine integration; dead_code is expected at this stage.
#![allow(dead_code)]

pub(crate) mod barrier;
pub(crate) mod caps;
pub(crate) mod device;
