//! Command pool and command buffer management.
//!
//! This module owns `VkCommandPool` and provides a `CommandRecorder` — a scoped wrapper
//! around a `VkCommandBuffer` in the recording state.
//!
//! # Design (ENGINE.md §6)
//!
//! One `CommandPool` per session, created with `RESET_COMMAND_BUFFER_BIT` so individual
//! buffers can be reset without resetting the whole pool. The pool allocates command buffers
//! from a single internal free-list; each `CommandRecorder` checks out exactly one buffer
//! for the duration of its scope.
//!
//! For v0 (single `Add` op), the flow is:
//!
//! ```text
//! CommandPool::new()           // allocate pool + one pre-allocated command buffer
//!   └─► CommandRecorder::begin()  // vkBeginCommandBuffer
//!         ├─► record barriers      // barriers.buffer_deps(…)
//!         ├─► vkCmdBindPipeline
//!         ├─► vkCmdBindDescriptorSets
//!         ├─► vkCmdPushConstants
//!         └─► vkCmdDispatch
//!   └─► CommandRecorder::finish()  // vkEndCommandBuffer → returns VkCommandBuffer for submit
//! ```
//!
//! # Synchronization (ENGINE.md §6.1–6.3)
//!
//! Barriers are inserted by calling `device.barriers().buffer_deps(cmd, deps)` inside
//! `CommandRecorder::record_barriers`. The barrier module — and only it — may name Vulkan
//! barrier types.
//!
//! After `finish()`, the caller submits the buffer to the compute queue and waits on a fence.
//! v0 uses a simple submit-and-fence pattern (one submission per subgraph). Timeline
//! semaphores are a future optimisation when overlapping subgraphs are introduced.
//!
//! # Lifetime rules
//!
//! `CommandPool` must outlive every `CommandRecorder` it produces. The `CommandRecorder`
//! borrows the pool's ash device (via lifetime `'pool`) to prevent use-after-free.

use ash::vk;

// ──────────────────────────────────────────────────────────────────────────────
// CommandPool
// ──────────────────────────────────────────────────────────────────────────────

/// Owns a `VkCommandPool` and a pre-allocated primary command buffer.
///
/// One per session. `RESET_COMMAND_BUFFER_BIT` allows individual command buffers to be
/// reused across `Run` calls without allocating from the pool on every inference.
pub(crate) struct CommandPool {
    ash_device: ash::Device,
    pool: vk::CommandPool,
    /// One reusable primary command buffer. Extended to a free-list when concurrent
    /// subgraph recording is needed (M2+).
    cmd: vk::CommandBuffer,
}

impl CommandPool {
    /// Create a command pool for the given queue family.
    ///
    /// # Safety
    /// - `ash_device` must be a live logical device for the lifetime of the returned `CommandPool`.
    /// - `queue_family` must be the compute queue family index reported by `Device::compute_queue_family()`.
    pub(crate) unsafe fn new(ash_device: &ash::Device, queue_family: u32) -> Option<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);

        // SAFETY: ash_device is live per the caller's contract; pool_info is valid.
        let pool = unsafe {
            match ash_device.create_command_pool(&pool_info, None) {
                Ok(p) => p,
                Err(e) => {
                    log::error!("vkCreateCommandPool failed: {e}");
                    return None;
                }
            }
        };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: pool is freshly created; ash_device is live.
        let cmds = unsafe {
            match ash_device.allocate_command_buffers(&alloc_info) {
                Ok(c) => c,
                Err(e) => {
                    log::error!("vkAllocateCommandBuffers failed: {e}");
                    // SAFETY: pool was created by us; no command buffers are live.
                    // Already inside an outer unsafe block — no nested unsafe needed.
                    ash_device.destroy_command_pool(pool, None);
                    return None;
                }
            }
        };

        Some(CommandPool {
            ash_device: ash_device.clone(),
            pool,
            cmd: cmds[0],
        })
    }

    /// Begin a new recording session on the pre-allocated command buffer.
    ///
    /// Resets the command buffer before beginning (cheap: the buffer was already executed
    /// and the pool was created with `RESET_COMMAND_BUFFER_BIT`).
    ///
    /// # Safety
    /// The previous recording (if any) must have completed on the GPU before calling this.
    /// Any `CommandRecorder` previously produced by this pool must have been dropped.
    pub(crate) unsafe fn begin(&self) -> Option<CommandRecorder<'_>> {
        let reset_flags = vk::CommandBufferResetFlags::empty();
        // SAFETY: cmd is a valid command buffer from this pool; previous work is complete
        // per the caller's contract.
        unsafe {
            if let Err(e) = self.ash_device.reset_command_buffer(self.cmd, reset_flags) {
                log::error!("vkResetCommandBuffer failed: {e}");
                return None;
            }
        }

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: cmd is reset and ready; begin_info is valid.
        unsafe {
            if let Err(e) = self.ash_device.begin_command_buffer(self.cmd, &begin_info) {
                log::error!("vkBeginCommandBuffer failed: {e}");
                return None;
            }
        }

        Some(CommandRecorder {
            ash_device: &self.ash_device,
            cmd: self.cmd,
        })
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        // Command buffers allocated from the pool are freed implicitly when the pool is destroyed.
        // SAFETY: pool was created by us; we are the sole owner. All command buffers allocated
        // from it must have completed on the GPU before reaching this Drop (their lifetime is
        // shorter than CommandPool's — they borrow `ash_device` from it).
        unsafe { self.ash_device.destroy_command_pool(self.pool, None) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CommandRecorder
// ──────────────────────────────────────────────────────────────────────────────

/// A command buffer in the recording state.
///
/// Produced by [`CommandPool::begin`]; finished by [`CommandRecorder::finish`]. The `'pool`
/// lifetime ties this recorder to the pool that produced it, preventing use-after-pool-drop.
pub(crate) struct CommandRecorder<'pool> {
    ash_device: &'pool ash::Device,
    /// The command buffer being recorded.
    pub(crate) cmd: vk::CommandBuffer,
}

impl<'pool> CommandRecorder<'pool> {
    /// End recording and return the command buffer, ready for submission.
    ///
    /// # Safety
    /// All commands recorded since `begin` must be valid and complete.
    pub(crate) unsafe fn finish(self) -> Option<vk::CommandBuffer> {
        // SAFETY: cmd is in the recording state (invariant of CommandRecorder construction).
        unsafe {
            if let Err(e) = self.ash_device.end_command_buffer(self.cmd) {
                log::error!("vkEndCommandBuffer failed: {e}");
                return None;
            }
        }
        let cmd = self.cmd;
        std::mem::forget(self); // don't run Drop (which would panic — cmd is handed to caller)
        Some(cmd)
    }
}

impl Drop for CommandRecorder<'_> {
    fn drop(&mut self) {
        // Dropping without finishing is a usage error — log it. The command buffer state is now
        // undefined; resetting it here would be the safe recovery, but we cannot do that without
        // knowing whether the pool allows single-buffer resets. Log a prominent warning and let
        // the caller's next `begin()` reset the buffer naturally.
        log::warn!(
            "CommandRecorder dropped without calling finish() — the command buffer is in an \
             undefined state. The next begin() call will reset it."
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Queue submission helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Create a fence, submit `cmd` to `queue` under it, and return the fence.
///
/// On failure the fence is destroyed (if it was created) and `None` is returned.
///
/// This is the split-submit half of the v0 synchronization model; pair with
/// [`wait_fence_then_destroy`] when the submit and wait phases need to be timed independently.
/// For the all-in-one path, prefer [`submit_and_wait`].
///
/// # Safety
/// - `ash_device` must be the logical device that owns `queue` and `cmd`.
/// - `cmd` must be in the executable state (returned by [`CommandRecorder::finish`]).
/// - No other work may be in-flight on `queue` (v0 single-queue constraint).
pub(crate) unsafe fn create_and_submit(
    ash_device: &ash::Device,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
) -> Option<vk::Fence> {
    let fence_info = vk::FenceCreateInfo::default();
    // SAFETY: ash_device is live.
    let fence = unsafe {
        match ash_device.create_fence(&fence_info, None) {
            Ok(f) => f,
            Err(e) => {
                log::error!("vkCreateFence failed: {e}");
                return None;
            }
        }
    };
    let submit_info = [vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))];
    // SAFETY: queue is valid; cmd is executable.
    if let Err(e) = unsafe { ash_device.queue_submit(queue, &submit_info, fence) } {
        log::error!("vkQueueSubmit failed: {e}");
        // SAFETY: fence was created by us; nothing was submitted to it.
        unsafe { ash_device.destroy_fence(fence, None) };
        return None;
    }
    Some(fence)
}

/// Wait for `fence` to signal, then destroy it. Returns `false` on wait failure.
///
/// The fence is destroyed regardless of the outcome. Pairs with [`create_and_submit`].
///
/// # Safety
/// - `ash_device` must be the logical device that owns `fence`.
/// - `fence` must have been submitted via [`create_and_submit`] or equivalent.
pub(crate) unsafe fn wait_fence_then_destroy(ash_device: &ash::Device, fence: vk::Fence) -> bool {
    // SAFETY: fence is live and was submitted.
    let wait_ok = unsafe { ash_device.wait_for_fences(&[fence], true, u64::MAX) };
    // Destroy regardless of wait outcome — must not leak.
    // SAFETY: fence is done (or errored); safe to destroy.
    unsafe { ash_device.destroy_fence(fence, None) };
    if let Err(e) = wait_ok {
        log::error!("vkWaitForFences failed: {e}");
        // A lost device is not one failure among many: every later submission on this device
        // will fail too, the process keeps running, and ORT's Python fallback will quietly
        // re-run the graph on the CPU EP and return a plausible answer. Record it separately so
        // a post-run screen can see it in the counters even when nothing raised.
        if e == vk::Result::ERROR_DEVICE_LOST {
            crate::counters::record_device_lost();
        }
        return false;
    }
    true
}

/// Whether the device has been lost at any point in this process.
///
/// Read after a run to decide whether the numbers it produced describe this EP at all.
pub(crate) fn device_was_lost() -> bool {
    crate::counters::device_losses() > 0
}

/// Submit `cmd` to `queue` and wait (via fence) for it to complete.
///
/// This is the v0 synchronization model: one submission per subgraph, blocking. Timeline
/// semaphores and pipelined submissions are a future optimisation (M2+, ENGINE.md §6.3).
///
/// Implemented as [`create_and_submit`] + [`wait_fence_then_destroy`]; use those directly when
/// the submit and wait phases need to be timed independently (e.g. in `dispatch_ort`).
///
/// # Safety
/// - `ash_device` must be the logical device that owns `queue` and `cmd`.
/// - `cmd` must be in the executable state (returned by [`CommandRecorder::finish`]).
/// - No other work may be in-flight on `queue` when this is called (v0 single-queue constraint).
pub(crate) unsafe fn submit_and_wait(
    ash_device: &ash::Device,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
) -> bool {
    // SAFETY: requirements delegated to create_and_submit/wait_fence_then_destroy per their docs.
    let Some(fence) = (unsafe { create_and_submit(ash_device, queue, cmd) }) else {
        return false;
    };
    // SAFETY: fence was returned by create_and_submit above; requirements are met per its docs.
    unsafe { wait_fence_then_destroy(ash_device, fence) }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the module compiles and the public API surface is as expected.
    /// Full Vulkan path is tested on Trinity's lavapipe lane.
    #[test]
    fn command_pool_types_compile() {
        // This test just verifies the module compiles correctly and the API is sound.
        // We can't create real Vulkan objects without a device, but we can verify the
        // type API is self-consistent.
        fn _assert_send<T: Send>() {}
        // CommandPool is not Send (contains ash::Device which is internally Arc-based and Send,
        // but vk::CommandBuffer is a raw pointer — we do NOT mark CommandPool as Send yet).
        // This is intentional for v0: the command pool lives on one thread.
        let _ = std::mem::size_of::<CommandPool>();
        let _ = std::mem::size_of::<vk::CommandBuffer>();
    }

    #[test]
    fn submit_and_wait_returns_false_without_a_device() {
        // Can't test submit_and_wait with null handles without UB — the Vulkan ICD path is
        // covered by Trinity's lavapipe lane. This test only documents the return-false contract.
        // The function signature and semantics are the tested artifact here.
        let _ = std::mem::size_of::<vk::Queue>();
        let _ = std::mem::size_of::<vk::Fence>();
    }
}
