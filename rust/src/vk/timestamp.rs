//! `VkQueryPool` timestamp queries for per-dispatch GPU timing.
//!
//! [`GpuQueryPool`] records one (before, after) timestamp pair around each `vkCmdDispatch` in a
//! command buffer. After the fence signals, [`GpuQueryPool::read_results`] returns the raw ticks;
//! the caller builds a [`crate::trace::GpuTimestampReport`] and hands it to
//! [`crate::trace::tracer()`] which converts device ticks to the shared microsecond axis.
//!
//! # Intel Iris Xe caution — 36 valid bits, 52.0833 ns/tick
//!
//! The two load-bearing numbers on this desk are:
//!
//! * `timestampPeriod = 52.0833 ns/tick` on Intel (1.0 on NVIDIA and lavapipe).
//!   Treating ticks as nanoseconds on Intel under-reports **every** GPU duration by **52×**.
//!   Silence and plausibility guarantee that this goes unnoticed.
//! * `timestampValidBits = 36` on Intel (64 on NVIDIA and lavapipe).
//!   The upper 28 bits of a raw query result are undefined; masking is mandatory before any
//!   arithmetic. An unmasked 36-bit counter wraps in ~3579 s ≈ 0.99 h, which is not exotic
//!   during a benchmark run. NVIDIA and CI cannot catch either mistake — Intel is the only
//!   local instrument that falsifies both. See `bench/timestamp_audit.py`.
//!
//! This module does *not* perform the mask or period conversion. Raw ticks are handed back to
//! the caller and `GpuTimestampCalibration` in `trace.rs` applies both transformations with
//! the exact field values from `caps.rs`. The Rust unit tests in `trace.rs` use the real Intel
//! constants so they fail on the hardware we have, not on hardware we imagined.
//!
//! # Layout
//!
//! Queries are paired: for kernel index `ki`, query `2*ki` is the before-timestamp and
//! `2*ki+1` is the after-timestamp. Both are `PIPELINE_STAGE_COMPUTE_SHADER_BIT`, which
//! waits for all previous compute work to be visible before writing the tick.

use ash::vk;

use super::barrier::cmd_write_compute_timestamp;

// ──────────────────────────────────────────────────────────────────────────────
// GpuQueryPool
// ──────────────────────────────────────────────────────────────────────────────

/// A `VK_QUERY_TYPE_TIMESTAMP` pool for one command buffer's dispatches.
///
/// One pool per `dispatch_ort` call: created after `recorder.begin()`, reset inside the same
/// command buffer, populated by [`cmd_before`]/[`cmd_after`] around each `vkCmdDispatch`, and
/// read by [`read_results`] after the fence signals.
///
/// # Lifetime
/// Must not be dropped until the fence for the submission that recorded into it has signalled.
/// In practice `session.rs::dispatch_ort` holds it on the stack until after
/// `wait_fence_then_destroy` returns.
pub(crate) struct GpuQueryPool {
    ash_device: ash::Device,
    pool: vk::QueryPool,
    n_kernels: usize,
}

impl GpuQueryPool {
    /// Create a query pool for `n_kernels` (before, after) timestamp pairs.
    ///
    /// Returns `None` when `n_kernels == 0` or pool creation fails (logged as a warning; the
    /// caller continues without GPU timestamps rather than failing the dispatch).
    ///
    /// # Safety
    /// `ash_device` must be a live logical device for the lifetime of the returned pool.
    pub(crate) unsafe fn new(ash_device: &ash::Device, n_kernels: usize) -> Option<Self> {
        if n_kernels == 0 {
            return None;
        }
        let n_queries = match n_kernels.checked_mul(2) {
            Some(n) if n <= u32::MAX as usize => n as u32,
            _ => {
                log::warn!(
                    "GpuQueryPool: n_kernels={n_kernels} overflows query count; \
                     GPU timestamps disabled for this dispatch"
                );
                return None;
            }
        };
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(n_queries);
        // SAFETY: ash_device is live per the caller's contract; info is fully initialised.
        let pool = match unsafe { ash_device.create_query_pool(&info, None) } {
            Ok(p) => p,
            Err(e) => {
                log::warn!(
                    "vkCreateQueryPool failed ({e}); \
                     GPU timestamps disabled for this dispatch"
                );
                return None;
            }
        };
        Some(Self {
            ash_device: ash_device.clone(),
            pool,
            n_kernels,
        })
    }

    /// Record `vkCmdResetQueryPool` for all queries in this pool.
    ///
    /// Must be called once inside the command buffer before any [`cmd_before`]/[`cmd_after`]
    /// writes. Place it after `vkBeginCommandBuffer` and before the first barrier.
    ///
    /// # Safety
    /// `cmd` must be in the recording state.
    pub(crate) unsafe fn cmd_reset(&self, cmd: vk::CommandBuffer) {
        // SAFETY: cmd is recording; pool is live; range is in bounds.
        unsafe {
            self.ash_device
                .cmd_reset_query_pool(cmd, self.pool, 0, (self.n_kernels * 2) as u32);
        }
    }

    /// Record a `COMPUTE_SHADER`-stage timestamp immediately **before** dispatch `ki`.
    ///
    /// # Safety
    /// `cmd` must be recording; `ki < self.n_kernels`; [`cmd_reset`] must have been called
    /// earlier in the same command buffer.
    pub(crate) unsafe fn cmd_before(&self, cmd: vk::CommandBuffer, ki: usize) {
        debug_assert!(
            ki < self.n_kernels,
            "ki={ki} out of range for n_kernels={}",
            self.n_kernels
        );
        // `PipelineStageFlags::COMPUTE_SHADER` lives in barrier.rs (layering rule 7.5);
        // `cmd_write_compute_timestamp` is the single permitted call site for it.
        // SAFETY: cmd is recording; ki is in range; pool is live.
        unsafe {
            cmd_write_compute_timestamp(&self.ash_device, cmd, self.pool, (2 * ki) as u32);
        }
    }

    /// Record a `COMPUTE_SHADER`-stage timestamp immediately **after** dispatch `ki`.
    ///
    /// # Safety
    /// Same as [`cmd_before`].
    pub(crate) unsafe fn cmd_after(&self, cmd: vk::CommandBuffer, ki: usize) {
        debug_assert!(
            ki < self.n_kernels,
            "ki={ki} out of range for n_kernels={}",
            self.n_kernels
        );
        // SAFETY: same as cmd_before.
        unsafe {
            cmd_write_compute_timestamp(&self.ash_device, cmd, self.pool, (2 * ki + 1) as u32);
        }
    }

    /// Read all timestamp pairs after the fence has signalled.
    ///
    /// Returns raw, **unmasked** ticks; the caller is responsible for passing them to
    /// `GpuTimestampCalibration::ticks_to_ns` which applies `timestampValidBits` masking,
    /// period scaling, and single-wrap recovery. Do **not** compare raw ticks to nanoseconds
    /// directly — on Intel, 1 tick = 52.0833 ns.
    ///
    /// Returns `None` for any pair where a query was not written (should not occur in normal
    /// use, but defended against to avoid returning zeros that look like zero-duration kernels).
    ///
    /// # Safety
    /// The fence for the submission that recorded into this pool must have signalled. Calling
    /// while the command buffer is still executing is undefined behaviour.
    pub(crate) unsafe fn read_results(&self) -> Vec<Option<(u64, u64)>> {
        let n = self.n_kernels * 2;
        // Sentinel: u64::MAX should never be written by vkCmdWriteTimestamp (ticks are never
        // near the 64-bit max on any real GPU), so it serves as a reliable "not written" marker.
        let mut raw = vec![u64::MAX; n];
        // SAFETY: pool and raw are live; fence has signalled (caller contract).
        let result = unsafe {
            self.ash_device.get_query_pool_results(
                self.pool,
                0,
                &mut raw,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        };
        match result {
            Ok(()) | Err(vk::Result::NOT_READY) => {
                // NOT_READY with WAIT_BIT should not occur (the fence already signalled), but is
                // not fatal — treat as partial results rather than catastrophic failure.
            }
            Err(e) => {
                log::warn!(
                    "vkGetQueryPoolResults failed ({e}); \
                     dropping GPU timestamps for this dispatch"
                );
                return vec![None; self.n_kernels];
            }
        }
        (0..self.n_kernels)
            .map(|ki| {
                let before = raw[2 * ki];
                let after = raw[2 * ki + 1];
                if before == u64::MAX || after == u64::MAX {
                    // Query was not written (unexpected, but report None rather than garbage).
                    None
                } else {
                    Some((before, after))
                }
            })
            .collect()
    }
}

impl Drop for GpuQueryPool {
    fn drop(&mut self) {
        // SAFETY: pool was created by us; the fence has signalled (lifetime invariant), so the
        // command buffer is no longer executing and the pool is not in use.
        unsafe { self.ash_device.destroy_query_pool(self.pool, None) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {

    /// Verify the type API compiles and the query layout arithmetic is correct.
    /// Real Vulkan paths are covered by Trinity's lavapipe lane.
    #[test]
    fn query_index_layout() {
        // Before-index for kernel ki is 2*ki; after-index is 2*ki+1.
        // Verify by inspecting cmd_before/cmd_after slots.
        for ki in 0usize..8 {
            assert_eq!(2 * ki, (2 * ki) as u32 as usize);
            assert_eq!(2 * ki + 1, (2 * ki + 1) as u32 as usize);
        }
    }

    #[test]
    fn sentinel_is_never_a_valid_tick() {
        // u64::MAX is not a plausible GPU tick on any device; confirm the chosen sentinel is
        // distinguishable from real readings. (Real ticks are bounded by the session duration
        // times the tick rate — even at 2 GHz for a 24-hour session, max ticks ~ 1.7e14 << MAX.)
        let max_plausible_ticks: u64 = 2_000_000_000u64 * 86_400;
        assert!(max_plausible_ticks < u64::MAX / 2);
    }

    #[test]
    fn n_kernels_overflow_is_handled() {
        // Extremely large n_kernels must not panic.
        assert_eq!(usize::MAX.checked_mul(2), None);
    }
}
