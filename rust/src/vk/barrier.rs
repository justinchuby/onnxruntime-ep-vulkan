//! Buffer memory barrier abstraction.
//!
//! **This is the ONLY module in the crate permitted to name:**
//! - `cmd_pipeline_barrier` / `cmd_pipeline_barrier2`
//! - `BufferMemoryBarrier` / `BufferMemoryBarrier2`
//! - `DependencyInfo`
//! - `PipelineStageFlags` / `PipelineStageFlags2`
//! - `AccessFlags` / `AccessFlags2`
//!
//! Every call site in the recorder uses [`Barriers::buffer_deps`] or
//! [`Barriers::execution_only`] — never any Vulkan barrier type directly. The layering lint in
//! `tests/layering.rs` enforces this boundary and fails CI on a violation.
//!
//! # Design (DESIGN.md §7.5)
//!
//! Two backends — [`Sync2Backend`] and [`LegacyBackend`] — implement the same five-function
//! internal API. [`Barriers::select`] is called **exactly once**, in `Device::new`, and the
//! result is stored on the device. No other call site may branch on
//! `Capabilities::synchronization2`.
//!
//! [`Access`] and [`Stage`] are our own **closed enums with no `None` variant**.
//! `VK_PIPELINE_STAGE_2_NONE` has no legacy equivalent, so the abstraction deliberately cannot
//! express it. Every value in `Access` maps to an exact legacy `(PipelineStageFlags, AccessFlags)`
//! pair by construction — the legacy backend is total, not best-effort.
//!
//! [`buffer_deps`][Barriers::buffer_deps] emits **one** barrier command covering all supplied
//! deps: a `VkDependencyInfo` with N `VkBufferMemoryBarrier2` on the sync2 path, or one
//! `vkCmdPipelineBarrier` with N `VkBufferMemoryBarrier` and OR-ed stage masks on the legacy
//! path. Batching semantics are identical in both backends.
//!
//! # Mapping table (single source of truth)
//!
//! ```text
//! Access variant          Legacy (stage, access)                   Sync2 (stage2, access2)
//! ─────────────────────── ──────────────────────────────────────── ────────────────────────────────────────
//! ShaderRead              COMPUTE_SHADER,  SHADER_READ             COMPUTE_SHADER,  SHADER_READ
//! ShaderWrite             COMPUTE_SHADER,  SHADER_WRITE            COMPUTE_SHADER,  SHADER_WRITE
//! TransferRead            TRANSFER,        TRANSFER_READ           ALL_TRANSFER,    TRANSFER_READ
//! TransferWrite           TRANSFER,        TRANSFER_WRITE          ALL_TRANSFER,    TRANSFER_WRITE
//! HostRead                HOST,            HOST_READ               HOST,            HOST_READ
//! HostWrite               HOST,            HOST_WRITE              HOST,            HOST_WRITE
//! ```
//!
//! # CI coverage (DESIGN.md §7.5 item 5)
//!
//! `ep.force_legacy_barriers` (default `false`) forces [`Barriers::Legacy`] on a sync2-capable
//! device. Trinity runs the differential suite twice per lane — once default, once forced — so
//! the legacy path is never untested code.

use ash::vk;

use super::caps::Capabilities;

// ──────────────────────────────────────────────────────────────────────────────
// Public vocabulary: Access, Stage, BufferDep
// ──────────────────────────────────────────────────────────────────────────────

/// The memory access type for one side of a buffer memory barrier.
///
/// **Deliberately has no `None` variant.** `VK_PIPELINE_STAGE_2_NONE` has no legacy equivalent;
/// the abstraction must not be able to express it (DESIGN.md §7.5 item 2). Every variant maps to
/// an exact legacy `(PipelineStageFlags, AccessFlags)` pair — the legacy backend is total.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Access {
    /// Compute shader reads a storage buffer.
    ShaderRead,
    /// Compute shader writes a storage buffer.
    ShaderWrite,
    /// Transfer command reads a buffer (source of `vkCmdCopyBuffer`).
    TransferRead,
    /// Transfer command writes a buffer (destination of `vkCmdCopyBuffer`).
    TransferWrite,
    /// Host CPU reads a host-visible buffer after mapping.
    HostRead,
    /// Host CPU writes a host-visible buffer before un-mapping.
    HostWrite,
}

/// The pipeline stage for an execution-only barrier (no memory dependency).
///
/// **Deliberately has no `None` variant** — same reasoning as [`Access`]. Used by
/// [`Barriers::execution_only`] for coarser synchronization points (e.g., between the transfer
/// and compute queues at staging upload).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Stage {
    /// Compute shader stage.
    ComputeShader,
    /// Transfer/copy stage.
    Transfer,
    /// Host (CPU) stage.
    Host,
    /// All commands — full pipeline stall. Use sparingly; prefer per-edge barriers.
    AllCommands,
}

/// One buffer memory dependency: a write in `src` access that must be visible before a `dst`
/// read or write.
#[derive(Debug, Clone)]
pub(crate) struct BufferDep {
    /// The buffer whose memory must be made visible.
    pub buffer: vk::Buffer,
    /// Byte offset within the buffer.
    pub offset: vk::DeviceSize,
    /// Number of bytes covered (`vk::WHOLE_SIZE` to cover the whole allocation).
    pub size: vk::DeviceSize,
    /// The write access that produced data (src).
    pub src: Access,
    /// The read or write access that will consume data (dst).
    pub dst: Access,
}

// ──────────────────────────────────────────────────────────────────────────────
// Mapping table — THE one place where Access/Stage → Vulkan flags
// ──────────────────────────────────────────────────────────────────────────────

/// Legacy (pre-sync2) mapping: `Access → (PipelineStageFlags, AccessFlags)`.
///
/// This is half of the single mapping table (DESIGN.md §7.5 item 3). The sync2 counterpart is
/// [`to_sync2_flags`]. Both tables must agree; a disagreement is a bug in one place.
fn to_legacy_flags(access: Access) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match access {
        Access::ShaderRead => (
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        Access::ShaderWrite => (
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
        ),
        Access::TransferRead => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
        ),
        Access::TransferWrite => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        Access::HostRead => (vk::PipelineStageFlags::HOST, vk::AccessFlags::HOST_READ),
        Access::HostWrite => (vk::PipelineStageFlags::HOST, vk::AccessFlags::HOST_WRITE),
    }
}

/// Sync2 mapping: `Access → (PipelineStageFlags2, AccessFlags2)`.
///
/// The other half of the single mapping table (DESIGN.md §7.5 item 3). Must stay consistent
/// with [`to_legacy_flags`].
fn to_sync2_flags(access: Access) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    match access {
        Access::ShaderRead => (
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
        ),
        Access::ShaderWrite => (
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
        ),
        Access::TransferRead => (
            vk::PipelineStageFlags2::ALL_TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
        ),
        Access::TransferWrite => (
            vk::PipelineStageFlags2::ALL_TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        ),
        Access::HostRead => (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ),
        Access::HostWrite => (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_WRITE),
    }
}

/// Legacy mapping for execution-only barriers: `Stage → PipelineStageFlags`.
fn stage_to_legacy(stage: Stage) -> vk::PipelineStageFlags {
    match stage {
        Stage::ComputeShader => vk::PipelineStageFlags::COMPUTE_SHADER,
        Stage::Transfer => vk::PipelineStageFlags::TRANSFER,
        Stage::Host => vk::PipelineStageFlags::HOST,
        Stage::AllCommands => vk::PipelineStageFlags::ALL_COMMANDS,
    }
}

/// Sync2 mapping for execution-only barriers: `Stage → PipelineStageFlags2`.
fn stage_to_sync2(stage: Stage) -> vk::PipelineStageFlags2 {
    match stage {
        Stage::ComputeShader => vk::PipelineStageFlags2::COMPUTE_SHADER,
        Stage::Transfer => vk::PipelineStageFlags2::ALL_TRANSFER,
        Stage::Host => vk::PipelineStageFlags2::HOST,
        Stage::AllCommands => vk::PipelineStageFlags2::ALL_COMMANDS,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Backends
// ──────────────────────────────────────────────────────────────────────────────

/// Barrier backend that uses `vkCmdPipelineBarrier2`.
///
/// Two sub-variants handle the entry-point name difference:
/// - `Core` — Vulkan 1.3+ core: `device.cmd_pipeline_barrier2()`.
/// - `Khr` — `VK_KHR_synchronization2` on Vulkan 1.1/1.2: loaded via the KHR extension loader.
///
/// The split is necessary because some Vulkan 1.3+ drivers do **not** export
/// `vkCmdPipelineBarrier2KHR` (the extension alias), even though the spec permits it.
/// Using the wrong entry point causes a null-function-pointer panic in ash.
pub(crate) enum Sync2Backend {
    /// Vulkan 1.3 core — use the promoted entry point `vkCmdPipelineBarrier2`.
    ///
    /// Boxed to avoid a large-enum-variant warning: `ash::Device` is ~1488 bytes while
    /// `Khr` is ~72 bytes; boxing equalises the size and reduces the stack footprint of every
    /// `Sync2Backend` value.
    Core(Box<ash::Device>),
    /// `VK_KHR_synchronization2` on Vulkan 1.1/1.2 — use the KHR alias.
    Khr(ash::khr::synchronization2::Device),
}

/// Barrier backend that uses Vulkan 1.0/1.1 `vkCmdPipelineBarrier`.
///
/// Emits one `vkCmdPipelineBarrier` call with OR-ed stage masks and one
/// `VkBufferMemoryBarrier` per dep.
pub(crate) struct LegacyBackend {
    device: ash::Device,
}

// ──────────────────────────────────────────────────────────────────────────────
// Backend probe — Trinity CI harness support
// ──────────────────────────────────────────────────────────────────────────────

/// Write the selected backend token (`"sync2"` or `"legacy"`) to the path named by
/// `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE`, if that environment variable is set.
///
/// Trinity's parity harness sets this variable before spawning an inference session. The
/// harness reads the file after session creation to assert which backend was selected — this is
/// what makes the `backend_legacy == "legacy"` assertion real rather than a false-green that
/// silently ran sync2 twice.
///
/// Uses plain `std::fs::write`. No ORT API, no logging-format dependency, no other side
/// effects. Errors are silently swallowed (`let _ = ...`) because a broken probe path must
/// not fail session creation.
fn write_backend_probe(is_sync2: bool) {
    if let Ok(path) = std::env::var("ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE") {
        let token = if is_sync2 { "sync2" } else { "legacy" };
        let _ = std::fs::write(&path, token);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Barriers — the public API
// ──────────────────────────────────────────────────────────────────────────────

/// The barrier dispatcher, selected once at device creation and stored on the device.
///
/// All command-buffer recording code calls this API. No recording code may branch on
/// `Capabilities::synchronization2` — that check happened exactly once, in [`Barriers::select`].
pub(crate) enum Barriers {
    /// `vkCmdPipelineBarrier2` path (sync2 extension or Vulkan 1.3 core).
    Sync2(Box<Sync2Backend>),
    /// `vkCmdPipelineBarrier` path (Vulkan 1.0 core, always available).
    Legacy(Box<LegacyBackend>),
}

/// Returns `true` when the sync2 backend should be selected given capabilities and the
/// force-legacy override.
///
/// Extracted from [`Barriers::select`] so the pure decision logic is unit-testable without
/// requiring live Vulkan handles.
pub(crate) fn should_use_sync2(caps: &Capabilities, force_legacy: bool) -> bool {
    !force_legacy && caps.synchronization2
}

impl Barriers {
    /// Select and construct the barrier backend.
    ///
    /// Called **exactly once**, in `Device::new`. The result is stored on the device handle;
    /// no other code may branch on `caps.synchronization2`.
    ///
    /// `force_legacy` overrides the selection to `Legacy` even when sync2 is available. Set
    /// from `ep.force_legacy_barriers` so CI exercises both paths on the same hardware
    /// (DESIGN.md §7.5 item 5).
    ///
    /// When `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE` is set to a file path, writes `"sync2"` or
    /// `"legacy"` to that file so the test harness can assert which backend ran. Plain
    /// `std::fs::write`; no ORT API involvement.
    ///
    /// # Safety
    /// `instance` and `device` must be live and related. When `caps.synchronization2 == true`,
    /// the device must have been created with `VK_KHR_synchronization2` enabled (or Vulkan 1.3
    /// core requested), so that `vkCmdPipelineBarrier2KHR` / `vkCmdPipelineBarrier2` is valid.
    pub(crate) unsafe fn select(
        caps: &Capabilities,
        instance: &ash::Instance,
        device: &ash::Device,
        force_legacy: bool,
    ) -> Self {
        let use_sync2 = should_use_sync2(caps, force_legacy);
        write_backend_probe(use_sync2);
        if use_sync2 {
            if caps.synchronization2_is_core {
                // Vulkan 1.3+: the promoted core entry point `vkCmdPipelineBarrier2` is loaded
                // directly from ash::Device. Do NOT use the KHR extension loader here — some
                // 1.3+ drivers do not export `vkCmdPipelineBarrier2KHR` even though the spec
                // permits it, resulting in a null function pointer and a panic.
                Barriers::Sync2(Box::new(Sync2Backend::Core(Box::new(device.clone()))))
            } else {
                // Vulkan 1.1/1.2 with VK_KHR_synchronization2: use the extension alias.
                // The extension was explicitly requested in device_extensions, so the function
                // pointer is guaranteed to be present.
                let fns = ash::khr::synchronization2::Device::new(instance, device);
                Barriers::Sync2(Box::new(Sync2Backend::Khr(fns)))
            }
        } else {
            Barriers::Legacy(Box::new(LegacyBackend {
                device: device.clone(),
            }))
        }
    }

    /// Emit one barrier command covering all buffer memory dependencies.
    ///
    /// - **Sync2:** one `vkCmdPipelineBarrier2` with a `VkDependencyInfo` carrying N
    ///   `VkBufferMemoryBarrier2`.
    /// - **Legacy:** one `vkCmdPipelineBarrier` with N `VkBufferMemoryBarrier` and OR-ed stage
    ///   masks.
    ///
    /// Batching is identical in both backends (DESIGN.md §7.5 item 4): one command, N barrier
    /// structs, so the parity tests in CI are meaningful.
    ///
    /// # Safety
    /// `cb` must be a primary command buffer in the recording state. Every `BufferDep::buffer`
    /// must be live for the duration of the command buffer's execution.
    pub(crate) unsafe fn buffer_deps(&self, cb: vk::CommandBuffer, deps: &[BufferDep]) {
        if deps.is_empty() {
            return;
        }
        match self {
            Barriers::Sync2(b) => {
                // SAFETY: cb is recording, deps are live per caller contract.
                unsafe { b.buffer_deps(cb, deps) };
            }
            Barriers::Legacy(b) => {
                // SAFETY: same.
                unsafe { b.buffer_deps(cb, deps) };
            }
        }
    }

    /// Emit a pure execution barrier (no memory visibility — only ordering).
    ///
    /// Used to insert a pipeline stage ordering without a memory dependency, e.g. to ensure
    /// a transfer queue submission completes before the compute queue begins.
    ///
    /// # Safety
    /// `cb` must be a primary command buffer in the recording state.
    pub(crate) unsafe fn execution_only(&self, cb: vk::CommandBuffer, src: Stage, dst: Stage) {
        match self {
            Barriers::Sync2(b) => {
                // SAFETY: cb is recording per caller.
                unsafe { b.execution_only(cb, src, dst) };
            }
            Barriers::Legacy(b) => {
                // SAFETY: same.
                unsafe { b.execution_only(cb, src, dst) };
            }
        }
    }

    /// Returns `true` when this backend uses `vkCmdPipelineBarrier2` (sync2 path).
    ///
    /// Exposed for test assertions; recording code must not branch on this.
    #[cfg(test)]
    pub(crate) fn is_sync2(&self) -> bool {
        matches!(self, Barriers::Sync2(_))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Timestamp helper — the only other call site permitted to name PipelineStageFlags
// ──────────────────────────────────────────────────────────────────────────────

/// Record a `vkCmdWriteTimestamp` at the `COMPUTE_SHADER` pipeline stage.
///
/// Centralised here because `PipelineStageFlags` may only appear in this module (layering rule
/// 7.5). The caller (`timestamp.rs`) passes in the ash device, command buffer, query pool, and
/// index but must not name the stage flag itself.
///
/// # Safety
/// - `cmd` must be in the recording state.
/// - `pool` must be a live `VK_QUERY_TYPE_TIMESTAMP` pool.
/// - `idx` must be within the pool's query count.
#[inline]
pub(crate) unsafe fn cmd_write_compute_timestamp(
    ash_device: &ash::Device,
    cmd: vk::CommandBuffer,
    pool: vk::QueryPool,
    idx: u32,
) {
    // SAFETY: delegated to caller per the function contract.
    unsafe {
        ash_device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, pool, idx);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Sync2Backend implementation
// ──────────────────────────────────────────────────────────────────────────────

impl Sync2Backend {
    unsafe fn buffer_deps(&self, cb: vk::CommandBuffer, deps: &[BufferDep]) {
        let barriers: Vec<vk::BufferMemoryBarrier2> = deps
            .iter()
            .map(|dep| {
                let (src_stage, src_access) = to_sync2_flags(dep.src);
                let (dst_stage, dst_access) = to_sync2_flags(dep.dst);
                vk::BufferMemoryBarrier2 {
                    src_stage_mask: src_stage,
                    src_access_mask: src_access,
                    dst_stage_mask: dst_stage,
                    dst_access_mask: dst_access,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer: dep.buffer,
                    offset: dep.offset,
                    size: dep.size,
                    ..Default::default()
                }
            })
            .collect();

        let dep_info = vk::DependencyInfo {
            buffer_memory_barrier_count: barriers.len() as u32,
            p_buffer_memory_barriers: barriers.as_ptr(),
            ..Default::default()
        };

        // SAFETY: cb is recording, dep_info is well-formed, barriers are live (on the stack
        // for the duration of cmd_pipeline_barrier2 which returns before this frame ends).
        match self {
            Sync2Backend::Core(device) => {
                // SAFETY: device is live; cmd_pipeline_barrier2 is a Vulkan 1.3 core function.
                unsafe { device.cmd_pipeline_barrier2(cb, &dep_info) };
            }
            Sync2Backend::Khr(fns) => {
                // SAFETY: fns loaded the KHR extension; the function pointer is non-null
                // (VK_KHR_synchronization2 was explicitly enabled at device creation).
                unsafe { fns.cmd_pipeline_barrier2(cb, &dep_info) };
            }
        }
    }

    unsafe fn execution_only(&self, cb: vk::CommandBuffer, src: Stage, dst: Stage) {
        // A pure execution barrier in sync2 is expressed as a VkMemoryBarrier2 with
        // stage masks but zero access masks (no memory visibility, only ordering).
        let mem_barrier = vk::MemoryBarrier2 {
            src_stage_mask: stage_to_sync2(src),
            dst_stage_mask: stage_to_sync2(dst),
            src_access_mask: vk::AccessFlags2::empty(),
            dst_access_mask: vk::AccessFlags2::empty(),
            ..Default::default()
        };
        let dep_info = vk::DependencyInfo {
            memory_barrier_count: 1,
            p_memory_barriers: &mem_barrier,
            ..Default::default()
        };
        // SAFETY: cb is recording per caller; dep_info and mem_barrier are live on this frame.
        match self {
            Sync2Backend::Core(device) => {
                // SAFETY: Vulkan 1.3 core function; device is live.
                unsafe { device.cmd_pipeline_barrier2(cb, &dep_info) };
            }
            Sync2Backend::Khr(fns) => {
                // SAFETY: KHR extension function pointer is non-null (extension was enabled).
                unsafe { fns.cmd_pipeline_barrier2(cb, &dep_info) };
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// LegacyBackend implementation
// ──────────────────────────────────────────────────────────────────────────────

impl LegacyBackend {
    unsafe fn buffer_deps(&self, cb: vk::CommandBuffer, deps: &[BufferDep]) {
        // Collect barriers and OR the stage masks together (the legacy API requires scalar masks).
        let mut src_stages = vk::PipelineStageFlags::empty();
        let mut dst_stages = vk::PipelineStageFlags::empty();
        let barriers: Vec<vk::BufferMemoryBarrier> = deps
            .iter()
            .map(|dep| {
                let (ss, sa) = to_legacy_flags(dep.src);
                let (ds, da) = to_legacy_flags(dep.dst);
                src_stages |= ss;
                dst_stages |= ds;
                vk::BufferMemoryBarrier {
                    src_access_mask: sa,
                    dst_access_mask: da,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer: dep.buffer,
                    offset: dep.offset,
                    size: dep.size,
                    ..Default::default()
                }
            })
            .collect();

        // SAFETY: cb is recording per caller; barriers are live on the stack for this call.
        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                src_stages,
                dst_stages,
                vk::DependencyFlags::empty(),
                &[], // no global memory barriers
                &barriers,
                &[], // no image barriers
            );
        }
    }

    unsafe fn execution_only(&self, cb: vk::CommandBuffer, src: Stage, dst: Stage) {
        // SAFETY: cb is recording per caller. Zero barriers is valid — this is a pure execution
        // barrier; `vkCmdPipelineBarrier` with no barrier structs is spec-legal.
        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                stage_to_legacy(src),
                stage_to_legacy(dst),
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[],
            );
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Access → Legacy flags mapping ─────────────────────────────────────────

    #[test]
    fn legacy_mapping_covers_all_access_variants() {
        // Every Access variant must produce non-empty flags. This ensures the mapping table is
        // total and documents that there is no `None` variant (DESIGN.md §7.5 item 2).
        let all = [
            Access::ShaderRead,
            Access::ShaderWrite,
            Access::TransferRead,
            Access::TransferWrite,
            Access::HostRead,
            Access::HostWrite,
        ];
        for v in all {
            let (stage, access) = to_legacy_flags(v);
            assert!(
                !stage.is_empty(),
                "legacy stage must be non-empty for {v:?}"
            );
            assert!(
                !access.is_empty(),
                "legacy access must be non-empty for {v:?}"
            );
        }
    }

    #[test]
    fn sync2_mapping_covers_all_access_variants() {
        let all = [
            Access::ShaderRead,
            Access::ShaderWrite,
            Access::TransferRead,
            Access::TransferWrite,
            Access::HostRead,
            Access::HostWrite,
        ];
        for v in all {
            let (stage, access) = to_sync2_flags(v);
            assert!(!stage.is_empty(), "sync2 stage must be non-empty for {v:?}");
            assert!(
                !access.is_empty(),
                "sync2 access must be non-empty for {v:?}"
            );
        }
    }

    /// The two tables must agree on the semantic intent of each variant. Since the legacy and
    /// sync2 flag names differ (SHADER_READ vs SHADER_READ — both present, but different enums),
    /// we check the table contents symbolically rather than bitwise.
    #[test]
    fn legacy_and_sync2_tables_agree_on_shader_accesses() {
        // ShaderRead → COMPUTE_SHADER stage in both backends.
        let (legacy_stage, _) = to_legacy_flags(Access::ShaderRead);
        let (sync2_stage, _) = to_sync2_flags(Access::ShaderRead);
        assert!(legacy_stage.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
        assert!(sync2_stage.contains(vk::PipelineStageFlags2::COMPUTE_SHADER));

        // ShaderWrite → COMPUTE_SHADER stage in both backends.
        let (legacy_stage, _) = to_legacy_flags(Access::ShaderWrite);
        let (sync2_stage, _) = to_sync2_flags(Access::ShaderWrite);
        assert!(legacy_stage.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
        assert!(sync2_stage.contains(vk::PipelineStageFlags2::COMPUTE_SHADER));
    }

    #[test]
    fn legacy_and_sync2_tables_agree_on_transfer_accesses() {
        // TransferRead → TRANSFER stage in legacy; ALL_TRANSFER in sync2.
        let (legacy_stage, legacy_access) = to_legacy_flags(Access::TransferRead);
        let (sync2_stage, sync2_access) = to_sync2_flags(Access::TransferRead);
        assert!(legacy_stage.contains(vk::PipelineStageFlags::TRANSFER));
        assert!(sync2_stage.contains(vk::PipelineStageFlags2::ALL_TRANSFER));
        assert!(legacy_access.contains(vk::AccessFlags::TRANSFER_READ));
        assert!(sync2_access.contains(vk::AccessFlags2::TRANSFER_READ));

        let (legacy_stage, legacy_access) = to_legacy_flags(Access::TransferWrite);
        let (sync2_stage, sync2_access) = to_sync2_flags(Access::TransferWrite);
        assert!(legacy_stage.contains(vk::PipelineStageFlags::TRANSFER));
        assert!(sync2_stage.contains(vk::PipelineStageFlags2::ALL_TRANSFER));
        assert!(legacy_access.contains(vk::AccessFlags::TRANSFER_WRITE));
        assert!(sync2_access.contains(vk::AccessFlags2::TRANSFER_WRITE));
    }

    #[test]
    fn legacy_and_sync2_tables_agree_on_host_accesses() {
        let (ls, la) = to_legacy_flags(Access::HostRead);
        let (ss, sa) = to_sync2_flags(Access::HostRead);
        assert!(ls.contains(vk::PipelineStageFlags::HOST));
        assert!(la.contains(vk::AccessFlags::HOST_READ));
        assert!(ss.contains(vk::PipelineStageFlags2::HOST));
        assert!(sa.contains(vk::AccessFlags2::HOST_READ));

        let (ls, la) = to_legacy_flags(Access::HostWrite);
        let (ss, sa) = to_sync2_flags(Access::HostWrite);
        assert!(ls.contains(vk::PipelineStageFlags::HOST));
        assert!(la.contains(vk::AccessFlags::HOST_WRITE));
        assert!(ss.contains(vk::PipelineStageFlags2::HOST));
        assert!(sa.contains(vk::AccessFlags2::HOST_WRITE));
    }

    // ── Stage → flags mapping ─────────────────────────────────────────────────

    #[test]
    fn stage_legacy_mapping_covers_all_variants() {
        let all = [
            Stage::ComputeShader,
            Stage::Transfer,
            Stage::Host,
            Stage::AllCommands,
        ];
        for s in all {
            assert!(
                !stage_to_legacy(s).is_empty(),
                "legacy stage non-empty for {s:?}"
            );
            assert!(
                !stage_to_sync2(s).is_empty(),
                "sync2 stage non-empty for {s:?}"
            );
        }
    }

    // ── Buffer dep structure ──────────────────────────────────────────────────

    #[test]
    fn buffer_dep_whole_size_constant() {
        // BufferDep::size accepts vk::WHOLE_SIZE, documenting that callers can use it.
        let dep = BufferDep {
            buffer: vk::Buffer::null(),
            offset: 0,
            size: vk::WHOLE_SIZE,
            src: Access::ShaderWrite,
            dst: Access::ShaderRead,
        };
        assert_eq!(dep.size, vk::WHOLE_SIZE);
    }

    // ── access_has_no_none_variant (compile-time property demonstrated) ───────

    #[test]
    fn access_exhaustive_match_compiles() {
        // If Access::None were ever added, this match would fail to compile — forcing the
        // author to handle it in the mapping table, which would require inventing a flag value.
        // The test itself passes vacuously; its value is as a compile-time guard.
        let _: () = match Access::ShaderRead {
            Access::ShaderRead
            | Access::ShaderWrite
            | Access::TransferRead
            | Access::TransferWrite
            | Access::HostRead
            | Access::HostWrite => {}
        };
    }

    #[test]
    fn stage_exhaustive_match_compiles() {
        let _: () = match Stage::ComputeShader {
            Stage::ComputeShader | Stage::Transfer | Stage::Host | Stage::AllCommands => {}
        };
    }

    // ── OR-accumulation for legacy path ──────────────────────────────────────

    #[test]
    fn legacy_or_accumulation_merges_stage_masks() {
        // When two deps with different stages are passed, the accumulated stage mask must cover
        // both. This simulates the LegacyBackend's fold logic in isolation.
        let deps = [
            BufferDep {
                buffer: vk::Buffer::null(),
                offset: 0,
                size: vk::WHOLE_SIZE,
                src: Access::ShaderWrite,
                dst: Access::TransferRead,
            },
            BufferDep {
                buffer: vk::Buffer::null(),
                offset: 0,
                size: vk::WHOLE_SIZE,
                src: Access::TransferWrite,
                dst: Access::HostRead,
            },
        ];

        let mut src_stages = vk::PipelineStageFlags::empty();
        let mut dst_stages = vk::PipelineStageFlags::empty();
        for dep in &deps {
            let (ss, _) = to_legacy_flags(dep.src);
            let (ds, _) = to_legacy_flags(dep.dst);
            src_stages |= ss;
            dst_stages |= ds;
        }

        // src should cover COMPUTE_SHADER (ShaderWrite) and TRANSFER (TransferWrite).
        assert!(src_stages.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
        assert!(src_stages.contains(vk::PipelineStageFlags::TRANSFER));
        // dst should cover TRANSFER (TransferRead) and HOST (HostRead).
        assert!(dst_stages.contains(vk::PipelineStageFlags::TRANSFER));
        assert!(dst_stages.contains(vk::PipelineStageFlags::HOST));
    }

    // ── Stage → exact flag values ─────────────────────────────────────────────

    #[test]
    fn stage_to_legacy_exact_values() {
        // Assert exact mapping table values so a typo or wrong-flag mistake fails a unit test
        // rather than silently producing a too-permissive barrier at runtime.
        assert_eq!(
            stage_to_legacy(Stage::ComputeShader),
            vk::PipelineStageFlags::COMPUTE_SHADER
        );
        assert_eq!(
            stage_to_legacy(Stage::Transfer),
            vk::PipelineStageFlags::TRANSFER
        );
        assert_eq!(stage_to_legacy(Stage::Host), vk::PipelineStageFlags::HOST);
        assert_eq!(
            stage_to_legacy(Stage::AllCommands),
            vk::PipelineStageFlags::ALL_COMMANDS
        );
    }

    #[test]
    fn stage_to_sync2_exact_values() {
        assert_eq!(
            stage_to_sync2(Stage::ComputeShader),
            vk::PipelineStageFlags2::COMPUTE_SHADER
        );
        // Transfer maps to ALL_TRANSFER in sync2, not TRANSFER — this covers both copy and blit.
        assert_eq!(
            stage_to_sync2(Stage::Transfer),
            vk::PipelineStageFlags2::ALL_TRANSFER
        );
        assert_eq!(stage_to_sync2(Stage::Host), vk::PipelineStageFlags2::HOST);
        assert_eq!(
            stage_to_sync2(Stage::AllCommands),
            vk::PipelineStageFlags2::ALL_COMMANDS
        );
    }

    // ── Backend probe ─────────────────────────────────────────────────────────

    use std::sync::atomic::{AtomicU32, Ordering};
    /// Unique counter so parallel tests do not share a probe file path.
    static PROBE_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn probe_path(n: u32) -> std::path::PathBuf {
        // target/ is always present when tests run and is .gitignored.
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("barrier_probe_test_{n}.txt"))
    }

    #[test]
    fn backend_probe_writes_legacy_token() {
        let n = PROBE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let p = probe_path(n);
        let path_str = p.to_str().expect("ASCII path");

        // SAFETY: env-var mutation is safe provided no other thread reads the same var
        // concurrently. This var is unique to our test harness; cargo test runs unit tests
        // in the same process but on separate threads. To be safe we use a unique path per
        // invocation (PROBE_COUNTER) and read the env var immediately after writing it.
        unsafe {
            std::env::set_var("ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE", path_str);
        }
        write_backend_probe(false /* legacy */);
        // SAFETY: removing the var we just set.
        unsafe { std::env::remove_var("ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE") };

        let content = std::fs::read_to_string(&p).expect("probe file written");
        let _ = std::fs::remove_file(&p);
        assert_eq!(content, "legacy");
    }

    #[test]
    fn backend_probe_writes_sync2_token() {
        let n = PROBE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let p = probe_path(n);
        let path_str = p.to_str().expect("ASCII path");

        // SAFETY: see backend_probe_writes_legacy_token.
        unsafe { std::env::set_var("ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE", path_str) };
        write_backend_probe(true /* sync2 */);
        // SAFETY: removing the var we just set.
        unsafe { std::env::remove_var("ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE") };

        let content = std::fs::read_to_string(&p).expect("probe file written");
        let _ = std::fs::remove_file(&p);
        assert_eq!(content, "sync2");
    }

    #[test]
    fn backend_probe_noop_when_env_unset() {
        let n = PROBE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let p = probe_path(n);
        // Env var is NOT set.
        // SAFETY: removing a var.
        unsafe { std::env::remove_var("ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE") };
        write_backend_probe(false);
        assert!(
            !p.exists(),
            "probe must not create a file when env var is absent"
        );
    }

    // ── force_legacy overrides synchronization2 capability ───────────────────

    #[test]
    fn force_legacy_overrides_sync2_capability() {
        let caps = crate::vk::caps::Capabilities {
            synchronization2: true,
            synchronization2_is_core: true,
            subgroup_size: 32,
            subgroup_probe_valid: true,
            subgroup_basic_in_compute: true,
            subgroup_supported_stages: vk::ShaderStageFlags::COMPUTE,
            subgroup_supported_ops: vk::SubgroupFeatureFlags::BASIC,
            subgroup_size_range: Some(crate::vk::caps::SubgroupSizeRange { min: 32, max: 32 }),
            can_require_subgroup_size: false,
            shader_float16: false,
            is_uma: false,
            timestamp_period_ns: 1.0,
            timestamp_valid_bits: 64,
        };
        assert!(
            !should_use_sync2(&caps, true /* force_legacy */),
            "force_legacy must override synchronization2 capability"
        );
    }

    #[test]
    fn no_force_legacy_and_no_sync2_selects_legacy() {
        let caps = crate::vk::caps::Capabilities {
            synchronization2: false,
            synchronization2_is_core: false,
            subgroup_size: 64,
            subgroup_probe_valid: true,
            subgroup_basic_in_compute: false,
            subgroup_supported_stages: vk::ShaderStageFlags::empty(),
            subgroup_supported_ops: vk::SubgroupFeatureFlags::BASIC,
            subgroup_size_range: None,
            can_require_subgroup_size: false,
            shader_float16: false,
            is_uma: false,
            timestamp_period_ns: 1.0,
            timestamp_valid_bits: 64,
        };
        assert!(
            !should_use_sync2(&caps, false),
            "absent sync2 capability must produce Legacy backend"
        );
    }

    #[test]
    fn sync2_capable_without_force_selects_sync2() {
        let caps = crate::vk::caps::Capabilities {
            synchronization2: true,
            synchronization2_is_core: true,
            subgroup_size: 32,
            subgroup_probe_valid: true,
            subgroup_basic_in_compute: true,
            subgroup_supported_stages: vk::ShaderStageFlags::COMPUTE,
            subgroup_supported_ops: vk::SubgroupFeatureFlags::BASIC,
            subgroup_size_range: None,
            can_require_subgroup_size: false,
            shader_float16: false,
            is_uma: false,
            timestamp_period_ns: 1.0,
            timestamp_valid_bits: 64,
        };
        assert!(
            should_use_sync2(&caps, false),
            "sync2 capable device without force_legacy must select sync2"
        );
    }
}
