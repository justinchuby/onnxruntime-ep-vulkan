//! Pipeline cache and compute pipeline management.
//!
//! # Design (ENGINE.md §5)
//!
//! Every compute dispatch needs a `VkPipeline` + `VkPipelineLayout` + `VkDescriptorSetLayout`.
//! These are expensive to create but identical across all dispatches of the same shader variant
//! with the same specialization constants. The [`PipelineCache`] avoids redundant creation by
//! keying on `(shader_stem, spec_constants)`.
//!
//! ## Object ownership
//!
//! ```text
//! PipelineCache
//!   └─► HashMap<PipelineKey, PipelineEntry>
//!         └─► VkPipeline
//!         └─► VkPipelineLayout
//!         └─► VkDescriptorSetLayout
//! ```
//!
//! All three objects share a lifetime: they are created together and destroyed together when
//! the `PipelineCache` drops (or when `PipelineCache::clear` is called).
//!
//! ## Descriptor management
//!
//! For v0, each dispatch uses a **single descriptor set** allocated from a per-dispatch pool
//! that is reset after the submit fence signals. This is the "allocate, use, reset pool"
//! pattern: simple, correct, and fast enough for the M0 `Add` target.
//!
//! The long-term design (M2+) is a persistent descriptor pool with per-frame recycling, but
//! that depends on the multi-subgraph scheduler which is out of M0 scope.
//!
//! ## Specialization constants (ENGINE.md §4.4)
//!
//! Each `KernelRequest` carries a `spec_constants: Vec<u32>` in binding order. The pipeline
//! manager maps these to `VkSpecializationMapEntry` records and creates a variant pipeline
//! per unique combination. This is how the same GLSL source produces both `wg_x=64` and
//! `wg_x=256` variants without runtime branching.
//!
//! ## Push constants (ENGINE.md §4.4)
//!
//! All pipelines use a single push-constant range covering the entire 128-byte budget
//! (the Vulkan minimum guarantee). Op handlers pack their per-dispatch scalars (tensor
//! dimensions, strides, etc.) into `KernelRequest::push_constants` and the recorder calls
//! `vkCmdPushConstants` immediately before `vkCmdDispatch`.
//!
//! The recorders push the **entire** [`PUSH_CONSTANT_RANGE_BYTES`], zero-padding whatever the
//! kernel packed. A declared range that is only partly written leaves the remainder undefined —
//! see the note at the push site in `session.rs`.

use std::collections::HashMap;

use ash::vk;

/// Size in bytes of the single push-constant range every pipeline layout in this engine declares.
///
/// 128 is the Vulkan minimum guarantee for `maxPushConstantsSize`, so a uniform range of this size
/// is portable to every conformant implementation. It is a constant rather than a per-kernel value
/// because the pipeline cache is keyed on `(shader, spec_constants)` only; a per-kernel range would
/// have to become part of that key, and a layout disagreeing with its dispatch would be a hard
/// error rather than a warning.
///
/// Anything that records `vkCmdPushConstants` against these layouts must write all of it.
pub(crate) const PUSH_CONSTANT_RANGE_BYTES: usize = 128;

// ──────────────────────────────────────────────────────────────────────────────
// Pipeline key
// ──────────────────────────────────────────────────────────────────────────────

/// Cache key: shader stem + specialization constants (in binding order).
///
/// Two dispatches with the same key share one `VkPipeline`. The spec-constants are part of the
/// key because Vulkan bakes them into the pipeline at creation time.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PipelineKey {
    pub(crate) shader: &'static str,
    pub(crate) spec_constants: Vec<u32>,
}

// ──────────────────────────────────────────────────────────────────────────────
// PipelineEntry
// ──────────────────────────────────────────────────────────────────────────────

/// One cached `(pipeline, layout, descriptor_set_layout)` triple.
pub(crate) struct PipelineEntry {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    /// Number of storage-buffer bindings (= number of `BufferView`s in the dispatch).
    pub(crate) binding_count: u32,
}

// ──────────────────────────────────────────────────────────────────────────────
// PipelineCache
// ──────────────────────────────────────────────────────────────────────────────

/// Compute pipeline cache.
///
/// Lazily builds and caches `(VkPipeline, VkPipelineLayout, VkDescriptorSetLayout)` triples.
/// All entries are destroyed when the cache drops.
pub(crate) struct PipelineCache {
    ash_device: ash::Device,
    /// Vulkan pipeline-cache object (enables driver-side serialisation).
    vk_cache: vk::PipelineCache,
    entries: HashMap<PipelineKey, PipelineEntry>,
}

impl PipelineCache {
    /// Create a pipeline cache backed by a new `VkPipelineCache`.
    ///
    /// `initial_data` is optional driver-serialised cache data from a previous run
    /// (from `ep.pipeline_cache_path`). Pass `&[]` to start fresh.
    ///
    /// # Safety
    /// `ash_device` must be a live logical device for the lifetime of this `PipelineCache`.
    pub(crate) unsafe fn new(ash_device: &ash::Device, initial_data: &[u8]) -> Option<Self> {
        let cache_info = vk::PipelineCacheCreateInfo::default().initial_data(initial_data);
        // SAFETY: ash_device is live; initial_data is valid bytes.
        let vk_cache = unsafe {
            match ash_device.create_pipeline_cache(&cache_info, None) {
                Ok(c) => c,
                Err(e) => {
                    log::error!("vkCreatePipelineCache failed: {e}");
                    return None;
                }
            }
        };
        Some(PipelineCache {
            ash_device: ash_device.clone(),
            vk_cache,
            entries: HashMap::new(),
        })
    }

    /// Look up or create the pipeline for `key`.
    ///
    /// `spirv` must be the SPIR-V bytes for `key.shader`, pre-validated by `build.rs`.
    /// `binding_count` is the number of storage-buffer descriptor bindings needed by this shader.
    ///
    /// Returns a reference to the cached entry, or `None` if pipeline creation fails.
    ///
    /// # Safety
    /// `spirv` must be valid aligned SPIR-V. `ash_device` behind `self` must be live.
    pub(crate) unsafe fn get_or_create(
        &mut self,
        key: PipelineKey,
        spirv: &[u8],
        binding_count: u32,
    ) -> Option<&PipelineEntry> {
        if self.entries.contains_key(&key) {
            return self.entries.get(&key);
        }
        // SAFETY: spirv is valid SPIR-V per the caller's contract; ash_device is live.
        let entry = unsafe {
            Self::create_entry(&self.ash_device, self.vk_cache, &key, spirv, binding_count)?
        };
        self.entries.insert(key.clone(), entry);
        self.entries.get(&key)
    }

    /// Serialise the driver-side pipeline cache to bytes.
    ///
    /// Returns an empty vec on failure (the cache is advisory).
    ///
    /// # Safety
    /// `ash_device` behind `self` must be live.
    pub(crate) unsafe fn serialise(&self) -> Vec<u8> {
        // SAFETY: vk_cache is valid; ash_device is live.
        unsafe {
            self.ash_device
                .get_pipeline_cache_data(self.vk_cache)
                .unwrap_or_default()
        }
    }

    /// Create one `PipelineEntry` from scratch.
    ///
    /// # Safety
    /// All handles must be valid and live.
    unsafe fn create_entry(
        ash_device: &ash::Device,
        vk_cache: vk::PipelineCache,
        key: &PipelineKey,
        spirv: &[u8],
        binding_count: u32,
    ) -> Option<PipelineEntry> {
        // 1. Descriptor-set layout: one STORAGE_BUFFER binding per input/output.
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..binding_count)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: ash_device and dsl_info are valid.
        let descriptor_set_layout = unsafe {
            match ash_device.create_descriptor_set_layout(&dsl_info, None) {
                Ok(d) => d,
                Err(e) => {
                    log::error!(
                        "vkCreateDescriptorSetLayout failed for '{}': {e}",
                        key.shader
                    );
                    return None;
                }
            }
        };

        // 2. Push-constant range: full 128-byte budget.
        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_CONSTANT_RANGE_BYTES as u32)];

        // 3. Pipeline layout.
        let set_layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_range);
        // SAFETY: descriptor_set_layout is valid; ash_device is live.
        let pipeline_layout = unsafe {
            match ash_device.create_pipeline_layout(&layout_info, None) {
                Ok(l) => l,
                Err(e) => {
                    log::error!("vkCreatePipelineLayout failed for '{}': {e}", key.shader);
                    // Already in an outer unsafe block — no nested unsafe needed.
                    ash_device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    return None;
                }
            }
        };

        // 4. Shader module.
        // SPIR-V bytes must be u32-aligned — copy into an aligned vec.
        let spirv_u32 = spirv_bytes_to_u32(spirv);
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spirv_u32);
        // SAFETY: spirv_u32 is valid SPIR-V.
        let shader_module = unsafe {
            match ash_device.create_shader_module(&shader_info, None) {
                Ok(m) => m,
                Err(e) => {
                    log::error!("vkCreateShaderModule failed for '{}': {e}", key.shader);
                    // Already in outer unsafe block.
                    ash_device.destroy_pipeline_layout(pipeline_layout, None);
                    ash_device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    return None;
                }
            }
        };

        // 5. Specialization constants.
        // `build_spec_info_data` returns the raw storage (entries + data bytes). We construct
        // VkSpecializationInfo here so it borrows from the storage, which lives in this scope.
        let spec_storage = build_spec_info_data(&key.spec_constants);
        let spec_info_holder = spec_storage.as_ref().map(|(entries, data)| {
            vk::SpecializationInfo::default()
                .map_entries(entries)
                .data(data)
        });

        // 6. Compute pipeline.
        let stage_builder = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        let stage = if let Some(ref si) = spec_info_holder {
            stage_builder.specialization_info(si)
        } else {
            stage_builder
        };

        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)];

        // SAFETY: pipeline_info is valid; vk_cache and ash_device are live.
        let pipelines = unsafe {
            match ash_device.create_compute_pipelines(vk_cache, &pipeline_info, None) {
                Ok(p) => p,
                Err((_, e)) => {
                    log::error!("vkCreateComputePipelines failed for '{}': {e}", key.shader);
                    // Already in outer unsafe block.
                    ash_device.destroy_shader_module(shader_module, None);
                    ash_device.destroy_pipeline_layout(pipeline_layout, None);
                    ash_device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    return None;
                }
            }
        };
        let pipeline = pipelines[0];

        // Shader module is only needed during pipeline creation; destroy it now.
        // SAFETY: shader_module is no longer needed after the pipeline is created.
        unsafe { ash_device.destroy_shader_module(shader_module, None) };

        Some(PipelineEntry {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            binding_count,
        })
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        for (_, entry) in self.entries.drain() {
            // SAFETY: these handles were created by us and are no longer in use (the EP
            // session is tearing down).
            unsafe {
                self.ash_device.destroy_pipeline(entry.pipeline, None);
                self.ash_device
                    .destroy_pipeline_layout(entry.pipeline_layout, None);
                self.ash_device
                    .destroy_descriptor_set_layout(entry.descriptor_set_layout, None);
            }
        }
        // SAFETY: vk_cache was created by us.
        unsafe { self.ash_device.destroy_pipeline_cache(self.vk_cache, None) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Descriptor pool (per-dispatch pool-and-reset model)
// ──────────────────────────────────────────────────────────────────────────────

/// A descriptor pool sized for one dispatch.
///
/// Allocated, used for one `vkCmdBindDescriptorSets`, then reset after the fence signals.
/// This is the simplest correct descriptor management: allocate one set per dispatch from a
/// dedicated pool, reset the whole pool after the GPU drains. No free-list, no recycling.
///
/// M2+ replaces this with a persistent pool and per-frame recycling when pipelined
/// multi-subgraph scheduling arrives.
pub(crate) struct DispatchDescriptorPool {
    ash_device: ash::Device,
    pool: vk::DescriptorPool,
}

impl DispatchDescriptorPool {
    /// Create a pool capable of allocating one descriptor set with up to `max_bindings` storage
    /// buffer descriptors.
    ///
    /// # Safety
    /// `ash_device` must be live for the lifetime of this pool.
    pub(crate) unsafe fn new(ash_device: &ash::Device, max_bindings: u32) -> Option<Self> {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: max_bindings,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        // SAFETY: ash_device is live; pool_info is valid.
        let pool = unsafe {
            match ash_device.create_descriptor_pool(&pool_info, None) {
                Ok(p) => p,
                Err(e) => {
                    log::error!("vkCreateDescriptorPool failed: {e}");
                    return None;
                }
            }
        };
        // Counted at the vkCreateDescriptorPool call site, not at the caller (issue #88): this
        // counts pools that were actually created, so a caller that starts reusing one makes the
        // number fall without anything else being edited. Counted only on success — a failed
        // create allocated nothing.
        crate::counters::record_descriptor_pool_created();
        Some(DispatchDescriptorPool {
            ash_device: ash_device.clone(),
            pool,
        })
    }

    /// Allocate one descriptor set for the given layout, then write storage-buffer descriptors
    /// for each buffer in `buffers`.
    ///
    /// Returns the allocated descriptor set, or `None` on failure.
    ///
    /// # Safety
    /// - `layout` must be valid and compatible with the number of bindings in `buffers`.
    /// - Each `(buffer, size)` in `buffers` must be a valid `(VkBuffer, byte_range)`.
    pub(crate) unsafe fn allocate_and_write(
        &self,
        layout: vk::DescriptorSetLayout,
        buffers: &[(vk::Buffer, u64)],
    ) -> Option<vk::DescriptorSet> {
        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);
        // SAFETY: pool and layout are valid; ash_device is live.
        let sets = unsafe {
            match self.ash_device.allocate_descriptor_sets(&alloc_info) {
                Ok(s) => s,
                Err(e) => {
                    log::error!("vkAllocateDescriptorSets failed: {e}");
                    return None;
                }
            }
        };
        let set = sets[0];
        // Counted at the vkAllocateDescriptorSets call site (issue #88). `descriptor_writes`
        // counts the individual buffer descriptors written into it, because "355 sets per warm
        // call" and "355 descriptors per warm call" are very different problems and the fix for
        // one is not the fix for the other.
        crate::counters::record_descriptor_set_allocated(buffers.len() as u64);

        // Write one VkDescriptorBufferInfo + VkWriteDescriptorSet per binding.
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|&(buf, size)| vk::DescriptorBufferInfo {
                buffer: buf,
                offset: 0,
                range: size,
            })
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        // SAFETY: writes reference buffer_infos which are valid for this call.
        unsafe { self.ash_device.update_descriptor_sets(&writes, &[]) };

        Some(set)
    }

    /// Reset the pool, freeing all descriptor sets allocated from it.
    ///
    /// Called after the GPU fence signals (i.e., the descriptor set is no longer in use).
    ///
    /// # Safety
    /// All descriptor sets allocated from this pool must have completed on the GPU.
    pub(crate) unsafe fn reset(&self) {
        // SAFETY: pool is valid; all sets are done per caller's contract.
        unsafe {
            if let Err(e) = self
                .ash_device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
            {
                log::error!("vkResetDescriptorPool failed: {e}");
            }
        }
    }
}

impl Drop for DispatchDescriptorPool {
    fn drop(&mut self) {
        // SAFETY: pool was created by us; all allocated sets are freed by reset() before drop.
        unsafe { self.ash_device.destroy_descriptor_pool(self.pool, None) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Convert a byte slice of SPIR-V into a `Vec<u32>`, padding with zeros if not aligned.
fn spirv_bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    let words = bytes.len().div_ceil(4);
    let mut out = vec![0u32; words];
    // SAFETY: dst has the right capacity and alignment; src overlaps is impossible.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast::<u8>(), bytes.len());
    }
    out
}

/// Build specialization constant raw data: map entries + packed u32 bytes.
///
/// The caller constructs `VkSpecializationInfo` from these, so the storage lives in the
/// caller's scope and can be borrowed by the ash struct.
fn build_spec_info_data(
    spec_constants: &[u32],
) -> Option<(Vec<vk::SpecializationMapEntry>, Vec<u8>)> {
    if spec_constants.is_empty() {
        return None;
    }
    let map_entries: Vec<vk::SpecializationMapEntry> = spec_constants
        .iter()
        .enumerate()
        .map(|(i, _)| vk::SpecializationMapEntry {
            constant_id: i as u32,
            offset: (i * 4) as u32,
            size: 4,
        })
        .collect();

    let data: Vec<u8> = spec_constants
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();

    Some((map_entries, data))
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_key_equality() {
        let a = PipelineKey {
            shader: "add_f32",
            spec_constants: vec![64, 1, 1],
        };
        let b = PipelineKey {
            shader: "add_f32",
            spec_constants: vec![64, 1, 1],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn pipeline_key_different_spec_constants_not_equal() {
        let a = PipelineKey {
            shader: "add_f32",
            spec_constants: vec![64],
        };
        let b = PipelineKey {
            shader: "add_f32",
            spec_constants: vec![128],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn spirv_bytes_to_u32_aligns_correctly() {
        // 8 bytes → 2 u32s.
        let bytes: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let words = spirv_bytes_to_u32(&bytes);
        assert_eq!(words.len(), 2);

        // 5 bytes → padded to 2 u32s.
        let bytes2: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        let words2 = spirv_bytes_to_u32(&bytes2);
        assert_eq!(words2.len(), 2);
        // The last 3 bytes of words2[1] must be zero-padded.
        assert_eq!(words2[1] & 0xFFFF_FF00, 0);
    }

    #[test]
    fn build_spec_info_data_empty_returns_none() {
        let result = build_spec_info_data(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn build_spec_info_data_nonempty_has_correct_entries() {
        let constants = vec![64u32, 1u32, 256u32];
        let result = build_spec_info_data(&constants);
        assert!(result.is_some());
        let (entries, data) = result.unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(data.len(), 12); // 3 × 4 bytes
        // Verify map entry offsets are sequential.
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[1].offset, 4);
        assert_eq!(entries[2].offset, 8);
        // Verify constant IDs are sequential.
        assert_eq!(entries[0].constant_id, 0);
        assert_eq!(entries[1].constant_id, 1);
        assert_eq!(entries[2].constant_id, 2);
    }
}
