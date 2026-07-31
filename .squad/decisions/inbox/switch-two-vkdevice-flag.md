# Two-VkDevice architectural question — formal flag

**Date**: 2026-07-30  
**Author**: Switch  
**For**: Coordinator → Morpheus

## The situation

`alloc_device_authoritative_spans` is 0 and will remain 0 under the current architecture. The reason is structural, not a code gap:

- **Tank's device-backed memory provider** (`rust/src/vk/host_device_memory.rs`) creates its own `VkDevice` — a full second Vulkan logical device on the same physical GPU, separate from the compute session's `VkDevice`.
- **The compute session** (`rust/src/vk/session.rs`) creates and owns its own `VkDevice` for command buffers, pipelines, descriptor sets, and now the weight-tensor GPU buffer cache.
- A `VkBuffer` allocated on device A cannot be bound or accessed from device B. These are separate logical device contexts sharing the same physical hardware but with no cross-device buffer visibility at the Vulkan API level.

## The architectural question

**Two `VkDevice`s in one process for the same physical GPU is a design decision nobody made deliberately.** It emerged from two engineers solving independent subproblems (Tank: ORT allocator integration; Switch: Vulkan compute session) without a shared device ownership plan.

Consequences:
- `alloc_device_authoritative_spans` cannot be non-zero without either sharing the `VkDevice` between Tank's allocator and the session, or importing/exporting memory via external memory handles (`VK_KHR_external_memory`).
- The weight-tensor GPU buffer cache (my work) is correct but session-local — it eliminates the re-upload cost but does not use Tank's ORT-allocator-backed buffers.
- `alloc_device_backed_shared_with_engine` is explicitly 0 and the counter documents this gap.

## What I am NOT doing

Not solving this unilaterally. The options (share device, external memory import, accept the gap) each have correctness and lifecycle implications that span Tank's and my code. This needs Morpheus's ruling on the intended architecture.

## What Morpheus needs to decide

1. Should the session and Tank's allocator share a `VkDevice`? If yes, which owns it and what is the teardown order?
2. Or should they remain separate, with the session's weight-tensor cache filling the persistent-residency role and `alloc_device_authoritative_spans` being structurally 0 for this milestone?
3. Is external memory handle import the right path for M1+?

The weight-tensor GPU buffer cache (`cdcc349`) already captures most of the persistent-residency benefit (2642× reduction in per-warm-inference upload bytes) without needing device sharing. The counter gap is a real architectural gap, but it does not represent an unaddressed performance problem at this milestone.
