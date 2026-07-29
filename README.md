# onnxruntime-ep-vulkan

A **cross-platform Vulkan compute execution provider for ONNX Runtime**, written in Rust and
shipped as an out-of-tree **plugin EP**. It is loaded by a stock, unmodified ONNX Runtime through
the plugin-EP C ABI — no ORT fork, no ORT rebuild, no link against `libonnxruntime`.

> **Status: pre-implementation.** The architecture is settled and written down; no Rust code has
> landed yet. See the milestone plan in [`docs/DESIGN.md`](docs/DESIGN.md) §10.

| | |
|---|---|
| Registered EP name | `VulkanExecutionProvider` |
| Artifact | `libonnxruntime_vulkan_ep.so` / `onnxruntime_vulkan_ep.dll` / `libonnxruntime_vulkan_ep.dylib` |
| ORT ABI | plugin-EP C ABI, `ORT_API_VERSION 27` (ONNX Runtime 1.27.x) |
| Backend | Vulkan compute · GLSL → SPIR-V · [`ash`](https://github.com/ash-rs/ash) |
| Target platforms | Windows · Linux · Android · macOS (MoltenVK) |
| Target hardware | NVIDIA · AMD · Intel · Adreno · Mali, plus lavapipe / SwiftShader for GPU-less CI |

## How it works

```
ONNX graph
   └─ GetCapability   claim only node forms the Vulkan translator implements exactly
   └─ fuse            maximal convex connected clusters
   └─ Compile         build a plan · prepack + upload constant weights once · warm pipelines
   └─ Compute         record (or replay) one command buffer · one submission · fence
Everything unclaimed runs on ORT's CPU EP.
```

Conservative claiming with clean CPU fallback is a hard requirement, not a stopgap: an unclaimed
op is always correct, and a wrongly-claimed one is silently wrong.

## Intended usage

```python
import onnxruntime as ort
import onnxruntime_ep_vulkan

onnxruntime_ep_vulkan.register_execution_provider_library()
sess = ort.InferenceSession(
    model,
    providers=["VulkanExecutionProvider", "CPUExecutionProvider"],
)
```

## Documentation

| Document | Owner | Contents |
|---|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Morpheus | **Architecture of record.** Goals and non-goals, ORT integration, crate layout, module boundaries, execution flow, tensor/memory model, Vulkan baseline, op strategy, testing, milestones, open questions. |
| [`docs/ENGINE.md`](docs/ENGINE.md) | Switch | Vulkan runtime: device and context, memory and allocators, shader strategy, pipelines and descriptors, command submission and synchronization. |
| [`docs/PLATFORMS.md`](docs/PLATFORMS.md) | Link | Platform and driver support matrix, Vulkan version reality per platform, capability detection, toolchains, CI lanes. |
| `docs/OP_ARCHITECTURE.md` | Mouse | Op registry design and the authoritative op-coverage contract. *(forthcoming)* |
| `docs/BENCHMARKS.md` | Niobe | Benchmark methodology and published baselines. *(forthcoming)* |

## Design in one paragraph

This project deliberately mirrors the architecture of
[`onnxruntime-mlx`](https://github.com/justinchuby/onnxruntime-mlx), a proven ONNX Runtime plugin
EP, with Vulkan replacing MLX as the backend. The registry, claim/fuse/compile/run pipeline, module
split, FFI ownership model, and repository layout are the same. The one structural difference that
drives everything else: MLX runs on Apple unified memory, while Vulkan has explicit, non-unified
device memory — so this EP owns an allocator, a data-transfer implementation, weight prepacking,
and a real Vulkan engine layer that MLX supplied for free. Every deliberate divergence is
enumerated in [`docs/DESIGN.md`](docs/DESIGN.md) §12.

## License

MIT — see [`LICENSE`](LICENSE).
