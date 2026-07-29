# onnxruntime-ep-vulkan

A **cross-platform Vulkan compute execution provider for ONNX Runtime**, written in Rust and
shipped as an out-of-tree **plugin EP**. It is loaded by a stock, unmodified ONNX Runtime through
the plugin-EP C ABI — no ORT fork, no ORT rebuild, no link against `libonnxruntime`.

> **Status: early implementation.** The architecture is settled and written down; the crate
> scaffolding, the test harness and CI have landed and M0 is in progress. See the milestone plan in
> [`docs/DESIGN.md`](docs/DESIGN.md) §10.

| | |
|---|---|
| Registered EP name | `VulkanExecutionProvider` |
| Artifact | `libonnxruntime_vulkan_ep.so` / `onnxruntime_vulkan_ep.dll` / `libonnxruntime_vulkan_ep.dylib` |
| ORT ABI | plugin-EP C ABI · built against ONNX Runtime **1.28** · minimum runtime API **1.24** |
| Backend | Vulkan compute · GLSL → SPIR-V · [`ash`](https://github.com/ash-rs/ash) |
| Target platforms | Windows · Linux · Android · macOS (MoltenVK) — *Windows and Linux are exercised in CI; the Android and macOS paths are designed for but not yet validated on physical hardware* |
| Device requirement | **Vulkan 1.1 core + a compute queue.** No required extensions — see [`docs/DESIGN.md`](docs/DESIGN.md) §7. |
| Target hardware | NVIDIA · AMD · Intel · Adreno · Mali, plus lavapipe for GPU-less CI (Linux **and** Windows) |
| Operator domains | `ai.onnx` and `com.microsoft` — the contrib domain is in scope because the ORT GenAI model builder emits contrib ops directly; see [`docs/DESIGN.md`](docs/DESIGN.md) §1.4 for the claim-safety constraints |

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
| `docs/OP_COVERAGE.md` | Mouse | **Authoritative op-coverage plan** — 174 ops across 16 families, driven by model families (LLM/Qwen3.5 → int4 → MoE → multimodal → linear attention → conv), with model-level exit criteria per tier. Ratified by Morpheus 2026-07-28. |
| `docs/THIRD_PARTY.md` | Rai | Third-party licence compliance and attribution requirements for adapted code. |
| `docs/OP_ARCHITECTURE.md` | Mouse | Op registry design and the per-op claim contract. *(forthcoming)* |
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
