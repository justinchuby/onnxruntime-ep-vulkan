# onnxruntime-ep-vulkan

[![CI](https://github.com/justinchuby/onnxruntime-ep-vulkan/actions/workflows/ci.yml/badge.svg)](https://github.com/justinchuby/onnxruntime-ep-vulkan/actions/workflows/ci.yml)

> **CI is the only place shaders execute.** A red CI lane means we have zero empirical
> evidence that the EP works. Before landing a commit: run `gh run list --limit 5` and
> confirm the badge above is green. A red badge blocks all merges — not by a GitHub branch
> protection rule (not yet configured), but by team discipline enforced by this notice.
> See [`.github/CI_POLICY.md`](.github/CI_POLICY.md) for the full policy.

A **cross-platform Vulkan compute execution provider for ONNX Runtime**, written in Rust and
shipped as an out-of-tree **plugin EP**. It is loaded by a stock, unmodified ONNX Runtime through
the plugin-EP C ABI — no ORT fork, no ORT rebuild, no link against `libonnxruntime`.

> **Status: early implementation.** The architecture is settled and written down; the crate
> scaffolding, the test harness and CI have landed and M0 is in progress. **M0 is not met.** As of
> 2026-07-29 a single `Add` node **is claimed by the EP and executes on the GPU through ONNX
> Runtime**, on two desktop GPUs (Intel Iris Xe, NVIDIA RTX 4060), with `VulkanExecutionProvider` in
> the profiling JSON and output matching the ORT CPU EP. That is the first result traversing the
> whole stack, and its scope is exactly one op, **fp32, static shapes, one OS, two devices**.
> Everything else — every other kernel, every contrib op, all quantized paths, every fused
> multi-node island, and the legacy barrier backend — is unexecuted; CI has no GPU hardware; and on
> real hardware the differential suite currently reports 125 failures, every one of them the harness
> refusing to score a CPU-fallback run as a pass. M0 also requires this on Windows **and** Linux, on
> a software rasterizer, in CI, under both barrier backends — none of which is done. See
> [`docs/DESIGN.md`](docs/DESIGN.md) §9.1.2 for the full execution status and §10 for the
> criterion-by-criterion M0 assessment.

| | |
|---|---|
| Registered EP name | `VulkanExecutionProvider` |
| Artifact | `libonnxruntime_vulkan_ep.so` / `onnxruntime_vulkan_ep.dll` / `libonnxruntime_vulkan_ep.dylib` |
| ORT ABI | plugin-EP C ABI · built against ONNX Runtime **1.28** · minimum runtime API **1.24** |
| Backend | Vulkan compute · GLSL → SPIR-V · [`ash`](https://github.com/ash-rs/ash) |
| Target platforms | `Windows · Linux · Android · macOS (MoltenVK)` — **Windows and Linux are verified in CI (lavapipe). The Android and macOS paths are designed for but have zero CI coverage on physical hardware; every Android and macOS entry in [`docs/PLATFORMS.md`](docs/PLATFORMS.md) §5 is explicitly marked untested.** |
| Device requirement | **Vulkan 1.1 core + a compute queue.** No required extensions — see [`docs/DESIGN.md`](docs/DESIGN.md) §7. |
| Target hardware | NVIDIA · AMD · Intel · Adreno · Mali — *none of these is covered by CI today; the only verified lanes are the lavapipe software rasterizer on Linux and Windows* |
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

## Building

**Prerequisite: the Vulkan SDK, or `glslc` on `PATH`.** Shaders are compiled from GLSL to SPIR-V at
build time and embedded in the library; there is no checked-in SPIR-V, deliberately — a checked-in
binary that drifts from its source would silently change what runs
([`docs/DESIGN.md`](docs/DESIGN.md) §7.8). A build without `glslc` fails with a message naming what
to install.

```powershell
cargo build --release        # from rust/
cargo test                   # lib + layering + capability-dump tests
```

`ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1` builds a **shader-less** artifact for lint-only and
docs-only lanes. It can create no compute pipeline, it advertises no device, and it must never be
shipped.

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
