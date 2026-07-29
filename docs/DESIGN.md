# onnxruntime-ep-vulkan — Architecture Design

**Status:** v0 architecture of record — accepted for M0/M1 implementation. **§7 (Vulkan baseline) is frozen.**
**Date:** 2026-07-28T17:59:54-07:00 · **Last revised:** 2026-07-28T22:28:08-07:00 (**OQ-4 resolved — §7.8**, SDK is a hard build prerequisite; **OQ-M6 accelerant ruling — estimates hold**, §8.4; **OQ-3 resolved — §6.4**, reserved-VA handle registry, no BDA; C2 shape confirmed + release-gate; `retain_viable` placement fixed in §5.4; eleven contrib ops; OQ-16 raised; **quantized-path oracle empirically validated — §9.1.1**, and **§9.1.2 execution-status disclosure**: no shader has yet run on any device)
**Author:** Morpheus (Lead / EP Architect)
**Repo:** `onnxruntime-ep-vulkan`
**Reference architecture:** `onnxruntime-mlx` (Justin Chu's MLX plugin EP for Apple Silicon)
**Sibling documents:** [`ENGINE.md`](./ENGINE.md) (Switch — Vulkan runtime & shaders), [`PLATFORMS.md`](./PLATFORMS.md) (Link — platform & hardware matrix), [`OP_COVERAGE.md`](./OP_COVERAGE.md) (Mouse — **authoritative op-coverage plan**, ratified), `THIRD_PARTY.md` (Rai — licence compliance)

---

## 0. TL;DR

`onnxruntime-ep-vulkan` is an **out-of-tree ONNX Runtime plugin Execution Provider** that runs
fused ONNX subgraphs on any Vulkan compute device. It is loaded by a **stock, unmodified** ORT
build through the plugin-EP C ABI. No ORT fork, no ORT rebuild, no link against
`libonnxruntime`.

| Field | Value |
|---|---|
| Repository / vendor string | `onnxruntime-ep-vulkan` |
| Cargo crate | `rust/` — `onnxruntime-ep-vulkan` |
| Library artifact | `libonnxruntime_vulkan_ep.so` / `onnxruntime_vulkan_ep.dll` / `libonnxruntime_vulkan_ep.dylib` |
| Registered EP / device name | **`VulkanExecutionProvider`** |
| Crate type | `cdylib` |
| ORT ABI | plugin-EP C ABI. **Built against ORT 1.28** (`ORT_API_VERSION`); **minimum runtime API 1.24** (`ORT_API_VERSION_MIN = 24`) with version negotiation. ORT 1.28 fixes a null-allocator PrePack crash that would have hit us on 1.27; 1.24 is where the `OrtEpFactory` surface we rely on stabilized. |
| Version scheme | `0.<ORT_API_VERSION>.<patch>` → `0.28.0` |
| Backend | Vulkan compute, GLSL → SPIR-V, `ash` bindings |
| **Device requirement** | **Vulkan 1.1 core + a compute queue + four limits (§7.2).** No required extensions. Everything else is probed and degrades op coverage, never device availability. |

The architecture is **deliberately the same shape as `onnxruntime-mlx`**: a registry-driven,
claim → fuse → compile → run plugin EP with conservative node claiming and clean CPU fallback.
Every module in `onnxruntime-mlx` has a counterpart here. Where we diverge, §12 records the
divergence and the reason.

**The single biggest divergence from the MLX reference:** MLX runs on Apple unified memory, so
the MLX EP advertises *no device allocator* and copies out with one `memcpy` at the subgraph
boundary. Vulkan has **explicit, non-coherent, non-unified device memory**. That reshapes the
tensor/memory contract (§6), the factory surface (§2.5), the compile step (weight prepacking,
§5), and the milestone plan (§10). It is the reason this document exists rather than a
find-and-replace of the MLX design.

---

## 1. Goals and non-goals

### 1.1 Goals for v1

1. **Cross-platform GPU inference from one codebase.** Windows, Linux, Android, and macOS
   (MoltenVK) on NVIDIA / AMD / Intel / Adreno / Mali, plus the **lavapipe** software rasterizer so
   CI is possible without a GPU runner. No vendor-specific code path is
   permitted to be load-bearing for correctness.
2. **Zero ORT fork.** Ship a single shared library that a stock ORT loads via
   `RegisterExecutionProviderLibrary`.
3. **Correctness before performance.** The ORT CPU EP is the oracle. A claimed op must match it
   within a stated tolerance on every supported platform before we quote a single speedup number.
4. **Conservative claiming with clean CPU fallback.** Claim only node forms whose exact
   dtype / attribute / shape / layout contract the Vulkan translator implements. Everything else
   runs on ORT CPU. Falling back is a feature, not a gap.
5. **Compile-once, replay-many execution.** Weight upload and prepacking happen at `Compile`
   time; per-inference work is command-buffer submission, not graph construction.
6. **A layering that survives contact with contributors.** The ORT C ABI never reaches op code;
   raw Vulkan handles never reach op code. Enforced by module privacy, not by review vigilance.

### 1.2 Non-goals for v1 — explicit and ruthless

These are **out of scope**. Each is a decision, not an oversight. Re-opening any of them requires
a decision record.

> **SUPERSEDED IN PART BY USER RULING, 2026-07-28T20:54:42-07:00.** Justin ruled directly:
> **「contrib op 要做」 — the `com.microsoft` contrib operator domain is in scope.** That decision is
> settled and is not mine to re-open; the `ai.onnx`-only position originally stated in this section
> is void. Record: `.squad/decisions/inbox/copilot-directive-contrib-ops.md`. What remains mine, and
> what §1.4 now contains, is **how we admit it safely** — claim-predicate discipline, the schema-drift
> protocol, and clean CPU fallback. Admission settled; constraints binding.

> **Amended 2026-07-28T19:16:08-07:00 by the ratification of [`OP_COVERAGE.md`](./OP_COVERAGE.md)
> (OQ-11).** Four rows below were reversed. The reversal is not a loosening of ambition-control; it
> is a correction of a factual error on my part. I wrote these rows believing that "run a Qwen
> graph" and "support the `com.microsoft` domain" were separable. Mouse verified from the ONNX
> Runtime GenAI model builder source that they are not: the builder **emits** contrib ops directly,
> so an EP that declines `com.microsoft` cannot run a Qwen graph at all — not slowly, not partially,
> at all. A non-goal that makes the project's named target unreachable is not ruthless, it is wrong.
> The reversed rows are marked **REVERSED** below and the constraints that now attach to them are in
> §1.4.

| Non-goal | Why |
|---|---|
| **Training / gradients** | ORT training EPs are a different ABI surface and a different correctness problem. Inference only. |
| **ONNX opset completeness** | Still a non-goal *as stated* — we do not chase the spec index. But see §8: coverage is now driven to ~174 inventoried ops by model family, and the reason the original row gave ("Vulkan supplies nothing — every op is a shader we write") is answered by kernel-template leverage, not by refusing breadth. What remains a non-goal is claiming ops **no target graph contains**. |
| **Dynamic shapes in the fast path (M0–M2)** | **PARTIALLY REVERSED.** Still a non-goal for M0–M2 generally. But **LLM-path kernels take their dimensions in push constants from tier 3 onward**, so a recorded command buffer is length-agnostic for the decode loop. This is structural, not an optimization: KV length grows every token. See OQ-M1 / §1.4. |
| **Data-dependent output shapes** | `NonZero`, `Unique`, value-dependent `Reshape` targets, `NonMaxSuppression`. These need a mid-graph host readback that a recorded command buffer cannot express. Permanent CPU fallback. Mouse inventoried and permanently declined these. |
| **fp64** | Most consumer GPUs have no usable double precision and Vulkan makes `shaderFloat64` optional. Permanent CPU fallback. |
| **Quantized ops (int4/int8 matmul, `MatMulNBits`, `GatherBlockQuantized`)** | **REVERSED — in scope, tier 4 (M3).** An int4 Qwen graph is the variant people actually run; without `MatMulNBits` it shatters into hundreds of islands and the EP is worse than useless on it. Constraints in §1.4. |
| **Attention fusion (GQA / MHA / SDPA / flash attention)** | **REVERSED for `GroupQueryAttention` — in scope, tier 3 (M2/M3).** My original row conflated two things. Performing a fusion is out of scope. `GroupQueryAttention` is not a fusion we perform: it **arrives as a single node from the exporter**, and decomposing it would materialize a `[B,H,S,S]` score matrix in VRAM — gigabytes at S=4096. Implementing it is the *conservative* choice. Fusion patterns we would have to *detect* remain out of scope. |
| **Graph-level op fusion** | Unchanged non-goal for patterns we must *detect* (llama.cpp's `MUL_MAT+ADD`, `RMS_NORM+MUL`). Note this is distinct from a single ONNX node whose semantics are internally fused (`Softmax`, `LayerNormalization`, `SkipSimplifiedLayerNormalization`, GQA) — implementing those as one kernel is not graph fusion, it is implementing the op. |
| **Mobile-first tuning** | Android must *work* (M3) and must not be architecturally excluded (§7). Tile sizes, memory budgets, and Adreno/Mali-specific tuning are not v1. |
| **Images / texture-backed tensors** | Buffers only in v1 (see `ENGINE.md` §3.6). Named trigger for re-evaluation: `Conv` at tier 5c/6. |
| **Multi-GPU / multi-queue overlap** | One `VkDevice`, one compute queue, one submission per subgraph execution. |
| **Cooperative matrix / tensor-core paths** | Optional extension on every baseline. Capability-probed later, never required. |
| **Custom / contrib domain ops (`com.microsoft`)** | **REVERSED — NO LONGER A NON-GOAL. `com.microsoft` is in scope by user ruling, 2026-07-28T20:54:42-07:00.** The nine ops needed for the Qwen3.5 target are enumerated in §1.4 and are the *admitted set for v1*; admitting a tenth is now a scoping decision within an in-scope domain (a decision record, not a re-opened non-goal). The engineering discipline of §1.4 — per-op claim predicates, no domain-wide opt-in in code, census-in-CI — is unchanged and binding, because it was never about whether the domain was permitted; it is about not claiming a schema we have not read. |
| **Shipping wheels on PyPI in v1** | The Python package exists for testing from M0. Publishing is a release decision, not an architecture one. |

### 1.3 Why "conservative claiming" is a hard requirement, not a preference

Because the fallback is not free but it *is* always correct. The failure mode we are designing
against is not "we didn't claim enough ops" — it is "we claimed a node form our shader gets
subtly wrong on one driver, and a user gets silently wrong logits." Every claim predicate is a
promise. The rule from the MLX reference stands verbatim: **when in doubt, do not claim.**

Nothing in the ratified coverage plan relaxes this. A faster schedule changes *how many* ops land
and *in what order*; it does not change what claiming means. If anything, the higher the op
throughput, the more load-bearing this rule becomes.

### 1.4 `com.microsoft` contrib ops — in scope; the constraints that attach

**Scope status: settled by user ruling, 2026-07-28T20:54:42-07:00.** The `com.microsoft` domain is
in scope. The deciding fact, verified by Mouse from the ORT GenAI model builder source, is that the
builder *emits* contrib ops directly, so an EP that declines the domain cannot run a Qwen graph at
all — and Qwen3.5 is a named target. I am not re-litigating that here.

**The admitted set for v1 is these eleven.** Nine were admitted on 2026-07-28T19:16:08-07:00 as the
ops a Qwen3.5-class GenAI-built graph actually contains:

`GroupQueryAttention`, `RotaryEmbedding`, `SimplifiedLayerNormalization`,
`SkipSimplifiedLayerNormalization`, `MatMulNBits`, `LinearAttention`, `CausalConvWithState`,
`QMoE`, `GatherBlockQuantized`.

**Two more are ratified here, 2026-07-28T22:28:08-07:00**, both staged by Mouse in `5ae991a` and
both accepted — this is the "a tenth requires a decision record" clause working as intended, and the
record is this paragraph:

- **`MultiHeadAttention`** — the non-GQA fused attention form that ViT and BERT-style encoders
  export. Admitted for the same reason as `GroupQueryAttention` and not by analogy to it: it
  *arrives* as a single node, and decomposing it materializes a `[B,H,S,S]` score matrix. It is
  tier 5c's vision-tower dependency (§10 M3+), and it should share `ops/attention.rs`'s kernel with
  GQA rather than becoming a twelfth thing to write.
- **`MoE`** — the float-expert sibling of `QMoE`. Admitted because a routing implementation that
  only exists in its quantized form cannot be differentially tested against a float oracle, which
  makes `QMoE` much harder to verify, not easier. Cheaper to carry both than to debug one.

Adding a twelfth is the same clause again: a decision record and a tier assignment, inside an
in-scope domain. What follows is what I still own, and it matters *more* now
that the domain is admitted, not less. Contrib ops are a genuinely more dangerous surface than
`ai.onnx` and the discipline below is the whole reason admitting them is safe.

#### C1 — No domain-wide opt-in exists anywhere in the code

`com.microsoft` being in scope as a *product* decision must not become "we accept
`node.domain == "com.microsoft"`" as a *code* decision. There is no such predicate and there must
never be one. The registry is keyed by `(domain, op_type)` and an unregistered contrib op declines
through exactly the same path as an unregistered `ai.onnx` op. **The test that enforces this**: a
graph containing a fabricated `com.microsoft::NotARealOp` node must be declined with the ordinary
`NoHandler` reason and must run correctly on CPU — Trinity, as an M-tier regression test from the
first contrib op onward.

#### C2 — Contrib ops have no opset, so version-gating is by ORT release plus a drift alarm

This is the substantive difference from `ai.onnx` and the constraint everything else hangs off.
`ai.onnx` gives us a monotonic opset number we can range-check in the claim predicate; a schema
change is *visible* as a number we did not accept. Contrib schemas carry no such number — they are
versioned by ORT *release*, and several of the eleven (`LinearAttention`, `CausalConvWithState`,
`QMoE`, `MoE`) are new and still moving. A contrib schema change therefore does not bump anything we
can test against: it silently changes what our claim predicate *should* accept, while our predicate
goes on accepting what it always did.

The protocol, and it is a hard precondition on the first contrib op landing — not on tier 3
generally, on **op #1 of the eleven**:

1. **The ORT version is pinned per release** and recorded in `Cargo.toml`, `docs/`, and the CI
   matrix. Trinity has pinned 1.28. A contrib claim predicate is only ever validated against a
   pinned version.
2. **Each of the eleven records the ORT version its predicate was written against**, in the registry
   entry, next to the predicate. Not in a comment in a design document — in the table, where it can
   be printed by `--dump-capabilities` and diffed.
3. **`tools/graph_census.py` runs in CI against pinned `.onnx` artifacts** and reports per-op claim
   rate. This is the drift alarm: when a schema shifts under us, the census claim rate for that op
   moves, and CI fails on the delta rather than on a numerical comparison months later.
4. **On an ORT version bump, the census delta is a review gate.** A bump that changes any contrib
   claim rate does not merge until the owning op's predicate has been re-read against the new
   schema. The failure we are buying insurance against is the quiet one: a new optional attribute
   appears, our predicate does not know to reject it, and we claim a node whose semantics changed.
5. **When a schema changes shape under us, the correct response is to narrow the predicate and
   decline, not to guess.** Declining is a performance regression that CI reports as a claim-rate
   drop; guessing is wrong logits that nothing reports at all. This ordering is not negotiable and
   it is the contrib-specific restatement of §1.3.
6. **A contrib row whose baseline is not a released ORT version may not be flipped from `Staged` to
   `Live`.** *Added 2026-07-28T22:28:08-07:00 — see below.*

**C2's implemented shape is confirmed** (Mouse, `registry.rs`; Tank, `sys.rs`; both in `5ae991a`).
`SchemaBaseline` is nested *inside* `ContribSchema` and surfaced through `OpSpec::schema_baseline()`,
with build-failing tests requiring a baseline on every `com.microsoft` row and forbidding one on
every `ai.onnx` row. This is better than the flat side table I would have accepted, and the reason is
worth stating as a general rule: **the nested placement makes it impossible to record a schema shape
without recording where the shape came from.** A parallel table can be half-filled; a nested field
cannot. Tank arrived at the same requirement from `sys.rs` within the hour, they reconciled as *sys
owns the type, registry owns the data*, and he deleted his side table — the right call, because two
places recording the same fact is a hazard best fixed by deleting one rather than by testing that
they agree. Ratified as the C2 mechanism of record. The asymmetry (default-domain rows print
`n/a (opset-versioned)`) is also right: their compatibility contract *is* the opset window, and a
baseline there would dilute the signal on the rows where it matters.

**On Tank's 18-line cross-owner edit to `registry.rs`: ratify now, Mouse reviews after.** The shape
is ratified on its merits and does not depend on the wording of one replaced test. Blocking a
ratification on a review of a test that replaced a now-uncompilable test would stall four other
people to protect a low-risk change that is already green and already covered from Tank's side in
`layering.rs` (every contrib row has a baseline, no default-domain row does, no baseline claims a
release newer than the one we compile against). Mouse should still review it — it is his file, he
should own the final wording, and Tank asked for exactly that — but as a follow-up, not a gate.
Tank yielding placement to Mouse's design and *saying why it was better* rather than just yielding
is the behaviour I want between owners; that is how a boundary gets stronger instead of just being
defended.

**The main-branch-only schemas — ruling.** Four of the eleven (`LinearAttention`,
`CausalConvWithState`, `QMoE`, `MoE`) carry `MAIN_BASELINE` — `main (post-1.28.0)` — because they do
not appear in the pinned release at all. Recording that honestly is exactly what C2 is for, and
Mouse recording it rather than writing `1.28.0` across the board is the constraint doing its job on
its first real test. Three rulings follow:

1. **The situation is contained today and must stay contained by mechanism, not by intention.** All
   four rows are `OpStatus::Staged`, so `claim_decision` declines them with a machine-readable
   `[staged]` reason and the graph runs on CPU. Nothing is claimed against an unreleased schema, so
   there is no correctness exposure right now. But "we will remember not to flip these" is not a
   control, so: **C2 item 6 — a contrib row whose baseline is not a released ORT version may not be
   flipped from `Staged` to `Live`.** Enforce it the same way C1 is enforced, as a test rather than
   a convention: a row that is `Live` with a non-release baseline fails the build. This is the same
   pattern as A2 — convert the warning into a gate, because a warning is what gets dropped under
   schedule pressure.
2. **The registry test asserting agreement on the verification *date* while not comparing release
   strings is correct**, and I want the reasoning recorded rather than left as a compromise.
   Comparing release strings would force `MAIN_BASELINE` to lie in order to pass. The date is the
   fact that actually matters — *when did a human last read this schema* — and it is the only field
   both baselines can honestly share.
3. **The milestone consequence, which is real: `LinearAttention` and `CausalConvWithState` gate
   T5a, the named Qwen3.5 target, and they are targeting a schema that has never shipped.** That is
   a genuine schedule risk and it is not Mouse's to absorb quietly, so it is now **OQ-16** in §11:
   the release these ops first appear in, and whether their schema changed between our fingerprint
   and that release, is a tracked question with an owner. Practically, this means the T5a kernels
   may be written twice, and the fingerprints will need re-verification against the release the
   moment it exists. The correct report upward if that slips is *not* that Qwen3.5 slipped for
   Vulkan reasons — it is that we are gated on an upstream schema stabilizing, which is a different
   risk with different mitigations.

#### C3 — Contrib declines use the ordinary machine-readable decline path, never a special case

Every decline — contrib or not — emits Mouse's machine-readable reason through the same mechanism
(`ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1`, and the profiling-JSON claim assertion Trinity landed). No
contrib-specific logging path, no contrib-specific decline enum, no `if domain == "com.microsoft"`
anywhere in the diagnostics. Two reasons: a bespoke path is one more thing that can be wrong in the
place we look when something is already wrong; and the whole value of a uniform decline vocabulary
is that a user debugging a Qwen graph and a user debugging a ResNet read the same output. Any new
decline reason a contrib op needs (`UnsupportedAttributeValue`, `UnverifiedSchemaVersion`) is added
to the shared enum for everyone.

#### C4 — Every one of the eleven gets a hand-written claim predicate

Not the shared `caps`-column helper. Their attribute surface — `num_heads`, `kv_num_heads`,
`do_rotary`, `rotary_interleaved`, `scale`, `softcap`, `bits`, `block_size`, `accuracy_level`,
`g_idx`, the `LinearAttention` recurrence rule — is exactly where a silently-accepted-but-
unimplemented variant produces wrong output rather than a decline. The generated table still owns
their dtype/capability gating; the *semantic* gate is written by hand and reviewed by hand.

#### C5 — Claim only the configuration Trinity has a real artifact for

No claiming a `LinearAttention` recurrence rule we have never seen emitted. Mouse's UNVERIFIED-row
discipline (`OP_COVERAGE.md` §2.1) is ratified and extended: **an UNVERIFIED contrib-op row may not
be claimed at all**, not merely kept off exit criteria. An op moves from UNVERIFIED to VERIFIED by
`graph_census.py` finding it in a pinned artifact, at which point its real attribute values are
known and the predicate can be written against them instead of against the schema docs.

#### C6 — CPU fallback is the safety net and it must stay intact

ORT's own CPU implementation of every one of the eleven is the reference and the fallback. **A contrib
op we claim but get subtly wrong is strictly worse than one we decline** — the decline costs a
transfer boundary and shows up in `largest_island_flops`; the wrong claim costs correctness and
shows up nowhere. Concretely: the eleven are claimed incrementally, one at a time, each landing with
its differential test before the next begins; and **numerical verification for contrib ops is
per-layer, not final-logits**. Comparing final tokens on a 1.7B model against ORT CPU will pass with
a broken kernel far more often than anyone expects — sampling hides a great deal. Trinity owns the
per-layer mechanism; this is a binding constraint on the coverage plan, not a suggestion.

#### C7 — The XL kernels are budgeted as XL work, not as coverage

Three of them (`GroupQueryAttention`, `MatMulNBits`, `LinearAttention`) have no kernel-template
leverage and each gates a separate tier. They must never be counted in an op-coverage number as if
they were eleven of the 174. §10.0 and §1.5 state the schedule consequence honestly.

### 1.5 Two different claims, deliberately kept apart

Ratified from `OP_COVERAGE.md` §11.1 because it is the single most important thing in that document
for expectation-setting:

- **"High op coverage" — the ~121-op elementwise/shape/indexing/reduction/GEMM surface — is a
  weeks-scale goal**, contingent on the kernel-template infrastructure being built *before* op #1.
- **"Qwen3.5 runs end-to-end on Vulkan" is a months-scale goal**, gated on three XL kernels and on
  M2's device allocator.

The op count will look excellent long before any LLM runs. That gap is precisely where a coverage
project deceives itself, which is why the metric of record is `largest_island_flops`, not op count
(§9.2).

---

## 2. How it plugs into ONNX Runtime

### 2.1 The plugin-EP model

ORT exposes a public C ABI for registering an out-of-tree EP as a shared library. The host
resolves two symbols by name:

```c
OrtStatus* CreateEpFactories(const char* registered_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories,
                             size_t max_factories,
                             size_t* num_factories);

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);
```

`rust/src/lib.rs` exports both. ORT is reached **only** through the `OrtApi` function-pointer
table handed to `CreateEpFactories`; we never link `libonnxruntime`. Ownership crosses the C
boundary with `Box::into_raw` / `Box::from_raw`. Every `extern "C"` entry point that runs real
logic is wrapped in a panic guard that converts a Rust panic into an `ORT_EP_FAIL` status —
unwinding into ORT's C++ is undefined behaviour and a plugin must never take down its host.

Usage from the application side:

```python
import onnxruntime as ort
import onnxruntime_ep_vulkan

onnxruntime_ep_vulkan.register_execution_provider_library()
sess = ort.InferenceSession(model, providers=["VulkanExecutionProvider", "CPUExecutionProvider"])
```

### 2.2 Object lifecycle

```
dlopen / LoadLibrary
   └─ CreateEpFactories ────────────► VulkanEpFactory      (process-lived, one per registration)
        ├─ GetName / GetVendor / GetVendorId / GetVersion
        ├─ GetSupportedDevices(OrtHardwareDevice[]) ──────► OrtEpDevice[]   (device enumeration)
        ├─ CreateAllocator(OrtMemoryInfo) ────────────────► OrtAllocator    (device memory)
        ├─ CreateDataTransfer() ──────────────────────────► OrtDataTransferImpl
        └─ CreateEp(devices, session_options, logger) ────► VulkanEp        (one per session)
                ├─ GetName
                ├─ GetDefaultMemoryDevice ────────────────► OrtMemoryDevice
                ├─ GetCapability(OrtGraph, OrtEpGraphSupportInfo)      ← node claiming
                ├─ Compile(OrtGraph[], OrtNode[], OrtNodeComputeInfo[]) ← plan build + prepack
                │     └─ OrtNodeComputeInfo { CreateState, Compute, ReleaseState }  ← inference
                └─ ReleaseNodeComputeInfos
   └─ ReleaseEpFactory
```

The `VulkanEpFactory` struct embeds `OrtEpFactory` as its **first field** under `#[repr(C)]`, so
the pointer ORT holds is pointer-identical to our Rust struct at offset 0. Same for `VulkanEp` and
`OrtEp`, and for the per-subgraph compute-info object and `OrtNodeComputeInfo`. This is the exact
pattern proven in `onnxruntime-mlx/rust/src/factory.rs`.

### 2.3 Device enumeration — where Vulkan differs from MLX

The MLX EP's `GetSupportedDevices` picks *the first* `OrtHardwareDeviceType_GPU` ORT presents and
advertises exactly one `OrtEpDevice`. Apple Silicon has one GPU; that is sufficient there.

Vulkan does not have that luxury. The factory must:

1. Create a `VkInstance` (once per plugin load) and enumerate physical devices.
2. For each physical device, evaluate the **capability gate** (§7): does it meet the required
   feature set? If not, it is not advertised — an unusable device must never be offered to ORT.
3. Correlate each usable `VkPhysicalDevice` with the `OrtHardwareDevice` entries ORT presents.
   The correlation key is vendor ID + device ID, both of which appear in
   `VkPhysicalDeviceProperties` and in ORT's hardware-device metadata. Where correlation fails
   (software rasterizers, virtualized GPUs, MoltenVK), we fall back to type matching (GPU, then
   CPU) and record which strategy was used in the EP device metadata.
4. Create one `OrtEpDevice` per usable device via `EpApi::CreateEpDevice`, attaching EP metadata
   (Vulkan API version, device name, driver version, vendor) and EP options.

Consequences that follow, and which the MLX design never had to answer:

- **Device selection is a user-visible session option.** Multi-GPU machines are normal on Windows
  and Linux. `ep.device_index` selects among advertised devices; the default is the
  highest-scoring device (discrete > integrated > virtual > CPU), matching `ENGINE.md` §2.2.
- **`VkInstance` lifetime is factory-scoped, `VkDevice` lifetime is EP-scoped.** Two sessions on
  the same physical device share the instance but get independent logical devices, queues,
  allocators, and pipeline caches. Sharing a `VkDevice` across sessions is a post-v1
  optimization with real thread-safety cost; we do not take it now.
- **Enumeration must never abort the host.** A machine with no Vulkan loader, no ICD, or a broken
  driver must produce zero advertised devices and a warning — not a crash and not an error status
  that fails session creation. This is a tested requirement (Trinity, M0).

### 2.4 Session options

Prefixed `ep.` and read in `CreateEp` from the `OrtSessionOptions`. The v1 set is small on
purpose:

| Option | Type | Default | Meaning |
|---|---|---|---|
| `ep.device_index` | int | auto | Which advertised Vulkan device to bind. |
| `ep.enable_validation` | bool | `false` (release), `true` (debug) | Enable `VK_LAYER_KHRONOS_validation`. |
| `ep.pipeline_cache_path` | string | platform cache dir | On-disk `VkPipelineCache` blob location. |
| `ep.max_claim_ops` | string list | unset | Restrict claiming to a named op set. Debugging and bisecting only. |
| `ep.disable_device_memory` | bool | `false` | Force the M0 host-memory I/O path (see §6.3). Escape hatch for driver bugs. |
| `ep.force_legacy_barriers` | bool | `false` | Force the legacy `vkCmdPipelineBarrier` backend on a device that supports `synchronization2` (§7.5). Exists so CI exercises both barrier backends on the same hardware; also an escape hatch for a broken sync2 driver. |

Environment variables mirror the MLX EP's convention for observability, and are *not* a
configuration surface: `ONNXRUNTIME_EP_VULKAN_VERBOSE`, `ONNXRUNTIME_EP_VULKAN_TRACE=<path>`,
`ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG`, `RUST_LOG=onnxruntime_ep_vulkan=<level>`.

### 2.5 Node claiming (`GetCapability`) and subgraph compilation

**Claim.** For each node in the `OrtGraph`, `ep.rs` builds a `NodeView` (a read-only FFI wrapper
over `OrtNode` exposing op type, domain, since-version, input/output slot info, and attributes)
and asks the registry a single question: `claimable(&NodeView)`. There is **no per-op logic in
`ep.rs`**. This is the invariant that makes "claimed" and "translatable" impossible to
desynchronize, and it is inherited directly from the MLX reference.

**Fuse.** Claimed nodes are grouped into **maximal convex connected clusters** — the union-find +
reachability-bitset algorithm from the MLX EP. Convexity is not optional: a non-convex fusion
creates a cycle in the partitioned graph and ORT rejects it. Each cluster is handed to
`EpGraphSupportInfo_AddNodesToFuse`.

**Compile.** ORT calls `Compile` with one fused `OrtGraph` per cluster. For each we:
1. Extract a `NodeDesc` per node — op type, domain, since-version, generically-copied attributes
   (ints / floats / int arrays / float arrays / strings / tensors), and input/output tensor refs.
2. Build a `Plan`: the topologically ordered `NodeDesc` list plus the I/O binding table.
3. **Prepack**: read every constant initializer, convert it to the layout the shader wants, and
   upload it into a device-local buffer owned by the plan. This happens **once**. (§5.4)
4. Create or fetch the `VkPipeline` for every node in the plan, so the first inference does not
   pay shader compilation.
5. Hand ORT an `OrtNodeComputeInfo` that owns the plan.

**Run.** `Compute` binds ORT's input tensors, records (or replays) a command buffer, submits it,
waits on a fence, and makes the outputs visible to ORT.

### 2.6 CPU fallback

Unclaimed nodes are assigned to ORT's CPU EP by the ORT partitioner. ORT inserts the required
memcpy nodes at partition boundaries, using the `OrtDataTransferImpl` our factory supplies
(§6.2). Nothing about fallback is our code path — which is precisely why it is trustworthy.

The cost model, which the op-coverage strategy in §8 is built around: **one unclaimed node in the
middle of a graph splits it into two islands with a device round-trip between them.** Claim rate
is a bad metric; fused-region compute volume is the good one. The MLX EP learned this the
expensive way and we inherit the lesson rather than repeating it.

---

## 3. Repository and crate layout

Mapped one-to-one against `onnxruntime-mlx`. New paths carry a ✨.

```text
onnxruntime-ep-vulkan/
├── README.md                          # what it is, how to build, how to run
├── LICENSE
├── docs/
│   ├── DESIGN.md                      # ← this file: architecture of record
│   ├── ENGINE.md                      # ✨ Switch: Vulkan runtime, memory, shaders, sync
│   ├── PLATFORMS.md                   # ✨ Link: platform/driver matrix, toolchains, CI lanes
│   ├── OP_ARCHITECTURE.md             # Mouse: op registry + authoritative coverage table
│   └── BENCHMARKS.md                  # ✨ Niobe: methodology + published baselines
├── rust/
│   ├── Cargo.toml                     # cdylib crate, lib name onnxruntime_vulkan_ep
│   ├── build.rs                       # bindgen(ORT C ABI) + GLSL→SPIR-V compile+embed
│   ├── README.md                      # crate-level notes for contributors
│   ├── shaders/                       # ✨ GLSL compute sources (Switch)
│   │   ├── include/                   #    shared GLSL headers (indexing, broadcast, dtype)
│   │   ├── elementwise_binary.comp
│   │   ├── elementwise_unary.comp
│   │   └── ...
│   └── src/
│       ├── lib.rs                     # CreateEpFactories / ReleaseEpFactory, panic guards
│       ├── factory.rs                 # OrtEpFactory vtable: devices, allocator, data transfer
│       ├── ep.rs                      # OrtEp vtable: GetCapability, clustering, Compile, Compute
│       ├── engine.rs                  # Plan, NodeDesc, DispatchContext — the op-facing API
│       ├── recorded.rs                # ✨ command-buffer recording cache (shape-keyed replay)
│       ├── registry.rs                # op registry + NodeView / GraphView + claim helpers
│       ├── allocator.rs               # ✨ OrtAllocator over device memory
│       ├── transfer.rs                # ✨ OrtDataTransferImpl: host↔device staging copies
│       ├── vk/                        # ✨ the Vulkan layer — Switch owns, nothing else enters
│       │   ├── mod.rs                 #    re-exports the safe surface only
│       │   ├── instance.rs            #    VkInstance, layers, debug messenger
│       │   ├── device.rs              #    physical-device scoring, VkDevice, queues
│       │   ├── caps.rs                #    Capabilities struct: the single capability oracle
│       │   ├── memory.rs              #    gpu-allocator integration, DeviceBuffer, StagingPool
│       │   ├── pipeline.rs            #    VkPipeline creation, VkPipelineCache, spec constants
│       │   ├── descriptor.rs          #    descriptor set layouts and pools
│       │   ├── command.rs             #    command pool/buffer recording, barriers, submission
│       │   └── shaders.rs             #    embedded SPIR-V module table + variant selection
│       ├── ops/                       # per-family ONNX handlers + claim predicates (Mouse)
│       │   ├── mod.rs
│       │   ├── elementwise.rs
│       │   ├── math.rs
│       │   ├── reduction.rs
│       │   ├── shape.rs
│       │   ├── matmul.rs
│       │   └── norm.rs
│       ├── sys.rs                     # raw bindgen output for the ORT plugin-EP C ABI
│       ├── logging.rs                 # in-crate `log` subscriber, env-gated, silent by default
│       └── trace.rs                   # env-gated Chrome/Perfetto tracer + GPU timestamp queries
├── tests/
│   ├── README.md
│   ├── ops/                           # pytest op-correctness: Vulkan EP vs ORT CPU EP
│   │   ├── conftest.py                #    registers the plugin from ONNXRUNTIME_VULKAN_EP_LIB
│   │   ├── _models.py                 #    ONNX IR model builders, shared with bench/
│   │   └── test_*.py
│   ├── backend/                       # ONNX backend node tests through the EP
│   └── conformance/                   # opt-in broader conformance (onnx-tests harness)
│       ├── README.md
│       ├── RESULTS.md
│       ├── claimed_ops.txt
│       └── run_conformance.sh
├── bench/
│   ├── README.md
│   ├── bench.py                       # per-op-family timings, Vulkan vs CPU
│   ├── cases.py
│   └── compare.py                     # base-vs-PR regression table for CI comment
├── python/
│   ├── README.md
│   ├── pyproject.toml
│   ├── hatch_build.py                 # builds the cargo cdylib into the wheel
│   └── src/onnxruntime_ep_vulkan/
│       ├── __init__.py                # register_execution_provider_library(), EP_NAME, paths
│       └── py.typed
└── .github/workflows/
    ├── ci.yml                         # fmt, clippy, build matrix, op tests on lavapipe (Linux + Windows)
    ├── conformance.yml                # opt-in workflow_dispatch
    ├── bench.yml                      # informational perf comment on PRs
    └── publish.yml                    # wheel build + release
```

### 3.1 Naming decisions

- **EP name `VulkanExecutionProvider`.** Matches ORT's naming convention for every other EP and
  is what a user will guess. Frozen — changing it later breaks every user's provider list.
- **Library base name `onnxruntime_vulkan_ep`,** pinned via `[lib] name` so it is stable
  regardless of the crate name, exactly as the MLX EP does. Python, tests, CI, and any downstream
  runtime load it by that exact filename.
- **Version `0.<ORT_API_VERSION>.<patch>`.** A plugin EP is bound to one plugin-EP C-ABI version,
  so the version must state which ORT it works with. `0.27.0` pairs with ORT 1.27.x. When ORT
  ships API version 28, we move to `0.28.0`. The EP reports this to ORT from
  `env!("CARGO_PKG_VERSION")` so it can never drift from the manifest.
- **Vendor ID.** Unlike the MLX EP (which reports Apple's `0x106B`), there is no single hardware
  vendor here. The factory reports the **Vulkan `vendorID` of the bound physical device** from
  `VkPhysicalDeviceProperties`, so a user querying EP devices sees NVIDIA/AMD/Intel/Qualcomm/ARM
  correctly. Open question OQ-6 covers the no-device case.

---

## 4. Module responsibilities and boundaries

### 4.1 Layer map

```
┌──────────────────────────────────────────────────────────────────────────┐
│ L0  ORT C ABI boundary        lib.rs · factory.rs · ep.rs · allocator.rs │
│                               transfer.rs · sys.rs                       │
│     Owns: OrtEpFactory/OrtEp/OrtNodeComputeInfo/OrtAllocator vtables,     │
│           Box::into_raw ownership, panic guards, OrtStatus construction.  │
│     Owner: Tank                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│ L1  Plan & dispatch           engine.rs · recorded.rs · registry.rs      │
│     Owns: NodeDesc, Plan, DispatchContext, the op registry, NodeView,     │
│           claim dispatch, command-buffer recording cache.                 │
│     Owner: Morpheus (contract) · Tank (plumbing) · Mouse (registry)      │
├──────────────────────────────────────────────────────────────────────────┤
│ L2  ONNX op semantics         ops/*.rs                                   │
│     Owns: per-op claim predicates and translate handlers. Reads           │
│           attributes, validates dtypes/shapes, requests dispatches.       │
│     Owner: Mouse                                                         │
├──────────────────────────────────────────────────────────────────────────┤
│ L3  Vulkan engine             vk/*.rs · shaders/*.comp                   │
│     Owns: VkInstance/VkDevice/VkQueue, allocator, staging, descriptors,   │
│           pipelines, barriers, submission, SPIR-V modules.                │
│     Owner: Switch                                                        │
├──────────────────────────────────────────────────────────────────────────┤
│ L4  Raw bindings              ash · gpu-allocator · bindgen(ORT)         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 The two hard rules

**Rule 1 — The ORT C ABI never leaks into op code.**
No type from `sys::ort` may appear in a signature, field, or local in `rust/src/ops/`. Op handlers
see `NodeDesc`, `NodeView`, `TensorRef`, and `DispatchContext`. They never see `OrtNode`,
`OrtValue`, `OrtKernelContext`, `OrtStatus`, or an `OrtApi` function pointer. The exception that
proves the rule: `NodeView` and `NodeDesc` are *where* the ABI is translated into safe Rust, and
they live in `registry.rs` / `engine.rs`, not in `ops/`.

*Enforcement:* `sys` is `pub(crate)` but a CI lint (`ci.yml`) greps `rust/src/ops/` for `sys::`,
`Ort`, and `unsafe` and fails the build on a hit. A rule that is not mechanically checked is a
suggestion.

**Rule 2 — Op code never touches raw Vulkan handles.**
No `ash::vk::*` type may appear in `rust/src/ops/`. Op handlers cannot hold a `vk::CommandBuffer`,
call `vkCmdDispatch`, allocate memory, or create a pipeline. They express intent through
`DispatchContext`:

```rust
// The entire vocabulary an op handler has. Illustrative signature — the real one lands with M0.
pub trait DispatchContext {
    fn resolve(&mut self, r: &TensorRef) -> Result<BufferView, EpError>;
    fn bind_output(&mut self, o: &OutRef, desc: TensorDesc) -> Result<BufferView, EpError>;
    fn alloc_temp(&mut self, desc: TensorDesc) -> Result<BufferView, EpError>;
    fn dispatch(&mut self, k: KernelRequest) -> Result<(), EpError>;
    fn read_const_i64(&self, r: &TensorRef) -> Option<Vec<i64>>;
}
```

`BufferView` is an opaque handle. `KernelRequest` names a shader variant, its specialization
constants, its push-constant payload, its bindings, and a workgroup count. The engine decides
descriptor sets, barriers, pipeline selection, and submission. `ENGINE.md` §1 states the same
boundary from the engine side and enforces it with module privacy: the wrapper types in `vk/` are
not `pub` outside the engine.

*Enforcement:* the same CI lint greps `rust/src/ops/` for `ash`, `vk::`, and `unsafe`. It also
enforces the barrier seam of §7.5: `cmd_pipeline_barrier`, `cmd_pipeline_barrier2`,
`BufferMemoryBarrier`, `DependencyInfo`, `PipelineStageFlags*` and `AccessFlags*` may appear
**only** in `rust/src/vk/barrier.rs`, and `Capabilities::synchronization2` may be read **only** in
`rust/src/vk/barrier.rs` and `rust/src/vk/caps.rs`.

**Why this matters enough to reject a working PR over.** The MLX EP got a mature backend that
handled memory, scheduling, and dtypes. We do not. Every op we add is a shader, a descriptor
layout, a barrier, and a workgroup calculation. If those details are allowed to bleed into 60 op
modules, the first driver quirk Link finds becomes a 60-file change instead of a 1-file change.
The boundary is the only thing that keeps op coverage a linear cost.

### 4.3 What each module may depend on

| Module | May use | May **not** use |
|---|---|---|
| `lib.rs`, `factory.rs`, `ep.rs`, `allocator.rs`, `transfer.rs` | `sys::ort`, `engine`, `registry`, `vk` (safe surface) | `ash` directly, `ops::*` internals |
| `engine.rs`, `recorded.rs` | `vk` safe surface, `registry` | `sys::ort` FFI calls outside `NodeDesc` construction |
| `registry.rs` | `sys::ort` (for `NodeView` only), `engine` types | `ash`, `vk` |
| `ops/*.rs` | `engine::{DispatchContext, NodeDesc, TensorRef}`, `registry` helpers | `sys`, `ash`, `vk`, `unsafe` |
| `vk/*.rs` | `ash`, `gpu-allocator` | `sys::ort`, `ops` |

---

## 5. Execution flow, end to end

### 5.1 Library load

`RegisterExecutionProviderLibrary` → `dlopen`/`LoadLibrary` → `CreateEpFactories`.
Initialise logging, negotiate `ORT_API_VERSION 27` (fail with a clear status if the host is
older), construct one `VulkanEpFactory`. **No Vulkan work happens yet** — a plugin must be cheap
to load even on a machine that will never use it.

### 5.2 Device enumeration

ORT calls `GetSupportedDevices`. *Now* we create the `VkInstance`, enumerate physical devices,
apply the capability gate (§7), score and sort, correlate with ORT's `OrtHardwareDevice` list,
and create one `OrtEpDevice` per usable device. Zero usable devices → advertise none, log a
warning, return success. The instance is kept alive on the factory.

### 5.3 Session creation

ORT calls `CreateEp`. We read session options, select the physical device, create the `VkDevice`,
compute queue, `gpu-allocator` arena, command pool, descriptor pools, staging pool, and load the
on-disk `VkPipelineCache`. The `VulkanEp` owns all of it and drops it in `ReleaseEp`. RAII, not
manual teardown — this is where the MLX rewrite found a real per-session leak that three lines of
`impl Drop` fixed, and we take the same posture from day one.

`GetDefaultMemoryDevice` returns our device's `OrtMemoryDevice` (M2+) or null (M0/M1, §6.3).

### 5.4 `GetCapability` — claiming

Per node: build `NodeView` → `registry::claim_decision(&view)`. Rejections carry a *reason string*
and are aggregated per op type; with `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` or tracing on, the EP
prints exactly which ops were declined, how many, and why. This is the single most valuable
diagnostic the MLX EP has, and it is cheap. It ships in M0.

Claimed nodes → convex clustering → **minimum-viable-subgraph filter** →
`EpGraphSupportInfo_AddNodesToFuse` per surviving cluster.

**Where `ops::partition::retain_viable` is invoked — the answer to Mouse's question.** Exactly one
call site, in `ep.rs`'s `GetCapability`, positioned as the **third of four stages** and never
anywhere else:

```
GetCapability(OrtGraph) →
  1. per-node claim      registry::claim_decision(&NodeView)      → claimed / declined + reason
  2. cluster             maximal convex connected components of the claimed set → Vec<Island>
  3. FILTER              ops::partition::retain_viable(&islands, &model, &policy)
                                                                   → (kept, dropped + RejectReason)
  4. report              AddNodesToFuse(kept) ; declines for step 1 and step 3 both go to the
                         one decline vocabulary and the one CLAIM_DEBUG dump
```

Four properties of that placement are binding, not incidental:

1. **After clustering, never before.** The rule is about an *island*, not a node — whether the work
   inside pays for the transfers at its edges. There is no meaningful per-node form of it, and a
   per-node approximation would re-introduce exactly the shredding it exists to prevent.
2. **Before `AddNodesToFuse`, so a rejected island is never handed to ORT at all.** Filtering after
   the fact would mean un-claiming, which the ORT surface does not offer.
3. **A step-3 rejection is a decline like any other**, carrying `DeclineCode::Partition` with the
   modelled numbers (`~Xns compute against ~Yns transfer, below the Zx margin`) into the same
   `CLAIM_DEBUG` output as a step-1 decline. This is §8.4 A3's requirement, and it is why the
   dropped set is returned rather than discarded: a silently-declined region is otherwise
   indistinguishable at the console from a missing op, and we would spend real time mis-diagnosing
   one as the other.
4. **`retain_viable` stays pure — data in, verdict out.** It does not touch the graph API or the
   device. `ep.rs` builds `Island` values from the cluster and the `TransferModel` from the device
   calibration, calls the function, and translates the verdicts back into ORT calls. This is the
   §4.2 boundary rule applied to partitioning: policy in `ops/`, ABI in `ep.rs`.

The `TransferModel` handed to it is the one calibrated at device init (`OP_COVERAGE.md` §7.2;
§8.4 A3), not a constant — with `TransferModel::UMA` / `DISCRETE` as the pre-calibration defaults so
M0/M1 have a defined behaviour before Niobe's measurements exist. `CoverageReport`, computed from
the same `(kept, dropped)` pair, is what produces `largest_island_flops` for §10.0's milestone
reporting — the rule and its metric live in one module deliberately, because their drifting apart is
exactly how a coverage number becomes a lie.

### 5.5 `Compile` — plan build and prepacking

For each fused subgraph, in order:

1. **Extract.** `NodeDesc` per node, attributes copied generically into typed maps, inputs and
   outputs resolved to `TensorRef`/`OutRef` with dtype and static shape where known.
2. **Validate.** Every node must resolve to a registry entry with a claim predicate that still
   accepts it. A mismatch here is an internal invariant violation, not a user error — fail the
   compile loudly.
3. **Plan shapes.** Compute static shapes for every intermediate. Determine the workgroup counts
   and the temporary-buffer set. Assign temporaries to a shared arena with a greedy
   liveness-based packing (ExecuTorch's `SharedObject` idea) so a 40-node subgraph does not
   allocate 40 buffers.
4. **Prepack weights.** For each constant initializer: read host bytes through the ORT graph API,
   convert to the shader's expected layout/dtype, and upload into a **device-local buffer owned by
   the plan**, via a staging buffer. Done once, at compile time. This is the ExecuTorch
   `prepack()` model and the direct analog of the MLX plan's repacked-weight cache, and it is the
   reason inference does not re-upload weights.
5. **Warm pipelines.** Create every `VkPipeline` the plan needs, populating the `VkPipelineCache`.
   First-inference latency is a real user-visible cost on Vulkan; paying it at compile time is the
   right trade.
6. **Wrap.** Return an `OrtNodeComputeInfo` owning the plan.

### 5.6 `Compute` — inference

```
ORT calls Compute(state, OrtKernelContext)
 ├─ 1. Resolve inputs.
 │      M0/M1 (host I/O):   ORT hands host pointers → upload into device buffers via staging.
 │      M2+  (device I/O):  ORT hands device "pointers" our allocator produced → resolve to
 │                          (VkBuffer, offset) with no copy at all.
 ├─ 2. Shape key.  Hash the concrete input shapes. Look up recorded.rs.
 │      hit  → reuse the recorded VkCommandBuffer.
 │      miss → record: for each NodeDesc in topo order, registry::translate() runs the handler,
 │             the handler calls DispatchContext::dispatch(), the engine binds descriptors,
 │             emits the memory barrier the dependency edge requires, and records vkCmdDispatch.
 │             Cache the recording under the shape key.
 ├─ 3. Bind outputs. KernelContext_GetOutput for each subgraph output.
 ├─ 4. Submit once to the compute queue. One submission per subgraph execution.
 ├─ 5. Wait on the fence.
 └─ 6. Outputs.
        M0/M1: download device → staging → ORT's host output tensor.
        M2+:   nothing — the output already lives in the device buffer ORT allocated.
```

**Where CPU fallback happens.** Three distinct places, and it is worth keeping them straight:

1. **Claim time (the main one).** A node the registry declines is never in a plan. ORT assigns it
   to the CPU EP and inserts partition-boundary copies through our `OrtDataTransferImpl`. This is
   the designed path and it is always correct.
2. **Compile time.** If a plan cannot be built (a shape that cannot be resolved statically, an
   initializer that cannot be read, a pipeline that fails to create), `Compile` returns an error
   status and ORT falls the whole subgraph back to CPU. Loud, logged, and rare by construction —
   §5.5 step 2 makes it an invariant violation.
3. **Runtime.** A device-lost, out-of-memory, or panic condition returns `ORT_EP_FAIL` from
   `Compute`. ORT surfaces the failure. We do **not** attempt a silent per-node CPU rescue inside
   `Compute`: a partially-executed command buffer with half-written outputs is not a state we can
   reason about, and silently producing CPU results after a GPU fault hides real bugs.

---

## 6. Tensor and memory model

This is the section where Vulkan and MLX genuinely part ways, so it states the **contract**;
`ENGINE.md` §3 owns the implementation.

### 6.1 The problem

MLX: unified memory. An ORT CPU tensor and an MLX array can point at the same bytes. The MLX EP
therefore advertises no device allocator, returns null from `GetDefaultMemoryDevice`, and copies
out once per subgraph with a `memcpy`.

Vulkan: explicit device memory. Device-local memory is generally not host-visible; host-visible
memory is generally not device-local; and on discrete GPUs the two are across a PCIe bus. There is
no pointer we can hand ORT that a shader can also read. Everything below follows from that.

### 6.2 The contract

| Concern | Contract |
|---|---|
| **Tensor identity** | A device tensor is `(VkBuffer, offset, size, dtype, shape)`. The `VkBuffer` may be a suballocation of a larger arena. Op code sees this only as an opaque `BufferView`. |
| **Layout** | Row-major dense, ONNX semantics, no implicit padding, no implicit transposition. Any op needing a different internal layout materializes it explicitly as a temporary. Prepacked *constants* may use a shader-specific layout; **activations may not**. |
| **Alignment** | Every allocation satisfies `minStorageBufferOffsetAlignment` for the bound device. Enforced by the allocator, never by op code. |
| **Who allocates weights** | The plan, at `Compile` time, device-local, uploaded once, freed when the plan drops. |
| **Who allocates activations at partition boundaries** | ORT, through our `OrtAllocator` (M2+) or ORT's CPU allocator (M0/M1). |
| **Who allocates intermediates** | The engine, from a plan-owned arena sized at compile time, reused across inferences. |
| **Who transfers** | Only `transfer.rs` (`OrtDataTransferImpl`) and the engine's staging path. Op code never initiates a transfer. |
| **When transfers happen** | At partition boundaries (ORT-inserted) and, in M0/M1, at subgraph entry/exit. Never per node. |
| **Coherence** | Non-coherent host-visible memory is flushed/invalidated by the engine around every host access. This is not optional and it is not the op author's problem. |
| **Synchronization** | The engine emits the barriers the plan's dataflow edges imply. A read-after-write between two dispatches always gets a barrier. Correctness first; barrier-batching optimization is Niobe's ticket, not a design assumption. |

### 6.3 Avoiding a copy on every inference — the phased plan

This is the crux, so it is explicit.

**M0/M1 — host I/O (the MLX shape).** `GetDefaultMemoryDevice` returns null; the factory's
`CreateAllocator`/`CreateDataTransfer` return null (valid — ORT tolerates it, as the MLX EP
proves). Subgraph I/O lives in CPU memory; `Compute` uploads inputs and downloads outputs each
call. Weights are still uploaded once at compile time, so the per-inference traffic is
activations only.

*Why start here:* it removes the entire allocator/data-transfer/ORT-memory-placement surface from
M0, which is the highest-uncertainty part of the ABI. It gets a correct, cross-platform,
CPU-oracle-verified elementwise op running on Windows and Linux in the shortest path. It is
honestly slow for anything small, and we will say so rather than benchmark it.

**M2 — device I/O (the real model).** The factory implements `CreateAllocator` (returning an
`OrtAllocator` backed by the device arena) and `CreateDataTransfer` (host↔device staging copies).
`GetDefaultMemoryDevice` returns the device's `OrtMemoryDevice`. ORT then:
- places tensors that only ever cross Vulkan partitions in device memory, so **two adjacent Vulkan
  subgraphs separated by a CPU node still avoid one of the two round-trips**;
- uses our data transfer for the boundary copies it does need;
- lets a user with `IoBinding` keep inputs and outputs resident on the device across inferences,
  which is what makes a per-inference copy disappear entirely.

The one hard problem M2 must solve: **ORT's allocator API is pointer-based, and a `VkBuffer` is
not a pointer.** This is OQ-3, and it is now **decided** — see §6.4.

**M3+ — persistence.** Keep prepacked weights and, where a graph allows, activation buffers
resident across `Compute` calls; shapeless recording so a growing dimension does not retrace.

### 6.4 OQ-3 RESOLVED — reserved virtual address space, resolved through an opaque-handle registry

> **Decided 2026-07-28T22:28:08-07:00** on Tank's proposal (`.squad/decisions/inbox/tank-oq3-allocator-proposal.md`,
> D-T8). **Accepted in full, including the part that goes further than my own framing.** Binding on
> `allocator.rs`, `transfer.rs`, `ENGINE.md` and every op that receives an ORT data pointer.

**The decision.** `Alloc(size)` sub-allocates real Vulkan memory through `gpu-allocator`, carves a
matching span out of a large region of **reserved-but-uncommitted virtual address space**
(`VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` on Windows; `mmap(PROT_NONE, MAP_NORESERVE)` on
Linux/Android/macOS), records `span_base -> (VkBuffer, offset, size, generation)`, and returns
`span_base` as the `void*`. Resolution to `(VkBuffer, offset)` happens **once per binding when a
descriptor set is built** — not per element, not per dispatch — and is cached in the compiled plan.
`Free` quarantines the span and bumps a generation rather than recycling it immediately.

**`VK_KHR_buffer_device_address` is not carried at all.** I had written "registry primary, BDA an
optimization on top". Tank's argument that BDA is **not an optimization of the registry but a second
shader architecture** is correct and it changes the answer, so I am adopting his position rather
than the one I brought. A `VkDeviceAddress` is unusable by a descriptor-bound shader: consuming one
requires `GL_EXT_buffer_reference` / `PhysicalStorageBuffer` addressing, which is a second shader
family, a second variant axis in the manifest, and a second set of conformance runs — a permanent
cost on Mouse's and Trinity's surface, not a branch in Tank's. And it *does not even remove the side
table*, because building a descriptor set still needs a `VkBuffer`, so address → buffer resolution
survives regardless; BDA only pays off under a fully-bindless design, which is a much larger bet
nobody has proposed. The platform evidence points the same way: `PLATFORMS.md` MVK3 (BDA needs
Metal 3 / Apple Silicon — Intel Macs excluded), MVK4 (MoltenVK explicitly advises the explicit
binding model), and my own §7.2 freezing BDA as probed-not-required. **Two paths where one is
unreachable on Apple and on older Android drivers is not a dual design; it is one design plus an
under-tested liability**, which is the same failure shape I rejected the `synchronization2` layer
shim for. Nothing forecloses BDA later: if a bindless GEMM path is ever adopted and measurement
shows descriptor construction to be a real cost, BDA slots in *alongside* the registry for shaders
that opted into buffer references. Revisit then, **with numbers**.

**Why reserved VA rather than a synthetic token — the part that decided it.** My stated fear was
that ORT performs pointer arithmetic on values it believes are addresses, and that the resulting bug
would be invisible until some ORT-internal path `memcpy`s from an allocator pointer. Reserved VA
answers that **by construction rather than by convention**, which is a different quality of answer:

- `base + offset` from the memory-pattern planner stays inside the same allocation because the span
  *is* contiguous reserved VA of exactly that size; an interior pointer resolves to
  `(VkBuffer, offset + delta)` correctly. A token scheme cannot provide this at all.
- `align_up(ptr, 256)` works exactly; span bases are 2 MiB-aligned, which dominates
  `minStorageBufferOffsetAlignment` on all target hardware.
- Uniqueness against real heap pointers is **OS-guaranteed** — the kernel will not hand that range
  to anyone else. A hand-rolled "reserved" numeric range can collide; a real reservation cannot.
- A stray dereference is an **MMU fault at an address we recognise**, with a stack trace naming the
  culprit, instead of a read or write to whatever happens to live at `0x1000`. This converts my
  worst case from silent corruption into a crash at the scene.
- Use-after-free resolves to a quarantined span and becomes a loud `OrtStatus` rather than a silent
  alias onto a different live tensor.

I want the general principle recorded, because it will recur: **when a design can make a hazard
impossible by construction, prefer it to a design that makes the hazard merely unlikely, even if
the second is simpler.** The cost here is a page-table reservation that consumes no physical memory.

Two supporting points I specifically endorse. Tank notes the NV reference calls
`DisableMemPattern()`; **we must not rely on that** — an EP cannot force a caller's session options,
and a design that only works with mem-pattern off breaks the first time somebody leaves it on. And
the resolution cost (a subtract, a shift, one flat-array load, under an `RwLock` taken for write
only on `Alloc`/`Free`) is stated plainly now precisely so it cannot later be cited as a reason to
reopen BDA without measurement.

**The Android sub-question: a tuning parameter, not a blocking dependency.** Android's narrower
virtual address space (39-bit on many devices) means the reservation size must be **derived from the
platform at runtime, not hard-coded**. That is a requirement on the implementation, and it is
Tank's to satisfy now — not a dependency on Link's matrix. Rationale: the design does not change
with the answer, only a constant does, and the correct implementation is to *probe and back off*
rather than to look a number up in a table. Binding form:

1. The region size is chosen at allocator construction by attempting a reservation and **halving on
   failure** down to a documented floor, rather than by consulting a per-platform constant. A table
   of platform address-space widths is a table that will be wrong about some device we have never
   seen; a reservation that either succeeds or does not is correct on every device by construction.
2. The chosen size, the granularity, and the number of back-off steps taken are recorded in the
   capability dump. On a 39-bit device this is the line that will explain a later allocation
   failure.
3. Falling below the floor is a **clean allocator-construction failure with a specific message**,
   which degrades the EP to M0/M1 host-I/O behaviour or to no device at all — never a silent
   success with a region too small to serve the model.
4. Link's platform matrix should still record observed address-space widths, because it tells us
   whether the back-off is ever actually taken. That is *information*, and it does not gate M2.

**What this means for OQ-13.** An imported external buffer becomes an ordinary registry entry
(Tank's D-T9 step 6), so nothing downstream — descriptor construction, translate handlers, data
transfer — needs to know a buffer came from outside. That is the registry serving the importer, and
it is a further argument for a single resolution mechanism.

---

## 7. Vulkan API baseline — decision

> **Status: FROZEN as of 2026-07-28T19:16:08-07:00.** OQ-1 is **resolved** with measured data
> (Link, [`PLATFORMS.md`](./PLATFORMS.md) §8, vulkan.gpuinfo.org pulled 2026-07-28) and the answer
> **reversed the provisional §7.2 requirement set**. This section is now the binding contract for
> Switch's [`ENGINE.md`](./ENGINE.md) and Link's CI matrix. Changing it requires a new decision
> record, not an edit.
>
> **Governing directive (Justin, 2026-07-28):** 「如果 1.3 兼容性不好 那 1.2 更好。可以保证兼容性
> 是最好。」 — *broad device compatibility is the top-priority property of this decision.* Where
> device coverage and engine-code simplicity conflict, **coverage wins**. Every ruling below is
> made under that constraint, and where it costs Switch complexity, it costs Switch complexity.
>
> Justin separately ratified the *framing* — a capability set rather than a version number
> (「拿能力集很聪明，听你的」). What changed is where the bar sits, not how the bar is expressed.

### 7.0 The frozen principle

**The device gate is minimal. Capability shortfalls degrade op coverage, not device availability.**

This one sentence replaces the previous "require the two features Switch wants" posture. A device
that lacks an optional capability must still load, still be advertised, and still run every op
that does not need that capability. It declines the ops it cannot do correctly, and ORT's
partitioner sends those to CPU (§2.6). We never refuse a device for a reason that only affects
*some* ops.

Consequences, stated so they are not re-litigated per op:

- A hard device requirement must be justified by *"no op we will ever ship can work without it."*
- A per-op requirement is expressed as a claim predicate in `ops/` (§8), never as a device gate.
- Anything we make optional, we must be able to run **both ways in CI** (§7.5 item 5, §9.1).

### 7.1 The evidence

| Source | Finding |
|---|---|
| Justin's proposal | Vulkan 1.3, citing llama.cpp. |
| llama.cpp `ggml-vulkan.cpp` | Hard runtime floor is **Vulkan 1.2** — `if (api_version < VK_API_VERSION_1_2) throw`. `VkApplicationInfo::apiVersion` is set to *whatever the instance reports*, not to a hardcoded 1.3. |
| llama.cpp `vulkan-shaders-gen.cpp` (Fact Checker, claim 1, SHA `3e6b395`) | **Base shaders are compiled with `--target-env=vulkan1.2`.** Only the cooperative-matrix-2 (`_cm2`) variants — an NVIDIA Ampere+/Ada optimization path — target `vulkan1.3`. The `--target-env=vulkan1.3` in CMakeLists is an extension-availability probe, not the default. **Verdict: the "llama.cpp requires 1.3" claim is contradicted.** |
| ExecuTorch `vk_api/Runtime.cpp` (Fact Checker, claim 2, SHA `8001512`) | Hardcodes `VK_API_VERSION_1_1` in `VkApplicationInfo`; `Device.cpp` branches feature queries at `>= VK_API_VERSION_1_1`; VMA is initialized with `VK_API_VERSION_1_0`. **Verdict: contradicted** — ExecuTorch targets 1.1. |
| MoltenVK (Fact Checker, claim 3) | **Verified:** MoltenVK 1.3.0 (2025) advertises Vulkan 1.3 on macOS/iOS. Older MoltenVK does not. |
| Android (Fact Checker, claim 4 — *unverified*, plausible) | Vulkan 1.3 ≈ **26%** of active Android devices; Vulkan 1.1 ≈ **62%**. The Android CDD does not mandate 1.3 at any API level as of Android 15. Link (`PLATFORMS.md` §4) reports ~89% for 1.1 measured against *devices that expose Vulkan at all* — a different denominator, same conclusion. |
| lavapipe / SwiftShader (Fact Checker, claim 5) | **Verified:** both support Vulkan 1.3 and both pass 1.3 conformance. Adequate for GPU-less CI. *Update 2026-07-28T21:01:56-07:00: Trinity evaluated SwiftShader and **rejected** it — no usable prebuilts and a ~20-minute build from source. **Both CI lanes use lavapipe**: Linux via `apt`, Windows via mesa-dist-win 26.1.3.* |
| Link (`PLATFORMS.md` §4) | Recommends 1.2 core + mandatory device features. Explicitly does **not** recommend a hard 1.3 baseline if Android coverage is a goal. |
| Switch (`ENGINE.md` §8) | Exactly **two** features materially simplify the engine: `synchronization2` and `subgroup_size_control`. Both are core in 1.3 — **and both are available as standalone extensions on 1.1/1.2 drivers.** `shaderFloat16`, `bufferDeviceAddress`, and cooperative matrix must be capability-probed at runtime *regardless of baseline*. |
| **Link, OQ-1 (`PLATFORMS.md` §8), vulkan.gpuinfo.org, pulled 2026-07-28** | **`VK_KHR_synchronization2`: Android 68.57%, Windows 87.78%, Linux 99.05%, macOS 97.5%, iOS 100%.** A **31.43-point Android gap** and a **12.22-point Windows gap.** The Android shortfall is concentrated in Adreno 5xx (Snapdragon 625–660, frozen pre-2021 OEM blobs), Adreno 6xx on unupdated Android 10/11, and Mali Bifrost (G52/G57/G72/G76) especially on MediaTek — populations with no update cadence, so this does not decay with time on any schedule we control. Link's verdict: the hard requirement is **not safe**. |
| **Link, OQ-1** | **`VK_EXT_subgroup_size_control`: Android 85.88%, Windows 93.33%, Linux 98.81%, macOS/iOS 100%.** A **14.12-point Android gap.** |
| **Link, OQ-1 — the MoltenVK artifact** | The macOS/iOS 100% figure is **extension-string presence only**. MoltenVK reports Vulkan 1.3, which promotes `subgroup_size_control` to core, so the string is always there — but the `subgroupSizeControl` **feature flag is `VK_FALSE`**, because Metal cannot control SIMD-group width per pipeline. **Requiring the feature flag to be `VK_TRUE` would silently exclude all of macOS and iOS** — and probably lavapipe too, which has a single fixed CPU SIMD width. |
| **Link, OQ-1 — limits that *are* safe** | `maxComputeWorkGroupInvocations ≥ 256`: ~1% of 8,206 Android reports show 128. `maxComputeSharedMemorySize ≥ 16 KiB`: the Vulkan spec minimum, universal. Subgroup `BASIC`: spec-guaranteed in the compute stage on 1.1+. Subgroup `ARITHMETIC`: >95%, but *query, never assume*. |
| **Morpheus, layer-shim feasibility research (2026-07-28, primary sources below)** | The Khronos `VK_LAYER_KHRONOS_synchronization2` shim **cannot be shipped by us on Android.** The AOSP Vulkan loader does not read `VK_LAYER_PATH`, does not use JSON manifests, and searches only the **host application's** `nativeLibraryDir` (derived from the installed APK via `GraphicsEnv::getAppNamespace()`) plus `/data/local/debug/vulkan` (debuggable/userdebug only). Khronos' own `docs/synchronization2_layer.md` states the `.so` "needs to be packaged **inside the APK**". A plugin `.so` `dlopen`ed into someone else's process has no mechanism to add a layer search path. Sources: `developer.android.com/ndk/guides/graphics/validation-layer`; `KhronosGroup/Vulkan-Loader` `docs/LoaderLayerInterface.md` ("The Android loader does not use manifest files"; "There is No Support For Implicit Layers on Android"); `KhronosGroup/Vulkan-ExtensionLayer` `docs/synchronization2_layer.md`. |
| **Morpheus, prior-art check on barrier strategy** | **wgpu, Dawn, and Godot all use legacy `vkCmdPipelineBarrier` exclusively and none of them ships the sync2 layer.** `gfx-rs/wgpu` `wgpu-hal/src/vulkan/command.rs` calls `cmd_pipeline_barrier` in `transition_buffers`/`transition_textures` with no sync2 variant and no sync2 entry in its `Workarounds` bitflags; `google/dawn` `src/dawn/native/vulkan/CommandBufferVk.cpp` calls `fn.CmdPipelineBarrier` and mentions sync2 only in a spec comment; `godotengine/godot` `drivers/vulkan/rendering_device_driver_vulkan.cpp` calls `vkCmdPipelineBarrier`. The cited precedent for Option B does not survive contact with the source. |

The premise that motivated 1.3 — "llama.cpp requires it" — is contradicted by llama.cpp's own
source at both the runtime check and the shader target, and independently verified as contradicted
by Fact Checker. That does not make 1.3 wrong; it makes the *reason* wrong, and I would rather we
decide this on the two features Switch identified than on a misattribution.


### 7.2 Decision — the frozen capability set

**We require a capability set, not a version number.** The set is deliberately small.

A physical device is advertised to ORT **if and only if** it satisfies all of:

| # | Hard requirement | Why it is a *device* gate and not a per-op gate |
|---|---|---|
| R1 | Vulkan **≥ 1.1** core, instance and device | `VkPhysicalDeviceFeatures2` / `VkPhysicalDeviceProperties2` chains and `VkPhysicalDeviceSubgroupProperties` are core at 1.1. Below 1.1 we cannot even *ask* what a device can do, so no op can be claimed safely. This is also the Android floor. |
| R2 | A queue family with `VK_QUEUE_COMPUTE_BIT` | Without it there is nothing to dispatch to. |
| R3 | `maxComputeWorkGroupInvocations ≥ 256` | Every shader skeleton we will write assumes a 256-invocation workgroup. ~1% of Android reports fall below. |
| R4 | `maxComputeSharedMemorySize ≥ 16384` | The Vulkan spec minimum; universal. Listed so the assumption is written down, not because it filters anything. |
| R5 | Subgroup `BASIC` in the `COMPUTE` stage | Spec-guaranteed on 1.1+; listed for the same reason as R4. |
| R6 | At least one `DEVICE_LOCAL` memory type and at least one `HOST_VISIBLE` memory type | The staging path (§6) has no meaning otherwise. |

**That is the entire gate.** It is satisfied by essentially every device that exposes Vulkan 1.1
at all, on every platform, including MoltenVK and lavapipe.

**Everything else is capability-probed** into a single `vk::caps::Capabilities` struct, read once at
device init, and used in exactly two ways: (a) to select an implementation strategy inside the
engine, or (b) to gate an op's claim predicate. Nothing on this list may ever become a device gate
without a new decision record:

| Capability | Probed how | What it changes |
|---|---|---|
| `synchronization2` | 1.3 core **or** `VK_KHR_synchronization2` device extension | Selects the barrier backend (§7.3). **Not required.** |
| `subgroup_size_control` **properties** | 1.3 core **or** `VK_EXT_subgroup_size_control` — *properties queryable only* (§7.4) | Narrows the known subgroup-size range; enables the subgroup-cooperative shader variants. |
| Subgroup `ARITHMETIC` / `BALLOT` / `SHUFFLE` | `VkPhysicalDeviceSubgroupProperties::supportedOperations` | Gates the subgroup-reduction shader variants. Absent → shared-memory tree-reduction variant. |
| `shaderFloat16`, `storageBuffer16BitAccess` | `VkPhysicalDeviceVulkan12Features` / `VK_KHR_shader_float16_int8` + `VK_KHR_16bit_storage` | Gates fp16 op variants; absent → those ops are not claimed for fp16. |
| `shaderInt8`, integer dot product | extension probe | Gates future quantized ops. |
| Timeline semaphores | 1.2 core or `VK_KHR_timeline_semaphore` | Post-v0 multi-stream pipelining. Unused in v0. |
| `bufferDeviceAddress` | 1.2 core or `VK_KHR_buffer_device_address` | **Not used.** OQ-3 is resolved against it (§6.4); probed for diagnostics only, and a future bindless GEMM path would have to re-argue it with numbers. |
| Cooperative matrix | `VK_KHR_cooperative_matrix` / `VK_NV_cooperative_matrix2` | Post-v0 GEMM variants, llama.cpp's `_cm2` split. |

`VkApplicationInfo::apiVersion` is set to `min(vkEnumerateInstanceVersion(), VK_API_VERSION_1_3)`
— llama.cpp's pattern. We ask for the highest the loader will give us and then *use* only what the
device actually reports.

### 7.3 `synchronization2` — dropped from the hard requirement; Switch carries a legacy path

**Ruling: Option A.** `synchronization2` is **not required**. Switch implements a legacy
`vkCmdPipelineBarrier` backend alongside the `vkCmdPipelineBarrier2` backend, selected once at
device init (§7.5 defines the seam).

This reverses the provisional §7.2 of 2026-07-28T17:59:54-07:00, which required it.

**Why.** Under the compatibility-first directive, a 31.43-point Android exclusion and a
12.22-point Windows exclusion cannot be traded for one internal code path. The Windows number
matters as much as the Android one and is easy to overlook: nearly one desktop Windows device in
eight in Link's sample would be silently declined. The missing Android population is
*structurally* missing — Adreno 5xx blobs frozen before the 2021 extension, Mali Bifrost on
MediaTek with no update cadence — so it does not shrink on any timeline we control.

The cost is bounded and one-time: two implementations of a five-function internal API, written
once, tested in CI on every run (§7.5). The cost of the alternative is unbounded and permanent:
every device we decline is a device we can never win back with engineering.

#### Ruling on the layer-shim proposal (Link's Option B) — **rejected as a shippable mechanism**

The coordinator asked me to examine rather than adopt this. I did, and the concern is correct and
decisive.

| Platform | Can *our plugin* enable `VK_LAYER_KHRONOS_synchronization2`? | Basis |
|---|---|---|
| **Retail Android (non-rooted, non-debuggable)** | **No.** | The AOSP loader does not read `VK_LAYER_PATH`, does not use JSON manifests, and enumerates layers only from the **host application's** `nativeLibraryDir` (set by the framework at process launch from the installed APK via `GraphicsEnv::getAppNamespace()`) and from `/data/local/debug/vulkan`, which requires a debuggable app or a userdebug build. Khronos' own layer documentation says the `.so` must be "packaged inside the APK". We do not own the APK. |
| Windows | Conditionally yes | `SetEnvironmentVariable("VK_ADD_LAYER_PATH", …)` before *our* `vkCreateInstance` works, because the desktop loader re-scans layer paths at `vkCreateInstance`, not at load time. **Fails silently** if the host process runs at High Integrity Level (`loader_secure_getenv` returns NULL). Mutating the environment of a host process we do not own is also a `setenv`/`getenv` data race in any multi-threaded host. |
| Linux / macOS | Conditionally yes | Same mechanism; fails under setuid/setgid (`secure_getenv`). Manifest must carry an absolute `.so` path. Same race. |

Two independent reasons to reject it even where it technically works:

1. **It does not solve the platform it was proposed for.** Android is 100% of the reason we were
   considering it, and Android is the one platform where it cannot work from a plugin.
2. **The cited precedent does not exist.** wgpu, Dawn, and Godot were offered as evidence that
   shipping this layer is normal practice. Reading their source, all three use legacy
   `vkCmdPipelineBarrier` exclusively and none of them ships the sync2 layer. The precedent
   actually supports Option A.

Add to that: silently mutating a host process's environment variables from inside a `dlopen`ed
plugin is behaviour I would reject in code review on its own merits, independent of Vulkan.

**What survives.** Nothing that we ship. If an *Android integrator* independently packages
`libVkLayer_khronos_synchronization2.so` in their own APK and enables it, our sync2 backend will
light up automatically — because we probe the extension, and the layer's documented behaviour is
to advertise the extension and disable itself when the driver already provides it. That is a
**documented, optional, integrator-side deployment note** in `PLATFORMS.md`, not a mechanism we
depend on and not a substitute for the legacy path. Labelled as materially weaker, exactly as the
coordinator required.

**Option C (scope Android to a 2021+ population) is rejected outright** — it is the directive read
backwards.

### 7.4 `subgroup_size_control` — required as a *query*, never as a *feature*

**Ruling.** `subgroup_size_control` is **not** a device gate at all, and where we do consult it we
require only that the **properties struct is queryable**. We **never** require
`VkPhysicalDeviceSubgroupSizeControlFeatures::subgroupSizeControl == VK_TRUE`, and we never call
`vkCmdSetRequiredSubgroupSize`-style per-pipeline sizing as a correctness dependency.

Precisely what the engine does:

1. **Always** read `VkPhysicalDeviceSubgroupProperties::subgroupSize` and `supportedOperations`
   (Vulkan 1.1 core, universally available). This is the baseline knowledge.
2. **If** `VK_EXT_subgroup_size_control` is present *or* the device reports 1.3 core, chain
   `VkPhysicalDeviceSubgroupSizeControlProperties` into `vkGetPhysicalDeviceProperties2` and record
   `minSubgroupSize` / `maxSubgroupSize` / `requiredSubgroupSizeStages`. Treat this as *better
   information about the range*, nothing more.
3. **Only if** the `subgroupSizeControl` feature flag is additionally `VK_TRUE` may a pipeline be
   created with `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`. This is an *optimization
   path*, gated at pipeline-creation time.
4. **A shader whose correctness depends on a specific subgroup width may only be selected when the
   width is known exactly** — i.e. `minSubgroupSize == maxSubgroupSize`, or the required-size
   pipeline path from (3) is available and was used. Otherwise the engine selects the portable
   variant, which uses shared memory and workgroup barriers and makes no subgroup-width assumption.

Rule 4 is the substantive part and it is a correctness rule, not a performance rule. Assuming a
subgroup width silently produces wrong numbers in cooperative GEMM and reduction shaders; that was
the original reason for wanting this extension, and this formulation preserves the guarantee
without excluding anyone.

**Why this matters beyond macOS.** Requiring the feature flag would have excluded all of
macOS/iOS (MoltenVK reports `VK_FALSE`; Metal has no per-pipeline SIMD-group width control) and
very likely lavapipe — meaning both of our own CI lanes. A requirement that excludes the
machines you test on is a requirement you have not tested. Requiring the extension *string* would
still have cost 14.12 points of Android for information we can approximate from 1.1 core.

**Link's third open item — the 12.22% Windows `synchronization2` gap — is moot** under this
ruling. We accept nothing, because we require nothing; those devices run the legacy backend.

### 7.5 The barrier abstraction contract — binding on `ENGINE.md`

Switch wrote `ENGINE.md` §6.2 around a single `vkCmdPipelineBarrier2` path (§6.3 already noted a
fallback, but only as a sentence). This section is the contract that replaces it.

**Rule: one internal barrier API, two backends, selected exactly once at device init. Not
`if caps.sync2 { … } else { … }` at call sites.** A dual path scattered across the recorder is how
this decision turns into a bug farm; a single seam is how it stays a one-time cost.

The seam lives in **`rust/src/vk/barrier.rs`** and is the *only* file in the crate permitted to
name `vkCmdPipelineBarrier`, `vkCmdPipelineBarrier2`, `VkBufferMemoryBarrier`,
`VkBufferMemoryBarrier2`, `VkDependencyInfo`, or the `VK_PIPELINE_STAGE*` / `VK_ACCESS*` flag
families. The layering lint (§4.2) is extended to enforce this: those tokens outside
`vk/barrier.rs` fail CI.

Shape of the seam (illustrative — Switch owns the final signatures):

```rust
// rust/src/vk/barrier.rs — the ONLY module that names Vulkan barrier types.

/// Our own closed set. Deliberately contains no `None`/`NONE` variant: `VK_PIPELINE_STAGE_2_NONE`
/// has no legacy equivalent, so the abstraction must not be able to express it.
pub(crate) enum Access { ShaderRead, ShaderWrite, TransferRead, TransferWrite, HostRead, HostWrite }

pub(crate) struct BufferDep {
    pub buffer: vk::Buffer, pub offset: u64, pub size: u64,
    pub src: Access, pub dst: Access,
}

pub(crate) enum Barriers { Sync2(Sync2Backend), Legacy(LegacyBackend) }

impl Barriers {
    /// Chosen ONCE, in `Device::new`, from `Capabilities`. Never re-evaluated.
    pub(crate) fn select(caps: &Capabilities, dev: &ash::Device) -> Self;

    pub(crate) fn buffer_deps(&self, cb: vk::CommandBuffer, deps: &[BufferDep]);
    pub(crate) fn execution_only(&self, cb: vk::CommandBuffer, src: Access, dst: Access);
}
```

Binding requirements on the implementation:

1. **`Barriers::select` is called once, in `Device::new`, and the result is stored on the device
   handle.** `recorded.rs` and every op path call `dev.barriers().buffer_deps(...)`. No call site
   anywhere else may branch on `caps.synchronization2`.
2. **`Access` and `Stage` are our own closed enums, not Vulkan flag re-exports.** This is what makes
   the legacy backend total rather than best-effort: every value we can express has an exact 32-bit
   legacy equivalent by construction. `VK_PIPELINE_STAGE_2_NONE`, `SHADER_STORAGE_*`-only bits, and
   any other sync2-only concept are simply not representable.
3. **The mapping is one table, in one place.** `ShaderRead → (COMPUTE_SHADER, SHADER_READ)`,
   `ShaderWrite → (COMPUTE_SHADER, SHADER_WRITE)`, `TransferRead → (TRANSFER, TRANSFER_READ)`,
   `TransferWrite → (TRANSFER, TRANSFER_WRITE)`, `HostRead → (HOST, HOST_READ)`,
   `HostWrite → (HOST, HOST_WRITE)`. The sync2 backend widens the same table to the `_2_` flag
   names. If the two tables ever disagree, that is a bug in one file.
4. **Batching semantics are identical in both backends.** `buffer_deps` takes a slice and emits
   **one** barrier command covering all of them — `VkDependencyInfo` with N
   `VkBufferMemoryBarrier2` for sync2, one `vkCmdPipelineBarrier` with N `VkBufferMemoryBarrier`
   and OR-ed stage masks for legacy. The barrier-placement algorithm in `ENGINE.md` §6.2 does not
   change at all; only the emission does.
5. **Both backends are exercised on every CI run.** A new session option
   **`ep.force_legacy_barriers`** (bool, default `false`) forces `Barriers::Legacy` on a device
   that supports sync2. Trinity runs the differential suite twice per lane — once default, once
   forced — so the legacy path is never the untested path. Without this, the ~99%-Linux-coverage
   sync2 path is the only one our CI ever sees, and the 31% of Android we just bought would be
   running code no test has executed.
6. **`ENGINE.md` §6.2's worked example must be rewritten in terms of `buffer_deps`**, and §6.3's
   `vkCmdPipelineBarrier2` row must point at this section. §6.2's *reasoning* — per-edge barriers
   rather than one global barrier, one barrier per consumer edge — is correct and unchanged.

### 7.6 Why this and not the alternatives

**Why not a hard 1.3 baseline (Justin's original proposal).** It buys the two features Switch
wanted, which we have now stopped requiring anyway. It costs roughly **36 points of Android
installed-base coverage** (Fact Checker: 26% at 1.3 vs 62% at 1.1) and any MoltenVK older than
1.3.0, for zero engine simplification. The premise that motivated it — "llama.cpp requires 1.3" —
is contradicted by llama.cpp's own source at both the runtime check (floor 1.2) and the shader
target (base shaders `--target-env=vulkan1.2`), independently verified by Fact Checker (claims
1–2). It is also flatly incompatible with the compatibility-first directive.

**Why not a hard 1.2 baseline.** On Android the 1.2 tier barely exists — devices jumped 1.1 → 1.3
— so a 1.2 floor pays nearly the full Android cost of a 1.3 floor while getting less than 1.3
gives on desktop. The only 1.2 core feature we care about is timeline semaphores, which v0 does not
use and which are available as `VK_KHR_timeline_semaphore` on 1.1 when we do.

**Why not keep the two-extension requirement (the provisional 2026-07-28T17:59:54 position).**
Because Link measured it and it costs 31.43 points of Android and 12.22 points of Windows. It was
a defensible position under "we don't know the number"; it is indefensible now that we do.

**Why not require `synchronization2` on desktop only, and legacy on Android.** A per-platform
requirement is the worst of both: we still write both backends, and we additionally get a matrix
where a Windows-only contributor cannot reproduce an Android-only code path. If we are writing the
legacy backend at all, it must be the one that runs everywhere it is needed and gets tested
everywhere.

**What we lose by being this permissive.** Two things, both accepted: (1) `Barriers` has two
implementations forever, and the CI matrix doubles for barrier-sensitive tests (§7.5 item 5);
(2) a device can now be advertised, claim an op, and produce a slow result where a stricter gate
would have declined the device and let CPU handle it. Mitigation for (2) is Niobe's job, not the
gate's: if a device class is measurably worse than CPU, that is a *scoring* and *claim* decision
(§8), recorded per device class in `PLATFORMS.md` — not a reason to refuse to load.

### 7.7 Shader targets

SPIR-V is compiled with `--target-env=vulkan1.1` by default (SPIR-V 1.3), which every device
meeting §7.2 can consume. This is one notch below llama.cpp's `vulkan1.2` default and is the
conservative choice for Android breadth; if a base shader ever needs a 1.2-only SPIR-V capability
we raise the default and record it. Variants needing higher SPIR-V (fp16 arithmetic, integer dot
product, cooperative matrix) or a known subgroup width (§7.4 rule 4) are compiled as **separate
variants** and selected at runtime from `Capabilities` — the same split llama.cpp uses for its
`_cm2` shaders. Never a single fat module with runtime-dead capabilities; some drivers validate
the whole module.

Note that shader targets are **independent of the barrier decision**: `synchronization2` is a
host-side API, not a SPIR-V capability. Nothing in `shaders/` changes because of §7.3.

### 7.8 OQ-4 RESOLVED — the Vulkan SDK is a hard build dependency; there is no checked-in SPIR-V

> **Decided 2026-07-28T22:28:08-07:00.** This **changes the provisional decision** recorded in
> OQ-4 ("build-time `glslc` with a checked-in SPIR-V fallback so a plain `cargo build` works") to
> match what `build.rs` actually does. The doc was wrong, not the code. Binding on Switch, Tank,
> Link and Trinity.

**The decision.** `glslc` from the Vulkan SDK (or on `PATH`) is a **required build prerequisite**.
There is no checked-in `.spv` fallback and none will be added. A build without `glslc` fails with an
actionable error naming the SDK, `PATH`, and the escape hatch.

**Why I changed the decision rather than the code.** The fallback was meant to prevent exactly the
outcome the coordinator hit — a new contributor cloning the repo and getting a build failure. That
is a real cost and I am not dismissing it. But a checked-in fallback buys convenience by creating a
**silent-divergence hazard**, and it is a bad trade at our shader count:

1. **A checked-in `.spv` that no longer matches its `.comp` is undetectable at a glance and changes
   what runs.** CI (which has the SDK) would compile from source while a contributor without it ran
   a stale binary — and the two would differ in numerical behaviour with no signal. That is the
   "silently wrong" failure class this project's whole claim discipline exists to avoid, relocated
   into the build system. Freshness could be enforced by hashing, but then a stale artifact fails
   the build *anyway* and the fallback has bought nothing.
2. **"Reviewable diffs" — the original argument for checked-in SPIR-V — does not survive contact
   with the artifact.** SPIR-V binary diffs are not reviewable, and the shader-variant table already
   generates **168** modules from 69 rows. Every shader edit would rewrite a large set of binaries,
   making PR diffs unreadable and producing binary merge conflicts.
3. **The prerequisite is normal for this ecosystem and it is honest.** Several Vulkan projects
   require the SDK. A dependency stated in the README costs a contributor one install; a stale
   binary costs somebody a day of debugging a numerical difference that is not in the source.

This is the same principle as §6.4: **prefer the design where the hazard cannot occur to the one
where it is merely unlikely**, and pay a visible cost rather than accept an invisible one.

**Five binding conditions**, because a hard dependency is only defensible if it fails well:

1. **The failure message must be actionable without documentation.** It names the missing tool, how
   many shaders exist, the SDK, `PATH`, and the escape hatch. It already does; this pins it.
2. **The escape hatch (`ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1`) is for lint-only and
   docs-only lanes, and it must stay loud.** It emits a `cargo:warning` stating that the artifact
   can create no pipeline and must not be shipped. It already does; this pins it.
3. **A shader-less artifact must be inert at runtime, not subtly broken.** If `SHADER_MODULES` is
   empty, the EP must **advertise zero devices and claim nothing**, with a specific reason
   (`built without shaders`) in the log and in `CLAIM_DEBUG` — never load, claim nodes, and fail at
   pipeline creation. An escape-hatch build that looks like a working EP is worse than one that
   does not build. *Switch owns this guard; it is the one thing the current code does not yet
   enforce.*
4. **No release artifact may be produced from an escape-hatch build.** The release workflow asserts
   `SHADER_MODULES` is non-empty. *Trinity/Link.*
5. **The prerequisite is documented where a contributor meets it first** — root `README.md` (done),
   `rust/README.md` (*Tank*), and the CI setup step that installs it (*Link*, already true on both
   lanes). Plus a from-clean-clone lane that proves the documented prerequisites are sufficient,
   because a prerequisite list nobody tests is a prerequisite list that is wrong.

**What would reverse this.** Evidence that the SDK requirement is actually blocking contribution —
a platform where `glslc` is genuinely hard to obtain, or repeated contributor friction. The remedy
then is **not** checked-in SPIR-V; it is vendoring a compiler (`shaderc` as a Cargo dependency, or
`naga`), which removes the prerequisite without introducing a second source of truth. That is the
right escape route and it stays open.


---

## 8. Op coverage strategy and the v0 op set

### 8.1 Strategy

> **RATIFIED 2026-07-28T19:16:08-07:00 — OQ-11 closed. [`OP_COVERAGE.md`](./OP_COVERAGE.md) (Mouse)
> is the authoritative op-coverage plan and supersedes §8.2 and §8.3 of this document.** Verdict:
> **ratify with five amendments** (§8.4). §8.1's seven principles below are unchanged and Mouse
> endorses all seven; §8.2 is retained only as the M0/M1 floor; §8.3's qualitative fragmentation
> rule is **replaced** by Mouse's quantitative Minimum Viable Subgraph rule (`OP_COVERAGE.md` §7.2),
> which is a strict improvement — it converts a principle I could only state into a rule the
> partitioner can enforce, with a transfer cost *measured at device-init time* rather than a
> hardcoded constant that would be wrong on half our platform matrix.
>
> **Why I ratify.** Three things earn it. (1) The op list is derived from **emitted graphs** — the
> GenAI model builder source, contrib schemas, the ORT WebGPU EP registries — not from the ONNX spec
> index, which is the difference between a coverage plan and a wish list. (2) Every row carries a
> VERIFIED/UNVERIFIED mark and no UNVERIFIED row may be load-bearing for a tier exit criterion; that
> is the discipline I would have had to impose, arriving pre-imposed. (3) The central thesis —
> **87 tier-1 ops served by ~5 kernel templates, and the template infrastructure must exist before
> op #1** — is correct and is the only honest answer to "MLX did this in days". MLX did it in days
> because MLX already owned the kernels. Our leverage has to be manufactured, and §5 of that
> document is a credible manufacturing plan.
>
> **The part I most want on the record** is `OP_COVERAGE.md` §11.1's refusal to collapse two
> different claims into one: ~121 ops is weeks-scale; Qwen3.5 end-to-end is months-scale. A lead who
> lets those merge gets a project that reports 90% op coverage while running nothing. See §1.5.
>
> Constraints from the prior brief are all honoured: conservative claiming (§7.1 of that doc,
> restated verbatim), clean CPU fallback (unchanged), minimum viable subgraph size (§7.2, now
> quantified), and test-plus-platform-row on the same PR (§10). Ratification does not license
> shortcuts on any of them.

Mouse owns `docs/OP_COVERAGE.md`, `docs/OP_ARCHITECTURE.md` and the registry. The architectural
constraints on that work:

1. **One op = one handler + one claim predicate + one registration line, in one
   `ops/<family>.rs`.** Zero edits to `ep.rs`, `engine.rs`, or the registry core. If adding an op
   requires touching the boundary layer, the boundary layer is wrong — that is a bug report
   against L1, not a reason to edit it.
2. **Claim and translate share one table.** Claimed can never outrun translatable.
3. **Claim predicates validate everything:** domain, op type, opset range, input/output count and
   presence, dtypes, required attributes, static-shape availability, and broadcast form. When in
   doubt, do not claim.
4. **Every claimed op ships with a differential test against ORT CPU on the same PR.** No
   exceptions, no "tests in a follow-up."
5. **Every claimed op ships with a `PLATFORMS.md` row or an explicit "untested on X" note.** An op
   verified only on lavapipe is not verified.
6. **Coverage is grown in families, not alphabetically.** A family shares a shader skeleton, a
   descriptor layout, and a test file, so families amortize; scattered ops do not.
7. **fp32 first.** fp16 is a variant per family, gated on `shaderFloat16` +
   `storageBuffer16BitAccess`, added family by family once fp32 is green. No fp64, ever.
   *Amended:* this holds through tier 2. From tier 3 the LLM path is **fp16-native** — an fp32 KV
   cache for a real model is a memory-footprint failure, not a slow path. See §8.4 amendment A4.

### 8.2 The M0/M1 op floor

> Superseded as a *plan* by [`OP_COVERAGE.md`](./OP_COVERAGE.md), ratified 2026-07-28T19:16:08-07:00
> (§8.1). Retained as the minimum that must exist for the pipeline to be provable; nothing here is
> a ceiling. `OP_COVERAGE.md` T0/T1 subsume this and Mouse explicitly left T0 unchanged.

**M0 — one op, end to end.** `Add`, fp32, identical shapes, 2 inputs, 1 output, static shape.
That is the whole M0 claim set. It exists to prove the ABI, the device, the memory path, the
dispatch, and the test harness — not to be useful.

**M1 — the elementwise and shape families.**

| Family | Ops | Constraints |
|---|---|---|
| Binary elementwise | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Min`, `Max` | fp32; equal shapes or suffix/scalar broadcast |
| Unary elementwise | `Neg`, `Abs`, `Sqrt`, `Exp`, `Log`, `Reciprocal`, `Floor`, `Ceil`, `Round`, `Sign`, `Erf` | fp32 |
| Activations | `Relu`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Elu`, `HardSigmoid`, `Softplus`, `Clip`, `Gelu` | fp32; `Clip` with constant or absent min/max |
| Comparison / logic | `Equal`, `Greater`, `Less`, `GreaterOrEqual`, `LessOrEqual`, `And`, `Or`, `Not`, `Where` | fp32/bool; same broadcast rule |
| Cast | `Cast` | fp32 ↔ int32 ↔ bool only |
| Shape (metadata-only) | `Reshape`, `Squeeze`, `Unsqueeze`, `Flatten`, `Identity` | constant target shape; **no data movement** |
| Shape (copying) | `Transpose`, `Concat`, `Slice`, `Gather` | fp32/int32; constant axes/indices where the op allows |

Explicitly **not** claimed in M1, and each with a stated reason in the coverage table: anything
fp16/fp64, anything with a data-dependent shape, `Resize`, `Pad` with non-constant pads, and every
`com.microsoft` op.

**M2 — the compute families that make the EP worth using.**
`MatMul`, `Gemm`, `ReduceSum`/`ReduceMean`/`ReduceMax`/`ReduceMin`/`ReduceProd`, `Softmax`
(last-axis), `LogSoftmax`, `LayerNormalization`, `RMSNormalization`, `ArgMax`/`ArgMin`.
This is the first milestone where a real model (a small MLP, then a small CNN once `Conv` lands)
has a fused region big enough for a speedup to be meaningful rather than dispatch-bound.

### 8.3 The fragmentation rule (superseded — see `OP_COVERAGE.md` §7.2)

> Retained for its reasoning. The enforceable form is Mouse's Minimum Viable Subgraph rule.

A new op is worth claiming when it **connects** existing claimed regions or **extends** one at the
edge. An op claimed in isolation, in the middle of a graph of unclaimed ops, makes the graph
*slower* — two extra device round-trips for one dispatch. Mouse prioritizes by "does this merge
two islands", not by "is this op easy." Niobe's benchmark must report island count and largest
fused region, not just wall time, so this is measurable rather than folklore.

### 8.4 OQ-11 ratification — the five amendments

Ratified **with amendments**. Each amendment is binding on `OP_COVERAGE.md` at its next revision;
none of them requires a respin of the document.

**A1 — Contrib ops: in scope, with per-op discipline.** *Revised 2026-07-28T20:54:42-07:00 by user
ruling.* The domain question is settled above my level: `com.microsoft` is in scope. My amendment
survives the ruling intact because it was never about permission — it is about **how**. The three
parts that bind: an **UNVERIFIED contrib-op row may not be claimed at all** (Mouse's rule kept them
off *exit criteria*; I extend it to claiming, because a contrib op has no opset number to
range-check and a wrong claim there produces wrong logits rather than a decline); **there is no
domain-wide opt-in predicate in the code**, only per-op registry entries with hand-written semantic
gates; and **`tools/graph_census.py` running in CI against pinned artifacts is a precondition on
contrib op #1**, not on tier 3 — it is the only drift alarm we have for a surface with no version
number. The full protocol, including what to do when a schema changes shape under us, is §1.4
C1–C7. Contrib declines flow through the ordinary machine-readable decline path (§1.4 C3); no
special-casing.

**A2 — The template infrastructure is a milestone deliverable with its own exit criterion, not a
preamble.** `OP_COVERAGE.md` §11.1 is explicit that the schedule dies if ops #1–#20 are hand-written
before `indexing.glsl`, the `build.rs` variant generation, the `ops!` macro and the shared claim
helpers exist. I am making that structural rather than advisory: **M1 does not begin until the
template infrastructure is merged, and M1's exit criterion includes the ops-per-hand-written-kernel
ratio (≥ 8 in tiers 1–2) as a reported number.** A ratio that collapses is the earliest possible
signal that the thesis is failing, and it is measurable in week one rather than month three.

**A3 — `OP_COVERAGE.md` §7.2's MVS rule is adopted, with one addition.** The measured transfer-cost
calibration at device init is right, and the ~2 ms cost is acceptable. The addition: **the
calibration result and the MVS decision for every rejected candidate subgraph must be dumpable**
(`ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1`). A partitioning heuristic that silently declines a region is
indistinguishable from a missing op at the console, and we will otherwise spend real time
mis-diagnosing one as the other. Also: the SAFETY factor 3.0 and the `node_count >= 4` /
`64 KiB` floors are **provisional constants that must be re-derived from Niobe's measurements at the
M2 retrospective**, not settled values.

**A4 — fp16 is promoted from "a variant per family" to a tier-3 precondition, and its coverage risk
is escalated.** `OP_COVERAGE.md` §6.1 item 2 is correct that an fp32-upcast LLM path is a
memory-footprint failure, not a slow path. Under the frozen §7.2, `shaderFloat16` and
`storageBuffer16BitAccess` are **probed, not required** — so this is a live product question, not a
detail: if a meaningful fraction of Android lacks 16-bit storage, the LLM story is desktop-first no
matter what the op plan says. This is Mouse's OQ-M2 and I am escalating it to **OQ-14** in §11 with
Link as owner, because it decides a product boundary and not just a shader variant.

**A5 — Shape-agnostic push-constant kernel parameters are authorized for the LLM path from tier 3,
and this is a contract on Switch, not a request.** `OP_COVERAGE.md` §6.1 item 3 / OQ-M1 is right
that KV length growing every token makes this structural. It is also much cheaper to decide now
than to retrofit through every LLM kernel. Ruling: **from tier 3, every LLM-path kernel takes its
dimensions in push constants; the recorded command buffer is length-agnostic for the decode loop.**
The §1.2 non-goal on dynamic shapes is amended accordingly and narrowed to "no dynamic-shape *fast
paths* outside the LLM decode path in M0–M2". Switch must reflect this in `ENGINE.md`'s recording
model alongside the barrier seam. Note the interaction the coverage plan did not call out: a
push-constant-parameterized dispatch still needs a **workgroup count** that depends on the sequence
length, so either the recording is re-issued per shape bucket anyway or we need
`vkCmdDispatchIndirect` with a device-computed count. **I want indirect dispatch evaluated, not
assumed** — it is the same mechanism `QMoE`'s data-dependent routing will need (`OP_COVERAGE.md`
§11 risk 8), so one answer serves both. Assigned to Switch as OQ-15.

**Not amended, explicitly.** Mouse's `ops!` macro with the machine-readable `caps` column
(`OP_COVERAGE.md` §5.7) is a genuine improvement on the MLX reference and I endorse it without
qualification — a hand-maintained support matrix across five vendors, two dtypes and
optional-capability shader variants *is* the "claims support but silently gets it wrong" failure
mode my §8.1 item 2 exists to prevent. Generating `OP_SUPPORT.md` and `--dump-capabilities` from the
same table is exactly right. Likewise the compose-before-bespoke rule (§5.6) with its short fusion
allowlist, the three claim-policy traps inherited from the MLX project's scars (§7.1), and the
metric contract with Niobe (§7.3) are adopted as written.

**Licence dependency, now closed.** Mouse names reading llama.cpp's MIT-licensed Vulkan shaders as
the single largest available accelerant for the three XL kernels. **Rai ruled OQ-M6 🟢 Green on
2026-07-28T19:16:08-07:00** (`.squad/decisions/inbox/rai-oq-m6-license-ruling.md`,
`docs/THIRD_PARTY.md`): reading is permitted with no attribution obligation; substantial adaptation
triggers a file header, a `THIRD_PARTY_NOTICES.md` entry, a commit-message note, and distribution of
the notices file with any binary containing the adapted SPIR-V. My timeline view in §1.5 assumes
this accelerant is used — **read the tiling and subgroup strategies, do not port the code**, and
treat any shader that ends up substantially adapted as triggering Rai's conditions. Without it I
would widen the tier-3/4/5a estimates rather than hold them.

**Clarification, 2026-07-28T22:28:08-07:00 — the accelerant is still available and the timeline
does not widen.** `OP_COVERAGE.md` §13.2 records every XL kernel as an *independent implementation
with no obligation attaching*, which reads at first glance as though the accelerant I priced in had
evaporated. It has not, and the distinction is the same one I stated when I priced it. Mouse's rows
say, in their own text, "read llama.cpp's flash-attention Vulkan shaders for tiling and
subgroup-reduction strategy" and "read `mul_mat_vec_q*` for the memory-access strategy" — that *is*
algorithm study, which is what I assumed and what Rai's 🟢 permits without obligation. What he
declines is **source adaptation**, which I also declined ("do not port the code"), and his reason for
declining it on `MatMulNBits` is stronger than mine: llama.cpp's block formats are not ONNX's, so
adaptation would not even work there. His §13.2 closes by saying the study "removes the largest
single unknown from the XL-kernel estimates". Those are the same position, and no obligation
attaching is the *expected* outcome of using the accelerant correctly, not evidence of its absence.

So: **the estimates hold and I am not widening T3/T4/T5a.** What I would need from Switch's
independent read to keep them — stated now, before his answer arrives, so it is a test rather than a
rationalization:

1. **For `GroupQueryAttention`**: that the flash-attention tiling schedule and the subgroup-reduction
   shape are transferable *independently of data layout* — i.e. that the useful content is the loop
   structure, the online-softmax rescaling order, and what lives in shared memory versus registers.
   If Switch reports the schedules are entangled with ggml's KV layout such that a reader gains
   little, T3 widens.
2. **For `MatMulNBits`**: that dequant-in-register patterns and the memory-access strategy transfer
   even though the block format differs. This is the one I would most expect to be weaker, and it is
   the one Mouse already flagged. If the answer is that the access strategy is a consequence of the
   block format and does not survive changing it, T4 widens.
3. **For `LinearAttention`**: nothing — there is no reference to read, Mouse says so, and my estimate
   never assumed one. This tier's risk is OQ-16 (an unreleased schema), not licensing.

If Switch's read is negative on 1 or 2, I widen the corresponding tier and say so plainly rather
than absorbing it, per §1.5. A timeline that quietly absorbs a lost accelerant is a timeline that
has started lying.

**RULING, 2026-07-28T22:28:08-07:00 — Switch's read has landed and the estimates HOLD. T3, T4 and
T5a are not widened.** I am stating this explicitly rather than letting the estimates stand by
default, because a schedule that survives a challenge by silence has not actually survived it.

Switch (D-S4-10), who read llama.cpp's shader pipeline closely while writing `ENGINE.md`, judges
Mouse's "adaptation would not even work" too strong. His finding, against my pre-committed test
above:

| Test | Switch's answer | Result |
|---|---|---|
| 1. GQA — flash-attention tiling and subgroup-reduction shape transfer independently of layout | **Yes.** GEMV-vs-GEMM tile-size specialisation-constant structure and the per-lane partial dot → `subgroupAdd` reduction shape are layout-independent. | **T3 holds** |
| 2. `MatMulNBits` — dequant-in-register and access strategy survive a different block format | **Yes, partially and usefully.** The ONNX nibble layout genuinely is incompatible with llama.cpp's K-quant structs — no code moves — but dequant-in-register *patterns* and the memory-access strategy transfer as algorithmic reference. | **T4 holds** |
| 3. `LinearAttention` — no reference assumed | Unchanged; there is nothing to read. Its risk is OQ-16. | **T5a holds** (on this axis) |

**The two of them are not actually in conflict, and it is worth being precise about where they
differ**, because the difference is small and the licensing conclusion is identical. Both agree we
write our own code and that **no obligation attaches** — that is settled and Rai's 🟢 covers it.
They differ only on whether *reading* still saves time given the block-format mismatch. Mouse
reasoned from the artifact (the structs are different, so nothing can be copied), which is correct
and is exactly why no obligation attaches. Switch reasoned from the *schedule* (what does a reader
learn that they would otherwise have to derive), which is the question my estimates actually depend
on. **The distinction that resolves it: the block format dictates the innermost unpack, not the
tiling schedule or the reduction shape** — and it is the latter two that cost weeks to get right on
unfamiliar hardware, not the former.

So the accelerant survives in Switch's narrower form, and that narrower form is the one I priced in:
tiling and reduction structure as algorithmic reference, never the quantization layout. Two riders
on holding the estimates:

- **Budget the study time explicitly.** Switch asks that items 1–2 carry real algorithm-study time
  rather than being treated as free. Agreed — "we may read llama.cpp" is not the same as "reading it
  is instantaneous." That reading is inside the T3/T4 estimates, not a discount applied to them.
- **The `MatMulNBits` unpack is ours from first principles.** The one part Mouse is unambiguously
  right about gets no reference and no schedule relief; it is dictated by the ONNX contrib schema.

**General rule from this exchange, since it will recur:** when two owners appear to disagree, check
whether they are answering the same question before adjudicating. Here one answered "can this be
copied?" and the other "does reading it help?" — both correctly, about different things. My error
would have been to widen three tiers on the first answer when my estimates depended on the second.

---

## 9. Testing and benchmarking strategy

### 9.1 Differential testing against the ORT CPU EP — Trinity

The oracle is **ORT's own CPU EP**, running the same ONNX model. Not numpy, not a reference we
wrote. This is the single most important testing decision in the project: it means a test failure
is unambiguous, and it means we cannot accidentally encode our own misreading of an ONNX spec into
both the implementation and the expectation.

| Layer | Location | Purpose | Gate |
|---|---|---|---|
| Op correctness | `tests/ops/` (pytest) | Per-op, per-dtype, per-shape differential vs ORT CPU. Models built with the ONNX IR API in `_models.py`. | **Required on every PR.** |
| Claim assertion | `tests/ops/test_claim_diagnostics.py` | Asserts the node *actually ran on `VulkanExecutionProvider`*, via the claim diagnostics. Prevents vacuous CPU-fallback passes. | **Required.** |
| ONNX backend node tests | `tests/backend/` | The ONNX project's own node tests through the EP. | Required. |
| Conformance fuzzing | `tests/conformance/` | Bounded property-based fuzzing of claimed ops against the ONNX standard, one op per subprocess so a native crash cannot abort the run. | Opt-in `workflow_dispatch`. |
| Validation layers | all suites, debug builds | `VK_LAYER_KHRONOS_validation` clean is part of "done" for any engine change. | Required in the CI debug lane. |
| Leak / teardown | stress scripts across many sessions | RAII teardown leaves no `VkDeviceMemory`, no pipelines, no descriptor pools. | Required. |
| **Barrier-backend parity** | every lane, run twice | The full suite with the default backend and again with `ep.force_legacy_barriers=1`, asserting **identical** numerical results (§7.5 item 5). Without this, the legacy `vkCmdPipelineBarrier` path we carry for 31% of Android and 12% of Windows would never be executed by any test we own. | **Required on every PR.** |

**The vacuous-pass trap, stated plainly.** Because CPU fallback is always correct, a test that
merely compares outputs will pass whether or not the EP ran anything. Every op test **must** assert
the claim. This is non-negotiable and it is the first thing I will look for in a review.

**Tolerances.** Stated per family in `tests/ops/`, defaulting to `rtol=1e-5, atol=1e-5` for fp32
elementwise. Reductions and GEMM get looser tolerances tied to accumulation order — but a
tolerance is *derived and documented*, never widened to make a red test green. Widening a
tolerance requires Trinity's sign-off and a note in the test.

**Cross-platform.** The same suite runs on every CI lane. As landed by Trinity
(2026-07-28): a **Linux lavapipe lane** (via `apt`) and a **real Windows Vulkan lane** — she found
lavapipe Windows prebuilts via **mesa-dist-win 26.1.3**, driven by `VK_ICD_FILENAMES`, so our
primary development platform has genuine correctness coverage rather than build-only coverage, which
materially raises the value of a green CI run. There is also an always-on no-ICD fallback assertion,
which is exit criterion 4 of M0 tested continuously rather than once. **SwiftShader was evaluated
and rejected** — no usable prebuilts and a ~20-minute build from source — so lavapipe is the only
rasterizer we run and the only one this document should name. ORT is pinned to 1.28 across lanes
(§1.4 C2 depends on that pin). Claims are asserted from the profiling JSON, so "it ran on Vulkan" is
proven rather than assumed. A pass on a software rasterizer alone remains a smoke test, not a
correctness claim — lavapipe does not reproduce driver-specific subgroup, denorm, or precision
behaviour, and **there is currently no CI coverage of any physical GPU, on any platform**; §11.1's
on-device work is what begins to close that gap. Link owns which lanes exist; Trinity owns what runs
on them.

#### 9.1.1 The oracle is validated, not assumed — and what validating it cost

§9.1's premise had one genuinely open risk: the CPU EP is an excellent oracle for `ai.onnx` float
ops, but the quantized contrib path (§1.4) is a different animal, and if the CPU EP could not serve
as an oracle for a GenAI-built int4 graph then the differential strategy for the *entire* quantized
half of the project — the half containing all three XL kernels — would have needed rethinking
before M2. Trinity ran the experiment rather than reasoning about it. **Result (2026-07-28): the
ORT CPU EP is usable as a differential oracle for the quantized path. §9.1 stands unchanged and no
rework is required.**

Two findings emerged that only running it could have produced, and both are now binding:

1. **The oracle is pinned to `accuracy_level=1`.** `MatMulNBits`' `accuracy_level` 4 (int8/VNNI)
   diverges from levels 0–3 by ~3.6e-3 at K=1024, N=512. Unpinned, ORT would select a level based
   on the CPU it happens to be running on, so our reference values would have drifted **silently
   across CI runner hardware** — and the resulting flake would have looked like a bug in our
   kernel. Generalized as a rule: **any oracle knob whose value the runtime chooses from the host
   machine must be pinned explicitly and recorded in the test, not left to default.** An oracle
   that changes with the machine is not an oracle. This applies to future additions too — thread
   count, arena settings, graph-optimization level.
2. **fp16 activations produce NaN/Inf on ORT 1.27**, independently reproducing the null-allocator
   `PrePack` bug Fact Checker found and that drove Tank's version pin. The fp16 oracle test is
   gated on ORT ≥ 1.28 and runs in CI. Three independent lines of evidence — Fact Checker's source
   read, Tank's build experience, Trinity's numerical failure — now converge on 1.28, so the
   `ORT_API_VERSION_MIN = 24` compile floor with a **pinned runtime** of 1.28 is settled on
   evidence rather than on caution.

**The one documented exception to "the CPU EP is the oracle".** Trinity implemented Mouse's
three-regime tolerance policy but checks **dequantize bit-exact against NumPy, not against the CPU
EP**. This is correct and I am ratifying it as the general rule for *layout* semantics: where a
kernel's job is to interpret a bit layout defined by a schema, comparing against another
implementation of that same schema does not test the reading — a shared misreading passes on both
sides, which is precisely the failure mode §9.1 exists to prevent. So: **behaviour is checked
against the CPU EP; bit-layout interpretation is checked against an independently written
specification of the layout.** The two are complementary and neither substitutes for the other.
This is C6 (§1.4 — per-layer verification, never final-logits) applied one level down, and the two
now reinforce each other: with per-layer capture implemented, a wrong kernel is *locatable* without
comparing logits across a 150k vocabulary, and with bit-exact dequant checks a wrong *unpack* is
locatable without running a kernel at all.

**No autouse EP fixture.** `conftest` no longer registers the EP as an autouse fixture, so tests
that do not require Vulkan actually execute instead of skipping wholesale. This changes what a
green run means, which is why it belongs in this document: 206 collected and 8 passing **with no EP
built at all** is now a meaningful signal rather than a vacuous one — and among those 8 is the
runtime half of C1, confirming that `com.microsoft::NotARealOp` takes the ordinary decline path.
**C1 is therefore enforced from both ends**: Tank's static ban on the domain as a *value* in
`layering.rs` (no `==`, `!=`, `matches!`, `if let`, `starts_with` can express it) and Trinity's
runtime assertion that an unregistered contrib op declines like any other unregistered op. A
constraint checked only statically can be satisfied by code that never runs; a constraint checked
only at runtime can be reintroduced in a path no test reaches. C1 now has neither hole.

#### 9.1.2 Execution status — what has actually run, as of 2026-07-28T22:28:08-07:00

This document describes a design and a partially-implemented crate. It must not be read as
describing a working GPU pipeline, and the following is stated here so that no reader has to infer
it:

- **No shader in this repository has ever been executed on any device.** As of this revision the
  development machine has no Vulkan ICD installed and no `glslc` (§7.8), the shader corpus is a
  variant table plus GLSL sources that have not been compiled here, and Switch's device path is
  vocabulary, seams and stubs rather than a live submit loop.
- **Trinity's lavapipe lanes are the only place anything will execute on a device**, and they
  execute on a software rasterizer, which §9.1 already qualifies as a smoke test rather than a
  correctness claim.
- Every "green" count reported to date — 227 tests at the `cbb1a0d` level, 206 collected / 8 passing
  in the no-EP configuration — measures **host-side logic**: claim predicates, registry invariants,
  the layering lint, decline paths, and the harness itself. That is real work and it is exactly what
  has to be right before a kernel is worth writing, but it is not evidence about numerics on a GPU.
- The first genuine execution evidence is M0's exit criteria (§10); the first evidence about
  *vendor* hardware is §11.1's on-device experiment, which has no hardware yet.

The rule this encodes, and it applies to every document in `docs/`: **a test count is a claim about
what was executed, and it must not be allowed to imply more execution than occurred.** The same
discipline that produced the RAI-003 platform disclosure in `README.md` and Link's
unverified-usability statement in `PLATFORMS.md` §8 applies to our own test numbers.

### 9.2 Benchmarking — Niobe

- **Baselines are versus the ORT CPU EP on the same machine, same model, same ORT build.** Any
  other comparison is marketing.
- `bench/` reuses `tests/ops/_models.py` builders so the benchmark cannot drift from what is
  tested.
- Reported per case: median wall time on Vulkan, median on CPU, ratio, **and** the claim
  diagnostics — island count, largest fused region, node count claimed. A speedup number without
  those three is not accepted.
- GPU-side timing uses `VkQueryPool` timestamp queries once the engine exposes them, so we can
  separate submit overhead from actual GPU time. Sub-millisecond cases are dispatch-bound and will
  be slower than CPU; that is expected, must be labelled, and must not be hidden.
- `bench.yml` posts an informational base-vs-PR table. **It does not gate**, because shared-runner
  timings are noise. It flags a regression as a prompt to re-measure locally.
- **No performance claim leaves this repo before the corresponding op is green in `tests/ops/` on
  at least one real GPU.**

---

## 10. Milestones

Each milestone's exit criteria are verifiable by a command, not by an opinion.

### 10.0 What admitting contrib ops does to the milestones — stated without smoothing

M0–M3 were originally written under an `ai.onnx`-only assumption. The user ruling of
2026-07-28T20:54:42-07:00 changes that assumption, and the honest accounting is:

**M0 and M1 are unaffected.** Neither contains a contrib op. M0 is one `Add`; M1 is the 87-op T1
elementwise/shape/indexing surface, all `ai.onnx`. Nothing about the contrib ruling accelerates or
delays them, and nothing about it should be allowed to *reprioritize* them — the kernel-template
infrastructure that M1 gates on is what makes tiers 1–2 weeks-scale, and skipping ahead to a contrib
kernel because it is closer to the visible goal is the single most expensive mistake available to
this project.

**M2 gains one obligation and remains the critical path.** Its exit criterion is deliberately
BERT-base with *primitive* attention and no contrib ops, because M2's job is to prove the device
allocator and the transfer path, not to prove an attention kernel. The added obligation is §1.4 C2's
drift alarm: `graph_census.py` in CI, with pinned artifacts and per-op claim rates, must exist
before the first contrib op lands — which means it is built in M1/M2, not in M3 when it is needed.
M2 remains the critical path for everything after it, because a KV cache that round-trips to host
every token is not an inference engine.

**M3+ is where the whole cost lands, and it does not compress.** Tiers T3–T5c are gated on three XL
kernels with no template leverage — `GroupQueryAttention`, `MatMulNBits`, `LinearAttention` — plus
`CausalConvWithState`, `QMoE` and the vision path. These are each one person's deep work and they
are **not parallelizable away**; three people on three kernels is the maximum useful parallelism and
we do not have three people free. Rai's 🟢 on OQ-M6 (§8.4) is the one real accelerant: reading
llama.cpp's MIT-licensed Vulkan shaders for tiling and subgroup strategy is authorized, and my
estimates assume it is used. Without it I would widen T3/T4/T5a rather than hold them.

**The reconciliation with the ambition, stated plainly.** Justin's target is high coverage on a
weeks-to-months horizon with Qwen3.5 end-to-end as the real goal. Mouse's split is honest and I
ratify it unchanged: **tiers 0–2 (121 ops) are weeks-scale; Qwen3.5 end-to-end is months-scale.**
Admitting contrib ops does not shorten the second number — it is what makes the second number
*reachable at all*, which is a different and more important thing. Before the ruling, "Qwen3.5
end-to-end" had no completion path; now it has a long one. I am not going to present those as the
same kind of progress.

The guard against presenting them as the same kind of progress is `largest_island_flops` (§9.2).
The op count will look excellent for weeks before any LLM runs, and a milestone report that leads
with op count is a milestone report that is hiding something. Every milestone from M1 onward reports
`largest_island_flops` on the corpus artifacts alongside whatever else it reports — including on the
Qwen artifacts, where for most of M1 and M2 it will be near zero and *should* be, because that
number going up is the only thing that means the named target is getting closer.

### M0 — "It loads, it runs, it matches"

> **A stock ORT loads the plugin, enumerates a Vulkan device, runs a graph containing a single
> `Add` node on that device, and the output matches the ORT CPU EP within tolerance — on both
> Windows and Linux, on a software rasterizer, in CI.**

| Work | Owner |
|---|---|
| `Cargo.toml`, `build.rs` (ORT bindgen + GLSL→SPIR-V embedding), crate scaffolding | Tank |
| `lib.rs`, `factory.rs`, `ep.rs` — factory/EP/compute-info vtables, panic guards, RAII teardown | Tank |
| `vk/` — instance, physical-device scoring + capability gate (§7.2), device, queue, allocator, staging, descriptor pool, pipeline cache, command recording, single-fence submit | Switch |
| **`vk/barrier.rs` — the barrier seam of §7.5: `Access`/`BufferDep` enums, `Barriers::select`, and *both* the `Sync2Backend` and `LegacyBackend` implementations** | Switch |
| `vk/caps.rs` — `Capabilities` probe incl. `synchronization2`, `subgroup_size_control` **properties-only** query (§7.4), subgroup ops, fp16 | Switch |
| `shaders/elementwise_binary.comp` + the SPIR-V embedding pipeline | Switch |
| `registry.rs` + `NodeView`; `ops/elementwise.rs` with `Add` claim + handler; claim diagnostics | Mouse |
| `engine.rs` — `NodeDesc`, `Plan`, `DispatchContext`; per-run command recording (no cache yet) | Tank + Morpheus (contract) |
| `tests/ops/conftest.py`, `_models.py`, `test_elementwise.py`, claim assertion helper | Trinity |
| CI: fmt, clippy, build on windows-latest + ubuntu-latest, **lavapipe on both** (Linux via `apt`, Windows via mesa-dist-win 26.1.3 with `VK_ICD_FILENAMES`), **the `ep.force_legacy_barriers=1` duplicate lane (§7.5 item 5)**, Vulkan SDK provisioning, layering lint | Link + Trinity |
| `python/` package with `register_execution_provider_library()` | Tank |
| Baseline harness stub; no numbers published | Niobe |

**Exit criteria.**
1. `cargo build --release` and `cargo clippy -- -D warnings` clean on Windows and Linux.
2. `pytest tests/ops -q` green on both, with the claim assertion proving `Add` ran on
   `VulkanExecutionProvider`.
3. Validation layers report zero errors and zero warnings in the debug lane.
4. A machine with no Vulkan ICD loads the plugin, advertises zero devices, logs a warning, and the
   session still runs on CPU.
5. **A shader-less build (`ALLOW_MISSING_GLSLC=1`) advertises zero devices and claims nothing, with
   a `built without shaders` reason** — it never loads, claims, and then fails at pipeline creation
   (§7.8 condition 3).
6. `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` prints per-op decline reasons.
7. The layering lint is in CI and fails a deliberately-planted violation, including a planted
   `cmd_pipeline_barrier` outside `vk/barrier.rs` (§4.2, §7.5).
8. **The full test suite passes twice per lane — once with the default barrier backend and once
   with `ep.force_legacy_barriers=1` — with identical numerical results** (§7.5 item 5).
9. Both sibling docs and this one are consistent; §12 lists every divergence.

### M1 — "A useful elementwise EP" (`OP_COVERAGE.md` tier T1 — 87 ops)

**Gate: M1 does not begin until the kernel-template infrastructure is merged** (§8.4 A2) —
`indexing.glsl`, the `build.rs` dtype/capability variant generation, the `ops!` table macro, and the
shared claim helpers. Hand-writing ops #1–#20 before the templates exist forfeits the entire
leverage thesis and, with it, the schedule.

The T1 op set, claimed, tested, and documented; shape-keyed command-buffer recording cache
(`recorded.rs`); convex clustering with multi-node fused subgraphs; the generated `OP_SUPPORT.md`
coverage table as the authoritative contract.

| Work | Owner |
|---|---|
| **Kernel-template infrastructure (gate), then** op families + claim predicates + the `ops!` table | Mouse |
| Shader variants, spec constants, broadcast indexing, descriptor layout per family | Switch |
| Convex clustering, `recorded.rs`, arena-based temporaries | Tank |
| Per-family differential tests, tolerance policy, `tests/backend/` node tests | Trinity |
| `bench/` harness + first published baselines with island counts | Niobe |
| macOS/MoltenVK lane; first real-GPU lane if a runner is available; driver quirk log | Link |
| `tools/graph_census.py` + node histograms for all 7 corpus artifacts (`OP_COVERAGE.md` §2.2) | Mouse + Trinity |

**Exit criteria.** Every T1 op green vs CPU on ≥2 platforms; **a pure-elementwise graph of ≥20 nodes
compiles to one island, one submission**; a shape change re-records once and then replays;
`OP_SUPPORT.md` is generated from the registry and matches it by construction; `graph_census.py`
exists and has produced histograms for the full corpus; and **the ops-per-hand-written-kernel ratio
is reported and is ≥ 8** (§8.4 A2).

### M2 — "Real memory, real compute" (`OP_COVERAGE.md` tier T2 — 33 ops, cum. 121)

Device allocator + data transfer (§6.3), `GetDefaultMemoryDevice` returning a real device, and T2's
reductions / `MatMul` / `Gemm` / norms / softmax. This is the first milestone that can honestly
claim a speedup. **It is also the critical path for everything after it** — the LLM tiers are
worthless without the device allocator, because a KV cache that round-trips to host every token is
not an inference engine (`OP_COVERAGE.md` §6.1 precondition 1). That dependency is Tank's and
Switch's, not Mouse's, and it should be resourced accordingly.

| Work | Owner |
|---|---|
| `allocator.rs`, `transfer.rs`, **reserved-VA span registry (§6.4, OQ-3 resolved)** incl. platform probe-and-halve sizing, `OrtMemoryDevice` wiring | Tank |
| Device-local arena, staging ring, coherence handling, barrier batching | Switch |
| Reductions / `MatMul` / `Gemm` / `Softmax` / norms — semantics, claim, tiling; the fused Softmax and RMSNorm kernels from the §5.6 allowlist | Mouse + Switch |
| MVS transfer-cost calibration at device init + its claim-debug dump (§8.4 A3) | Mouse + Switch |
| Timestamp queries and the trace exporter | Switch + Niobe |
| Numerical tolerance policy for accumulation-order-sensitive ops (OQ-10) | Trinity |
| GPU CI lane; per-vendor result matrix | Link |
| First published speedup table, with island counts and CPU baseline | Niobe |

**Exit criteria.** A small MLP **and a BERT-base encoder using primitive attention (no contrib
ops)** run end-to-end on Vulkan; with `IoBinding`, per-inference host↔device traffic is **zero** for
a fully-claimed graph; a measured speedup vs the ORT CPU EP on at least one real discrete GPU,
published with methodology; and the MVS constants are re-derived from Niobe's measurements rather
than left at their provisional values (§8.4 A3).

### M3+ — "Breadth and platforms" (`OP_COVERAGE.md` tiers T3–T6)

Sequenced by `OP_COVERAGE.md` §6, not re-sequenced here. **Entry precondition for the whole of M3:**
`tools/graph_census.py` runs in CI against pinned `.onnx` artifacts and reports per-op claim rates
(§1.4 C2). No contrib op is claimed before that alarm exists. The shape of M3+:

| Tier | Target | Gating item |
|---|---|---|
| T3 | Qwen3-0.6B fp16, GenAI-built, KV cache, correct tokens end-to-end, ≤2 islands | `GroupQueryAttention` (XL); M2's allocator; fp16 (OQ-14); push-constant shapes + OQ-15 |
| T4 | Qwen3-1.7B int4, correct tokens, ≤2 islands, beats ORT CPU on ≥2 vendors | `MatMulNBits` (XL) + weight prepacking |
| T5a | **Qwen3.5 hybrid end-to-end — the named target of the directive** | `LinearAttention` `gated_delta` (XL) + `CausalConvWithState` |
| T5b | Qwen3-MoE int4 with the expert block on Vulkan | `QMoE`; likely needs indirect dispatch (OQ-15) |
| T5c | Qwen-VL vision tower + projector feeding the decoder in one session | `Conv` (patch-embed form), `MultiHeadAttention` |
| T6 | ResNet-50 / MobileNetV3 end-to-end, beating ORT CPU | General `Conv`/pooling breadth |

Android hardware validation (§11.1's OQ-12 experiment) runs in parallel with T3–T4 and is gated only
on devices, not on op coverage. The three XL kernels are **not parallelizable away** — each is one
person's deep work — and §1.5's months-scale claim rests on them.

---

## 11. Open questions

| # | Question | Decided by | Blocks |
|---|---|---|---|
| **OQ-1** | ~~How many real devices report Vulkan 1.1/1.2 **without** `VK_KHR_synchronization2` or `VK_EXT_subgroup_size_control`?~~ **RESOLVED 2026-07-28T19:16:08-07:00.** Link measured it (`PLATFORMS.md` §8, vulkan.gpuinfo.org 2026-07-28): `VK_KHR_synchronization2` is missing on **31.43% of Android** and **12.22% of Windows**; `VK_EXT_subgroup_size_control` on **14.12% of Android**, and its *feature flag* is `VK_FALSE` on all of macOS/iOS. **Ruling (§7.2–§7.5): both are dropped from the hard requirement.** `synchronization2` becomes a probed capability selecting one of two barrier backends behind a single seam (`vk/barrier.rs`); `subgroup_size_control` is consulted as a *properties query* only and never as a required feature. Link's layer-shim option is **rejected** — the AOSP loader cannot discover a layer we ship from a plugin `.so`, and the cited wgpu/Dawn/Godot precedent turned out to be legacy-barrier-only in all three. | Link investigated → **Morpheus decided** | — (§7 is frozen) |
| **OQ-2** | ~~Do llama.cpp and ExecuTorch's stated version floors survive verification?~~ **RESOLVED 2026-07-28T17:59:54-07:00.** Fact Checker claims 1–2: both "requires 1.3" claims **contradicted**. llama.cpp base shaders target `vulkan1.2` (only `_cm2` variants target 1.3); ExecuTorch hardcodes `VK_API_VERSION_1_1`. Claim 4 (Android share) remains *unverified but plausible*. | **Fact Checker** (done) | — |
| **OQ-3** | ~~The ORT allocator's pointer problem (§6.3): ORT allocators return `void*`, a Vulkan allocation is a `(VkBuffer, offset)` pair.~~ **RESOLVED 2026-07-28T22:28:08-07:00 — see §6.4.** `Alloc` returns a span of **reserved, never-dereferenceable virtual address space** (`VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` / `mmap(PROT_NONE, MAP_NORESERVE)`), resolved to `(VkBuffer, offset)` through an opaque-handle registry once per descriptor binding. **`VK_KHR_buffer_device_address` is not carried at all** — Tank's argument that BDA is a second *shader architecture* rather than an optimization is correct and superseded my "registry primary, BDA on top" framing. Reserved VA makes ORT's pointer arithmetic correct by construction and turns a stray dereference into an MMU fault instead of silent corruption. Android's narrower address space is handled by **probe-and-halve at construction**, not by a platform constant — a tuning parameter, not a blocking dependency on Link. | Tank proposed → **Morpheus decided** | — |
| **OQ-13** | **Zero-copy IO binding via `OrtEpFactory::CreateExternalResourceImporterForDevice`.** *New, 2026-07-28T19:16:08-07:00.* Verified by Fact Checker: the public vtable member is `CreateExternalResourceImporterForDevice` (the `…Impl` suffix is a local static in test code, not API), it landed in **ORT 1.24** — not 1.28 — and Tank has already set `ORT_API_VERSION_MIN = 24` with version negotiation, so **it costs us no ABI floor movement.** It is **orthogonal to OQ-3**: it is an OS-handle external-memory path in which the *caller* exports their `VkDeviceMemory` via `vkGetMemoryWin32HandleKHR` / `vkGetMemoryFdKHR` and we re-import it, answering "how does an external caller hand us their buffer as a graph input/output", not "what does our `Alloc()` return". Tank and Fact Checker independently reached this and Tank has recorded it as evaluated-and-rejected for OQ-3; **it is not to be re-proposed there.** Tracked here on its own merits: it is real, supported upstream, has an in-tree reference (`onnxruntime/test/providers/nv_tensorrt_rtx/nv_vulkan_test.cc`), and is the complete answer to zero-copy IO binding. **Scope: post-M2**, because it presupposes the device-memory tensor path exists. Known constraint to design around: the caller's memory must have been allocated with `VkExportMemoryAllocateInfo` up front — it cannot be retrofitted onto an ordinary allocation, so this is an integration contract we must document, not a transparent optimization. | **Tank** designs → Morpheus reviews | post-M2 |
| **OQ-14** | **What fraction of target devices support `shaderFloat16` + `storageBuffer16BitAccess`?** *Escalated from Mouse's OQ-M2, 2026-07-28T19:16:08-07:00.* Under the frozen §7.2 both are probed, not required. An fp32-upcast LLM path is a memory-footprint failure, not a slow path (§8.4 A4), so a low Android number means the LLM story is **desktop-first as a product boundary**, regardless of op coverage. This decides a product scope, which is why it is not a shader-variant detail. | **Link** measures → **Morpheus** rules on scope | tier 3 / M3 |
| **OQ-15** | **Indirect dispatch.** *New, 2026-07-28T19:16:08-07:00.* Shape-agnostic push-constant kernel parameters (§8.4 A5) make the *shader* length-agnostic but not the **workgroup count**, which still depends on sequence length — so either we re-record per shape bucket anyway, or we use `vkCmdDispatchIndirect` with a device-computed count. The same mechanism is what `QMoE`'s data-dependent expert routing needs on a pre-recorded command buffer. One evaluation should serve both. Evaluate, do not assume. | **Switch** evaluates → Morpheus decides | tier 3, tier 5b |
| **OQ-4** | ~~Shader compilation: build-time `glslc` vs checked-in pre-generated SPIR-V vs both with SDK preferred. Provisionally: build-time with checked-in fallback.~~ **RESOLVED 2026-07-28T22:28:08-07:00 — §7.8. The provisional decision is *changed*, not implemented: the Vulkan SDK is a hard build prerequisite and there is no checked-in SPIR-V fallback.** Found by the coordinator building on a machine without the SDK — `build.rs` panics with an escape hatch, which contradicted this row. The doc was wrong, not the code: a checked-in `.spv` that drifts from its `.comp` changes what runs with no signal, "reviewable diffs" is not true of SPIR-V binaries, and the variant table already generates 168 modules. Five binding conditions in §7.8, of which one is new work for Switch: **a shader-less artifact must advertise zero devices and claim nothing**. If the prerequisite ever proves to block contribution, the remedy is vendoring `shaderc`/`naga`, not checked-in binaries. | Switch proposed → coordinator found the divergence → **Morpheus decided** | — |
| **OQ-5** | `gpu-allocator` vs a hand-rolled suballocator. `ENGINE.md` §3.1 picks `gpu-allocator`; I concur provisionally. Confirm it cross-compiles cleanly for Android and works under MoltenVK. | **Switch** owns → Link validates | M0/M3 |
| **OQ-6** | What vendor ID does the factory report when it advertises zero devices, or before a device is bound? ORT calls `GetVendorId` on the factory, not per device. | **Tank** proposes → Morpheus decides | M0 |
| **OQ-7** | Do we need a real GPU CI runner for M2's exit criteria, and if so, self-hosted or a cloud GPU lane? Software rasterizers cannot validate a speedup claim. | **Link** proposes → Justin decides (cost) | M2 |
| **OQ-8** | ~~Is `com.microsoft` contrib-op support ever in scope?~~ **RESOLVED — and re-resolved at a higher level. 2026-07-28T20:54:42-07:00: Justin ruled directly that the `com.microsoft` domain is in scope** (`.squad/decisions/inbox/copilot-directive-contrib-ops.md`), superseding both the original `ai.onnx`-only non-goal and my narrower 2026-07-28T19:16:08 formulation of "nine named ops, never the domain". The admitted v1 set is still those nine; what changed is that a tenth is now a scoping decision inside an in-scope domain rather than a re-opened non-goal. The engineering constraints (§1.4 C1–C7) are unaffected by the elevation — they were always about safe claiming, not about permission. The deciding fact, verified by Mouse from the ORT GenAI model builder source: the builder *emits* these ops directly, so declining `com.microsoft` means a Qwen graph cannot run at all. | **Justin** (domain) / **Morpheus** (constraints) | — |
| **OQ-9** | Threading model: one `VkDevice` per session (chosen) vs a process-shared device with a mutex. Sharing saves memory and pipeline-cache warmth for multi-session hosts. | **Tank + Switch** propose → Morpheus decides | post-M2 |
| **OQ-10** | Tolerance policy for accumulation-order-sensitive ops (GEMM, reductions) across vendors, where fp32 associativity differs. Needs a stated, derived rule before M2's ops land, not after. | **Trinity** proposes → Morpheus ratifies | M2 |
| **OQ-11** | ~~Ratification of `OP_COVERAGE.md` (§8.1).~~ **RESOLVED 2026-07-28T19:16:08-07:00: ratified with five amendments** (§8.4). It supersedes §8.2/§8.3; §8.1's seven principles stand. **Its central question — whether to admit `com.microsoft` — was then settled above my level by Justin's ruling of 2026-07-28T20:54:42-07:00** (see OQ-8); A1 is revised accordingly and survives as the *discipline* rather than as the permission. | Mouse proposed → **Morpheus ratified** → **Justin ruled on the domain** | — |
| **OQ-12** | Does carrying the legacy barrier backend (§7.3) actually buy *usable* devices, or does the Adreno 5xx / Mali Bifrost population fail for some other reason? **The 31.43% figure is a database claim, not a usability claim, and until the experiment in §11.1 runs, that is exactly how much of it is unverified: all of it.** *Concurrence noted 2026-07-28T21:01:56-07:00: Link's `PLATFORMS.md` §8 rewrite now states the same position in his own words — the gpuinfo data proves those devices lack `VK_KHR_synchronization2`, not that a legacy barrier path makes them usable. The two documents agree on the honest position rather than each implying the other verified it.* The experiment, its pass/fail bar, and what would reverse the decision are specified in §11.1. Needs real hardware, which we do not have. | **Link** measures → Niobe benchmarks → Morpheus reviews | M3 Android scope |

| **OQ-16** | **When do `LinearAttention` and `CausalConvWithState` appear in a *released* ORT, and does their schema change between our main-branch fingerprint and that release?** *New, 2026-07-28T22:28:08-07:00.* Four of the eleven admitted contrib rows (`LinearAttention`, `CausalConvWithState`, `QMoE`, `MoE`) carry `MAIN_BASELINE` — they exist only on ORT main. All four are `Staged`, so nothing is claimed against an unreleased schema, and C2 item 6 now makes that structural rather than intentional. But `LinearAttention` and `CausalConvWithState` gate **T5a, the named Qwen3.5 target**, so we are partly gated on an *upstream schema stabilizing* — a different risk from "the Vulkan kernel is hard", with different mitigations, and it must be reported as such rather than folded into a Vulkan schedule slip. Practically: the T5a kernels may be written twice, and every main-baseline fingerprint needs re-verification the moment its release exists. | **Fact Checker** watches upstream → Mouse re-verifies → **Morpheus** rules on T5a scope | T5a |

### 11.1 OQ-12 — the minimum decisive experiment

Written now, so it can be executed the hour hardware exists rather than designed then. Until it
runs, **the entire 31.43% Android claim behind §7.3 is unverified as a *usability* claim** — Link's
data proves those devices lack `VK_KHR_synchronization2`, not that they can run a model. §7.3's
Windows argument (12.22%) is unaffected and stands on its own; this experiment is only about how
much of the Android half is real.

**Devices — four, chosen to be decisive rather than representative.** Two must be from the
sync2-missing population and two are controls:

| Slot | Device class | Why this one |
|---|---|---|
| A | **Adreno 5xx** — e.g. Snapdragon 660 (Adreno 512) or 636 (Adreno 509), Android 8–10, stock OEM blob | The single largest sync2-missing bloc. Frozen pre-2021 drivers. If any class fails for other reasons, it is this one. |
| B | **Mali Bifrost on MediaTek** — e.g. G52 (Helio G85) or G76 (Helio G90T), stock ROM | The second bloc, and a different vendor's driver bugs. MediaTek specifically, because the same Mali IP on a Samsung/Exynos ROM has a different update history. |
| C | **Adreno 6xx on Android 12+** — e.g. Snapdragon 865/888 | Control: *has* sync2. Isolates "is the legacy backend correct" from "is this device usable". |
| D | **Mali Valhall on Android 12+** — e.g. G78 / G710 | Second control, second vendor. |

Two physical units of A and B are worth more than four of C and D. If only two devices can be
obtained, take **A and C**.

**Workload — three stages, each of which can independently fail the experiment.**

1. **Gate check (minutes).** Does the device pass §7.2? Report `vkEnumerateInstanceVersion`,
   `maxComputeWorkGroupInvocations`, `maxComputeSharedMemorySize`, memory types, subgroup
   `supportedOperations` and `subgroupSize`, and — separately tracked, this is OQ-14's data point —
   `shaderFloat16` and `storageBuffer16BitAccess`. A device that fails §7.2 outright is the
   cleanest possible negative result.
2. **Correctness (hours).** The **full M1 differential suite** — the §8.2 elementwise/shape floor —
   run on device against ORT CPU, plus the M2 reduction/GEMM/softmax/norm set if it exists by then.
   Run it **twice**: once normally, once with `ep.force_legacy_barriers=1` on devices C and D, so
   any failure can be attributed to the legacy barrier backend rather than to the device. This is
   the same parity harness §9.1 already requires; no new test infrastructure is needed.
   Validation layers on. Record every failure with its op, dtype, shape and driver version.
3. **Usability (hours).** One bandwidth-bound elementwise chain and one GEMM-anchored subgraph,
   sized realistically, timed against **that device's own CPU** through the ORT CPU EP. Not against
   a desktop GPU — the question is whether the Vulkan path beats the CPU already in the phone.
   Report Mouse's `boundary_time_fraction` alongside wall time (`OP_COVERAGE.md` §7.3).

**Pass bar.** For the legacy backend to be vindicated on Android, devices A **and** B must:
(i) pass the §7.2 gate; (ii) pass the full differential suite with **zero** numerical failures and
zero validation-layer errors; and (iii) beat their own device's ORT CPU EP by **≥ 1.5×** on the
GEMM-anchored subgraph. 1.5× is the threshold below which the integration cost, the memory
footprint and the thermal cost are not worth a user's trouble.

**What reverses the decision, stated in advance so it is not rationalized afterwards.**

- **If A and B both fail stage 1 or stage 2** — i.e. the sync2-missing population cannot run correct
  compute at all, for reasons unrelated to barriers — then the Android half of §7.3's justification
  is void. The legacy backend **still stays**, on the strength of the 12.22% Windows gap alone, but
  Android's §7.2 gate should then be tightened per device class in `PLATFORMS.md`, and M3's Android
  scope narrows to the Adreno 6xx / Valhall tier. I would record that as a scope decision, not
  quietly.
- **If A and B pass stages 1–2 but fail stage 3 (< 1.5×)** — they are correct but not worth using.
  The legacy backend stays and the devices remain supported (correctness is free once written), but
  they are documented as "runs, not recommended" and get **no tuning budget**. This is the outcome I
  consider most likely.
- **If A and B pass all three stages** — §7.3 is fully vindicated and the Android tier gets a real
  tuning budget in M3.
- **If devices C or D fail the differential suite only under `ep.force_legacy_barriers=1`** — that
  is a bug in `LegacyBackend`, not a finding about devices, and it is the most valuable possible
  result of the whole experiment because it is the failure mode §7.5's parity lane exists to catch.

**Cost and owners.** Two to four used mid-range Android phones, obtainable second-hand for a modest
sum — this is the cheapest decisive experiment in the entire project and it retires the largest
unverified assumption in §7. **Link** owns device acquisition, the gate-check harness and the
`PLATFORMS.md` rows; **Trinity** owns running the differential suite on-device; **Niobe** owns
stage 3 and the `boundary_time_fraction` numbers; **Morpheus** rules on the outcome. Blocked only
on hardware, not on any other milestone — stages 1 and 2 can run the day M1 is green.

---

## 12. Divergences from the `onnxruntime-mlx` reference

Every deliberate difference, with its reason. Anything not listed here is intended to match the
reference.

| # | Divergence | Reason |
|---|---|---|
| D1 | **Real device allocator + data transfer** (`allocator.rs`, `transfer.rs`); MLX returns null stubs. | No unified memory. Without them, every partition boundary is a full host round-trip. Phased: null in M0/M1, real in M2 (§6.3). |
| D2 | **`GetSupportedDevices` advertises N devices, with a capability gate**; MLX advertises one. | Multi-GPU is normal on Windows/Linux, and an unusable Vulkan device must never be offered. |
| D3 | **A whole Vulkan engine layer (`vk/`) that has no MLX counterpart.** MLX supplies scheduling, memory, and op semantics; Vulkan supplies none. | Unavoidable. It is also why Rule 2 (§4.2) is enforced rather than encouraged. |
| D4 | **`recorded.rs` (command-buffer recording cache) replaces `compiled.rs` (`mlx_compile` tracing).** | Same intent — pay graph construction once, replay many. Different mechanism: Vulkan's unit of replay is a recorded `VkCommandBuffer`, following ExecuTorch's model rather than llama.cpp's re-record-every-eval. |
| D5 | **Weight prepacking is a first-class `Compile` step**, not a first-`Run` cache fill. | Vulkan uploads are explicit and expensive; ORT gives us initializer bytes at compile time; doing it lazily would put a staging copy on the first inference for no benefit. |
| D6 | **`shaders/` directory and a SPIR-V build step.** MLX explicitly *deleted* its `.metal` kernels. | We are the backend. This is the one place where the MLX project's history is an anti-pattern for us — its lesson was "don't hand-write kernels when a good backend exists", and for Vulkan no such backend exists. |
| D7 | **fp32-only v0; fp16 as a per-family gated variant.** MLX is dtype-generic for free. | MLX carries dtype through its ops with no per-dtype code. Every Vulkan dtype is a separate SPIR-V variant plus a device feature probe. |
| D8 | **`com.microsoft` contrib ops in scope; eleven named ops are the v1 admitted set (tiers 3–5).** MLX's highest-value ops are contrib ops. | *Revised 2026-07-28T20:54:42-07:00.* Originally "no contrib ops in v1"; reversed by the OQ-11 ratification and then settled directly by Justin's ruling that the domain is in scope — the ORT GenAI model builder emits them, so declining means no Qwen graph runs at all. The remaining divergence from MLX is *how*: we register ops **by name with a hand-written claim predicate each and no domain-wide opt-in in the code**, plus a graph census in CI as the drift alarm for a surface that has no opset number (§1.4 C1–C7). |
| D9 | **Vendor ID is read from the bound device**, not hardcoded to one vendor. | Cross-platform mandate. |
| D10 | **Validation layers are part of the definition of done.** No MLX equivalent. | Vulkan's error surface is enormous and mostly silent without layers; MLX's C API validates for us. |
| D11 | **`ash` (safe-ish Rust Vulkan bindings) rather than bindgen over `vulkan.h`.** MLX bindgens `mlx-c` directly. | `ash` is the ecosystem standard, handles the loader/extension-function-pointer problem correctly, and removes a large class of hand-written FFI bugs. The ORT side still uses bindgen, matching the reference. |
| D12 | **Two barrier backends behind one internal seam (`vk/barrier.rs`), selected once at device init**, plus a session option to force the legacy one. MLX has no counterpart — MLX's C API owns all synchronization. | §7.3. Requiring `synchronization2` would exclude 31.43% of Android and 12.22% of Windows; the compatibility-first directive makes that unacceptable. The cost is one file with two implementations and a doubled CI lane, which is bounded; the cost of the alternative is permanent device exclusion. wgpu, Dawn and Godot all ship legacy-only barriers, so this is the mainstream position, not an exotic one. |

---

## 13. References

- **Reference architecture:** `onnxruntime-mlx` — `docs/DESIGN.md`, `docs/OP_ARCHITECTURE.md`,
  `docs/COMPILED_CAPTURE.md`, `rust/src/{lib,factory,ep,engine,registry,compiled,sys}.rs`,
  `rust/{Cargo.toml,build.rs}`, `tests/`, `bench/`, `python/`.
- **Sibling docs:** [`ENGINE.md`](./ENGINE.md) (Switch), [`PLATFORMS.md`](./PLATFORMS.md) (Link),
  [`OP_COVERAGE.md`](./OP_COVERAGE.md) (Mouse), `OP_ARCHITECTURE.md` (Mouse, forthcoming),
  `BENCHMARKS.md` (Niobe, forthcoming).
- **Vulkan layer deployment (§7.3 ruling):** `KhronosGroup/Vulkan-Loader`
  `docs/LoaderLayerInterface.md` and `docs/LoaderApplicationInterface.md` (Android layer discovery;
  "The Android loader does not use manifest files"; elevated-privilege `secure_getenv` caveats),
  `loader/loader_environment.c`; `KhronosGroup/Vulkan-ExtensionLayer`
  `docs/synchronization2_layer.md` ("needs to be packaged inside the APK");
  `developer.android.com/ndk/guides/graphics/validation-layer`.
- **Barrier prior art (§7.3):** `gfx-rs/wgpu` `wgpu-hal/src/vulkan/command.rs`
  (`cmd_pipeline_barrier`, legacy only); `google/dawn`
  `src/dawn/native/vulkan/CommandBufferVk.cpp` (`fn.CmdPipelineBarrier`, legacy only);
  `godotengine/godot` `drivers/vulkan/rendering_device_driver_vulkan.cpp` (`vkCmdPipelineBarrier`,
  legacy only). None ships `VK_LAYER_KHRONOS_synchronization2`.
- **ORT plugin-EP C ABI:** `onnxruntime_ep_c_api.h`, `RegisterExecutionProviderLibrary`,
  `SessionOptionsAppendExecutionProvider_V2`, `CreateEpFactories`, `ReleaseEpFactory`.
- **Prior art — llama.cpp Vulkan backend:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
  `vulkan-shaders/vulkan-shaders-gen.cpp`. Per-node eager dispatch with graph-level fusion passes;
  re-records command buffers every eval; buffers only, no VMA; hard runtime floor Vulkan 1.2; base
  shaders compiled at `--target-env=vulkan1.2` with `vulkan1.3` reserved for cooperative-matrix-2
  variants; build-time SPIR-V with runtime binary patching.
- **Prior art — ExecuTorch Vulkan backend:** `backends/vulkan/`. Ahead-of-time partitioning
  (`vulkan_partitioner.py`), serialized graph, `prepack()` at init, command buffer recorded once
  and replayed, buffers **and** image textures, VMA, on-disk `VkPipelineCache`, hard floor
  Vulkan 1.1.
- **Decision records:** `.squad/decisions/inbox/morpheus-architecture-v0.md`,
  `.squad/decisions/inbox/morpheus-oq1-resolution.md`,
  `.squad/decisions/inbox/morpheus-oq11-ratification.md`,
  `.squad/decisions/inbox/link-oq1-extension-availability.md`,
  `.squad/decisions/inbox/mouse-op-coverage-plan.md`,
  `.squad/decisions/inbox/fact-checker-ort-1.28-api.md`,
  `.squad/decisions/inbox/rai-oq-m6-license-ruling.md`,
  `.squad/decisions/inbox/tank-m0-foundation.md`.
