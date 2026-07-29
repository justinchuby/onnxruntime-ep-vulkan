# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Lead / EP Architect — architecture, design docs, scope, review
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### 2026-07-28T17:59:54-07:00 — `docs/DESIGN.md` authored (architecture of record)

**The MLX reference is a pipeline, not a backend.** What transfers from `onnxruntime-mlx` is the
plugin-EP integration: `CreateEpFactories`/`ReleaseEpFactory`, the `#[repr(C)]` embed-ORT-struct-at-
offset-0 vtable pattern, `Box::into_raw`/`from_raw` ownership, panic guards at every `extern "C"`
entry, the `(domain, op_type, [min,max] opset) → {handler, claim}` registry, `NodeView`/`NodeDesc`,
convex clustering (union-find + reachability bitsets — non-convex fusion creates a cycle ORT
rejects), and the repo shape. What does **not** transfer is everything MLX supplied for free:
memory, scheduling, dtype genericity, and op semantics. Roughly: MLX gave the EP a backend; Vulkan
gives us a driver.

**The single structural divergence that drives all the others: unified vs. explicit memory.** The
MLX EP advertises no device allocator, returns null from `GetDefaultMemoryDevice`, and copies out
with one `memcpy`. Vulkan forces us to own `OrtAllocator`, `OrtDataTransferImpl`, staging, coherence,
barriers, and weight prepacking. Any future "just mirror MLX" instinct must stop at this line.

**ORT's allocator API is pointer-based; a `VkBuffer` is not a pointer.** This is the sharpest
concrete ABI problem in the project. Decided: opaque tagged-handle registry resolving to
`(VkBuffer, offset)`. Rejected `VK_KHR_buffer_device_address` — optional on every baseline, and
MoltenVK support is partial.

**Vulkan version floors: verify the premise before designing to it.** llama.cpp does **not** require
Vulkan 1.3 — its hard runtime floor is 1.2 (`if (api_version < VK_API_VERSION_1_2) throw`), it sets
`VkApplicationInfo::apiVersion` to whatever the instance reports, and `vulkan-shaders-gen.cpp`
compiles its **base shaders at `--target-env=vulkan1.2`**, reserving `vulkan1.3` for the NVIDIA
cooperative-matrix-2 variants; the `vulkan1.3` in its CMake is an extension-availability probe.
ExecuTorch hardcodes `VK_API_VERSION_1_1` with a 1.0 fallback path and initializes VMA at
`VK_API_VERSION_1_0`. Both widely-cited "requires 1.3" claims are wrong — independently confirmed
by Fact Checker (audit trail, claims 1–2, contradicted). Generalizable: when a design proposal
cites a project's requirement, read that project's source before building on it.

**The baseline decision generalizes: require a capability set, not a version number.** Switch found
that only two features materially simplify the engine (`synchronization2`, `subgroup_size_control`)
and both exist as standalone extensions on 1.1/1.2 drivers. So requiring the *features* gives the
single barrier code path and guaranteed subgroup sizing without a version floor's coverage cost.
Also learned from Link's data: **on Android the Vulkan 1.2 tier barely exists** — devices jumped
1.1 → 1.3 — so a 1.2 floor pays nearly the full Android cost of 1.3 while delivering less on desktop.

**Because CPU fallback is always correct, a plain output comparison is a vacuous test.** Every op
test must additionally assert the node ran on `VulkanExecutionProvider`. This is the highest-value
testing invariant in the project and the first thing to check in a review.

**Claim rate is a bad metric; fused-region compute volume is the good one.** One unclaimed node in
the middle of a graph splits it into two islands with a device round-trip between them. Op priority
is "does this merge two islands", not "is this op easy". Benchmarks must report island count and
largest fused region alongside wall time or the number is not interpretable.

**Prior-art split worth remembering:** llama.cpp re-records command buffers every eval (fine for a
few large matmuls, wrong for many small dispatches); ExecuTorch records once at init and replays,
with an explicit `prepack()` step for constants. For an ONNX EP the ExecuTorch model is right, and
it maps cleanly onto the MLX EP's `compiled.rs` (`mlx_compile`) role → our `recorded.rs`.

**Process:** Switch's `ENGINE.md` and Link's `PLATFORMS.md` already existed when I started, despite
the spawn prompt assuming they might not. Check the working tree before writing "pending X's
findings" — reading a sibling's actual output produced a materially better decision than reasoning
around its absence would have.


---

## 2026-07-28T19:16:08-07:00 — Freezing DESIGN.md §7 (OQ-1 resolution)

**A "provisional" decision is only honest if you actually reverse it when the data arrives.** My
§7.2 of two hours earlier required `synchronization2` and `subgroup_size_control` and said so
"pending Link's findings". Link's findings said 31.43% of Android and 12.22% of Windows would be
excluded. I reversed it. The lesson is not "I was wrong" — it is that marking something provisional
creates an obligation, and the whole point of the capability-set framing was that the cost is
*measurable*, unlike a version-number floor. Design in units you can later measure.

**Never require a feature flag when you only need a property.** Link caught that MoltenVK reports
the `subgroup_size_control` *extension* (Vulkan 1.3 promotes it to core) while
`subgroupSizeControl` is `VK_FALSE`, because Metal cannot control SIMD-group width per pipeline.
Requiring the flag would have silently excluded all of macOS/iOS — and probably lavapipe and
SwiftShader, i.e. our own CI. Generalized rule: **a requirement that excludes the machines you test
on is a requirement you have not tested.** Always ask "extension string, property value, or feature
flag?" — they are three different requirements with three different coverage numbers.

**The right formulation of a hardware requirement is usually a correctness rule, not a gate.** For
subgroup width the answer was not "require the extension" but "a shader whose correctness depends
on an exact subgroup width may only be selected when the width is *known* exactly, otherwise use
the portable variant". That costs nobody coverage and preserves the actual guarantee. Look for this
shape whenever a capability requirement is proposed.

**The frozen principle worth carrying to any future EP: the device gate is minimal; capability
shortfalls degrade op coverage, not device availability.** A hard device requirement must be
justified by "no op we will *ever* ship can work without it". Everything else is a claim predicate.
This falls straight out of conservative-claiming-with-clean-CPU-fallback, and it makes the failure
mode "runs fewer ops" instead of "device does not exist".

**Verify the mechanism before you accept the mitigation.** Link proposed bundling the Khronos
`VK_LAYER_KHRONOS_synchronization2` layer, citing wgpu/Dawn/Godot as precedent. Two things
collapsed on inspection: (1) the AOSP Vulkan loader ignores `VK_LAYER_PATH`, uses no JSON
manifests, and searches only the *host application's* `nativeLibraryDir` — so a plugin `.so`
`dlopen`ed into someone else's process cannot enable a layer on retail Android at all, which was
100% of the motivation; (2) all three cited projects use legacy `vkCmdPipelineBarrier` exclusively
and none ships the layer. **A precedent you have not read in the source is not a precedent.** We
are a plugin, not an application — that distinction invalidates a whole class of otherwise-standard
Vulkan advice (layers, environment variables, instance ownership), and it should be the first thing
I check on any proposal of this shape.

**When you authorize a dual code path, ship the seam and the test lane in the same decision.** A
dual path becomes a bug farm exactly when it is `if caps.x { } else { }` at every call site. The
decision that makes it survivable is: one internal API, our own closed enums (so the legacy backend
is *total* by construction — no `VK_PIPELINE_STAGE_2_NONE` to translate), backend selected once at
device init, one mapping table, and a session option forcing the minority path so CI executes it
every run. Without the forced lane, the path we carry for 31% of Android would be run by no test we
own, because our Linux CI has sync2 99% of the time.

**Coverage-count is the metric most likely to be gamed by an ambitious plan.** When the op-coverage
ambition was raised, the constraint I had to write down explicitly was minimum viable subgraph
size — high op count that shreds a graph into transfer-dominated fragments is a regression wearing
a coverage badge. Attach a metric (island count, largest fused region) or the constraint is a
slogan.

**Don't resolve an open question on a symbol name.** ORT 1.28's
`CreateExternalResourceImporterForDeviceImpl` looks like a better answer to OQ-3 than my
opaque-handle registry, and it may well be — but inferring semantics from a name is the same
mistake as "llama.cpp requires 1.3", which we had already made once this week. Recorded it as a
live alternative with its cost (a 1.28 ABI floor, which is itself a compatibility regression) and
left OQ-3 open pending Fact Checker.

---

## 2026-07-28T19:16:08-07:00 — OQ-11 ratification, the contrib-domain reversal, and OQ-12

**A non-goal that makes the project's named target unreachable is not ruthless, it is wrong.** I
wrote "v1 is `ai.onnx` only" believing "run a Qwen graph" and "support `com.microsoft`" were
separable concerns. They are not: the ORT GenAI model builder *emits* GroupQueryAttention,
RotaryEmbedding, MatMulNBits, LinearAttention and friends directly, so declining the domain means
the EP cannot run the named target model at all. Mouse found this by reading the program that
**emits** the graph rather than the ONNX spec index. Generalized lesson: **for any "do we support X"
scope question, go read the exporter, not the standard.** The standard tells you what is legal; the
exporter tells you what you will actually receive.

**The right shape of that reversal is "named ops, never a domain."** Contrib ops have no opset to
range-check — they version with ORT *releases* — so admitting a domain is admitting an unbounded,
silently-mutating surface. Nine named ops, each with a hand-written claim predicate, plus a graph
census in CI as the drift alarm. The general rule: **when you admit a surface you cannot
version-check, you must replace the version check with a mechanical drift alarm, or you have
admitted it blind.**

**"Performing a fusion" and "implementing a fused node" are different things and I conflated
them.** My non-goal list had "attention fusion" out of scope, which correctly excluded pattern
*detection* but incorrectly excluded GroupQueryAttention — which arrives as one node from the
exporter and whose decomposition would materialize a [B,H,S,S] score matrix in VRAM. Implementing
it is the conservative choice; decomposing it is the reckless one. Watch for this inversion
whenever a non-goal is phrased as a technique rather than as a behaviour.

**Ratify with amendments, and make the amendments the things that get dropped under pressure.**
Mouse's plan was better than the section it replaced, so the useful review work was not
finding faults — it was identifying which of his own correct warnings would quietly evaporate on a
tight schedule, and converting them into gates and metrics. His "build the templates before op #1"
became an M1 entry gate plus a reported ops-per-kernel ratio (≥ 8). A warning is advice; a number in
an exit criterion is a decision.

**Escalate anything that decides a product boundary, even when it arrives labelled as a detail.**
Mouse filed "what fraction of devices have shaderFloat16?" as an op-plan open question. Under the
frozen §7.2 fp16 is probed, not required, and an fp32-upcast LLM path is a memory-footprint failure
rather than a slow path — so that question decides whether the LLM story is desktop-first. It
belongs in the architecture doc's open-question table with a named decider, not in a subordinate
document.

**Look for the constraint the proposal solved halfway.** Push-constant kernel dimensions make the
*shader* length-agnostic but not the **workgroup count**, which still depends on sequence length —
so either we re-record per shape bucket anyway or we need `vkCmdDispatchIndirect`. That is the same
mechanism data-dependent MoE routing needs on a pre-recorded command buffer. One evaluation, two
problems. Half-solved constraints are where architecture review earns its keep; the author is
usually too close to the op to see that the engine has the same problem elsewhere.

**I propagated an unverified symbol name and an unverified version number into the architecture of
record, and it cost the team a wrong objection.** I recorded ORT's external-resource importer as
`…ForDeviceImpl`, "new in 1.28", and objected that adopting it would move our ABI floor. Fact
Checker found: wrong name (`Impl` is a test-code static), wrong version (1.24, not 1.28), and Tank
had *already* set the minimum to 24 — so the cost I objected to was zero. It was also solving a
different problem entirely (caller-exports-their-memory, not what-our-Alloc-returns). **An error in
the architecture of record propagates into everyone's assumptions; a lead's wrong entry costs more
than an engineer's.** Mark anything unverified as unverified *in the document*, not just in my head.

**Design the decisive experiment before the data exists, and write down in advance what would
reverse your decision.** §7.3 is my decision, so I am the person most at risk of designing an
Android experiment that confirms it. §11.1 therefore fixes the devices, the pass bar (≥1.5× vs the
phone's own CPU, zero numerical failures), and all four outcomes with their consequences — including
the one where the finding is "the legacy backend has a bug", which is the most valuable result and
the one the parity lane exists to catch. Also: state plainly how much of a claim is currently
unverified. Link's 31.43% proves an extension is absent, not that a device can run a model.

---

## 2026-07-28T20:54:42-07:00 — the user ruled on contrib ops; what a lead keeps when a decision is taken above him

**When a ruling supersedes your position, record it as superseding — do not quietly reinterpret it
as agreement.** Justin ruled 「contrib op 要做」 the day after I had reversed my own non-goal to "nine
named ops, never the domain". The tempting move was to read his ruling as endorsing my formulation,
since the practical op list is identical. That would have been a misreading dressed as continuity: he
ruled on the *domain*, and narrowing a ruling by reinterpretation is how a record stops being
trustworthy. §1.2 and OQ-8 now say plainly that his ruling supersedes both the original non-goal and
my narrower version, and §1.4 says what I retain.

**What a lead retains after a scope ruling is the "how", and it is worth more than the "whether".**
The useful reframing: the per-op discipline I had attached was never about *permission* — it was
about not claiming a schema nobody has read. So none of it died with the scope boundary. Separating
"is this allowed" from "how do we do this safely" before the ruling landed is what made the
constraints survive it intact. Write constraints so they are independent of the scope decision they
happen to accompany.

**The decisive distinction: a product-level "in scope" must never become a code-level
`domain == "com.microsoft"` predicate.** A domain-wide accept is a claim predicate that, by
construction, accepts every op Microsoft has ever put in that domain plus every one they add later —
an unbounded, silently-growing input set feeding the wrongly-claimed-node failure mode. The registry
key *is* the allowlist; a second hardcoded list is a drift bug waiting to happen. Enforced by a test
that fabricates `com.microsoft::NotARealOp` and requires an ordinary decline.

**When you admit a surface with no version number, you must replace the version check with a
mechanical drift alarm — or you have admitted it blind.** `ai.onnx` gives a monotonic opset the
predicate range-checks, so a schema change is visible as a number we did not accept. Contrib schemas
version with ORT *releases* and bump nothing: they silently change what the predicate *should*
accept while it goes on accepting what it always did. Hence pinning + per-op recorded ORT version in
the table (not in a comment) + census claim rates in CI + version-bump-as-review-gate. And the rule
that will be unpopular exactly when it matters: **on drift, narrow and decline, never guess** —
declining is a performance regression CI reports, guessing is a correctness regression nothing
reports. Those are not comparable costs.

**Do not special-case the diagnostics of the thing most likely to break.** Contrib declines flow
through the same machine-readable path as everything else. A bespoke diagnostic path is one more
thing that can be wrong in the exact place you look when something is already wrong.

**Beware weak oracles on the models you care most about.** An LLM appears to work with a broken
kernel far longer than intuition suggests — sampling and model redundancy hide a lot — so
final-token comparison against ORT CPU is weakest precisely on Qwen. Per-layer comparison, and land
the XL kernels one at a time rather than in parallel: parallel landing maximizes simultaneously
unverified kernels at the moment the oracle is weakest.

**Say what a scope expansion actually bought, in the right units.** Admitting contrib did not
shorten the months-scale number; it is what made that number *exist* — before the ruling "Qwen3.5
end-to-end" had no completion path at all. "It became possible" and "it became faster" are different
kinds of progress and blending them is the first self-deception available to a project whose
ambition was just raised. Same discipline as refusing a single blended date: the blended number is
the one that would be wrong, in the direction everyone wants to hear.

**Guard the schedule against its own visible goal.** The most expensive available mistake right now
is reprioritizing M1's template infrastructure or M2's device allocator to reach a contrib kernel
sooner, because contrib is what everyone can see. §10.0 states M0/M1 are *unaffected* by the ruling
as an explicit instruction, not as an omission.
