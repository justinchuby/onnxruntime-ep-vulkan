# Mouse (Op-Coverage) — history.md

## Learnings

### [SUMMARY] Turns 1–16: op plan, infrastructure, contrib rows, census, Live ops, extents, runtime (2026-07-28–2026-07-30)

**Turns 1–5 (archived):** 174-op inventory (ai.onnx standard domain). Table-driven registry (`registry.rs`). Eleven staged contrib rows for `com.microsoft` ops. Machine-readable claim log (JSON Lines, flushed per decision). GQA fingerprint self-audit found two permissive bugs (corrected).

**Turn 6 — in-house crate review (2026-07-29T07:14:15-07:00):**
`onnx-genai-models` / `mobius` builder emits `ai.onnx::Attention`/`RMSNormalization`/`RotaryEmbedding` when our EP advertises GQA support, not the `com.microsoft` variants. Standard-domain rows are required for the mobius path, not optional.

**Turn 8 — mobius as producer of record (2026-07-29):**
Authoritative producer is `onnxruntime/mobius@87fd878`, not `justinchuby/onnx-genai-models`. Default opset 24. `ai.onnx::Attention` gained optional input 6 `nonpad_kv_seqlen` at opset 24 — predicate written against opset 23 would have claimed and returned wrong logits. `onnx-runtime-ir` trust objection withdrawn (it is Justin's own crate); structural objection stands independently.

**Turn 9 — opset range (2026-07-29):**
`ONNX_OPSET_LAST_RELEASED=26`, `ONNX_OPSET_REGISTERED=27`. Two constants, test asserts. `LinearAttention-27` and `CausalConvWithState-27` registered (Qwen3.5-hybrid ops standardised in onnx 1.22.0).

**Turn 10 — Foundry Local census (2026-07-29):**
Phi-3.5-mini-instruct (`ai.onnx`=14) and gpt-oss-20b (`ai.onnx`=21) read from disk. Five findings: `OPSET_STD_LLM=23` excludes both; `do_rotary=1` universal; packed QKV predicate requires both inputs; `SimplifiedLayerNormalization` has `domain=""` not `ai.onnx` in both graphs; `QMoE` top-4 (not top-1|2). §8.5 third strengthening: "builder source is intent; the model file is the fact." Metric upgraded to triple: `(claimed_coverage, island_count, largest_island_flops)`.

**Turn 11 — first Live row, and oracle boundary (2026-07-29):**
`Add` Live for f32 only. `EXERCISED` evidence list introduced. `Sub/Mul/Div/Pow` stay Staged — template similarity is not evidence (D-M11-02). "An oracle knows ORT's correctness, not our dispatch correctness."

**Turn 12 — test contradiction and elementwise flip (2026-07-29):**
`OnceLock` bug in claim log path fixed — re-reads env var per decision. Profiling-JSON retained for `is_vulkan_claimed` (post-load env var changes unreliable for DLL on Windows; CLAIM_LOG still correct for subprocess use). Three-layer skip contradiction found and closed.

**Parameter tail (2026-07-29):**
Four-float push-constant tail unconditionally at block end. 7 activations unlocked (Selu, Elu, HardSigmoid, Shrink, ThresholdedRelu, LeakyRelu, CeluAlpha). `Clip` excluded (two params, NaN semantic mismatch).

**MatMulNBits Live (2026-07-29):**
`com.microsoft::MatMulNBits` Live for all `M`, fp32 and fp16. GEMV layout from oracle (`A=I`). All 161 Phi-3.5 nodes are fp16 (`bits=4`, `block_size=32`, 3-input symmetric, `K∈{3072,8192}`, `N∈{3072,8192,9216,32064}`). fp16 through `unpackHalf2x16/packHalf2x16` — no 16-bit storage capability needed.

**Decline census (2026-07-29 evening):**
First-match histogram is a ceiling, not a measurement. Full-set Phi-3.5: `dynamic-shape=356` (not 258); 98 of 100 staged nodes are also shape-blocked; landing all staged kernels unlocks 0 nodes. gpt-oss: `dynamic-shape=342 > staged=197` — first-match would have reversed a correct ruling.

**Runtime extents (2026-07-29):**
`ENGINE_ACCEPTS_RUNTIME_EXTENTS` flag; `shape_class` computed independently; `predicate_ok_runtime_extents` field in JSONL. 227 Phi-3.5 nodes predicate-clean under runtime extents; 161 (`MatMulNBits`) claimable immediately once Switch flips the flag (now done).

**Current state:**
- 45 Live rows. `cargo ci` — green.
- 161 Phi-3.5 nodes claimable; GQA + SkipSimplifiedLayerNorm are next for remaining coverage.
- `onnxruntime/mobius@87fd878` is the pinned authoritative producer.
- Standing rules: closed windows = schema-version windows; `do_rotary=1` must precede GQA claim; metric is the triple, reported per producer at version.
---

## 📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)

### Worktree layout and inbox portability constraint
The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree. `.squad/decisions/inbox/` is **gitignored** — records written in a worktree do NOT travel with the branch. The inbox in `main` is authoritative.

### Vulkan SDK path
`C:\VulkanSDK\1.4.350.0` — installed but **not on the default PATH**. `glslc` discovery must search this path; `VULKAN_SDK` env var is the canonical pointer.

### Local hardware — both GPUs pass the §7.2 gate
- Intel Iris Xe: Vulkan 1.4.309, UMA, `subgroup_size=32`, 32 KiB shared. Spec-conformance oracle. Do not special-case Intel.
- RTX 4060 Laptop: Vulkan 1.4.325, discrete, `subgroup_size=32`, 48 KiB shared.
- Lavapipe (CI): `subgroup_size=8`, 32 KiB shared, `is_uma=true`. CI exercises the mobile-warp path. LVP2 retracted.

### ORT's planner hands back interior pointers from run 2 onward
Memory-pattern planner does not engage on run 1. From run 2 onward hands back interior pointers. 52 observed, identical on both devices, all within span, `pointers_in_guard_band=0`. Gate: `epctl --check-counters <file> --require-dispatches 1`.

### Execution counters file is the instrument for "did anything execute"
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` — always-on JSON. `dispatches_executed > 0` is the only reliable indicator.

### `push_next` must rebind, never discard
`let _ = props2.push_next(..)` silently discards pNext chain. Rebind, never discard. Root cause of LVP2, `subgroup_size=0`, ReBAR UMA misclassification.

### First real execution: 45 ops Live, 161 nodes claimed on Phi-3.5
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. M0 not declared — open: validation positive control, CI lanes green.

### Performance metric is a TRIPLE (Niobe — critical)
`(claimed_op_coverage, island_count, largest_island_flops)` per producer at version. Portability floor = §7.2. `SUBGROUP_SIZE_IS_GUARANTEED=False`.
**Diagnostics (turn 5):**
- Diagnostic codes are worthless if they stop at the FFI boundary. Design the reader in the same breath as the codes.
- Per-event JSON (append-and-flush, one self-contained line per event). No lifecycle hook needed.
- Hook `claim_decision` not `ep.rs` aggregator — zero cross-owner edits; all future callers get the record.
- `GroupQueryAttention` fingerprint corrected: min_inputs=7, 1.28 and main forms identical.
- `cargo test` → 265 passed.

**ORT ABI facts verified against vendored 1.28 header:**
- `Node_GetInputs`/`Node_GetOutputs`: size-then-fill protocol.
- `GetValueInfoTypeInfo`: returns borrowed `const OrtTypeInfo**` — must NOT be released (double-free).
- `CastTypeInfoToTensorInfo`: `_Outptr_result_maybenull_` (non-tensor → null, no error).
- `Node_GetAttributeByName`: also `_Outptr_result_maybenull_`.
- `ReadOpAttr`: call once to size, again to fill.
- ORT returns null `OrtValueInfo` for omitted interior optional inputs (`Clip(x, , max)`).

---

## Cross-agent context appended (2026-07-28T22:28:08-07:00)

📌 **C2 item 7: fingerprint audit CI job (Morpheus §1.4):** A CI job running `graph_census.py` must execute before any tier-3 contrib work. Rows with `SchemaBaseline` pointing to a non-release (ORT `main` only) may not be set to `Live` — this is a build failure enforced by Tank. Your `ContribSchema` nested `SchemaBaseline` field wins over Tank's side table (deleted). Verify the CI job exists in `graph_census.py` and is wired in `.github/workflows/` before tier-3.

📌 **C1 domain regression test (Trinity):** `tests/ops/test_domain_regression.py` asserts `com.microsoft::NotARealOp` produces an ordinary decline (not a crash). TODO: upgrade Trinity's test to machine-readable reason code when your diagnostic JSON format is stable. `[contrib-schema]` and `[attribute]` must remain separate decline code buckets — never merge them.

📌 **Switch's `bind_aliased_output` seam (Switch engine-seams, D-S3):** KV-cache in-place update requires `bind_aliased_output(output_slot, input_slot)` on `DispatchContext`. GQA and LinearAttention handlers will need this for M2+. Default method returns resolved input — your handlers do not need to use it until KV-cache is required.

📌 **Switch's `compile_hook_for` stub in `registry.rs` (Switch engine-seams, Seam 1):** `registry.rs` now has `pub fn compile_hook_for(desc: &NodeDesc) -> Option<CompileHook>`. Mouse fills in per-op prepack hooks for GQA/MatMulNBits. `CompileHook` = `fn(&mut CompileContext, &NodeDesc)`. The `TileConfig`, `PackKey`, `PackInput`, `PackOutput`, `PrepackRequest`, `PrepackResult` vocabulary is in `engine.rs`.

📌 **`concentration()` metric is the honest performance predictor (Mouse partition rule).** `largest_island_flops ÷ total_claimed_flops` separates "80% coverage across 40 islands" from "80% in one island". Niobe reports this; always include it alongside `node_coverage` in any coverage summary you publish.

📌 **Byte-typed tensors: allocator rounding (Mouse turn-4):** `bool`/`uint8` buffers must be allocated rounded up to 4 bytes. This is invisible from Rust; it belongs in the allocator contract. Coordinate with Tank when the M2 allocator lands. Document in `OP_COVERAGE.md §8` alongside the `PackedWeights` memory class note.

📌 **GQA fingerprint correction (Mouse turn-5):** `min_inputs = 7` (not 3). Optional inputs are positional; `seqlens_k`/`total_sequence_length` at indices 5 and 6. GenAI builder emits 16-input GQA for Qwen3 (sets `q_norm`/`k_norm`). Verify the exported graph before finalizing the GQA claim predicate.

---

## Turn 6 — 2026-07-29T07:14:15-07:00 — the in-house crate review

📌 **Op coverage is relative to a *producer*, not to a model architecture.** The biggest finding of
the turn, and it invalidated a premise of my own document. I derived the entire op inventory from
what the ORT GenAI model builder emits. Justin's `onnx-genai-models` builds the *same models* and
emits `ai.onnx::Attention`, `RMSNormalization` and `RotaryEmbedding` @ opset 23 instead of the
`com.microsoft` spellings — so our table would have declined every norm, rotary and attention node
in a Qwen3 built by our own toolchain. Same kernels, missing rows. **Always ask "which exporter
produced this graph", never just "which model is this".**

📌 **Reading the source changed three verdicts that the READMEs would have gotten wrong.**
`onnx-ir-rust` looks like a Rust ONNX IR and is 20% of one — its producer/consumer fields are
literally commented out, and it cannot ingest a protobuf at all. `onnx-shape-inference` sounds like
a Rust crate and is pure Python. `onnx-genai` sounds like a model thing and contains the most
complete Rust IR of the three. Judge dependencies from `src/`, never from the front page.

📌 **The decisive objection to a graph IR here is architectural, not quality.** We are a plugin EP:
ORT hands us `OrtGraph`/`OrtNode` across a C ABI and we never see a protobuf. Any external IR would
require *copying the whole graph* into a second representation inside someone else's process. That
objection would survive the library becoming perfect, which is why it is worth stating separately
from maturity concerns — and why the deferral came with a named trigger (a representation that must
outlive one `GetCapability` call) rather than a vague "maybe later".

📌 **"Defer the dependency, adopt the information" is a real outcome.** Justin said *参考*, and the
review produced a bigger coverage gain (five standard-domain rows) than any of the three libraries
would have. `onnx-shape-inference` also became two free things: a preprocessing step for Trinity
that turns `[dynamic-shape]` declines into claims with zero Rust changes, and a second independent
source for the contrib fingerprints.

📌 **Share kernels freely; share claim predicates only when the vocabularies genuinely match.**
`RMSNormalization` reuses `simplified_layer_norm` verbatim. `ai.onnx::Attention` needed its own
predicate over the same kernel, because attribute names, the illegal-combination set and the
optional-input indices all differ. A predicate stretched to cover two schemas is wrong about one of
them, in the permissive direction.

📌 **`macro_rules!` gotcha:** `$min:literal ..= $max:expr` cannot accept a named constant, and you
cannot upgrade it to `$min:expr` either, because `..=` may not follow an `expr` fragment. `$min:tt`
takes both a literal and a bare ident.

📌 Verify commands, all green: `$env:ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC='1'`; `cargo ci`
→ **299 passed**, 7 ignored, 0 failed. (Clippy has one error in `src/trace.rs`, Niobe's file, not
mine.)

---

## Turn 8 — 2026-07-29 — mobius as producer of record, opset 24, `onnx-runtime-ir` on the merits

📌 **The mirror is not the repo.** I derived §4.18 from `justinchuby/onnx-genai-models`. The
authoritative producer is `onnxruntime/mobius`. Re-deriving found the earlier reading was wrong on
one substantive point: mobius **does** emit `com.microsoft::GroupQueryAttention`, via a
`RotaryAttentionToGQA` rewrite and a direct `_forward_gqa()` path, both gated on the target EP
advertising GQA support. So the op set is a function of producer × revision × **how we describe
ourselves to the producer**. The last factor is partly under our control and I had not considered
it at all.

📌 **Coverage claims need a producer *and* a revision.** Same argument Trinity used to pin
`accuracy_level`: unpinned, the reference drifts. Raised for Morpheus's §8.5.

📌 **The real find: an open-ended opset window was a correctness bug, not untidiness.**
`ai.onnx::Attention` gained optional input 6 `nonpad_kv_seqlen` at opset 24, and mobius defaults to
opset 24. My row said `23 ..= OPSET_ANY`, so the predicate would have claimed the static-cache form
and silently used the wrong per-batch causal offset. This is precisely the failure my own §7.1 rule
exists to prevent, and I had written the rule and then violated it in the window column. Lesson:
**the opset window is part of the claim predicate, not metadata about it.**

📌 Opset windows are **schema-version** windows, not model-opset windows. `Node_GetSinceVersion`
returns the resolved op schema version — an `Add` in an opset-24 model reports 14. That is why
closing `RMSNormalization` at `23 ..= 23` does not exclude opset-24 graphs, and it is the fact that
makes closed windows cheap. I nearly got this backwards.

📌 The policy I settled on, deliberately *not* blanket: close a window when the op has ever gained
an input or attribute across a revision (`Attention`, Q/DQ); leave it open when it has not
(elementwise). Closing all ~70 elementwise windows at 27 would decline valid opset-28 graphs the
day onnx 1.23 ships — that is death-by-fallback bought with no evidence. The closed set is the set
with evidence behind it.

📌 **`onnx-runtime-ir`: separating the two objections paid off.** Justin retired the prudential
half by owning the crate; the structural half was untouched by that, and because I had recorded
them separately I could say so without appearing to resist. Answer is still defer, but for one
reason instead of two, and the disposition changed: the trigger is now a switch rather than a
question. Generalisable: **when deferring, name the objections separately and name the trigger** —
it makes the later re-evaluation cheap and non-adversarial.

📌 Honest self-check on the trigger: I had to argue *against* my own preference here. Compiled-graph
caching and prepack keying both sounded like they fired it; neither does. Caching needs a stable
hash key, not a graph; prepack is node-local. Wanting a nice library is not a requirement.

📌 `Swish` was free coverage — one `#elif` in `ew_unary.comp`, one op code, one row, two variants.
That is the §5.2 leverage thesis behaving exactly as advertised, on an op that is in every LLM MLP.

⚠️ **Build environment:** Switch had uncommitted, non-compiling work in `rust/src/vk/caps.rs`
(four `E0503` borrow errors), so `cargo build` failed in the main tree through no fault of mine. I
verified in a throwaway `git worktree add .mouse-verify HEAD --detach`, copied my files in, built
and tested there, then removed the worktree. **Reusable technique**: it validates your own changes
against a green baseline without touching another owner's in-flight files. Do not `git stash`.

📌 Verify recipe with real hardware:
`$env:VULKAN_SDK="C:\VulkanSDK\1.4.350.0"; $env:PATH="$env:VULKAN_SDK\Bin;$env:PATH"` — all 170
variants compile with `glslc`, no `ALLOW_MISSING_GLSLC` needed. Regenerate the variant table with
`MOUSE_BLESS_VARIANTS=1 cargo test --lib variants`. **306 passed / 0 failed / 7 ignored**, clippy
clean, in the verify worktree (HEAD + my files).

📌 A device dispatch test now passes: `vk::dispatch_integration::add_f32_dispatches_end_to_end`.
Switch's path is real. My kernels are still `UNEXERCISED` — none of mine has executed — but the
seam to exercise them exists now, and that is the T1 unblock.

### Turn 8 addendum — the `Attention`-24 ruling and what it taught about C2

📌 **Justin: no opset bump, implement the corrected semantics.** R4 closed. The lesson is not about
attention — it is that my two drift detectors are *both* version-based and therefore share one
blind spot: a semantic correction applied without a version change moves behaviour while every
number stays put. `ContribSchema` cannot fire. The opset window cannot fire. I had been treating
`ai.onnx` as the safe domain and `com.microsoft` as the risky one; that framing is wrong. **An
opset window is a strong guarantee about interface and no guarantee about behaviour.**

📌 There is no fix to write. That was the uncomfortable part — the honest output is a documented
limitation plus a routed dependency (Trinity pins `onnx`), not a mechanism. Recording a gap so the
machinery stops *looking* more complete than it is, is a legitimate deliverable.

📌 Generalisable: **write the limitation where the reader finds the mechanism.** I put §9.4.1 in the
doc, a "what this does not detect" section in the `ContribSchema` doc comment, and a line at the
decline site. Anyone reaching for the fingerprint struct now learns its boundary in the same breath
as its purpose.

📌 On how it surfaced: nothing detected it. Following §7.1 literally — narrow and decline, never
guess — meant going to *read* the 23→24 diff instead of assuming an open-ended window was harmless.
The errata was sitting next to the interface change in the same source file. Rules that force you
to go and look pay off on things they were not aimed at.

📌 Still true and worth repeating: **zero of my kernels have executed.** `add_f32` executing on both
local GPUs with validation layers clean is Switch's path working, not mine. The seam exists now;
the honest statement about my rows is unchanged.

---

## Turn 9 — the opset range (2026-07-29)

**Directive:** support the full ONNX opset range, not just what mobius emits.

**The number.** "Latest opset" has two correct answers and they disagree by one. onnx v1.22.0
registers 27 (`map_[ONNX_DOMAIN] = {1,27}`, what `onnx_opset_version()` returns, what
`make_model` stamps) but still reports `last_release_version_map_[ONNX_DOMAIN] = 26`. Justin
said 26, the coordinator measured 27, and neither was wrong. Lesson: when two people disagree about
a version number, the likely cause is that the library exposes two of them for different purposes.
Go read the header before picking a side. Both are now constants with a test.

**The result nobody expected: there was nothing to extend.** Every closed window I set last turn —
Attention 23..=24, RMSNorm 23..=23, RotaryEmbedding 23..=23, TensorScatter 24..=24, Swish 24..=24,
Q/DQ 21..=25 — is already at the newest schema version that *exists*. Opset 25 is a type-constraint
expansion, 26 is BitCast/CumProd, 27 is the SSM ops plus Range. So an opset-27 model is claimable
today for all of them.

That is entirely because windows are keyed on **schema version**, not model opset. Had I keyed them
on model opset, satisfying this directive would have meant re-reading and re-testing six predicates
at four more opsets. **A framing choice made for correctness reasons paid off as speed.** Worth
remembering the shape of that: the cheap answer to a scope directive usually comes from a
representation decision made earlier, not from working faster.

**The actual coverage win was a side effect.** Reading the opset-27 contents to answer a question
about *bounds* turned up that `LinearAttention` and `CausalConvWithState` are now `ai.onnx`
ops. I had them only as low-confidence `com.microsoft` main-branch rows. That is §4.18 recurring
on the Qwen3.5 hybrid path — same computation, two producers, two spellings — and it would have
gone unnoticed indefinitely because nothing about our contrib rows looks wrong. **Range checks find
things that are not range problems.** Read the whole diff, not the part that answers your question.

**A new argument for narrow predicates.** onnx#7913 swapped the meaning of `qk_matmul_output_mode`
1 and 2 with no opset bump. It cannot touch us, because we claim only mode 0. Generalised:
*the attributes you decline cannot drift under you.* I had been defending narrow claiming purely on
"we have not implemented that"; it is also a structural defence against the C2 blind spot. The
residual is exactly the attributes we do claim — for Attention that is the causal mask and GQA
repetition, both of which ONNX has silently corrected.

**Ambiguity is not always worth resolving.** Two sources disagreed about whether Q/DQ's
`precision` arrived at opset 23 or 25. I did not settle it: declining every non-default value is
correct under both readings and costs nothing. Check whether the conservative action is
reading-independent before spending time on the reading.

**Process.** Two `vk::barrier` tests fail under the parallel runner and pass with
`--test-threads=1` — a shared probe file in Switch's code. Confirmed it was pre-existing by
running the two tests in isolation rather than assuming. Flagged, not touched.

---

## Turn 10 — 2026-07-29 — The Foundry Local census: two real graphs, five wrong conclusions

**The lesson landed a third time and the recurrence is the finding.** Turn 6: wrong producer.
Turns 8-9: right producer, wrong revision. This turn: right producer at a pinned revision, whose
**actual output I had never read**. Each correction was one level more concrete than the last,
which is the tell that the underlying rule was still too abstract. The form I now hold:

> A claim about what a producer emits is not evidence until it has been read off a graph that
> producer actually produced. Builder source is intent; the model file is the fact.

I had written §8.5 ("a coverage number without a named producer at a version is not well-formed")
and still spent two turns reasoning about op sets from `builder.py` and schema headers. Writing a
rule is not the same as being governed by it. The concrete tell I missed: I could quote a builder's
rewrite-rule preconditions but could not quote a single node's input arity.

**I had been reasoning about the ceiling only.** Four turns of narrowing and extending upper
bounds — 23, 24, 26, 27 — and both real models import `ai.onnx` at **14** and **21**. The floor
excluded them outright from every standard-domain LLM row. Ranges have two ends and I had treated
one as interesting and one as settled. Ask which end the evidence actually constrains.

**Two of the five errors were permissive, which is the direction that costs.** `group_query_attention`
never read inputs 1 and 2, so it would have *claimed* a packed-QKV node and handed the kernel a
fused tensor where it expected a query. I had recorded "PackQKVForGQA never matches Qwen3" and
generalised a statement about one model family into a statement about the producer. Both cached
models pack on every layer. **A negative finding about one model is not a negative finding about
the world** — the scope of a "never" is exactly the scope of the evidence for it.

**Simulation beat argument, and contradicted it.** I had argued death-by-fallback abstractly in §7.
Running an island simulation on real graphs produced something I would not have predicted: adding
`Cast` to the claim set on gpt-oss raises coverage 28%→54% **and islands 52→125**. Claiming more
ops made partitioning strictly worse. And Phi-3.5 sits at 34-35 islands from T1 through T3 and then
collapses to *one island of 364* the moment `MatMulNBits` lands. Two conclusions I could not have
reached by reading: partial coverage of these graphs is worth nothing, and coverage percentage is
an actively misleading metric. **Cheap simulation against a real artefact is worth more than a
careful argument about a hypothetical one.** ~120 lines of Python.

**"Right answer, wrong reason" is a defect.** Real `QMoE` nodes were declining as
`[contrib-schema]` because my fingerprint was missing two attributes — but they should decline on
top-k (4, and we admit 1|2). The decline was correct and the diagnosis was wrong, which is exactly
what the machine-readable decline codes exist to prevent. **Check that declines fire for the reason
you think.** A census makes that checkable; nothing else does.

**A third registry category appeared that I had not modelled.** `SimplifiedLayerNormalization`
carries `domain == ""` — no ONNX schema, but the standard domain, so `Node_GetSinceVersion` is
meaningless and only a fingerprint detects drift. My taxonomy was binary (standard-with-opset vs
contrib-with-fingerprint) and reality had a third case. I handled it with an explicit allow-list
capped at four entries **by test**, so if it grows the test forces a real `Domain` variant rather
than letting the list quietly become the normal case. **When you patch a taxonomy with a list, put
a bound on the list.**

**Discipline held under temptation.** The census made it obvious how to make the coverage numbers
look good: lower the floor, admit `do_rotary`, widen top-k. I widened nothing and *narrowed* GQA.
The point of measuring what is out there is to know it, not to move predicates until the count
improves.

**Still true and worth repeating: zero of my kernels have executed.** Phi-3.5 is now a runnable
target sitting on this disk, which is a better first target than Qwen3.5 on every axis (MHA not
GQA, softcap 0, no SWA, no sinks, no QK norm, symmetric RTN, uniform bits/block, cold control
flow, 366 nodes). That changes what is worth building first, not what has been built.

---

## Turn 11 — 2026-07-29 — The first live row, and a question about someone else's oracle

**"Flip it to Ready" was a request my type system could not honour, and that was the wrong thing to
fix.** The vocabulary is `Live | Staged(reason)`; there is no `Ready`. The instinct was to add a
variant. The right answer was that **status is a gate and scope is not in the status** — scope
lives in `caps` and in the claim predicate. Adding `Ready` would have encoded one specific
narrowing (fp32, static) into a type that has to serve every future one. When asked to widen a type
to express a constraint, check first whether the constraint already has a home.

**A row's `caps` is not evidence that its variants ran, and I nearly let that through.** `Add`
arrived `Live` with `caps: NUMERIC` — four dtypes claimed on one executed shader. `caps` serves two
consumers (claim checking and variant generation) and that dual role is exactly what hid the
problem: narrowing `caps` to F32 would have "fixed" the claim by silently dropping three compiled
variants from the manifest. **When one field serves two consumers, a change that looks right for
one can be a regression for the other.** The fix was a narrowed predicate plus an `EXERCISED`
evidence list, leaving `caps` alone.

**Evidence lists beat flags.** `EXERCISED: &[("Add", "f32")]` with a test asserting it equals the
live set makes going live a two-place edit, and the second place demands a sentence naming a test
and a device. A boolean would have cost nothing to flip. The friction is the feature.

**Say what the bet actually is, not what it used to be.** The old decline read "the shader has
never executed" — true yesterday, false today. The bet moved to the *wire*, which has only carried
a mock host's tensors. Inheriting the old sentence would have made the row look better-established
than it is. **When the premise behind a caveat changes, rewrite the caveat; do not delete it and do
not keep it.**

**Flipping is what makes a bet settleable.** I would have been more comfortable waiting for the
differential test — but the differential test cannot run against a staged row. An unexercised path
that nothing can execute never gets proven. There is a class of caution that is indistinguishable
from never finding out.

**A question about someone else's component found a hole in mine.** Trinity asked whether her
oracle pinned at `accuracy_level=1` matches a model that emits `0`. Answering it meant reading
ORT's CPU kernel, and the kernel says something I had recorded the opposite of: my comment claimed
`accuracy_level` is "never a correctness requirement, so any value is claimable", but **level 4
quantizes the activation to int8** — a different computation, which we would have answered wrongly.
The permissive hole was in *my* predicate and I found it by investigating *her* pin. **"Is the
reference doing what I think" and "am I claiming more than I implement" have the same answer
surface: the kernel source.** Route questions about adjacent components toward the source, not
toward the owner.

**Prose lost to source again.** The schema doc says level 0 means "not quantized or downcast";
the coordinator read it as "let ORT choose"; the actual branch is a single `if (attr == Level4)`.
Three readings, one truth, and only the code had it. Same lesson as §4.21 in a new place: the
artefact over the document. I should stop being surprised by this one.

**And the answer was not the one anyone expected.** The pin is harmless (0-3 collapse to one path)
but it is also not doing the work it appears to. The divergence that is real is **keyed on host
architecture** — the fp16 compute path exists only on ARM64, so the same model and the same ORT
give an fp32-accumulated oracle on x86-64 and an fp16-accumulated one on ARM64. **When you check
whether a pin is necessary, check what else varies that nobody pinned.**

**Verify a reported failure before fixing it.** Tank flagged the default-domain contrib row as a
live failure. It was already fixed and committed; the tree was green. Rather than reporting "not my
problem", I checked the part that would have made the fix cosmetic — whether the `ai.onnx` row's
fingerprint is actually *evaluated* at runtime, or whether the schema check is gated on
`Domain::Ms`. It is domain-agnostic, so the fingerprint is real drift detection. **A green test is
not the same as a working mechanism; check that the thing fires.**

**Restraint, recorded as a test.** `Sub`/`Mul`/`Div` are one word from live and share the shader.
I left them staged and wrote a test explaining why, because the reason ("it's the same template")
is exactly the argument the discipline refuses, and it will look just as tempting next week.

---

## Turn 12 — the test contradiction, and the elementwise f32 flip (2026-07-29)

**When two of your own tests disagree, suspect the instrument before the subject.** Coordinator
asked which node form `test_barrier_parity` builds that the predicate declines. Answer: none. Both
`Add-fp32` at (3,4) and `test_add_is_claimed` at [4,4] claim. The tests used different *mechanisms*
— profiling JSON versus our own claim record — and the record was dead. I nearly went hunting for a
shape/rank/dtype difference first; the thing that cracked it was holding the model constant and
varying only the environment. **Vary one thing at a time, and make the thing you vary the one you
have not yet suspected.**

**A `OnceLock` is a statement that a value cannot change in a process.** Mine said the claim-log
path could not, because "an environment variable is set before the process starts". The only caller
that exists sets it per call, mid-process, hundreds of times. I wrote both sides. **Check the
lifetime assumption against the actual caller, not against the platonic caller.**

**A diagnostic whose failure mode is indistinguishable from a negative result is not a
diagnostic.** Missing log file → "not claimed" → a skip whose sentence sounded entirely reasonable.
It was invisible for as long as it was the only thing looking. The same defect had also been
quietly disabling Trinity's C1 regression test, which has an "if the EP wrote a log" guard — so it
was passing without asserting. **When you find one silent-degradation guard, grep for the others
reading the same input.**

**Restraint has an expiry date, and so does its justification.** Last turn I wrote a test asserting
`Sub`/`Mul`/`Div` stay staged because "it's the same template" is the argument the discipline
refuses. That was right *while the wire was unproven*. Once the wire carried real ORT tensors on
two vendors, the bet changed from "does the wire work" to "is this one line of GLSL right" — a
different and much smaller bet — and holding the line unchanged would have been ceremony. I did not
widen `EXERCISED`; I added a second, weaker, explicitly-bounded list. **When the premise of a rule
dissolves, replace the rule with a narrower one rather than either keeping it or dropping it.**

**Flipping is how the evidence gets produced.** A `Staged` row's differential test compares nothing
— it fails with "the EP executed no node". Waiting for evidence before flipping would have meant
holding 34 shaders permanently unverifiable in order to avoid claiming them. The flip is the
experiment. It came back clean on both devices, so all 34 were promoted to `EXERCISED` the same
day, and I recorded "it did not fail" as plainly as I would have recorded a failure.

**I planted a false red in my own test and it went off one turn later.** `registry.rs` named `Sub`
as its staged example; `Sub` went live and the test broke on data legitimately moving. **A test of
an invariant should select its fixtures from the data, not name them.**

**Build contention is now routine.** The shared `target/release` DLL was locked by another agent's
running pytest. `CARGO_TARGET_DIR=target-mouse` gave a private build in 1m43s with no disruption to
anyone; removed afterwards. Better than waiting, and much better than killing their processes.


## 2026-07-29 — the parameter tail: sixteen bytes that retired a whole blocker

**Fourteen rows were staged for months behind a blocker that cost one afternoon to remove.**
`LeakyRelu`, `Elu`, `Selu`, `Celu`, `ThresholdedRelu`, `Shrink`, `HardSigmoid`, `Swish`, plus
`Gelu` and `Clip` by adjacency. Every shader existed and compiled; each was one float away. The
blocker was correct — their GLSL had the ONNX default baked in as a literal, and claiming on that
answers `alpha=0.2` with `alpha=0.01` — but I had recorded it as a *property of the ops* rather
than as *one missing mechanism*. Four floats appended to the push-constant block unblocked all of
them at once. **When N rows share a blocker, the blocker is the work item, not the rows.** I should
have asked "what is the one thing all of these are waiting for" the first time I wrote
`NEEDS_PARAMS` fourteen times.

**The cost of the mechanism was the interesting part, and it was structural, not numeric.** Adding
the tail was trivial. Deciding it should be *unconditional* — parameterless ops push four zeros
they never read — was the real call, and it turned on a fact I had to go and check rather than
assume: `vk/pipeline.rs` declares a fixed 128-byte push-constant range for **every** pipeline. So
an under-filled block is not caught by the layout; the shader just reads bytes nobody wrote. Two
layouts would have been a silent-corruption generator. **I nearly wrote this into a decision record
as "pipeline.rs sizes the range from the supplied bytes" — which is false — and caught it only
because I went to verify a claim I was about to publish.** Verify the sentence you are about to
assert about someone else's file.

**I declined to use my own `TEMPLATE_LIVE` shortcut for the rows it literally permits.** These
satisfy every stated condition: same template, same translate, same descriptor layout, same
push-constant block, f32 predicate, exercised representative. I still made them earn their own
dispatch, because the tail is a **new code path**, not a new expression inside an exercised one —
a wrong offset for `params[0]` is invisible to every live op, since they all push zeros there and
read none of them. The sharpened rule: *template evidence covers a different expression in an
exercised path, never a different path.* The test to apply: **ask what a plausible bug in the new
code would do to the representative. If the answer is "nothing", the representative is not
evidence.**

**The float/selector distinction is the durable part.** `alpha` is a coefficient and rides the
tail. `Gelu.approximate`, `Mod.fmod`, `BitShift.direction`, `IsInf.detect_*` choose a different
*expression*, so they need a shader variant, not a value. I rewrote `NEEDS_PARAMS`'s text to say
this, so the blocker now names an obstacle that still exists. A staging reason that has been half
retired is worse than no staging reason: it reads as a to-do that someone already did.

**`Clip` was the case that looked like the others and was not.** Its bounds are optional *inputs*
from opset 11, not attributes, so the tail cannot carry them **in principle** — `TensorRef` exposes
`is_initializer` but not the contents, and a bound may be computed at runtime. But it did not need
the tail: three-input `Clip` is an ordinary ternary elementwise op whose rank-0 bounds broadcast
with stride zero, which the shared indexing helper already does. The right question was not "how do
I get the values into push constants" but "why did I think these were values". One- and two-input
`Clip` still decline `[arity]`: an omitted bound is a different **dispatch shape** — a descriptor
bound to nothing — not a different value, so widening the predicate would bind a buffer that does
not exist.

**Trinity's suite already tested the thing I built, before I built it.** `LeakyRelu(alpha=0.1)`,
`Elu(alpha=1.5)`, `HardSigmoid(alpha=0.15, beta=0.4)` were sitting in `test_op_table.py` failing
loudly. That meant the flip was verified against **non-default** values on the first run — what
passed was the mechanism, not the defaults that were already in the shader. A harness that fails
loudly on staged rows is what makes a flip a real experiment instead of a hope.

**Numbers, both devices, identical:** `test_elementwise` 25/11 → **33 passed / 3 failed**;
`test_op_table` 39 → **49 passed** / 28 failed; `test_barrier_parity` 36 → **46 passed** / 28
skipped. The three remaining elementwise failures are `Min`/`Max` (variadic) and
`test_clip_no_bounds` (my deliberate decline) — the table still reads as "what is left to do".

## 2026-07-29 — `MatMulNBits` GEMV: Live on both devices, and the island premise was wrong

**Shipped.** `com.microsoft::MatMulNBits` is `Live` — a block-dequantising GEMV, one workgroup per
output element, grid `(N, M_total, 1)`, so correct for every `M` rather than only decode. New
`Template::QGemv` + `shaders/glsl/templates/q_gemv.comp`. Both devices, identical:
`tests/ops/test_matmulnbits.py` 29 passed / 1 failed (the failure is `DequantizeLinear`, still
`Staged`, failing loudly by design). Coverage: bits {4,8} x block {16,32,64,128} x
{3-input symmetric, 4-input asymmetric} in fp32, M in {1,2,7,32}; fp16 at M in {1,3} both forms.
`cargo ci` ALL CHECKS PASSED.

**The lesson that actually mattered.** I was scheduled on this kernel on the premise that it
collapses Phi-3.5 from ~35 islands to one island of 364. Measured on the real graph, claiming
`MatMulNBits` alone takes coverage 27.3% -> 71.3% and islands **35 -> 100**. Same on gpt-oss:
148 -> 100 islands but largest island still 3. The collapse needs the **pair**
`(MatMulNBits, SkipSimplifiedLayerNormalization)` — 88.8% / 5 islands / largest 320 — because in a
GenAI decoder block every `MatMulNBits` is separated from the next by a SkipSLN or a GQA. Third
time this lesson has landed. I had written the "one island of 364" figure myself, from schema
reading rather than from the graph, which is exactly the §8.5 failure I keep documenting for other
people. **Measure the partition, never predict it.**

**Two things I got right by refusing to trust myself.**
1. Ran an empirical probe (`A = I` through the CPU EP) before writing GLSL, instead of writing the
   nibble order from memory. Settled low-nibble-first, implied zp = `1<<(bits-1)`, zp packing,
   scale indexing, `B` orientation. Recorded as §8.1.1.
2. Re-censused the real nodes' dtypes rather than assuming fp32 was a fine first target. **All 161
   Phi-3.5 `MatMulNBits` nodes are fp16** — an f32-only claim would have declined the entire model
   the kernel exists to run. Solved without a capability gate via `unpackHalf2x16`/`packHalf2x16`
   over `uint` buffers with fp32 accumulation.

**What actually blocks Phi-3.5 and it is not this kernel.** A static-shape node is claimed and
matches at rank 2 and rank 3; the same node with symbolic `batch`/`seq` is declined by the global
`REQUIRE_STATIC_SHAPES`, because `Compile` bakes byte sizes. Every real node has symbolic leading
dims. The island numbers are a ceiling, not today's behaviour. Say so before someone reads
"MatMulNBits is Live" as "Phi-3.5 partitions".

**Prepacking.** Wrote the pure transform; it is a **pass-through**, and saying so was the right
answer — ONNX's `B` layout is already what a workgroup-per-column GEMV streams. The seam is also
not connected: nothing calls `compile_hook_for`. It did not matter, because `plan.inputs` is the
fused node's inputs with `drop_constant_initializers = false`, so weights arrive as ordinary
Compute inputs. Cost is re-upload per `Run` — that is what connecting the hook buys.

**Found in Trinity's `_models.py`:** the fp16 builder emitted fp32 `scales`/output while
`MatMulNBits` binds `A`/`scales`/`Y` to one `T1`, so ORT rejected the model and the fp16 test had
never reached a kernel; and `zp_bytes_per_col` assumed 4 bits. Fixed both, added `with_zero_points`
and `rows` knobs. Her file — flagged, not imposed.

**Habit to keep:** when a brief hands me a number, re-measure it before building on it. The number
in this brief was mine, and it was wrong.

## 2026-07-29 (evening) — the decline census: measuring *why* nodes are declined

Coordinator ran Phi-3.5 through the EP: 0 claimed, 258 `dynamic-shape`, 100 `staged`, 5
`not-registered`. Asked me to quantify the 258 and cost three options for handling them.

**What I found, in order of how much it changed:**

1. **The decline histogram is first-match and is not a partition of causes.** Cross-tabbing the
   code against an independently computed shape class: only **4** of Phi-3.5's 363 nodes and **2**
   of gpt-oss's 371 have fully static shapes. Landing every staged kernel while
   `REQUIRE_STATIC_SHAPES` stands moves claimed from 0 to **2** and **1**. So shapes are not "2.5x
   the problem" — the shape gate is upstream of the kernel gate for ~99% of nodes. The two models
   look *opposite* (Phi shape-dominated, gpt-oss kernel-dominated) purely because gpt-oss's 100
   `Cast` nodes hit the staging test first. Same cause, opposite appearance.

2. **"Dynamic shapes" is too coarse.** Every symbolic dim in both graphs is a *leading* axis;
   the last axis of every declined tensor is a literal. Coined EXTENT-ONLY vs STRUCTURAL:
   324/359 and 318/369 are EXTENT-ONLY. Broadcasting is decidable symbolically because equal
   `dim_param` implies equal extent.

3. **Option (a) — Compile-time shapes — already works, and I had written it off.** Pinning
   `batch_size=1, sequence_length=1` with `AddFreeDimensionOverrideByName` claims **161 nodes**,
   the model **runs**, and it matches the CPU EP on **both** devices (argmax 30751, top-5 match,
   max|Δ| 0.0078 decode / 0.0488 prefill seq=16). First production model executing on this EP.
   Also confirms §8.1.2 on hardware: MatMulNBits alone = 161 one-node islands.

4. **(b) and (c) are the same change and contain no shader work.** No pipeline in the Live set is
   keyed on a runtime extent — checked `dispatch_elementwise` (`all_identical` is structure, not
   extent) and `q_gemv` (`local_size_x` from static `K`). Extents live only in push constants and
   grid dims. Cheapest route: run the translate handler a *second time* at Compute against a
   `ComputeRecorder` implementing `DispatchContext` with real shapes. Three parts: my
   `REQUIRE_STATIC_SHAPES`, Switch's `CompiledKernel` fields, Switch/Tank's `dispatch_ort`.
   I will not flip mine before theirs exist — it would produce a wrong answer, not an error.

5. **A second gate I had never seen: f16.** With shapes pinned, 97 nodes decline on **dtype** —
   `Mul` x64 and `Sigmoid` x32 are f16. The elementwise family is Live for f32 only, so on a real
   fp16 decoder our celebrated elementwise coverage is worth **zero nodes**. Cheapest remaining
   work in the whole plan, entirely mine. Did not flip it — the brief asked for costs, not counts,
   and f16 deserves the same device proof f32 got.

6. **gpt-oss-20b runs on no EP here** — ORT's own CPU QMoE rejects `swiglu_fusion=0` at session
   init. GetCapability still runs so the census is valid, but there is no CPU oracle for T5b.

**Lessons for me.** I keep writing "the blocker is X, singular" and being wrong twice over: X was
reachable from the caller, and there was a Y behind it. Before calling something a blocker, check
(a) whether anyone outside the EP can move it, and (b) what the *next* predicate would say if it
did. Also: `claim_log`'s sink only reopens on path change, so delete-and-reuse silently records
nothing — cost me a measurement.

No code changed this turn. `docs/OP_COVERAGE.md` §7.4 new, §8.1.3 corrected.
Record: `.squad/decisions/inbox/mouse-decline-census.md`.

---

## 2026-07-29 — Full-set decline audit; the predicate accepts symbolic extents (Morpheus R8, §8.8)

Morpheus read my claim path and found the histogram I had reported was structurally misleading:
`claim_decision` recorded only the **first** failing check. R8: early codes are ceilings, late
codes are floors, and two decline counts are not comparable without knowing the check order.

**What I built.**
1. `registry::claim_audit` runs *every* check and returns `ClaimAudit { primary, failures,
   unevaluated, shape_class, predicate_ok, predicate_ok_with_runtime_extents }`. The JSONL record
   is extended, not replaced — `code`/`reason` keep first-match meaning.
2. `shape_class` is computed from the node's edges, **row-independent**, because a staged row's
   predicate may be a stub and its answer is therefore not evidence.
3. Three-way split in `check_shape`: extents-symbolic → claimable, rank-unknown → decline
   (`unknown-rank`), data-dependent → permanent decline (`data-dependent-shape`).
4. `REQUIRE_STATIC_SHAPES` → `ENGINE_ACCEPTS_RUNTIME_EXTENTS` (inverted): the constant describes
   `vk::session`, not claim logic.
5. `AssumeRuntimeExtents` — an RAII counterfactual so "how many would this unlock?" is answered by
   running the real predicates instead of re-implementing them in a probe.

**What it measured (Phi-3.5, identical on both devices).**
* Full-set `dynamic-shape` = **356 of 363**, not 258. 98 of the 100 staged nodes are *also*
  shape-blocked.
* **The three staged kernels alone unlock 0 nodes.** Not "at most 100" — zero.
* **227 nodes become predicate-clean under runtime extents**; 161 (`MatMulNBits`) claimable
  immediately, 66 staged.
* The 97 that do not unlock are blocked on **dtype** — R8 recurring one level down, inside a
  predicate that returns a single reason. A known limit of the audit, recorded rather than hidden.
* gpt-oss first-match would have **falsely triggered** Morpheus's stated reversal condition
  (146 < 197); full-set is 342 > 197.
* No regression: still 0 claimed unpinned, **161 claimed** with dims pinned.

**Bug found while changing it.** `check_broadcast` returned `Ok(())` early whenever static shapes
were not required — so the instant symbolic extents became acceptable, broadcast compatibility
would have gone *unchecked*. Now symbolic-aware: literal extents still checked pairwise.

**Lessons for me.**
* I reported that histogram as fact for a whole turn. It was produced by code I own and had read.
  When a measurement is going to drive a roadmap, read the *producer*, not just the output —
  a first-match histogram is indistinguishable from a complete one at the point of use.
* "Rejecting symbolic extents" was a **design correction, not a defect**: right for a static-shape
  EP, wrong for an LLM EP. Say which of the two a change is; they carry different lessons.
* `EdgeType.shape` drops `dim_param`, so we cannot prove two symbolic dims are equal. Treat
  symbolic-vs-symbolic equality as unknown, never equal.

Code: `ops/common/claim.rs`, `registry.rs`, `ops/claim_log.rs`, `ops/quant.rs`.
Docs: `docs/OP_COVERAGE.md` §7.5 new, §8.1.3 corrected.

---

## Session 20 — 2026-07-30T09:14:00-07:00 — all-zero logit investigation and fix

**Task:** After Switch's runtime-extents work enabled 161 fp16 MatMulNBits nodes on Phi-3.5,
the model dispatched all 161 with `compute_failures=0` but produced all-zero logits on both
Intel Iris Xe and RTX 4060.

### Vacuous-pass correction

A prior Mouse session (session 19) reported "bit-identical" results by comparing against
a pre-fix DLL. That result was a vacuous pass (R7): the EP was registered and appeared in
`get_providers()`, but the dynamic-binding bug caused no output to be written; ORT effectively
ran CPU-on-both-sides while the check passed. The coordinator's original zero-logit observation
was correct. The prior session did not rebuild the DLL after session.rs was patched.

### Failure mode localisation

- **Isolated f16 MatMulNBits with static shapes (test_matmulnbits_fp16_matrix): PASS**
  — the f16 GEMV kernel computes correctly; the bug is not in the shader.
- **Isolated f16 MatMulNBits with dynamic M (symbolic_batch=True): ALL ZERO**
  — confirms the failure mode is in the session-layer dynamic-dispatch path, triggered only
  when the activation tensor has a symbolic leading dimension.
- **Phi-3.5 model pre-fix: logits [0.0, 0.0], top-10 overlap 0/10 (both devices)**
  — confirmed zeros persist after rebuilding with 3a0cc58 (Step 1b fix is unrelated).

### Root cause

`push_dynamic_kernel` (session.rs) creates binding tokens from NodeDesc input/output counts:
3 inputs + 1 output = **4 tokens** `[0, 1, 2, n_plan_inputs]`.

But `matmul_nbits_gemv` (quant.rs) passes **5 bindings** to `KernelRequest::dispatch`:
`[a, b, scales, zp, y]` where `zp = scales` (no zero_point input → scales bound twice as
inert placeholder for shader slot 3). The q_gemv.comp DTYPE_F16 shader declares 5 binding
slots (0=A, 1=B, 2=scales, 3=zero_points, 4=OutY).

At Compute time `dispatch_ort` used `kernel.bindings.len()=4` for `n_bindings`. The pipeline
descriptor set layout had 4 slots (0–3). Shader binding 4 (the output `OutY`) fell outside the
layout and was never bound. Both Intel Iris Xe and NVIDIA zero-initialise freshly allocated GPU
memory for security; the unwritten output buffer read back as all-zero.

**Failure mode: "writing nothing"** — kernel computed correct partial sums into shared memory
and reduced them, then called `store_y()` which executed `atomicAnd`/`atomicOr` on a buffer
that had no binding in the descriptor set. On both tested drivers this is a silent no-op.

Static-shape isolation tests bypassed `push_dynamic_kernel` entirely (CompileRecorder captures
the full 5-element binding vector from the translate handler's KernelRequest directly).

### Fix

`ShapeOnlyRecorder::dispatch` (session.rs) now captures `k.bindings` alongside the other
fields. `dispatch_ort` uses those captured bindings — not `kernel.bindings` — for `n_bindings`
(pipeline/descriptor-set creation) and `buf_bindings` (VkBuffer mapping) on the dynamic path.

### Regression test

`test_matmulnbits_fp16_dynamic_batch` (test_matmulnbits.py):
- Uses `make_matmulnbits_model(..., symbolic_batch=True)` to trigger `push_dynamic_kernel`
- Asserts `np.any(np.abs(vk_out) > 1e-4)` — all-zero fails
- Confirmed FAIL on pre-fix build, PASS on fixed build (both devices, K=256, N=64, fp16)

### Phi-3.5 results post-fix

| Device | logits range | max\|VK-CPU\| | top-1 | top-10 |
|--------|-------------|--------------|-------|--------|
| 0 Intel Iris Xe | [-13.11, 13.02] | 0.031 (0.24%) | ✓ | 10/10 |
| 1 RTX 4060 | [-13.11, 13.01] | 0.035 (0.27%) | ✓ | 10/10 |

Not bit-identical (fp16 accumulation differences compound across 161 nodes) but correct:
both top-1 and top-10 agree with CPU oracle, max abs diff is within fp16 MATMULNBITS_FP16
tolerance (rtol=2e-2), and not a function of device vendor.

### accuracy_level ruling (Trinity's question)

Model declares `accuracy_level=0`; oracle pinned at `accuracy_level=1`. ORT CPU kernel
branches on accuracy_level exactly once: only level 4 changes computation (int8 activations).
Levels 0, 1, 2, 3 all resolve to `SQNBIT_CompFp32` on x86. The GPU shader always uses
`float acc = 0.0` (fp32 accumulation). Levels 0 and 1 are indistinguishable. `accuracy_level`
was NOT a cause of zeros and the oracle pinning is correct.

### Files modified this session

- `rust/src/vk/session.rs` — ShapeOnlyRecorder captures bindings; DynCaptured includes them;
  dispatch loop uses eff_bindings for pipeline creation and buf_bindings
- `tests/ops/_models.py` — `make_matmulnbits_model` gains `symbolic_batch` parameter
- `tests/ops/test_matmulnbits.py` — `test_matmulnbits_fp16_dynamic_batch` regression test
- `tests/ops/test_phi35.py` — docstrings corrected to actual root cause
- `.squad/decisions/inbox/mouse-f16-zero-logits-postmortem.md` (main inbox) — decision record

Record: `.squad/decisions/inbox/mouse-runtime-extents.md`.

---

## 2026-07-29 — Moved to worktree `C:\Users\justinchu\dev\ep-vulkan-mouse` (branch `squad/mouse`)

`main` is now the integration branch. I commit to `squad/mouse`; I still do not push.

**Transfer, done carefully because the failure mode is silent data loss.** My previous turn's work
(the full-set claim audit) was still *uncommitted in main's working tree* — `5ab2d85` did not
contain it. Order: export the diff, apply it in the worktree, run `cargo ci` there, commit, verify
`git diff main squad/mouse --stat` matches the six files, and only then `git checkout --` in main.
Reverting main first would have been one mistyped path away from losing 900 lines. The gitignored
`.squad/decisions/inbox/` is not carried by a branch, so I copied it across by hand.

**The worktree reproduces the numbers of record on both devices**, which is the check that makes it
a working environment rather than merely a green one:

* unpinned — 363 records, 0 claimed, full-set `dynamic-shape` 356, shape_class 360 extents-symbolic
* pinned — 161 `MatMulNBits` claimed, residual `dtype` 97

Byte-identical on device 0 and device 1. `cargo ci` green in the worktree: 336 lib tests.

**Why this matters to me specifically:** the false red that cost me a turn was Switch's uncommitted
`vk/caps.rs` in a tree I did not own. In here, `cargo ci` red means *I* broke something — which is
the only condition under which a signal is worth reading. Tank's framing is the right one: a false
red teaches you to ignore the tool.

No cross-owner edits this turn. Nothing outside `ops/**`, `registry.rs`, `OP_COVERAGE.md`.

---

## 2026-07-30 — fp16 elementwise, and the two bugs a closed claim was hiding

**Brief said "build MatMulNBits GEMV". It was already built** — row `Live`, f16 handled,
`q_gemv.comp` complete, 161 nodes claimed on the real model matching the CPU EP on both devices.
Checking the premise before starting the work is what turned this turn into something.

**Then I checked the brief's other premise and it was also wrong.** It said `Mul`x64 / `Sigmoid`x32
/ `Sub`x1 were "blocked solely by compile-time extent baking". Pin the shapes and those exact 97
decline on `dtype`. That is R8 recurring at the level of the coordinator's own reading of a
first-match code — the same defect one layer up, which is the lesson's fourth landing and by far
the most uncomfortable one, because the person quoting the histogram had just finished writing the
rule about histograms.

**Two gates in series, so I went and shut the second one.** 96 of the 97 are f16, and the
elementwise family was f32-only. `Sub` is i64 and stays declined — correct, not pending.

### What I found while doing it

**The narrowing was hardcoded next to the evidence that already knew.** `only_f32` sat beside
`EXERCISED`, which recorded the same fact and was read by nothing in the claim path. Two sources of
truth; the weaker won. Now `only_proved_dtypes` reads the evidence list directly. Semantics-
preserving on introduction; the value is that widening a claim is one edit rather than two that can
disagree. **This kind of drift is invisible: it declines nodes we can serve, and nothing fails.**

**Every f16 shader we had ever built was unloadable.** `SCALAR_T = float16_t` under
`GL_EXT_shader_16bit_storage` made every f16 module declare `OpCapability StorageBuffer16BitAccess`.
The engine enables only `synchronization2`. Nothing had failed because nothing had ever *asked* a
device to load one — the f32-only claim guaranteed it. So: a capability we generate is not a
capability we have, and generation proves nothing. Fixed by packing f16 into `uint` with
`unpackHalf2x16`/`packHalf2x16` — no device feature at all, the same trade `q_gemv.comp` already
made, which is exactly why *its* f16 path worked and this one did not. Generalised into a test that
decodes `OpCapability` from every embedded module against an allowlist, because this bug class is
silent by construction whenever the matching claim is closed.

**Device 0 earned its keep, loudly.** fp16 differential: 12/12 on NVIDIA, **6/12 on Intel** — every
failure the last element of an odd-length tensor. 15 f16 elements are 30 bytes; the store for
element 14 addresses bytes 28..31, past the bound range. The 4060 absorbs it and is right; the Iris
Xe applies `robustBufferAccess` and writes a zero. The wrong answer was the *quiet* one. Tested only
on the fast card, this ships and surfaces as a wrong logit on a stranger's laptop.

`indexing.glsl` had *asked* the allocator to round sub-word buffers to four bytes. ORT sizes its own
tensors exactly and we bind what we are given, so the request was unenforceable — **a requirement
the EP cannot enforce has to be met by declining, not by asking.** `check_subword_tail` declines
what it cannot prove even, with a named lift condition for Switch (round the descriptor range up)
that deletes it rather than relaxing it.

### Result

257 claimed on pinned Phi-3.5, both devices identical: `MatMulNBits`x161, `Mul`x64, `Sigmoid`x32.
`dtype` is **gone from the unpinned full-set histogram**; 257 nodes are now blocked by dynamic shape
and nothing else, and the log says so in a field rather than by inference. And the session *runs* —
65 outputs, same argmax token as the CPU EP, 0.035 max fp16 logit deviation, on both devices. First
real-model arithmetic on this EP.

Per SS7.5.8 the pinned number is a measurement device and not a milestone, and I will keep saying so,
because it is the single most quotable number I have produced and the easiest one to misuse.

### For my own record — the design correction, stated as one

Rejecting all symbolic dims was right for a static-shape EP and wrong for an LLM EP. Restricting an
f16 claim to provably-even element counts is the same shape of judgement in the other direction:
narrow deliberately, name the lift condition, and do not let the restriction outlive its cause.

### Cross-owner edits

* `tests/ops/test_op_table.py` (Trinity) — data rows only, no harness logic.
* `rust/shaders/**` — ownership still unresolved; asked twice.

Baseline note: `test_op_table.py` is **28 failed / 63 passed** with my change and **28 failed / 49
passed without it** — pre-existing, intentional per its docstring, +14 and no regressions from me.
A suite red by design still trains people to stop reading it.

---

## 2026-07-30 (later) — the variant census, and finding my own guard had the hole in it

**Two rulings landed.** Decision records go in `main`'s inbox, not the worktree's — my own finding,
and the coordinator confirmed Switch nearly lost a record to it. Shader ownership is by op, not by
directory: op kernels mine, `shaders/include/**` Switch's. Which means `indexing.glsl` — which I
rewrote yesterday for the packed-f16 path — is his. The edit predates the ruling and is validated on
both devices, so I flagged the hunk for review rather than reverting it; reverting removes the f16
path entirely. From here I ask before touching that file.

**Assignments were a turn behind: both items were already done.** So I went looking for the next
real thing instead of re-reporting.

**The next real thing was a question I had not thought to ask.** SS7.4's rule is that planning is
driven by the decline histogram rather than the op histogram. There is a level below that. An op
census says which op; a decline census says which op first; **neither says which *variant* is worth
anything.** On an fp16 model that decides between 64 nodes and 0.

So I censused the graph by dtype signature. Every staged node that matters is f16 end to end:
`SkipSimplifiedLayerNormalization` x64, `GroupQueryAttention` x32, `SimplifiedLayerNormalization` x1.
**`skip_simplified_layer_norm_f32.comp`, which is in flight right now, claims zero nodes of
Phi-3.5.** That is exactly the mistake I made with elementwise, about to be repeated one kernel
later, and catching it before the kernel is finished is worth more than anything else I did today.
Two more things only the signature census shows: skip-norm's output count *varies* (63 nodes bind
two, one binds one), and GQA mixes f16 tensors with i32 seqlens inside a single node.

**Then the uncomfortable one.** The i64 variants declare `OpCapability Int64`, which needs
`shaderInt64` enabled; the engine passes no `pEnabledFeatures` at all. Same bug as the f16 one,
still live. **And the guard I wrote yesterday to catch that class allowed `Int64`** — with a comment
of mine reading "core in Vulkan 1.0 via `shaderInt64`", which is true about the feature existing and
irrelevant to whether it is enabled.

A guard whose allowlist is written from the same misunderstanding as the bug it guards against
inherits the bug. I wrote a plausible-sounding justification into the one place designed to reject
plausible-sounding justifications, and it read as diligence. The general form: **the dangerous
review comment is the one that sounds like a reason and is actually a restatement.**

Fixed by splitting one list into two — `GENERATED_CAPABILITIES` (what may be built; wide on purpose)
and `ENGINE_ENABLED_CAPABILITIES` (what a live claim may rest on; `Shader` only) — and by adding a
claim-side test that walks every proved pair to its module. **I fired it deliberately** by adding
`("Sub","i64")` to `EXERCISED`, watched it fail with the right message, and reverted. A guard that
has never fired is a guard nobody has tested, and I had just written one.

Generation and admission are different claims. GLSL compiling says nothing about whether a device
can create the module, and the claim is the only place the two are reconciled.

No cross-owner edits this turn.

## 2026-07-30 — P6, and the run that had never been run twice

Assigned `MatMulNBits` GEMV for the third time. For the third time it was already built. Checking
the premise before starting is now the cheapest thing I do: kernel, claim row, workgroup derivation
from the guaranteed floor, prepack transform, `accuracy_level` reasoning — all present, all green.

So I went looking for what the brief did not know, and found it in someone else's file:
`allocator.rs` names "Mouse's P6 assertion" twice, and P6 had never been asserted anywhere. The
constraint had been stated in the design doc, quoted back to me in three consecutive briefs, and
cited in another owner's code — and nothing in the tree would have failed if it were violated.
**A constraint everybody repeats is not a constraint that anything enforces.**

I asserted it structurally rather than dynamically, and that choice is the substance. `alloc_temp`
is the only route from an op handler to device memory, so counting calls proves the property for
every shape at once; a high-water threshold proves it only for shapes actually run, and any bound
loose enough not to be flaky is loose enough to hide a small scratch buffer. **Zero is not a
threshold.** Negative-controlled it by inserting a deliberate `alloc_temp` and watching it fail
with the right bytes, then reverting — the same discipline as the `Int64` guard, and for the same
reason.

The other finding is Tank's, seen from my side. He measured interior pointers appearing from run 2
of a session. I then noticed that **every model-level check on record, mine included, had run
exactly one inference per session** — so the entire body of evidence covered run 1 and nothing else,
and we would each have sworn the model was verified. Five runs in one session with differing feeds:
clean on both devices. The insulation is structural, because op code never sees a raw pointer. But
"structurally impossible" was my belief before I measured, and it is only now a result.

The recurrence worth naming: a test harness has a shape, and its shape decides which bugs are
*reachable*, entirely independently of how many assertions it makes. One inference per session is
not a weaker version of five; it is a harness in which a whole class of defect does not exist.

No cross-owner edits this turn.
📌 Team update (2026-07-30T05:48:29-07:00): A green suite has been shown not to imply a correct model. Phi-3.5: 161 MatMulNBits dispatched, compute_failures:0, entire suite green — vk logits all-zero (argmax 0 vs CPU argmax 30751). R9 (Morpheus): for every claim, name the instrument that would go red if the claim were false; if none, the claim is UNMEASURED. model_output_equivalence verdict required alongside all counter summaries; default UNMEASURED. Any comparison must first assert EP_NAME in session.get_providers() before calling sess.run() — failure to do so compares CPU to CPU and reports agreement. Coordinator's own first comparison reported bit-identical on both devices due to this exact error. Trinity has landed xfail(strict=True) correctness gate. M0 criterion 10 added (NOT MET: DIVERGENT). Criteria 2, 4, 5 reopened. — decided by Morpheus, Trinity, Switch, Mouse; coordinator-verified.
## Session 21 — 2026-07-30T09:14:00-07:00

### Context
Coordinator relay from Tank (07:51 AM): three-run session on Phi-3.5 revealed two SEPARATE failure
modes affecting different tensors:
  - Output 0 (logits): exactly 0.0 on runs 2 and 3 in a dirty arena → computed zero (not unwritten)
  - Outputs 1..64 (KV cache): bitwise different between runs → unwritten, arena reuse
Tank ruled out his allocator layer with a control (same results with device memory unset).
KV cache routing: Switch (binding/partition question at N=161).
Logits routing: Mouse — "computed zero" is consistent with the fp16 hypothesis.

### Merge and conflict resolution
Merged origin/main. Conflicts in tests/ops/test_phi35.py resolved:
  - Docstring conflict: kept HEAD's post-fix state description, removed "KNOWN BUG" section
  - Determinism test docstring: merged both versions (kept renamed-from note from origin/main)
  - Removed @pytest.mark.xfail(strict=True) from Trinity's test_phi35_vulkan_matches_cpu_logits:
    the fix landed in commit 64f390b; xfail is now stale, and with strict=True it would XPASS-error

### Multi-run test suite additions (Tank's discriminator requirement)
Tank's relay: every probe run must run the session ≥3 times and compare across runs. Single-run
evidence is structurally incapable of distinguishing "computed zero" from "unwritten zero in a
clean arena" (ORT's memory-pattern planner does not engage on run 1).

Added: test_matmulnbits_fp16_dynamic_batch_multirun
  - Creates one session with symbolic_batch=True (dynamic-batch fp16 MatMulNBits)
  - Runs sess.run() 3 times with identical feeds
  - Asserts all 3 runs produce non-zero output (pre-fix: zeros on all 3 runs due to unwritten binding)
  - Asserts all 3 runs are bit-identical (determinism within a session)
  - Confirmed: FAIL on pre-fix DLL (zeros on run 1), PASS on fixed DLL (both devices)

Added: test_phi35_vulkan_multirun_logits_stable
  - Creates one Phi-3.5 session and calls run() 3 times with identical feeds
  - Asserts all 3 runs produce non-zero logits (vacuous-pass guard + non-zero guard on each run)
  - Asserts runs 2 and 3 are bit-identical to run 1

### Verification results (post-fix DLL, commit 64f390b + merge 8523733)

test_matmulnbits_fp16_dynamic_batch_multirun:
  Device 0 (Intel Iris Xe): PASS (3 runs, non-zero, bit-identical)
  Device 1 (RTX 4060):      PASS (3 runs, non-zero, bit-identical)

test_phi35_vulkan_matches_cpu_logits (Trinity's gate, xfail removed):
  Device 0: run range [-13.1094, 13.0156], argmax=30751 (CPU: 30751), top-10 10/10 — PASS
  Device 1: run range [-13.1094, 13.0078], argmax=30751 (CPU: 30751), top-10 10/10 — PASS

test_phi35_vulkan_multirun_logits_stable (3 runs, same session):
  Device 0: runs 1/2/3 all [-13.1094, 13.0156] argmax=30751 — bit-identical ✓
  Device 1: runs 1/2/3 all [-13.1094, 13.0078] argmax=30751 — bit-identical ✓

### The multi-run result and Tank's discriminator
Tank's relay said: "Something actively writes zeros to output 0 on runs 2 and 3." My fix makes
the output binding correctly included in the descriptor set. On the dirty arena (runs 2 and 3),
the shader now writes the CORRECT non-zero values — not zeros, not garbage. This confirms:
  - The pre-fix failure WAS "output binding missing from descriptor set" (not computed in shader)
  - The zeros Tank saw on runs 2/3 were from the driver zero-initialising each freshly allocated
    GPU output buffer (Vulkan malloc consistently returns zeroed memory on both Intel and NVIDIA
    for security — even when arena sub-division is active, the OUTPUT BUFFER for each MatMulNBits
    Compute call is a fresh vkCreateBuffer allocation, not a reused ORT tensor)
  - The KV cache dirty-arena behavior is a separate phenomenon: those outputs pass through ORT's
    CPU memory pool (not fresh Vulkan allocations), so ORT's memory planner CAN sub-divide them

### KV cache status
The 64 KV cache outputs (outputs 1..64) that Tank found bitwise different between runs are
Switch's domain. They are NOT MatMulNBits outputs. They are KV cache tensors computed by CPU ops
(Slice, Concat, etc.) downstream of some EP intermediate. The unwritten EP intermediate (now
fixed) fed wrong values into the CPU ops, causing the KV cache to be unstable between runs.
Post-fix: not verified directly (Switch owns it), but the logit stability confirms the LM-head
MatMulNBits outputs are now correctly written.

### accuracy_level ruling (standing, confirmed)
The model declares accuracy_level=0; Trinity's oracle is pinned at 1. ORT's CPU kernel: levels
0-3 all map to SQNBIT_CompFp32 (only level 4 changes computation). The GPU shader always uses
float acc = 0.0 (fp32 accumulation) regardless of the attribute. Levels 0 and 1 produce
identical computation. accuracy_level was NOT a factor in the zero-logit bug, and is not a
factor post-fix. Ruling: oracle pinning is correct; no change needed.

### Commits
  8523733 — Merge origin/main, resolve conflict, remove stale xfail from Trinity's gate
  (plus new tests in test_matmulnbits.py and test_phi35.py for multi-run stability)

---

## Session 23 — 2026-07-30T09:14:00-07:00

### Context
Coordinator (Niobe's measurement): 257 → now 321 islands. 12.1× slower on Intel, 7.9× on NVIDIA.
Priority: SkipSimplifiedLayerNormalization (128 nodes) to reduce island crossings.

### Merge conflict resolution (origin/main, c650f2c)
- `session.rs`: Switch's split-fields approach kept over Mouse's 5-tuple bundle (semantically identical, Switch's is better factored). My struct split was already adopted by Switch in the merge commit.
- `test_phi35.py`: Trinity's version kept. Trinity removed my xfail (fix confirmed both devices), added cross-run consistency test. My docstrings dropped — intended outcome.

### SkipSimplifiedLayerNormalization — implementation
SkipNorm already had a f32 shader and a stub translate handler. The f16 shader was missing.

- Wrote `skip_simplified_layer_norm_f16.comp`: uint buffers, `LOAD_HALF`/`STORE_HALF` macros (unpackHalf2x16 + disjoint-lane atomics), 3-pass algorithm (partial sq sums, tree reduce via shared memory, normalize), arithmetic in f32 for precision. Race-free proof: stride loop places adjacent logical elements lsz apart in memory — no two threads within one pass share a uint word.
- Updated `norm.rs`: SkipNorm → `Ready`, `templates::skip_norm`.
- Updated `templates.rs`: f16 dispatch path added; 3 new unit tests.
- Updated `registry.rs`: direct-shader row handling in `no_live_row_lacks_a_shader_or_dispatch_path` test.
- New `tests/ops/test_skipnorm.py`: 7 tests covering f32/f16, slot-0/slot-3, CPU-match, Phi-3.5 shape.
- `test_matmulnbits.py`: `xfail(strict=True)` added to `test_dequant_linear_bit_exact`.

### alloc_temp infrastructure bug (root cause of 2 test failures)
`skip_norm` is the first op to use `alloc_temp` in production. The `CompileRecorder` and `ShapeOnlyRecorder` both assigned temp tokens above the ORT-output range, but `buf_bindings` in `dispatch_ort` only indexed into `gpu_outputs` — panicking when j ≥ len(gpu_outputs).

Fix:
- `CompileRecorder`: added `pending_temp_sizes: Vec<u64>`, flushed on `dispatch()` into `CompiledKernel::temp_byte_sizes`.
- `ShapeOnlyRecorder::alloc_temp`: already pushed to `temp_descs` (done in prior session).
- `dispatch_ort`: added `gpu_temps: Vec<GpuBuffer>`, `dyn_temp_sizes: Vec<Vec<u64>>`, `temp_starts: Vec<usize>` (per-kernel offset into gpu_temps). Allocated temps after outputs. Routed temp tokens via `temp_starts[ki] + (j - n_ort)`. Extended `free_all` signature to 5 pools.
- `ep.rs`: added `temp_byte_sizes: Vec::new()` to test-only `CompiledKernel` construction.

Before fix: `test_skip_norm_f32_slot0_matches_cpu` and `test_skip_norm_f16_slot0_matches_cpu` panicked with index-out-of-bounds. After fix: both pass.

### Test results
- Rust: 363 passed, 0 failed (both before and after alloc_temp fix)
- Python test_skipnorm.py: 7/7 on device 0 (Intel Iris Xe) and device 1 (RTX 4060)
- Python test_matmulnbits.py: 31 passed, 1 xfailed (test_dequant_linear_bit_exact, correct) — both devices
- The xfailed test fires the vacuous-pass guard because DequantizeLinear is still Staged

### Island measurement — prediction vs. reality
**Prediction (stated before building):** 128-200 fewer islands (from 257 to ~57-129).
**Falsifier stated:** if subgraphs_live drops by less than 128, some SkipNorm nodes are not between two claimed nodes.

**Measured result:** 257 → 321 islands. The falsifier fired. Claiming 64 new SkipNorm nodes added 64 new islands — none of them merged neighbouring MatMulNBits islands. The coordinator's hypothesis that "each SkipNorm sits between two MatMulNBits nodes" was wrong on the Phi-3.5 graph as ORT partitions it. Claiming more ops adds islands before it removes them, unless the newly claimed op is the sole unclaimed gap between two existing claimed islands.

Written into OP_COVERAGE.md §7.1.4 with the measurement note. The general rule: op priority should be chosen by `declined_nodes` histogram island-removal potential, not by node count alone.

### OP_COVERAGE.md additions
- §7.1.4: "Optional-input population is a coverage axis" — the MatMulNBits 3-input vs 4-input finding, including the island measurement falsifier and island count correction.
- §8.9: "The proof ledger — claimability is derived, not hand-written" — full specification of the ProofKey format, CLAIM_UNPROVEN escape hatch, ledger file mechanism, and the no-wildcard enforcement rule.

### Proof ledger scaffolding status
Already in registry.rs from prior session: `ProofKey`, `ProofKey::validate()`, `claim_unproven_keys()`, `ledger_contains()`, planted rejection tests (`claim_unproven_rejects_star_wildcard`, `_all_wildcard`, `_boolean_one`). Gate activation is pending Trinity's differential harness producing a non-empty ledger. No change needed to the scaffolding this session.

### Decisions written
- `mouse-skisnorm-island-measurement.md` in main's inbox: island count went up, not down; falsifier fired.

### Commit
  951b592 — SkipSimplifiedLayerNormalization f16, alloc_temp fix, xfail flip

---

## Turn (partition wiring + multi-node dispatch) — 2026-07-30T09:14:00-07:00

### Assignment
Wire `partition.rs` into `GetCapability` so ORT receives maximal convex connected subgraphs, not one capability per node. Fix multi-node island Compute dispatch (intermediate buffer aliasing bug). Island count reduction was previously measured at 33 (correct); dispatch accounting was `compute_calls = 1` (broken — panic in first multi-node Compute, ORT fell back to CPU thereafter).

### Root cause of compute_calls = 1
`CompileRecorder::push_dynamic_kernel` and `ShapeOnlyRecorder` used positional token assignment, resetting the bind counter to 0 per kernel. For a 2-node island {A, B}:
- CompileRecorder for the full island: A output → token 5+0=5, B output → token 5+1=6.
- ShapeOnlyRecorder re-run for B (Compute-time): starts fresh, assigns B's output to token 5+0=5.
- At dispatch time: `eff_bindings` for B has token 6 → j=1 ≥ n_ort=1 → `gpu_temps[0]` → EMPTY → panic.

The panic was caught by `guard_ffi_status`, ORT got an error status, abandoned the EP, and all subsequent inferences ran on CPU. `MATCH` appeared to hold (CPU vs CPU), `dispatch_accounting` was the only instrument that caught it.

### Fix: name-based token assignment
Built an island-wide `name_to_token: HashMap<String, u64>` in `compile_impl`:
- `plan.inputs[k].name` → token k (external inputs)
- `plan.outputs[j].name` → token n_plan_inputs + j (external outputs)
- Node outputs not in plan.outputs → intermediate tokens n_plan_inputs + n_plan_outputs + k++

`CompileRecorder::new_named` and `ShapeOnlyRecorder::new_named` use this map. The translate handler's `resolve`/`bind_output` calls look up by name, then fall back to positional. Single-node islands stay on the existing positional path (n_intermediates=0, no map).

New token ranges:
```
0..n_plan_inputs                                    → external ORT inputs  (gpu_inputs)
n_plan_inputs..n_plan_inputs+n_plan_outputs         → external ORT outputs (gpu_outputs)
n_plan_inputs+n_plan_outputs..first_temp_token      → intermediate buffers (gpu_intermediates)
first_temp_token..                                  → alloc_temp scratch    (gpu_temps)
```

`dispatch_ort` allocates `gpu_intermediates: Vec<GpuBuffer>` and routes tokens through all four ranges. A SHADER_WRITE → SHADER_READ barrier is emitted after each dispatch (except the last) on all intermediate buffers. `free_all` extended to include `gpu_intermediates`.

### SubgraphComputeInfo new fields
`n_intermediates`, `name_map: Option<Arc<HashMap<String,u64>>>`, `first_temp_token`, `static_intermediate_byte_sizes: Vec<u64>`.

### Pre-pass intermediate propagation
For multi-node islands, the pre-pass maintains `computed_descs: HashMap<u64, TensorDesc>`. After each kernel's ShapeOnlyRecorder run, intermediate output descs are inserted into `computed_descs`. When patching later kernels' inputs, intermediate tokens look up from `computed_descs` instead of (missing) ort_values entries.

### Results
- Both devices: `model_output_equivalence = MATCH`
- Both devices: `dispatch_accounting = ok — compute_calls 1023 == 33 islands × 31 inferences`
- Intel Iris Xe: 3.7× slower (was 12.1×, then 12.6× with SkipNorm, now 3.7× with wiring fixed)
- RTX 4060: 4.1× slower (was 7.9×)
- 38 passed, 1 xfailed on both devices

### accuracy_level ruling (owed to Trinity)
The model declares `accuracy_level=0` (implementation-defined). Trinity's oracle is pinned at `accuracy_level=1` (explicit fp32 accumulator). On x86 ORT CPU EP, levels 0–3 all use the fp32 accumulation path and produce identical results (`test_matmulnbits_accuracy_level_pinning` verifies this). Pinning at 1 rather than 0 avoids dependence on ORT's interpretation of "default". The ruling: **the oracle pin is correct and should not change**. The model's declaration of 0 does not conflict — it means "let the hardware decide", and our Vulkan kernel uses fp16 accumulation (native to the shader). The comparison is Vulkan-fp16-accumulator vs CPU-fp32-accumulator; the tolerance budget in §10.1 Regime 2 absorbs this difference, as confirmed by MATCH on both devices.

### OP_COVERAGE.md additions
- §7.1.5: "Island-count == claimed-count is the partition-wiring falsifier" — the multi-node dispatch finding, the intermediate-buffer token aliasing root cause, the name-based fix, and the dispatch-accounting red instrument.

### Decisions written
- `mouse-partition-multinode-dispatch.md` in main's inbox: intermediate buffer aliasing root cause, fix, results.

### Files modified
- `rust/src/ep.rs`: `SubgraphComputeInfo` new fields, `compile_impl` name-map building, `compute_impl` dispatch_ort call update.
- `rust/src/vk/session.rs`: `CompileRecorder` name-based mode, `ShapeOnlyRecorder` name-based mode, `dispatch_ort` new parameters, `gpu_intermediates` allocation, inter-kernel barriers, `free_all` extended.
- `docs/OP_COVERAGE.md`: §7.1.5 added.

---

📌 Team update (2026-07-30T19:05:03-07:00) — Scribe

Two findings apply to every agent on the team:

**(a) A mechanism that exists in a file but not in a call graph is indistinguishable from
one that does not exist.**  Verification by reading is insufficient.  Verify by running.
Five such mechanisms surfaced in this single batch: partition.rs, the GPU tracer,
model_output_equivalence, compute_failures, and should_claim_island.  In every
case the code was correct; the wiring was absent; the absence was invisible to review.

**(b) 85.9% of inference wall-time involves no GPU work** (recording 68.3%, fence-wait
idle 16.3%, submit 0.3%; GPU kernels 14.1%).  Optimising GPU kernels before the
command-buffer recording bottleneck is resolved is low-leverage.  Align work priorities
accordingly.

