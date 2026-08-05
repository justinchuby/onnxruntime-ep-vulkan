# `Reshape`: what actually happened

**Author:** Mouse. **Run:** 2026-08-04, worktree `ep-vulkan-mouse`, branch `squad/mouse`,
NVIDIA RTX 4060 Laptop GPU (device 0), debug build, Windows.
**Prediction under test:** `bench/results/reshape-prediction.md`, committed `a842d57` *before*
any build containing a `Reshape` row existed.

---

## 1. Scoreboard, prediction against actual

| # | predicted | actual | verdict |
|---|---|---|---|
| 1 | cost is **a copy**, not an alias, not a spanned no-op | a copy, `ew_cast_f32_to_f32`, no new shader | **hit** |
| 2 | the op-type counterfactual is wrong by ~2 orders of magnitude | it is: `+167` predicted, `0` claimed | **hit** |
| 3 | I will claim **3** `Reshape` nodes on BERT | **0** | **miss** |
| 4 | `dispatches_executed` on BERT stays **4** | **4** | hit, but for the wrong reason |
| 5 | `claimed_nodes` 481 → **484** | 481 → **481** | **miss** |
| 6 | Phi-3.5 cannot move (zero `Reshape` nodes) | 355/355, unchanged | hit |

**The direction of the miss is the finding.** I predicted the op-type instrument would
over-predict and it did; I then under-corrected by exactly the amount of the two rules I had not
looked up. Both of the ways I was wrong are things the claim log already knew and I did not read
carefully enough:

* The 3 nodes I named as claimable are the token-id reshapes and they are **i64**, not f32. They
  decline `[dtype]`. I read `output_shapes` out of the claim log and did not read `input_dtypes`
  in the same pass, which is the same class of error as reading `shape_class` without reading
  what produced it.
* Two *other* nodes, which I had predicted would decline, passed my first gate — and passed it
  **only because I had reproduced, in brand-new code, the vacuous rank-0 defect I named last
  round.** `[]` -> `[-1, 768]` was claimed on the strength of having read nothing. See §3.

## 2. All four models: `claimed_nodes` and `dispatches_executed`

`dispatches_executed` is now the headline in `probe_model_op_census.py`. When no inference was
run the tool prints `not-measured`, never `0`.

| model | graph nodes | `claimed_nodes` | `dispatches_executed` | islands offered / retained | honest number |
|---|---|---|---|---|---|
| BERT-SQuAD-12 | 1167 | **481** | **4** | 4 / 4 | `dispatches_executed` |
| MobileNetV2-12 | 105 | **98** | **97** | 1 / 1 | either; they agree |
| Phi-3.5-mini-int4 | 366 | **355** | **355** | 1 / 1 | either; they agree |
| gpt-oss-20b | — | — | — | — | **not measurable on this box** |

**Where they disagree, and which to believe.**

* **BERT is the pathological case and the reason the metric changed.** 481 claimed, 4 executed —
  a ratio of 120:1. The claims are real claims; they are singletons and pairs stranded between
  unregistered ops, and `net_benefit_gate_clusters_seen 146` is the partitioner discarding them.
  Quoting 481 for BERT is quoting a number that describes the registry, not the run.
* **MobileNetV2 and Phi-3.5 agree**, and that is why the defect went unnoticed: on the two models
  that form one island, `claimed_nodes` *is* the executed count. 98 vs 97 on MobileNetV2 is one
  node claimed and then dropped, not a fragmentation story.
* **gpt-oss-20b cannot be censused on this machine at all.** ORT 1.28.0's CPU `QMoE` kernel
  refuses at session creation: `activation_type_ != ActivationType::SwiGLU || swiglu_fusion_ == 1
  was false`. No session, so no claim log and no counters. The census tool reports
  `ERROR(instrument)` and exits non-zero rather than emitting a row of zeros — which is the same
  repair as the `not-measured` rule, applied one level up. **This model is unmeasured by me this
  round and nothing below should be read as covering it.**

## 3. `Reshape` on BERT: 59 offered, 0 claimed, and the naming histogram

    Reshape   59 seen   0 claimed   59 declined
              {'unknown-rank': 53, 'dtype': 4, 'shape': 2}

* **53 `unknown-rank`.** ORT resolved no rank for the output. 48 of those have no input rank
  either. There is no `TensorDesc` to bind, no size to allocate, and no runtime-extent handling
  that recovers a rank that was never inferred.
* **4 `dtype`.** i64 token-id reshapes. The row is f32; widening it would add a proof key with no
  model behind it.
* **2 `shape`.** `bert/encoder/Reshape_1` (`[]` -> `[-1, 768]`) and
  `.../layer_0/attention/self/ExpandDims` (`[]` -> `[-1, 1, 256, 256]`).

**Those last two are the ones worth reading.** The first version of my predicate *claimed* them.
It claimed them because `EdgeType::is_static()` returns `true` over an empty dim list — "all dims
are non-negative" is vacuously true of no dims — so `claim::check_shape` passes a rank-0 edge,
and ORT emits rank 0 both for a genuine scalar and for a rank it never established. That is the
exact defect I named last round for `MatMul`, reproduced by me, in new code, one round later.

`MatMul` could settle it from the ONNX schema, which admits no rank-0 operand. **`Reshape`
cannot** — a scalar reshaped to `[1]` is legal ONNX. So it is settled by arithmetic instead:
**a rank-0 input holds exactly one element**, and `[-1, 768]` needs a multiple of 768. A genuine
scalar still passes, because its output really does hold one element. This is narrower *and*
stronger than the per-op minimum-rank table I declined to build last round, and it is unit-tested
in both directions (`a_rank_zero_input_is_one_element_and_cannot_become_a_768_wide_tensor`,
`a_genuine_scalar_still_passes_because_its_output_really_does_hold_one_element`).

## 4. Instruction #4: `Reshape` is structurally blocked, and here is where

Two independent blocks, both established by measurement rather than argument:

**(a) At the gate — no inferred output rank.** 58 of BERT's 71 graph `Reshape` nodes take their
shape operand from a runtime `Cast`/`Concat`/`Shape` chain. ORT does not constant-fold it, so it
infers no output shape. The escape hatch named after this exact op —
`DispatchContext::read_const_i64`, documented at `engine.rs:519-522` as being for "`Reshape`'s
shape input" — **returns `None` in all four implementations in the tree**. The seam exists and
has never been implemented, and implementing it would still answer `None` for 58 of 71, because
the operand genuinely is not constant.

**(b) At translate — no output descriptor for a free axis.** This one I did not predict and it
cost a ledger case. I built `reshape_f32_dyn` with input `[N,3,4]` and target `[-1, 4]`.
The node claimed. Then `dispatch_ort`'s dynamic re-run of translate refused:

    Unsupported("`Reshape` output has no declared shape; the reshape target is unknown")

**ORT hands the translate handler no output `TensorDesc` at all when it could not resolve the
output's extents.** The other route to the target — the shape operand's *value* — is (a). So for
a free axis the target is unknown at the one point it is needed, and the failure is not a
decline: it is a claimed island whose `Compute()` returns non-OK. A **broken commitment**, the
same failure as the 323-node Phi-3.5 round.

The gate now refuses a free axis outright, and the `reshape_f32_dyn` ledger case was rebuilt as
a symbolic *input* extent with a fully-resolved target — the only runtime-extent form the row
claims. This is the second time this round that building the falsifying case was worth more than
the reasoning that preceded it.

**Consequence:** `Reshape` is a `Ready` row that claims nothing on any model in the census. It is
not dead code — it is proven, dispatched and numerically exact on both ledger cases (§5) — but it
is waiting on ORT shape inference, not on a kernel. **`Concat` is the next pick and it is not
better** (§6).

## 5. The positive control: the row does claim, dispatch, and agree

Two ledger cases minted, both `MATCH`, both `worst_rel 0.0`:

    ai.onnx::Reshape/5+/f32,i64>f32/ew_cast_f32_to_f32/static/n2
        claimed_nodes 1, dispatches_executed 1, shader ew_cast_f32_to_f32
    ai.onnx::Reshape/5+/f32,i64>f32/ew_cast_f32_to_f32/runtime-extent/n2
        claimed_nodes 1, dispatches_executed 1, shader ew_cast_f32_to_f32

Without these the row would be indistinguishable from a row that cannot claim at all. A histogram
of honest declines and a proof that the claim path works are two different facts and the round
needs both.

## 6. The counterfactual instrument, repaired — and `Concat` is not the next win

`probe_island_counterfactual.py` ranked **op types**; the EP claims **nodes**. It now reports a
bracket. The baseline is per-node too (the old one treated all 364 BERT `Add` nodes as claimable
when 182 were).

BERT, against `_claim_log_bert_r16.jsonl` (claim log covers 794/1167 graph nodes by name, 68.0%,
and the tool prints that rather than hiding it):

| op | in graph | rank-resolved | optimistic Δ | gated Δ |
|---|---|---|---|---|
| `Reshape` | 59 | 3 | **+78** | **+3** |
| `MatMul` | 95 | 1 | **+128** | **+0** |
| `Concat` | 9 | 9 | +0 | +0 |
| `Transpose` | 49 | 0 | +0 | +0 |
| `ReduceMean` | 50 | 0 | +50 | +0 |
| `Softmax` | 12 | 0 | +12 | +0 |

Cumulative, `MatMul`+`Reshape`: **optimistic +184 in 27 islands, gated +3 in 30 islands.**

Last round's headline — *"`MatMul`+`Reshape` = 738 nodes in 17 islands"* — was the optimistic
column of an instrument with an optimistic baseline. The gated column would have said **+3**, and
`MatMul` shipped and delivered **+0 / 1 node**, and `Reshape` shipped and delivered **0**. The
gated column is the one that has been right twice.

**On instruction #4's suggestion that `Concat` is next:** the repaired instrument says `Concat` is
worth **+0 gated and +0 optimistic** on BERT — all 9 of its nodes are rank-resolved, and it still
moves nothing, because they do not adjoin claimed nodes. Pairs do not help: the largest gated
cumulative reading over the top ten candidates is **+3**, and it is reached by `ConstantOfShape`
and `OneHot`, one node each.

**There is no next op-registration pick on BERT.** The gated ceiling over every unregistered op
in that graph is three nodes. What is blocking BERT is that ORT infers no ranks through its
`Shape`/`Cast`/`Concat` chain, so 724 of 1274 claim rows read `static` having read nothing. That
is one problem with three symptoms — 94 `MatMul` declines, 53 `Reshape` declines, and the
`check_shape` `[unknown-rank]` branch being dead code on this whole class of model — and the next
round on BERT should be that, not another kernel.

The instrument does not overstate itself: `gated` models exactly one precondition (every operand
and the output has a rank). dtype, attribute and per-op rules are not modelled, and it prints
that under the table. On `Reshape` it said +3 and the answer was 0 — the 3 were i64, which is a
dtype rule, which it declares it does not model.

## 7. What this verification established, and what it did not

**Established, on this box** (RTX 4060 Laptop GPU, device 0, Windows, debug build, ORT 1.28.0):

* `cargo test --lib` — **607 passed / 0 failed / 4 ignored** (590 on `main`).
* `cargo clippy --all-targets` — clean.
* `cargo fmt --check` — **0 diffs.** It was **red when I first ran it, in my own new file**, on
  two hunks. Run before you push; it is a real CI gate.
* `gen_proof_ledger.py --check` — reading the **subject arithmetic** line, not the PASS line:
  `133 entries = 133 identical + 0 SOURCE-COSMETIC + 0 PROVEN-ELSEWHERE + 0 SUBJECT-CHANGED +
  0 SUBJECT-INDETERMINATE + 0 no-module-in-build`; loss invariant `176 ever MATCHed, 133 in
  ledger, 0 missing, 43 retired`; mintability `133/133 mintable, 43/43 retired mintable`.
  Baseline was 131/131.
* `counters_abi.py --check` — `PASS(the derived mirror is exactly the size and shape the DLL
  publishes)`.
* `pytest tests/ops` — see §8.
* `probe_model_output_agreement.py` on BERT — `AGREE: 0 of 3 outputs disagree`
  (max_rel 1.1e-06 on both float outputs, `EXACT` on the i64 output).
* Census with counters on BERT, MobileNetV2 and Phi-3.5, each with `--run`.

**NOT established:**

* **gpt-oss-20b is entirely unmeasured.** The session cannot be created on this ORT build. I did
  not work around it and I am not reporting a number for it.
* **No f16 `Reshape` anywhere.** The row is F32-capped, deliberately, and no f16 graph was tried.
* **No second device.** Everything above is device 0. The Intel Iris Xe (device 1) was not run,
  so nothing here says anything about a UMA allocator or a different driver's `ew_cast` codegen.
* **No release build.** Debug only.
* **No CI run.** Link owns CI and it has been red on `main` for a dozen pushes; I ran these gates
  locally, on this worktree, and a green local gate is not a green CI.
* **`DEVICE_MEMORY` and `KV_ARENA` were not enabled**, per the coordinator's instruction — so
  nothing here bears on Switch's `ctx-4096` `ep_inter_76` allocation failure, and the Phi-3.5
  numbers above are from the default configuration.
* **The `allowzero=1` form of `Reshape` is declined, not handled.** ORT applies it during shape
  inference so the declared output would already be correct, but no graph exists to test it on
  and an attribute claimed-but-untested is the `Gemm` transpose mistake.
* **The 68% claim-log name-match rate in §6 is a real limit on that table.** ORT's graph
  transformers rename and fuse before `GetCapability`; 373 graph nodes have no row. They are
  counted as not-claimable, which biases every delta *downward*. The tool prints the rate.

## 8. `pytest tests/ops`

Full-suite result is recorded in the history entry. One case changed by design:
`test_op_table.py::test_op_table[Reshape-fp32-static]` was an asserted **decline** with a written
ruling — *"`Reshape` and `Flatten` perform no arithmetic... the Phi-3.5 claim log contains zero
`Reshape` and zero `Flatten` nodes"* — and a named falsifier: *"If a graph ever asks, the
argument reverses and these two rows flip back."*

**The falsifier fired.** BERT has 71 `Reshape` nodes, 59 offered. The row is now `claim=True` and
passes against the CPU oracle. `Flatten` stays declined and the distinction is now measured
rather than inherited: BERT 71 `Reshape` / **0** `Flatten`, MobileNetV2 1 / 0, Phi-3.5 0 / 0.
