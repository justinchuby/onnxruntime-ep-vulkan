# `Reshape`: what I predict, written before the run

**Author:** Mouse. **Written:** 2026-08-04, before any build containing a `Reshape` row exists.
**Falsifiable by:** `bench/results/op_census_bert_r16_after.json` and the counters beside it.

## 1. What `Reshape` costs, stated before the kernel is written

Three candidate costs were on the table. The answer is **a copy**, and the reasoning is
code-level rather than preference.

* **A no-op the partitioner spans** — not available. ORT's `GetCapability` contract is that a
  claimed node is *executed* by the fused kernel. There is no "claim it but don't run it" state;
  the island's node list is the dispatch list.

* **An alias** — not available *at this seam*, and this is the part I checked rather than
  assumed. `DispatchContext::bind_aliased_output` exists (`engine.rs:565`) and `vk/session.rs`
  implements it, but `dispatch_ort` only honours a recorded pair when
  **`out_tok` is an external plan output and `in_tok` is an external plan input**
  (`vk/session.rs:1282-1290`). Every `Reshape` that is worth claiming is an *interior* edge of an
  island — that is the entire reason it ranks — so its pair would be recorded and then ignored,
  the output buffer would be allocated and never written, and every downstream consumer would
  read uninitialised device memory. That is a silent wrong answer of the right shape, which is
  the one failure mode my charter puts above all others.

  Switch's disjointness argument for the KV arena does **not** transfer, and not because it is
  wrong. His argument is about a *shader* that reads `past[t]` and writes `present[tok_pos]` in
  the same dispatch, and it is discharged by `tok_pos = past_len + s_local >= past_len` under one
  common stride. A `Reshape` alias has no write at all, so that hazard class does not arise —
  but a different one does, which his argument never had to address: **two live tensors sharing
  one allocation with independent lifetimes**, against a generation-stamped quarantine-on-free
  allocator. Nothing in the engine tracks that for interior tensors today. Aliasing `Reshape` is
  an engine change (Switch/Tank), not a kernel change (mine), and I am not going to make it look
  like one by writing it in `ops/`.

* **A copy** — what ships. One full-tensor read plus one full-tensor write, dispatched through
  **`ew_cast_f32_to_f32`**, a module the build already produces. No new shader. This is the
  `MatMul`-over-`gemm_f32` precedent: a row-major buffer reinterpreted is the same bytes, so the
  op's identity belongs in the *claim* and the *proof key*, not in a second `.comp` that
  distinguishes nothing (§8.9.23, and `form.rs` was deleted for committing that error).

  The copy is not free and I am not going to call it free. What makes it worth paying is the
  thing it replaces: a `Reshape` left unclaimed **splits an island**, and the split costs a
  device→host download of the tensor, a CPU-EP execution, and a host→device upload of the
  result. The copy is one device-local pass over the same bytes. It is strictly cheaper than the
  boundary crossing it removes, and that — not FLOPs — is the whole argument, the same shape of
  argument as `Gather` in `ops/indexing.rs`.

## 2. The two predictions

### 2a. What the existing instrument predicts (op-type counterfactual)

`bench/results/island_counterfactual_bert.json`, measured last round:

| added | retained nodes | retained islands | delta |
|---|---|---|---|
| baseline (pre-`MatMul`) | 473 | 29 | — |
| `+Reshape` | 640 | 66 | +167 |
| `+MatMul,+Reshape` | **738** | **17** | +265 |

`MatMul` is now registered, so the live baseline should read 608 / 34 and the post-`Reshape`
reading should be **738 nodes in 17 retained islands**.

### 2b. What I predict, which is not that

**I predict the op-type counterfactual is wrong by two orders of magnitude, and that I will
claim 3 `Reshape` nodes on BERT, not 59.**

The counterfactual ranks *op types*. The EP claims *nodes*. Last round that gap turned "MatMul
×95" into one node, and I attributed it to the partitioner's net-benefit gate. That was only
half of it. The other half is visible in the claim log I already have, and I am reading it
before writing the kernel rather than after:

Of BERT-SQuAD-12's 59 `Reshape` claim rows (`_claim_log_bert_r15_after.jsonl`, `output_shapes`
as ORT reported them at `GetCapability`):

| input shapes | output shape | nodes |
|---|---|---|
| `[]` | `[]` | 48 |
| `[-1,768]` / `[-1,256]` | `[]` | 6 |
| `[]` | resolved | 2 |
| resolved | resolved | **3** |

**53 of 59 have no output rank at all**, and 48 have no input rank either. A `Reshape` whose
output rank is unknown cannot be given an output `TensorDesc`, cannot be sized, and cannot be
handed to a downstream consumer. Two of the remaining six have a resolved output but an
unresolved input, so the element count that would fix the free dimension is unknown.

That leaves **three**: `[-1,256]→[-1]`, `[-1,256]→[-1,256,1]`, `[-1,256,1]→[-1]`.

So:

* `claimed_nodes` on BERT: **481 → 484**.
* `dispatches_executed` on BERT: **4 → 4**. All three sit in the embedding preamble, which is
  not the region the four executed nodes are in, and three nodes is below the partitioner's
  net-benefit floor unless they attach to an existing retained island. If I am wrong here I
  expect 5–7, not 40.
* MobileNetV2 (1 `Reshape`), Phi-3.5 (**0** `Reshape` nodes in the whole graph), gpt-oss-20b:
  no change from `Reshape`. Phi-3.5 in particular cannot be moved by this op at all, and saying
  so now removes the temptation to read any Phi-3.5 movement as caused by it.

### 2c. Why the shape input is not the way out

58 of BERT's 71 graph `Reshape` nodes take their target shape from a runtime `Cast` output, not
an initializer. `DispatchContext::read_const_i64` is documented in `engine.rs:519-522` for
exactly this — *"for ops whose shape depends on a constant input (`Reshape`'s shape input,
`Slice`'s starts/ends)"* — and **every one of the four implementations in the tree returns
`None`**: `vk/session.rs:317`, `vk/session.rs:502`, and two test stubs. The seam named after
this op has never been implemented, and even implemented it would answer `None` for 58 of 71
because the operand genuinely is not constant.

This is the fourth instance of the family: **a mechanism that is clean because nothing looks
through it.**

## 3. What I will report

`claimed_nodes` **and** `dispatches_executed`, on all four models, and which of the two is the
honest one for each. `probe_model_op_census.py` currently headlines `claimed_nodes` and does not
read the counters at all; that is the metric everybody quotes and nobody executes, and it is
getting fixed in the same change.
