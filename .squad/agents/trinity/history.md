# Trinity (Test-Conformance) — history.md

## Learnings
<!-- CONDENSED-AT: 741a0c06dd42d59d8b6053a73bc34114c67289ea -->

### [SUMMARY] Rounds 27-36 (2026-08-02): union_check tiering, criterion 11(c)/11 wiring, ULP residual unit, criterion 3(a), fatal-log liveness, criterion 10 route/attribution, model-scale seam

- **Round 27, union defect:** a merge collision (`guard` kwarg vs a caller added on Mouse's branch) resolved as `guard=None` meaning "build one," not "run unguarded." `union_check.py`'s tiers 1/2/2b are preconditions only; tier 3 (real merge + rebuild) is the only gate. Near-miss: a stale DLL after merging almost read as "ledger UNWIRED" -- rebuild-after-merge became a standing obligation (root of Morpheus's later R12 gen-4 "frame is the binary that ran it").
- **Round 28, criterion 11(c):** `ledger_hits` moves on `shape_class` (static vs dynamic-extent), not on `MatMulNBits` scales/zero_points pairing (both HIT). `model_output_equivalence: MATCH` beside `UNMEASURED` was a field nobody set -- keys off record *presence*, not token value. `union_check` FAIL(union_red) on the merged tree attributed to Mouse's own 9-entry ledger being red on unmodified main.
- Team update (2026-08-02T02:03:46, Scribe/Morpheus): R12 gen-4 traces partly to Trinity's own round-27 stale-DLL near-miss.
- (Prior compressed block, 2026-08-02): Round 29 criterion 12 (12 uncensused surfaces), Round 30 per-output attribution, Round 31 ORT's `disable_cpu_ep_fallback` flag exists, Round 32 merge/rebuild discipline reapplied, second Round 29 (two unfalsified guards screened into the always-on lane).
- **Round 33, ULP residual unit:** built `ulp_residual()`/`ulp_outliers()`; headline moved `max_ulp_diff` -> `median_ulp_diff` (ULP can blow up near zero from cancellation, non-monotonicity needs a cancellation element). KV cache flat 0-3 ULP over 62/64 outputs (confirms Morpheus's prediction) but **output 0 (logits) reads median 12 ULP, 12x baseline, both vendors** -- a located, vendor-independent defect in the head, not the layers.
- **Round 31, criterion 3(a):** discharged both devices; a bare `device_losses: 0` is `UNOBSERVABLE` unless the messenger is proven live -- used `BEST_PRACTICES_EXT` messages as proof (14 armed vs 0 clean). Device selector is a request not an identity. Corrected the brief: `ledger_gate` is MIXED not ALL-PROVEN; `dispatches_executed` is per-claimed-node (355) not per-dispatch (8875).
- **Round 34, fatal-log liveness:** `_verdict.FATAL_LOG_MARKERS` never matched ORT's real fallback line -- five prior "second witness" corroborations were the test's own docstring echoed into the log, never ORT (message wraps across lines; ORT's sink writes UTF-16LE into a UTF-8 file, unfindable after UTF-8 decode). Fix: `assert_fatal_log_check_is_live()` requires a witness prove it can go red before any verdict is trusted.
- **Round 35, criterion 10 route:** premise false -- `BIND_OUTPUTS` ships OFF, a second path beside the criterion; forced on, all 65 residuals identical both vendors. Nearly published a fictitious "peaks at layer 9" caused by reading names out of a sorted JSON file instead of session order -- fixed with `assert_names_are_session_order()`. Constructed the exact historical-defect shape as a named reachable falsifier.
- **Round 36, logits step inherited:** no smooth climb (flat ~2 ULP, dips at layer 29); moves only in the last two hops (L31 3 -> RMSNorm 6 -> logits 12). First float64 reference: at RMSNorm Vulkan is bit-exact, CPU EP is the 1-ULP side. Nearly shipped a false defect: arm F compared a float64 reference from the CPU EP's own taps against Vulkan's in-situ output, "vulkan" by construction -- isolation means identical inputs on both sides or it means nothing. The "intermittent" failure was three real races, found because a 21-test pool was more sensitive than the 505-test pool.
- Team update (2026-08-03T04-55, Link): a DLL hash proves nothing about content.
- Team update (2026-08-03T10-35, Link): an accepted red and a new red are indistinguishable when the acceptance record lives only in someone's head.

## Round 37 — 2026-08-04 — which side is wrong, and my own prediction written down as a reading

**§8.9.24 refuted my unsatisfiability finding and needed no run to do it.** I quoted `atol`
alone out of `|a−b| ≤ atol + rtol·|b|` — one term of a two-term sum — and divided by the
spacing at the **tensor maximum** while the predicate evaluates **per element**. The
corollary inverts my reading of layer 31: its key and value do not fail within one
representable step, they fail by **640–1266 element-basis ULP** against an allowance of
**58–297**. What made them look sub-step was a step borrowed from a value ~500× larger.
`ULP-at-scale` is fenced, not withdrawn, and I own the fence: any row reporting it now
carries the allowance in the same unit and the failing set on the element basis, checked by
a function that **raises**.

**What surprised me most: `np.spacing` returns `inf` at fp16's largest finite value.** It
looks upward and above 65504 there is nothing but `inf`, so `|a−b|/inf == 0` and **a
504-unit error read `0.0` ULP on both bases**. Every prior instrument defect here made a
*sound* residual look wrong. **This is the first one that makes a wrong residual look
sound** — strictly worse, because nobody re-derives a clean number. It moved no verdict
(`np.allclose` reads no ULP), but the report could have acquitted a saturating tensor while
the gate failed it, on the same row. `format_spacing` takes the step downward at the
boundary. numpy's answer was outside the format; the algebra was fine.

**I was the specimen for A PREDICTION IS NOT A READING, in the same round it was minted.**
I reasoned: position 0 → rotation by angle 0 → `present.31.key` is the K slice verbatim.
Sound reasoning, false conclusion — Phi-3.5 folds a long-rope factor into the cache and
`cos_cache[0]` is **1.1904296875**. Because the sine is zero it degenerates to a single
multiply, still one correctly-rounded operation, so the reference stayed envelope-free and
**the conclusion that mattered survived; the stated reason for it did not.** With a nonzero
sine the same assumption would have reported a rotation defect as a copy defect. The form is
now classified from the tapped caches and `reference_is_exact` is a *consequence* of it.

**The answer, identical bytes to both EPs, float64 reference from those same bytes, both
devices agreeing.** Outputs 63/64: **both EPs bit-exact** — neither side is wrong, and arm E
reconstructs each EP's own in-situ output from its own QKV tap **bit-equally**, so the
divergence is inherited as an *identity*, not a story. Criterion 10's two tail failures
contain **no arithmetic of their own**. Output 0 at the isolated `lm_head`: **ORT's CPU EP is
the further side**, unanimously (2 vs 11 ULP max, 11 vs 47 elements, 0.00024 vs 0.0039 abs).
Second node where "the Vulkan EP is wrong until proven otherwise" runs backwards.

**Round 36 said `neither (equal)` for that same node and I nearly published a changed
result.** Round 36 discriminated on the *median*, which is 0.0 on both sides — 32053/32064
bit-exact — so it could not have separated them whatever the answer was. R11's shape again.
The record now reports five discriminators at once. **Building that machinery immediately
caught a conflict in my own current result**: the `qkv_proj` arm splits — CPU wrong in 18
elements vs Vulkan's 2, but Vulkan's worst element 32× further in absolute terms. There is
no fact of the matter without a further choice, so `unanimous_direction` is null and the
**split is the finding**. The single-discriminator version reported `neither (equal)` and hid
it entirely.

**What I did NOT establish:** which side is wrong for these outputs **as criterion 10 sees
them**. Every answer is per-hop with identical inputs; at model scale the EPs reach the
`lm_head` with hidden states 6 ULP apart, and a float64 reference built from either is my own
discarded arm F — `"vulkan"` by construction. It needs a float64 forward pass of all 355
nodes (~30 GB dense, infeasible here; ~0.7 GB layer-at-a-time, feasible) with a liveness bar
at **every** layer, since a reference that dies mid-graph agrees with everything downstream.
Written into the record, not just this report, so the limit travels with the answer. I
proposed nothing that follows — §8.9.24(4), enforced by `assert_record_proposes_no_motion`
running as a gate, with a test proving it can still refuse.

**`atol`/`rtol` untouched for the fourth round running.** Arm D's "ORT is 11 ULP from true"
is precisely the budget a loosening argument would want. It is a reading and nothing else.

**Verified:** 152 passed / 0 FAIL / 0 ERROR across the criterion-10 and ULP lanes; probe
selftests 11 and 5 arms with no GPU; **9 deliberate corruptions of the records, 9 caught** by
the new gate — which also went red unprompted on three schema mismatches and on my own wrong
assumption that fp32's tolerances match fp16's (they are 10× tighter). `union_check --run` 5
red, none mine: criterion 10's known `DIVERGENT`, one unrelated binding test, and two lanes
that pass in isolation and are red only under the full run. **Not run: `cargo test --lib`,
clippy — no Rust moved, and I say so rather than implying coverage I do not have.**

## Round 38 — 2026-08-04 — the model-scale answer, and an instrument that called a healthy build wedged

**§8.9.24(4)'s remaining half is built and measured.** A float64 forward pass of all 355
nodes, layer-at-a-time (~0.7 GB, not the ~30 GB dense pass), 32/32 layers live, both
devices, both reference variants. **Output 0 `logits`: `cpu` — ORT's CPU EP is the further
side, unanimous on all five separating discriminators, under both variants, on both
devices.** Outputs 63 and 64: **`null` — no direction**. The discriminators conflict inside
a variant and the variants disagree with each other, and that is the reading, not a gap in
it. Round 37 established those two outputs carry **no locally-made error**; at model scale
they still do not acquire a side.

**What surprised me: at model scale both EPs are ~6× further from true than from each
other.** Logits: Vulkan 70 and CPU 83 element-basis ULP from true, **12 apart**. Their error
is largely **common**, not opposed — which is a fact about the composed graph and about
fp16, and is recorded as `how_far_both_sides_are_from_true_vs_from_each_other` with an
explicit note that **it is not a statement about any threshold.** It is exactly the number a
loosening argument would want, which is why it is a reading and nothing else.

**The seam is where arm F comes back, so it is structural.** The chain reads *initialisers
and `input_ids` only*; layer L's input is layer L−1's **reference** output, never an EP tap,
so neither EP appears anywhere in its derivation. `assert_chain_never_reseeded` digests
every boundary and raises if any chain state equals an EP tap there. The **liveness bar** is
re-seeded per layer per side from that side's own tap — and its result **never reaches the
chain**. Re-seeding is what made arm F dishonest *when its output fed a verdict*; it is what
makes a liveness bar mean anything. The distinction is which structure the number is allowed
to reach, and it is enforced by data flow rather than by a comment.

**Reading the first real run exposed a defect in my own aggregation — the R11 shape one
level up, inside the code written to prevent it.** A variant whose discriminators *conflict*
was being dropped so the *decisive* variant could speak for both, and output 64 printed
`direction='cpu'` with `variants_agree_on_direction=True`. Dropping a `None` is
single-discriminator reasoning wearing the multi-discriminator machinery.
`direction_across_variants` now refuses it.

**`union_check`'s `wiring_census` red was never contention, and I ran the experiment instead
of arguing.** It reproduces **in isolation with nothing else running**. A cold `cargo test
--test layering` prints `Compiling onnxruntime-ep-vulkan` and then **nothing at all** while
rustc works: 12015 silent units against a 12000 budget. The guard's own stated premise —
*cargo emits a line per crate, so a crawling build keeps beating* — is true of many crates
and false of one big one. **`test_proof_ledger` is green under the full run**, so the
"contention pair" was one instrument defect and one coincidence.

**The repair is a second beat source, never a bigger number** — the file says the number is
gone rather than bigger, and a bigger budget makes a wedged child *harder* to catch.
`guarded_run` now watches what a `KIND_TOOLCHAIN` child has **written**, consulted only
after half the budget of silence. **My first version of the fix was wrong and the run said
so**: depth 1 beat 4098 times and the census **still stalled at 12011 units**. Sampling a
cold build directly: longest blind interval **195.5 s at depth 1, 18.6 s at depth 3** —
rustc touches `incremental/<hash>/s-*-working/` and almost nothing above it. Depth is now a
measured constant.

**Cold before: ERROR(instrument), 649 s. Cold depth-1: ERROR(instrument), 775 s. Cold
depth-3: 7 passed, 957 s.** The green run is the **slowest of the three**. The failing
quantity was silence, not slowness — which is precisely the property the work clock was
built to have, defeated by a premise about stdout rather than by the clock. The refusal arm
keeps the witness **live** and still requires `Stalled` on a child that writes nothing; a
bigger budget could not pass that arm. A `KIND_MECHANISM` stall stays a **detection**.

**What my verification established and what it did not.** It established which side is
further from the true value at model scale, for the three outputs criterion 10 reads, with
every layer proven to contribute and the reference proven never to be seeded from either EP.
It did **not** establish that either EP is within any tolerance, that any tolerance should
move, or that outputs 63/64 have a side — they do not, on this evidence. It says nothing
about sequence positions beyond 0, nothing about batch or longer prompts, and nothing about
what follows: **Morpheus rules on that**, and `assert_record_proposes_no_motion` gates both
records so this one cannot smuggle a proposal into a reading.

**`atol`/`rtol` untouched for the fifth round running.**

**Verified:** chain probe selftest 10 arms; `test_criterion10_chain.py` 30 passed;
`test_stall_guard.py` 21 passed including the two new refusal arms; `test_wiring_census.py`
7 passed **cold**; contention gate **GREEN**, 0/20 red across all five pools (and it is a
**cargo/libtest** gate, so it is *not* the instrument for a pytest pair — I say that rather
than imply coverage I do not have). **Not run: `cargo test --lib` and clippy — no Rust
moved this round.**
📌 Team update (2026-08-04T12:25:00-07:00): Morpheus's ruling — the screening question for a
criterion motion is not "is this true?", it is "what does it admit?" (the unsatisfiability finding
was refuted without a run; the false premise was harmless, but relaxing the criterion on its
strength would have admitted two of three real failures with no element moving). This applies to any
motion you adjudicate against a criterion or a prior ruling. — decided by Morpheus
