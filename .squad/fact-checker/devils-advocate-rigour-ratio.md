# Devil's Advocate — is the measurement discipline worth its cost?

**Owner:** Fact Checker, Mode 2 (Devil's Advocate). **Advisory. Never a veto.**
**Written:** 2026-08-03, against `main` = `8ac1172`.
**Asked for by:** the coordinator, on the grounds that this project measures everything except this.

He asked for the strongest version, not a reassuring one. What follows is the strongest version I
can build, and I have tried to make it hurt. The rebuttal is in §4 and it is shorter than the
steelman, because it is weaker than the steelman on the axes the user named.

---

## §1 — The steelman

### 1.1 The ratio, measured rather than asserted

Counted at `8ac1172`, whole tree:

| category | lines | |
|---|---|---|
| GLSL compute kernels (12 `.comp` + 2 includes) | **2,287** | the thing that makes it fast |
| `rust/src/ops` | 10,625 | the thing that makes it correct |
| `rust/src/vk` | 9,906 | the thing that makes it run |
| **implementation subtotal** | **22,818** | |
| `rust/src/counters.rs` + `trace.rs` + `disclosure.rs` | 6,970 | in-EP instrumentation |
| `tests/` (57 files) | 21,586 | |
| `bench/` (80 files) | 27,373 | |
| `ci/` | 13,384 | |
| `docs/` (architecture of record + rulings) | 16,677 | |
| `.squad/` (governance) | 17,908 | |
| **apparatus subtotal** | **103,898** | |

**4.55 lines of apparatus per line of implementation.** Excluding `docs/` and `.squad/` entirely —
the most generous framing available to the defence — it is still **2.73 : 1**.

I will not use the 45:1 figure that falls out of comparing apparatus to GLSL alone. That is a
denominator error of exactly the kind this project catalogues, and using it would make the argument
easier and wrong. **4.55 : 1 is the honest number and it is damaging enough.**

### 1.2 The competitor comparison, from primary source

`gh api repos/ggml-org/llama.cpp/contents/ggml/src/ggml-vulkan`, read today:

| | this project | llama.cpp Vulkan backend |
|---|---|---|
| Vulkan shader sources | **12** | **164** files in `vulkan-shaders/` |
| backend dispatch source | `registry.rs` 5,159 lines | `ggml-vulkan.cpp` **997,161 bytes** |
| models proven end-to-end | **1** (Phi-3.5) | dozens of architectures |
| GPUs exercised | **2** (RTX 4060 Laptop, Iris Xe) | NVIDIA / AMD / Intel / Adreno / Mali / Apple, in the wild |
| platforms with a green op suite | **1** (Windows; Linux compiled today and its ledger faults on a `glslc` version difference) | 5+ |
| users | **0** | many |

✅ verified by API call. The op-parity narrative around llama.cpp is ⚠️ unverified secondary
reporting and I have not leaned on it.

### 1.3 The argument

> **A competitor shipping a Vulkan EP with half this rigour is not "less correct". They are
> *further ahead on every axis the user named*, and the gap is widening at the rate of one
> instrument per round.**
>
> The user's standing instruction is that **performance must be very high** and **compatibility
> outranks API elegance.** Neither is served by:
>
> - a proof ledger with **103 entries** whose current state is `SPEC-UNRECORDED`, on one model;
> - `PROVEN-ELSEWHERE`, a state invented to describe evidence from a **second GPU on the same desk**;
> - a `contention_gate.py` whose detection power at the operating point is **0.397** and which
>   exists to measure the reliability of tests of a machine that is permanently contended;
> - eight unnumbered obligations, thirteen numbered rules, a decline tally, an audit of the decline
>   tally, and now **this file, which is an instrument measuring the instruments**.
>
> **Compatibility is a coverage problem and coverage is a headcount problem.** 71 kernel-carrying op
> rows against ONNX's opset-26 surface is not compatibility; it is one model's node histogram. Every
> hour spent proving that a `1.000000` is a measurement rather than an identity is an hour not spent
> on `Conv`, `Resize`, `MoE`, or the second model that would have found ten real defects by
> arriving.
>
> **And the discipline has a self-referential tell.** Round 10's decision record contains five
> entries. **Four are about instruments** — an ABI mirror, a kernel-identity witness, a provenance
> class, a decline audit. **One is about the EP doing something new for a user.** The most cited
> rules in the codebase, `R13` (419) and `R12` (256), are both rules *about instruments*. When the
> highest-traffic artifacts in a compute project are its rules about measurement, the measurement
> has become the product.
>
> **The strongest form of the charge:** this project can tell you, with unusual confidence, that a
> `1.000000` is real. It cannot run a second model. **It has achieved a very high signal-to-noise
> ratio on a very small signal**, and confidence about a small thing is not a moat — it is a moat
> around a small thing.

---

## §2 — Load-bearing assumptions the team is treating as fixed

These are choices being carried as constraints. Naming them is the point of this mode.

1. **That the proof ledger must be per-form-per-device.** This produced `PROVEN-ELSEWHERE`, the
   toolchain-key crisis, the two-digest schema, and the Linux blocker — a large fraction of the
   session's total governance output. **Alternative never costed: a per-op differential suite run at
   bring-up on each new device, with no persistent ledger at all.** llama.cpp ships to six vendors
   with no artifact resembling this.
2. **That one model is a sufficient conformance surface.** Phi-3.5's `Nq/Nkv = 1.00` has already
   been identified *by the team* as hiding a GQA defect that Llama-3 would expose. **The team knows
   its oracle is degenerate on the axis that matters and has added instruments rather than a model.**
3. **That the contended box is a constraint to be measured around.** Four rounds of clock work,
   `STEADY_UNCERTIFIED`, tenant counters, SM-clock sampling, a contention gate — to reach the
   conclusion that wall-clock is unusable here and everything must move to counts. **A $200/month
   cloud runner with a dedicated GPU was never priced against that.**
4. **That the register must be authored by one agent.** Fixed by role, not by argument. It produced
   the sole-author/sole-judge defect twice, on two different people, in two days.
5. **That correctness is established before coverage grows.** Defensible — and it is a choice, and
   the compounding runs the wrong way: each new op now costs a ledger entry, two digests, a
   provenance class, a positive control, and a decline row.

---

## §3 — Pre-mortem: it is 2026-09-02 and this project is dead

**The post-mortem, written now.**

> **It was never used, and the reason it was never used was decided in the first week.**
>
> The EP reached August with one model, two GPUs on one desk, and a governance apparatus that grew
> 4.5× faster than the kernels. In the last week of August someone tried Llama-3 8B. It failed — the
> GQA path had been certified against `Nq/Nkv = 1.00`, the one ratio at which the defect is
> invisible, and the team had a written note saying so. Fixing it meant a new ledger entry, a
> second-model tolerance ruling, a new provenance class for the KV quantisation error budget, and a
> `PROVEN-ELSEWHERE` promotion path that had been specified but not built. **The repair was four
> days of governance and one day of kernel.**
>
> Meanwhile ONNX Runtime's WebGPU EP — same portability story, same vendors, an order of magnitude
> more ops, already in the box — got good enough. The compatibility argument that justified a Vulkan
> EP evaporated, because compatibility was always going to be won by whoever had more ops, and we
> had chosen to have fewer ops that we were more certain about.
>
> **Three proximate causes, in order of contribution:**
>
> 1. **The oracle was one model and everyone knew it.** Every gate, tolerance, ledger entry, ULP
>    series and roofline figure is conditioned on Phi-3.5. The apparatus does not generalise; it is
>    a very precise measurement of one graph. Adding the second model was always "next round".
> 2. **The marginal cost of an op rose every round.** In July an op was a kernel and a test. By
>    August it was a kernel, a test, a ledger entry, two digests, a spec digest, a provenance class,
>    a positive control, an extent declaration, a decline row and a disclosure state. **Rigour that
>    is added to the per-item cost rather than to the per-project cost is a tax on the exact thing
>    the project needed most of.**
> 3. **The machine was never fixed.** Every performance conclusion was reached by routing around a
>    permanently contended shared box. The team built an entire counter-based methodology because
>    the clock was unusable — excellent engineering, in service of not spending $200.
>
> **And the thing that will read worst in hindsight:** on 2026-08-03 the team had `518 passed / 0
> failed`, `521`, `527` — and **zero users**. The number that went up every round was the test count.

---

## §4 — The rebuttal, and it is real but partial

**Where the steelman is wrong:**

1. **The instruments have paid, in refuted claims, at a measurable rate.** They are not
   self-referential in the way the charge alleges. This session alone, the apparatus caught: a
   roofline anchor that was an identity (`x/x`); a "bit-identical" claim that was 62 of 65; a
   `--check PASS` that checked a file against itself; a `0/363` from a stale binary; an ABI
   insertion that made `dispatches_executed` read `device_losses` — **stable, plausible, and zero on
   every healthy run**; six DLL hashes from one tree; and 41 test reds from a rename. **Every one of
   these is a defect that ships silently in a project with half the rigour.** The ABI one is the
   decisive example: it is *invisible to a passing test suite by construction*, and no amount of
   op-coverage velocity finds it.
2. **"Ship more ops" is not free of the same tax; it defers it.** llama.cpp's Vulkan backend is
   ~1 MB of dispatch logic and 164 shaders because it has absorbed six vendors' worth of quirks over
   years. The choice is not rigour-vs-coverage, it is **rigour now vs. rigour after the bug reports**,
   and a plugin EP that returns silently-wrong logits inside someone else's runtime is a
   reputational event ORT will not forgive.
3. **The user's instruction cuts both ways.** *Performance must be very high* is exactly the claim
   this apparatus exists to make honestly. The session's largest performance finding — the
   present/past round-trip, **2.21×, then 4.06× with int4 KV** — was found by the byte model, an
   instrument, on a machine with no usable clock. **The apparatus did not slow that down; it was the
   only thing that could have produced it here.**
4. **The 4.55:1 ratio is inflated by a category error of mine.** `.squad/` (17,908) is team process,
   not project rigour, and `docs/` (16,677) is the architecture of record any EP needs. Strip both
   and it is **2.73:1** — high, but within range for systems software with a correctness obligation
   inside a third-party runtime.

**Where the steelman is right, and I will not soften it:**

- **Assumption 2 is indefensible.** The team has *written down* that its only oracle is degenerate
  on the axis its flagship op depends on, and has responded with instruments. One model is not a
  conformance surface. **This is the single highest-expected-loss item in the project and it is not
  anybody's assignment.**
- **Assumption 3 is a false economy** and it has consumed more engineering than the alternative
  would have cost.
- **The per-op marginal cost is rising and nobody is tracking it.** There is a counter for almost
  everything in this repo and **no counter for how expensive a new op has become.** Given the
  project's own doctrine, the absence of that counter is the finding.

---

## §5 — Alternative approach, so that the current direction is a chosen one

**"Two models, one box, frozen register."** For the next three rounds:

1. **Freeze the rule register.** No new numbered rules, no new unnumbered obligations, no new
   provenance classes. The register is at 13 + 8 and its marginal value is now below its marginal
   cost. Re-open after a second model runs.
2. **Land Llama-3 8B as a second oracle** and let it break things. Predicted: it refutes the GQA
   certification, the KV byte model's `Nq/Nkv` assumption, and at least two tolerances. **That is
   three real defects for one model's cost, against zero for another instrument.**
3. **Rent a dedicated GPU runner.** Retire `STEADY_UNCERTIFIED` as the default verdict rather than
   building a fourth instrument to characterise it.
4. **Add a counter for the marginal cost of an op** — lines, artifacts, and ledger obligations per
   newly-claimed op, per round. If it is rising, the discipline is eating the project, and by the
   project's own standards that must be measurable rather than argued.
5. **Cap the ledger's ambition at bring-up.** A per-op differential re-run on a new device, with no
   persistent cross-device state, would have made `PROVEN-ELSEWHERE`, the toolchain key, the
   two-digest schema, and the Linux blocker all unnecessary. That is most of two rounds.

---

## §6 — Risks for the team to consciously accept or mitigate

| risk | severity | mitigation, if accepted |
|---|---|---|
| **One-model oracle**; the flagship op is certified at the one ratio that hides its defect | 🔴 highest expected loss in the project | assign a second model this round; it is nobody's job today |
| **Rising per-op marginal cost**, untracked | 🟠 | add the counter; the project's own doctrine demands it |
| **Register growth outrunning register navigation** — 13 numbered, 8 unnumbered-but-binding, two namespaces | 🟠 | freeze; see `rule-register-derived.md` §2.3 |
| **Contended box as a permanent constraint** | 🟡 | price a dedicated runner against four rounds of clock work |
| **Zero users** while the test count is the number that rises | 🟡 | pick one external consumer and make their model run |
| **Ledger ambition** (cross-device persistent proof state) | 🟡 | cap at bring-up re-run |

**None of this is a veto and none of it is a recommendation to reduce rigour.** The recommendation
is to **stop adding rigour to the per-item cost and start adding it to the per-project cost**, and
to spend the next round on the one thing the apparatus cannot substitute for: **a second model.**
