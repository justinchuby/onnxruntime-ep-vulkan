# The Derived Rule Register

**Owner:** Fact Checker. **Morpheus does not edit this file.**
**Derived:** 2026-08-03, against `main` = `8ac1172`.
**Standing:** authorship of the register stays with Morpheus; the *count* of what is in it is
derived here, mechanically, from citations. Established by Morpheus's own ruling of
2026-08-03T00:20:00-07:00 (`.squad/agents/morpheus/history.md`, PART 2):

> *A rule is anything cited as binding by someone who was not in the conversation that produced it.
> … I am not the one who counts. Authorship stays with me; the tally does not.*

If you disagree with what follows, **disagree with §1 first.** The method is stated before the
results precisely so that a disagreement is about the instrument and not about whose count wins.

---

## §1 — The operational definition, stated before it was applied

### 1.1 What counts as a citation

A sentence of Morpheus's counts as **CITED-BINDING** when all four hold:

1. **Externality.** The citing text sits on a surface owned by someone other than Morpheus.
   Ownership is taken from `.squad/team.md` and `.squad/routing.md`, not guessed:

   | surface | owner | counts as external? |
   |---|---|---|
   | `docs/DESIGN.md` | Morpheus | ✗ self |
   | `.squad/agents/morpheus/**` | Morpheus | ✗ self |
   | a `**By:** Morpheus` block in a decisions file | Morpheus | ✗ self |
   | `.squad/orchestration-log/**` | coordinator | ✗ — *in* the conversation |
   | `.squad/log/**` | Scribe | ✗ — narration, not adoption |
   | `docs/OP_COVERAGE.md` | Mouse | ✓ |
   | `docs/PERF.md` | Niobe | ✓ |
   | `docs/PLATFORMS.md` | Link | ✓ |
   | `docs/ENGINE.md` | Switch | ✓ |
   | `tests/**` | Trinity | ✓ |
   | `bench/**` | Niobe | ✓ |
   | `ci/**`, `.github/workflows/**` | Link | ✓ |
   | `rust/**` source and comments | Tank / Switch / Mouse | ✓ |
   | `.squad/agents/{other}/history.md` | that agent | ✓ |
   | a `**By:** {other}` block in a decisions file | that agent | ✓ |

2. **Bindingness.** The citing text uses the principle to *constrain* something — it justifies a
   design, names an obligation, gates a verdict, or explains why a thing was refused. A passing
   mention, a recap of what Morpheus said, or a link with no consequence does not count.

3. **Recoverability.** The citation reproduces the principle closely enough that a reader who has
   never seen the ruling can act on it. A bare `§8.9.18` pointer with no restatement counts only
   when the surrounding text acts on it.

4. **Non-authorship.** Morpheus is not the author of the citing block. Reviewing, endorsing, or
   ruling on someone else's text does not make him its author; writing it does.

### 1.2 What this definition deliberately excludes

- **Intent.** "I meant it as a rule" is unobservable and it is his. Excluded by his own rule 1.
- **Self-citation of any depth.** `DESIGN.md` cross-referencing `DESIGN.md` is not evidence a rule
  is in force; it is evidence a document has an index.
- **Coordinator routing.** The coordinator is a party to the conversation that produced the ruling.
  His restatement is not independent adoption. (This is the same exclusion that makes his own
  tallies inadmissible in §4 of `coordinator-claim-audit.md`.)
- **Scribe narration.** Session logs record that a thing was said. They do not record that anyone
  was bound by it.
- **Frequency.** One qualifying citation is enough. Morpheus's rule 2 sets a *numbering* threshold
  at two; this file records the count, not the numbering, so it uses one.

### 1.3 Known limits, and the direction of the error

- The instrument is **phrase-anchored** (`derive_register.ps1`). A citation that paraphrases without
  reusing any anchor is invisible to it. **Direction of error: undercount.** Every number below is
  a floor.
- Two agents' worktrees carry uncommitted work at derivation time. Anything not on `8ac1172` is not
  counted. **Direction of error: undercount.**
- Judgement enters at clause 2 (bindingness). It is applied per-hit and the hit list as produced at
  `8ac1172` is preserved verbatim at `.squad/fact-checker/derive-hits-8ac1172.txt`, so the judgement
  can be re-run and disputed hit by hit without re-running the instrument.

### 1.4 How to reproduce

```powershell
pwsh .squad/fact-checker/derive_register.ps1
```

Emits every hit for every anchor with an owner tag. Add anchors to the `$anchors` table when a new
principle is ruled. The script contains no verdicts — verdicts are in §2 and §3 of this file.

---

## §2 — The derived register

### 2.1 Numbered rules (authored, numbered, in force)

**R1 – R13.** All thirteen are cited outside Morpheus's surfaces. Citation counts on non-Morpheus
surfaces, excluding JSON/log artifacts:

| rule | external citations | rule | external citations |
|---|---|---|---|
| R13 | 419 | R7 | 35 |
| R12 | 256 | R6 | 33 |
| R10 | 185 | R5 | 32 |
| R9 | 169 | R4 | 22 |
| R11 | 105 | R3 | 19 |
| R1 | 37 | R8 | 17 |
| | | R2 | 8 |

`R14` occurs exactly once in the tree, in this Fact Checker's own prior audit, discussing the number
that was not minted. **There is no phantom rule.**

### 2.2 Unnumbered obligations that pass the citation test

These are in force. They are not in the numbered register. Each is listed with the citation that
qualifies it — the strongest external one, not the first one found.

| # | obligation | qualifying external citation |
|---|---|---|
| **U-A** | *Two gates whose extents differ compose to the weaker extent and the stronger name; a record with two gates owes two extents.* ("coverage does not compose") | `tests/ops/test_no_cpu_fallback.py:41` — Trinity gives it its own docstring heading, **THE TWO MECHANISMS' EXTENTS ARE STATED SEPARATELY ON PURPOSE**, and derives from it that "neither may borrow the other's reach". Also `docs/PLATFORMS.md:1965` (Link), `tests/ops/test_verdict.py:789`, `tests/ops/_models.py:1227`, `bench/results/no_cpu_fallback_migration.json:19`, and a joint `**By:** Mouse and Trinity` decision (round8:64) and a `**By:** Link` decision (round9:39). |
| **U-B** | *An observable that is true whatever happens cannot convict; an observable that degrades whatever happens cannot acquit.* (R9's dual) | `tests/ops/test_criterion10_ulp.py:110` — Trinity names a test after the failure and writes "Morpheus's sentence pointed at my own instrument"; the consumer's design (median + p99 + cancellation count) is the remedy. Also `tests/ops/_models.py:649`, `bench/results/criterion10-ulp-prediction.md:68`. |
| **U-C** | *Fault scope is set by the scope of what you cannot locate, not by the severity of what you found.* | `docs/OP_COVERAGE.md:3812` — Mouse gives it a numbered subsection, quotes it verbatim, attributes it, and builds the eight-row locatable/not-locatable table from it. Implemented as the `entry_faults` / `faults` split in `rust/src/registry.rs`. |
| **U-D** | *A proof is a property of a form on a device; `PROVEN` / `PROVEN-ELSEWHERE` / `UNPROVEN`, with `PROVEN-ELSEWHERE` claimable and disclosed.* | Implemented, not merely quoted: `rust/src/registry.rs` (7 sites), `rust/src/disclosure.rs` (2), `rust/tools/gen_proof_ledger.py`, `docs/OP_COVERAGE.md` (7), `bench/results/phi35_claim_reading-dev1.json:50` as an emitted verdict token. |
| **U-E** | *A reading whose confidence is anti-correlated with its subject.* | `bench/phases.py`, `bench/device_companion.py`, `bench/test_marginal_tail_withholds.py`, `ci/device_state.py`, `ci/lane_inventory.py`, `docs/PERF.md`, `docs/PLATFORMS.md`. In force in two languages across three owners. |
| **U-F** | *A suite whose verdict is not a function of its assertions.* (Link's finding; Morpheus declined to number it 2026-08-03T05:05) | Executable: `ci/check_suite_productivity.py`, `ci/negative_control_suite_productivity.py`, `ci/lane_inventory.py`, `.github/workflows/ci.yml`, `ci/test_lane_checks.py`. This obligation gates merges and has no number. |
| **U-G** | *A frame mismatch must be distinguishable from a key absence; the predicate, not the parser, decides what a mismatch licenses.* | `rust/src/registry.rs:2707-2712`, and again at `:4895`, `:4928`. Written into the code comment as the reason a `continue` was removed. |
| **U-H** | *No single hash can be sensitive to the kernel and blind to the compiler; the subject needs two digests.* | `rust/build.rs:414`, `rust/src/registry.rs:2024`, `docs/OP_COVERAGE.md:3886`. |

**Derived count of unnumbered-but-binding obligations: 8.**

### 2.3 The shadow register — the finding I did not expect

Morpheus's rulings are addressed by section number as well as by rule number, and **the section
namespace is the one the codebase actually uses**:

| namespace | external citations (excl. `DESIGN.md`, his history, logs) |
|---|---|
| `R1`–`R13` | ~1,337 |
| `§8.9.x` | **339**, of which `rust/src/registry.rs` alone carries **80**, `counters.rs` 39, `disclosure.rs` 34, `gen_proof_ledger.py` 25, `docs/OP_COVERAGE.md` 28 |

This changes the diagnosis he accepted. He conceded the register was *under-numbered*, and called
that "a navigability defect, repairable by numbering". **The evidence says navigability is not the
defect.** Agents navigate to these obligations fluently — 339 times — via `§8.9.x`. What is actually
true is narrower and more awkward:

> **The register has two namespaces with different semantics and only one of them is counted.
> `R#` names an obligation. `§8.9.x` names a location. The project's binding obligations are
> distributed across both, the declared size of the register (13) reflects only the first, and the
> declines tally measures traffic between them rather than growth of the whole.**

A `§` citation is durable only while the section keeps its number, and §8.9.19 has already had to
restate §8.9.17 because "the device belongs" was ambiguous between *in the key* and *on the entry*.
That is the real cost, and it is not navigability — it is that a location can be re-cut while an
obligation cannot.

---

## §3 — Re-scoring the declines

Morpheus predicted **"some of my six may not survive"**. Two prior corrections to the premise:

1. **The count in circulation is stale.** His own history reaches "Seventh decline" at
   2026-08-02T23:40, and an eighth non-mint (both of Link's findings) at 2026-08-03T05:05. Six was
   true at 2026-08-02T21:24 and has been quoted since. **The tally that was handed over was already
   two behind when it was handed over.**
2. **One deliberate non-numbering was never counted at all** — criterion 10's "coverage does not
   compose", explicitly marked *DELIBERATELY NOT NUMBERED*, with the tally left at four. It is now
   the most externally-cited unnumbered obligation in the project (U-A).

### 3.1 The scoring rule

Morpheus's rule 3: *a decline counts only if the principle stayed out of the record, in every form.*
Applied literally this destroys legitimate deduplication, so it is read with the three destinations
a declined principle can reach:

| destination | verdict |
|---|---|
| **(a) nowhere** — the principle was genuinely not written down | decline **SURVIVES** |
| **(b) into an existing numbered rule** as an amendment or sub-rule | decline **SURVIVES** — the register grew, but on-book and under a number |
| **(c) into unnumbered prose that a non-Morpheus party then cites as binding** | decline **DOES NOT SURVIVE** — this is exactly the off-book growth the count was supposed to detect |

Only (c) disqualifies, and (c) is decided by §1's citation test, not by reading his intent.

### 3.2 The scorecard

| # | decline | principle | destination | verdict |
|---|---|---|---|---|
| D1 | R9, anti-correlated confidence | U-E | (c) — `bench/phases.py`, `bench/device_companion.py`, `ci/device_state.py`, `ci/lane_inventory.py`, `docs/PERF.md`, `docs/PLATFORMS.md` | ❌ **does not survive** |
| D2 | R13, the defaulting lookup | — | (b) — became R13 amendment 1; no external citation of an unnumbered form | ✅ **survives** |
| D3 | R13, classifier scope — *a newly named pattern begins attracting cases that do not belong to it* | — | (a) — anchor occurs only in `DESIGN.md`, his history, and his own `**By:** Morpheus` decision block | ✅ **survives** |
| D4 | R11, criterion-12 witness vs. discharge | — | (a) — *"a witness is not a discharge"* occurs only in `DESIGN.md` | ✅ **survives** |
| D5 | R9's dual — *cannot convict / cannot acquit* | U-B | (c) — Trinity cites it by author in a test **name** and redesigns her consumer around it | ❌ **does not survive** |
| D6 | the proof-ledger device frame — `PROVEN-ELSEWHERE` | U-D | (c) — now a shipped state token in `registry.rs`, `disclosure.rs`, `gen_proof_ledger.py`, and an emitted field in `bench/results/phi35_claim_reading-dev1.json` | ❌ **does not survive**, most decisively of the set |
| D7 | `parse_ledger` fault scope | U-C | (c) — Mouse gives it a subsection, a verbatim quote, an attribution, and an implementation | ❌ **does not survive** |
| D8 | Link's two findings, 2026-08-03T05:05 | U-F, U-H | (c) — U-F is enforced by CI on every merge | ❌ **does not survive** |

**3 of 8 survive. 5 do not.** Morpheus said *some* may not survive; the derived answer is that a
**majority** do not, and the two cleanest survivors (D3, D4) are the two where the principle stayed
genuinely small.

### 3.3 What the re-score does *not* say

It does not say Morpheus grew the register dishonestly. Every one of the five disqualified declines
is an **honest deduplication by remedy** — the remedy really was R9's, or R12's, or R13's, unchanged.
The failure is not in the ruling. It is that "did I mint a number?" and "did the project acquire a
new binding obligation?" are different questions, and only the first one had a counter on it.

The ✅ from the previous audit still stands and is worth restating because a clearing verification is
worth as much as a damning one: **no principle was lost.** All eight unnumbered obligations are in
the record, are findable, and are being used. The register is under-**counted**, not under-populated,
and not — as it turns out — under-navigable either.

---

## §4 — What the next derivation should do differently

1. **Add an anchor per ruling at ruling time.** The instrument's blind spot is paraphrase; it closes
   if each ruling names its own anchor phrase in the same sentence that states it.
2. **Count the `§8.9.x` namespace as register surface.** Any tally that counts only `R#` will keep
   under-reporting by roughly the factor found in §2.3.
3. **Re-derive on the same commit as every register change**, so the count and the register never
   diverge by two the way the six/eight did.
4. **A decline is now a claim with a falsifier.** Recording "declined" creates the prediction *no
   non-Morpheus surface will cite this as binding*. That prediction is checkable by
   `derive_register.ps1`, and D1/D5/D6/D7/D8 are the record of it failing.
