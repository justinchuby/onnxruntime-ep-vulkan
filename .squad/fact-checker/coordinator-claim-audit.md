# Coordinator Claim Audit — derived, not self-reported

**Owner:** Fact Checker. **The coordinator does not edit this file.**
**Derived:** 2026-08-03, against `main` = `8ac1172`.
**Standing:** the coordinator ruled that *"the count leaves my hands"* is the general repair for
sole-author-and-sole-judge, and asked to be held to it beyond the rule register. This is the first
exercise of that on him.

---

## §1 — Method, stated before the results

### 1.1 The unit

One **error** = one distinct proposition the coordinator asserted as fact, which was subsequently
established to be false or materially mis-scoped, by any party including himself.

- A turn may carry several. Turn 127 carries three and says so.
- **Relaying another agent's error does not count.** He must have asserted it in his own voice.
- **Mis-scoping counts.** A true number quoted with a false extent is the failure mode this project
  spent a session cataloguing; excluding it would be scoring the easy half.
- **An epistemic-status error counts even when the value survives.** See §3, item 16.

### 1.2 The evidence base

| source | what it contains | admissible? |
|---|---|---|
| local session store, session `c6bec1a7`, turns 0–163, 2026-07-29 → 2026-08-03 | the coordinator's user-facing turn summaries — **136 turns carry a response** | ✓ primary |
| `.squad/agents/{agent}/history.md` | receiving agents quoting what they were told | ✓ secondary — establishes routing-prompt claims |
| `.squad/decisions-archive/**`, `.squad/decisions.md` | adjudicated corrections | ✓ secondary |
| the coordinator's routing prompts, verbatim | — | ✗ **not observable to me** |

### 1.3 The limit that matters most

**I cannot read the routing prompts.** Roughly half of the coordinator's factual assertions are made
to agents, not to the user, and those survive only where a receiving agent chose to quote them. Two
of the five errors he self-reported (the amplification status, and "bit-identical to the CPU EP")
are recoverable *only* because Morpheus wrote them into his history.

**Direction of error: undercount.** Every figure below is a floor. This is the same shape as the
register instrument's blind spot in `rule-register-derived.md` §1.3, and it should be read the same
way: the honest reading of a floor is not "the number is small".

---

## §2 — The headline

| | |
|---|---|
| Errors derived | **21** |
| Errors self-reported in the tasking | **5 classes, covering 7 propositions** |
| Self-report recovery rate | **33%** |
| Turns with a response | 136 |
| Rate | **0.15 errors per user-facing turn**; ~2.1 per round across 10 rounds |
| Corrected in the record | **20 of 21 (95%)** |
| Denied, defended, or quietly dropped | **0** |
| Self-caught and published as a named correction | **5** |
| Median time to correction | **within the same or the next turn** |
| Open / unreconciled | **1** (item 20) |

**The self-report was the least reliable instrument in this audit, exactly as predicted — and it
failed in the direction that flatters, recovering one error in three.** It is worth being precise
about *how* it failed: the five he listed are the five that were adjudicated by a named agent in a
written ruling. The thirteen he missed are the ones he corrected himself, in passing, inside a turn
that was mostly about something else. **He under-reported his own catches, not his misses.** That is
an unusual failure direction and it is the opposite of self-flattery.

---

## §3 — The derived list

Legend: ❌ established false · ⚠️ materially mis-scoped · ★ not in the self-report.

| # | turn / where | the proposition | verdict | caught by | latency |
|---|---|---|---|---|---|
| 1 ★ | t59, 07-30 | claimed nodes = **161** | ❌ 257 | Niobe, by re-reading counters in the run that produced the number | quoted "all day" |
| 2 ★ | t68, 07-31 | `DEVICE=0` is the Intel Iris Xe | ❌ `DEVICE=0` is the NVIDIA 4060; two index spaces, one print | Tank, control experiment | 1 turn |
| 3 ★ | t68 | second broadcast figure in the same correction ("我今天广播的两个数字是错的") | ❌ | Tank | 1 turn |
| 4 ★ | t89, 08-01 | §6.5 closes on selector 0; he had verified the cast himself | ❌ the cast was real, the closure was coincidence | Tank, from his own artifact | 1 turn |
| 5 ★ | t92, 08-01 | the machine load is his own orchestration | ❌ it was the user's other project | **the user** | immediate |
| 6 ★ | t98, 08-01 | R11 already covers the `gpu_steady_tail` specimen, so don't mint R14 | ❌ R11's maxim fits, its four obligations would have *certified* the specimen | Morpheus | 1 turn |
| 7 ★ | t107, 08-02 | `instrument_census.json` lives in `bench/results/` | ❌ `rust/tools/` | self | same turn |
| 8 ★ | t119, 08-02 | there is coverage of all 65 outputs against CPU | ❌ that gate is cross-run identity; CPU compare is 1 of 65 | Fact Checker, then confirmed by him in source | 1 turn |
| 9 ★ | t121, 08-02 | criterion 12 is closed — `unwired: []` | ❌ measured with the census's own denominator; 33 surveyed, 12 instrumented with no census entry | Link (independent enumeration), Morpheus | ~2 turns |
| 10 | t127, 08-02 | **0 of 363** claimed nodes — routed to two agents as an emergency | ❌ stale binary; 323/363, 33/33 islands, ALL-PROVEN | Mouse | same round |
| 11 ★ | t127 | no number on this project is quotable | ❌ over-generalisation of the R13 withdrawal; `12.1847 ms` device-counter GPU-busy was never withdrawn, `quotable = true`, n=41, reproduced to 12.1869 | self + Switch | same turn |
| 12 ★ | t127/129 | the fusion lever he had proposed exists | ❌ intermediates are 0.47% of traffic; the lever does not exist | Switch | same round |
| 13 ★ | t131, 08-02 | the KV divergence is a Vulkan defect (ruling `618d6a4`) | ❌ all 32 GQA nodes were declined; `present.*` was computed on **CPU** | self — published as a correction **naming the superseded commit** rather than letting it lapse | ~3 turns |
| 14 | t148, 08-03 | ops: **Live 48 / Staged 42** | ❌ | self, next turn | 1 turn |
| 15 | t149, 08-03 | ops: **live 47 / staged 22 / 69 total** | ❌ 91 rows / 71 kernel-carrying / 20 staged (Morpheus, artifact-derived). Root cause named: he counted `-match` lines (90), case-insensitively, including legend and header | Morpheus | ~1 round |
| 16 | routing → Morpheus, 08-02 | `weight_reread_amplification = 1.000000` is **"exact, not approximate"**, and anchors the roofline | ⚠️ **the value survived; the status did not.** All four artifact fields were literals; the ratio was `x/x` by construction and would print `1.000000` for a broken kernel. Niobe later made it a real measurement over the compiled SPIR-V, with three positive controls, and it **re-derived to exactly 1.000000** | Morpheus (status), Niobe (value) | ~4 turns |
| 17 | routing → Morpheus, 08-02 | **"bit-identical to the CPU EP"** on the model | ❌ 62 of 65; `logits_max_abs_diff 0.0625`; `criterion10-dev0.json` = `DIVERGENT`. The true, narrower results are GQA `MATCH 0.00072939` and the KV residency probe's `rel = 0.0` | Morpheus, on the artifact | same round |
| 18 | ≥6 occurrences | `gen_proof_ledger.py --check PASS` as merge evidence | ❌ it checks the file against itself; `--append` printed `UNMEASURED … no unlockable keys` then `PASS`, having written nothing | Switch | **longest-lived of the set** |
| 19 | t127, t116, t137, t157 | a DLL hash witnesses that the binary changed | ❌ six builds of an unchanged tree, six distinct hashes; the Linux `.so` was byte-identical across four. Identical hash means nothing relinked; differing hash means nothing at all | Link, who retired **his own** Session-13 method to do it | ~1 round |
| 20 ★ | t156, 08-03 | "six-step logits **bit-identical**, same token chain as the CPU EP" — asserted in the same message as "criterion 10 stays `DIVERGENT`" | ⚠️ **OPEN.** The two are probably compatible (different path, different measurement, different outputs) but the message does not say so, and a reader cannot tell which claim governs | nobody yet | **unresolved at derivation time** |
| 21 ★ | t161, 08-03 | his `Select-Object -Last 2` tail slice was a diagnosis | ❌ the slice recovered nothing; now a Trinity `--selftest` arm that replays a real committed red and asserts the tail-2 capture produces nothing | Trinity | ~1 round |

---

## §4 — What the count says that an impression would not

### 4.1 The rate is low; the persistence is the problem

0.15 errors per turn on a session that pushed ten rounds of merges is not a high rate, and 95%
correction with zero defended errors is a good record by any standard I could apply. **If the
question is "is this coordinator careless", the derived answer is no, and I would say so if the
count had said otherwise.**

But the count is the wrong summary statistic. Weight each error by how long it stayed in circulation
and the picture inverts. Three errors account for nearly all the damage — **#1 (161 nodes, quoted
all day), #16 (the amplification status, headlined into `DESIGN.md` §0), #18 (`--check PASS`, quoted
at least six times across rounds)** — and they share one shape:

> **A number or a green token that arrived pre-formed from an instrument, and was re-quoted rather
> than re-derived.**

That is R13's family and R9's family. The coordinator has spent the session diagnosing exactly this
in other people's instruments. **His own dominant failure mode is not arithmetic; it is trusting a
token he did not produce, and the arithmetic slips (#14, #15) are the least costly things on the
list even though they are the most embarrassing.**

### 4.2 The self-supply correlation is real and it is the structural finding

Of the 21, **at least 8 (#4, #8, #9, #10, #11, #13, #14, #15) occur in turns where the coordinator
was both the producer of the evidence and the reader of it.** That is 38% of errors from what is
certainly a much smaller fraction of his claims.

This is the same defect he diagnosed in Morpheus — sole supplier and sole judge — measured on
himself, and it comes out at roughly the same strength. **He was right that the repair generalises,
and the count is the evidence for it, not the rhetoric.**

Note what the correlation does *not* support: it is not an argument that he should stop verifying
things himself. #10's stale binary was caught because he rebuilt, hashed and re-ran. #13 was
self-caught. The pattern is not "his verification is bad" — it is "his verification is good and his
adjudication of his own verification is not independent of it."

### 4.3 The clearing findings, stated as plainly as the damning ones

- **Zero denials.** In 21 cases I found no instance of the coordinator disputing a refutation,
  softening it, or restating the original claim afterwards. Where a refutation arrived he adopted it
  in the same turn and routed the repair.
- **#13 is a better-than-required behaviour.** He wrote the correction as a record *naming*
  `618d6a4` rather than letting the superseded ruling quietly lapse, with the reason: an argument
  that disappears silently is the path by which it returns as a known fact.
- **#5 and #11 are corrections against his own interest** made without external pressure, in turns
  whose subject was something else entirely.
- **He recused himself repeatedly.** He declined to close criterion 10 on his own artifacts, handed
  the RAI-008(a) tally to Rai rather than moving it himself, and put a conclusion he had supplied
  evidence for to an adversary. Items #8 and #9 were *found because he arranged for them to be*.

### 4.4 The one that is still open

**#20.** `criterion10-dev0.json` reads `DIVERGENT`. Turn 156 says six-step logits are bit-identical
to the CPU EP. Both may be true of different measurements — but this is the second time in this
session that "bit-identical to the CPU EP" has been asserted at a scope the artifact does not carry,
and the first time it was #17. **A claim that has already been refuted once at the wrong extent
should not be re-asserted without its extent.** Owner: the coordinator. Falsifier: name the path,
the output set, and the step count in the same sentence as the word "bit-identical", or do not use
the word.

---

## §5 — Standing derivation

This file is re-derived, not appended to, on request. The method in §1 is the thing to argue with.

The instrument that would make it much better is cheap and does not exist: **routing prompts are not
retained anywhere I can read.** Until they are, half the coordinator's factual surface is audited
only through whatever the receiving agent happened to quote — which means the count is a floor set
by other agents' note-taking habits, not by his accuracy.
