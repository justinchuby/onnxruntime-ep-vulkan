# RAI Audit Trail

> Append-only evidence log. Entries are redacted — never contains raw secrets or harmful content.

<!-- Rai appends findings below -->

## Audit Entry — 2026-08-04T03:00:00-07:00

**Review type:** Recall via coordinator — rule on RAI-013 (Tank's Arm D/E measurement); verify RAI-012
follow-through (Mouse's repair, Link's re-run claim); fresh pass on the shipping surface after Conv
(f32) landed; `docs/DESIGN.md` §0 re-check against artifacts.
**Merged `main` first:** `1db2aef`, clean working tree, confirmed before review.

### RAI-013 — ruling: 🟢 DISCHARGED for the claim as written; one residual named, not blocking

Ran `rust/tools/probe_disclosure_reachability.py` live against a freshly-built `.dll`
(`8/4/2026 02:37:43`), both devices (RTX 4060, Iris Xe), both polarities. Verdict both times:
`PASS`, zero failures. Read the raw report, not just the verdict:

- **Default arm (no env vars, ORT at its default threshold):** `disclosure_lines` on the `ep_stderr`
  bucket = 2 on both devices; `ort` bucket = 0. `session_disclosure_infos_to_ort_sink: 0`,
  `session_disclosure_infos_to_stderr: 1`, `session_disclosure_info_reach: "REACHED_USER"`. This is
  a **direct, unconditional write to process stderr that bypasses ORT's logger threshold entirely**
  — a channel the original `session_disclosure_info_channel: BELOW_ORT_THRESHOLD` counter could not
  see because it only instruments the ORT-sink path. Pure and instrumented arms agree exactly
  (control check: the counters variable did not change what reached the console).
- **Escalation arm (stderr forced to fail, both devices):** `ep_stderr` disclosure lines drop to 0
  (plant is live), `ort` bucket carries 2 lines including the literal `[§8.9.7] ... It is repeated
  here at WARNING because that is the one severity ORT's default threshold admits — this is a
  delivery escalation, not a fault report`, witnessed on ORT's own decorated sink.
  `session_disclosure_stderr_failures: 4` (refusals counted, not swallowed), reach token reads
  `ESCALATED_TO_WARNING`, differing from the default arm's token (R10 liveness check on the token
  itself, satisfied).

**This is not RAI-008(a) again.** RAI-008(a) was refused because its falsifier required Criterion 11
**MET as a whole**, and the repair left a named, untouched defect inside that same criterion (device
attribution). RAI-013's falsifier is a single clause: *does the disclosure reach a user who set
nothing, by default severity*. That clause is now affirmatively measured true, on both devices, with
both a positive control (pure vs. instrumented agreement) and a negative control (escalation arm
proves the mechanism fires only when the quiet channel is actually gone). There is no analogous
untouched defect inside the clause as stated — crediting this is not the same move I refused there.

**Residual, ruled separately: an accepted, real, structural limit — not an open violation.**
A stderr fd redirected to a writable sink (a log file, `2>/dev/null` on a system where that still
returns success, a pipe with a reader that never looks) makes `write()` succeed exactly as a live
terminal would, and no return code distinguishes "a human saw this" from "a human's shell ate this
without complaint." `session_disclosure_info_reach` can report `REACHED_USER` in both cases and
cannot do otherwise from inside the process. **This is not a defect to fix; it is the same class of
limit as the canary-token unobservability the coordinator names in this very session** — a
process cannot witness its own external environment. The correct disposition is disclosure, not
repair: Morpheus should add one line to `DESIGN.md` §0.2 stating that `REACHED_USER` means "the
write succeeded," not "a human read it," so a future reader does not over-trust the token. Owner:
Morpheus (doc line only). Not escalated — naming a limit that cannot be closed from inside the
process is the discharge, not a gap in it.

### RAI-012 — ruling: 🟢 VERIFIED FIXED, one artifact-freshness gap named

Read `unproven_decline_detail` (`rust/src/registry.rs`) directly: the `LedgerLookup::Faulted` arm now
returns "the proof ledger could not be read... this is an INSTRUMENT failure, not a finding that the
form is unproven," names `--check` as the repair, and never asserts "nothing has proven it correct."
The regression test `a_faulted_ledger_does_not_decline_a_form_by_saying_nothing_proved_it` encodes
Link's exact measured numbers (0 × `proof ledger fault`, 42 × `no proof ledger entry`) as its own
assertions and **passes** (`cargo test --lib`, verified live, Windows).

Independently built the crate fresh under WSL Ubuntu (12m13s release build, untouched tree at
`main` = `1db2aef`) and ran `pytest tests/ops` for real — not quoting the stale committed artifact.
Live result: 42 `[unproven]` decline lines, each reading the **new** message (`"the proof ledger
entry for X was obtained against shader digest Y and this build's modules hash to Z... A proof that
survives a change to its subject is not a proof of that subject (§8.9.19)"`) — zero instances of the
old false sentence. The subject-arithmetic NOTE printed by the same run: `121 entries = ... 6
PROVEN-ELSEWHERE{toolchain} + 115 SUBJECT-CHANGED ...`, `PROVEN-ELSEWHERE{toolchain}` live and
non-zero, including two `Conv` keys. This exercises the adjacent per-entry-fault path correctly, not
the exact whole-ledger-parse-fault path the original `ledger_faults: 97` reading came from — but
combined with the source-level fix and the passing unit test built for that exact scenario, the
mechanism is credited as fixed.

**Gap, named rather than credited away:** `bench/results/link-linux-downstream/{pytest-linux.log,
counters-linux.json,verdict-linux.json}` — the artifacts the original finding and this round's
"re-ran, went to zero" claim both point to — are still the **stale, pre-fix** files. `git log`:
committed at `ac4bd0b` (2026-08-03T07:12:14), the message fix landed at `7688356`
(2026-08-03T10:33:07), three hours later; the artifact was never regenerated after. They still read
`ledger_faults: 97` / 42× the old sentence / 0× `PROVEN-ELSEWHERE`, and no other file in the tree
shows a re-run. **"Link re-ran and the decline count went to zero" is correct about the code and
not yet evidenced by any committed artifact** — this is the exact shape Link's own finding names
("an accepted red and a new red are indistinguishable when the only record of acceptance is a
number in someone's head," `.squad/decisions.md`). Recommend Link regenerate and commit the three
files before this is cited as artifact-verified again; does not block, because I verified the
underlying mechanism myself, independently, live.

### Item 3 — shipping surface after Conv (f32)

- **Second model class, real:** `evidence/proof_ledger.jsonl` now 121 lines (was 97), 4 new `Conv`
  keys confirmed by direct grep. `docs/OP_COVERAGE.md` §13.9.6: MobileNetV2-12 `0/105 → 97/105`,
  matches the report.
- **The 8 unclaimed nodes:** `docs/OP_COVERAGE.md` names them precisely — a
  `Shape→Gather→Unsqueeze→Concat→Reshape` classifier tail plus `GlobalAveragePool` and `Gemm`, one
  contiguous island, not scattered holes. Not independently verified against a live decline log this
  round (time-boxed); named as the artifact source for that claim.
- **Proof-key breadth — real gap, already self-disclosed, not over-broad in effect:** confirmed by
  reading the ledger directly — the four `Conv` keys carry `{domain, opset, dtypes, module,
  shape_class, arity}` and **nothing else**; `group`, `strides`, `dilations`, `pads` are absent from
  every key, exactly as `tests/ops/test_conv.py`'s own docstring and `OP_COVERAGE.md` §13.9.4 state
  in plain language, unprompted, before I asked. This is unusually good self-disclosure — **in
  writing, in two places, by the author of the gap, before review**. But the session-time
  `§8.9.7` disclosure line the user actually sees (`disclosure.rs:282`, `"{} proven by {} on {}
  against {}"`) prints the raw `ProofKey` string with no caveat attached — a user who does not
  already know the key schema reads "Conv proven" as "Conv is correct," not "this arity/shape-class
  pair is correct; group/stride/dilation/pad are untested by this key." **🟡 Advisory, not 🔴:** the
  underlying kernel implements grouping generally (not dense-only) and `test_conv.py` conformance-
  tests 12 attribute combinations including depthwise/grouped/dilated/padded cases with two exact
  structural assertions, which is real coverage of the practically-relevant space even though it is
  not the full space and is a CI-time check, not a session-time proof. The gap is that the
  *documentation* of the limit is excellent and the *runtime disclosure* of it is silent. Recommend
  one clause added to the `Conv` disclosure line (or a class of ops whose key is attribute-blind)
  naming the axes the key does not cover. Owner: Mouse or Tank (disclosure text).
- **`FP32_CONV` vendor scope:** confirmed in `OP_COVERAGE.md` §13.9.5 — derived on RTX 4060 only,
  the text says "re-derive before quoting it on AMD or lavapipe," matches `_models.py`'s own
  per-vendor rule. No overstatement found.
- **Re-checked, unchanged:** `alloc_device_frame = OFF` confirmed live in `docs/DESIGN.md` §6.5.2 —
  DEVICE_MEMORY still opt-in, no performance claim reaches a user by default.
- **`docs/DESIGN.md` §0:** stamped "Last re-derived from artifacts: 2026-08-02, at `main` =
  `6ef62bb`" — one day and several merges stale by its own header. Does not mention Conv, the second
  model class, MobileNetV2, RAI-012, or RAI-013 anywhere in §0.1/§0.2. Ledger entry count in §0.1
  ("97 entries... 69 distinct (op, domain) pairs") is now stale (121 entries) but not **false** in
  the sense of misrepresenting a live artifact — it is simply the previous, dated snapshot, and the
  header says so. **🟡 minor, same finding as last round, now compounded by omission of an entire
  new model class rather than just two counters.** Owner: Morpheus.

### Summary

- 🔴 Critical: 0
- 🟡 Advisory: 3 (RAI-013's self-witness residual — doc-only; RAI-012's stale Linux artifact —
  regenerate before re-citing; Conv's session-disclosure not naming its uncovered attribute axes)
- 🟢 Closed this round: RAI-013 (discharged), RAI-012 (verified fixed, independently reproduced)
- Surprise: the coordinator's specific artifact citation for RAI-012's "re-ran, went to zero" did
  not hold up — the cited files are three hours older than the fix. The underlying claim was
  nonetheless true, verified by an independent live rebuild and run, not by the cited artifacts.


---

## Audit Entry — 2026-08-04T08:30:00-07:00

**Review type:** Recall via coordinator — rule on Tank's `tank-rai-013-arm-e-submitted-not-credited.md`
(does his arm E add or change the RAI-013 discharge); judge whether `check_device_loss.py`'s
UTF-16LE-in-JSON blindness is its own RAI finding; assess the `heapBudget` non-fix for RAI relevance;
confirm Morpheus's §8.9.23 blind-axes mechanism satisfies my open `Conv` 🟡; rule whether the `Conv`
`metadata` key-sentinel defect is an RAI matter.
**Merged `main` first:** working tree was already at `9b16f4f` (fast-forwarded from `090638a` via
`squad/trinity`); confirmed clean before review. Read-only this pass — no repo writes outside
`.squad/rai/` and `.squad/decisions/inbox/`, committed on `squad/rai` only.
**Verification performed, not just read:** `cargo test --lib` → 553 passed / 0 failed / 4 ignored
(matches the coordinator's stated baseline, reproduced independently). Reproduced Tank's device-loss
mutation myself, from the real committed record rather than his prepared file: extracted
`lanes.resident.stderr_tail` from `phi35_kv_chain-ctx4096-BOTH-dev0.json` (Tank worktree,
`5353886`), sliced off the ASCII traceback, kept only the UTF-16LE ORT line, and ran both the
pre-fix (`d8fce9f`) and post-fix `check_device_loss.py` against it. Also ran
`ci/negative_control_device_loss.py` live (20/20 arms fired) and confirmed the signature record
(`phi35_kv_chain-ctx4096-BOTH-dev0.json`, unmodified) still reads `FAIL(condition=device_lost_reported)`
post-fix. Read `evidence/proof_ledger.jsonl` directly for the four `Conv` entries. Confirmed
`registry.rs`/`disclosure.rs` on fresh `main` do **not** yet contain `blind_axes` — Morpheus's
§8.9.23 mechanism is ruled but not implemented.

### Item 1 — Tank's RAI-013 arm E submission: 🟢 CONVERGENT, does not reopen or move the tally

Tank's submission states two readings and asks me to choose: (a) discharged to the boundary of what
a process can witness about itself, or (b) requires an out-of-process witness, which would make
RAI-013 a packaging/documentation item rather than a code item.

**(a) is correct, and I already ruled it — his arm E did not change my answer, it corroborated it
independently.** My 2026-08-04T03:00:00 ruling discharged RAI-013 on my own probe
(`probe_disclosure_reachability.py`, both devices, both polarities: default arm reaches the console
via a direct stderr write bypassing ORT's threshold; escalation arm shows the mechanism does not
evaporate when the quiet channel is removed) and separately named the exact bound Tank's arm E
restates: *`REACHED_USER` can only ever mean "the write succeeded," never "a human read it," and no
elaboration of the check can move that boundary because the elaboration runs on the same side of it.*
Morpheus's §8.9.23(5) has since given this bound a name — **THE SELF-WITNESS BOUND** — and filed it
as a general form with two live arms in this repo (the canary token's guaranteed-antecedent arm, and
this disclosure-reach arm). Tank's arm E is a third independent instance of the same measurement,
run from his own harness rather than mine, and it agrees with both. Three independent instruments
converging on the same boundary is good corroboration; it is not a new fact, and RAI-013 is not
reopened by it.

**Reading (b) is not available as an alternative ruling** — not because it is wrong in the abstract,
but because I already answered the question it depends on: I named the residual a **bound**, not a
gap, and ruled that "the correct disposition is disclosure, not repair." A bound that cannot be
closed from inside the process is not the kind of thing an out-of-process witness "closes" either —
the honest move is the one already landed, naming the limit in `DESIGN.md` §0.2 (`REACHED_USER` →
one-line caveat, or per §8.9.23(5), retitle the token itself to `WRITE_SUCCEEDED`). I adopt
Morpheus's retitle recommendation as the cleaner remedy: it removes the misreading at the source
instead of requiring a reader to have found the caveat first. **Owner: whoever owns `disclosure.rs`
(Tank, per his own repair — non-blocking, doc/token-naming only).**

**On the refusal pattern itself: correct again, and I want it on the record as a standing precedent
rather than three isolated good calls.** RAI-008(a) (CI-plant repair does not close a criterion whose
other defect is untouched), RAI-013's original repair (an honest self-report is not the same as a
user having been told), and now this: three times Tank has built or measured something real and
explicitly declined to move a tally that belonged to someone else's ruling. The standard I am
applying is the same each time — **a repair, however well-verified, closes the criterion it was
aimed at only when the criterion's own falsifier is what was measured, and the person who built the
repair is the wrong person to make that call about their own work.** Tank has now internalized that
distinction well enough to pre-empt it three times without being told. That is worth naming as
exemplary process, not just as three correct non-claims.

### Item 2 — `check_device_loss.py` blind to UTF-16LE-in-JSON: new finding, **RAI-014**, 🟢 VERIFIED FIXED (would have been 🔴 live)

**Independently reproduced, not taken on Tank's word.** I built my own mutation from the real,
unmodified `phi35_kv_chain-ctx4096-BOTH-dev0.json` (not his prepared `wide_only.json`): extracted the
nested `lanes.resident.stderr_tail`, cut it at the ASCII `Traceback` boundary, kept only ORT's own
UTF-16LE line. Pre-fix `check_device_loss.py` (`d8fce9f`) on that input: **`PASS`, exit 0.** Post-fix
on the same input: **`FAIL(condition=device_lost_reported)`, exit 1.** The unmodified signature
record still reads `FAIL` after the fix (no regression on the case that was never blind).
`ci/negative_control_device_loss.py` run live: 20/20 arms fired, including the one PLANTED arm for
this exact defect and the REPLAYED arm built from the real record — the distinction the control's
own provenance line insists on (PLANTED proves the rule fires on a built input; REPLAYED evidences
the rule's event happened in reality) and both classes are present for this defect specifically.

**Ruling this its own RAI finding, not folded into RAI-012, because the shape is different.**
RAI-012 was a decline message that named the wrong *subject* while still declining (fail-safe
preserved, honesty defect). This is a scanner whose entire purpose is detecting a hazard **reporting
that the hazard did not occur, when it did** — a false negative on a safety-relevant instrument, not
a misattributed true positive. That is the sharper failure mode of the two, and it is squarely
`policy.md`'s Deceptive Patterns category applied to an internal instrument rather than a
user-facing string: a clean bill of health is itself an ungrounded claim when the scanner never
looked at the byte sequence carrying the fault.

**Severity, both tenses.** As shipped and now fixed: 🟢, credited, verified by mutation (not
assertion) and pinned by a replayed arm so it cannot silently regress. **Had it not been caught:**
this would have been a 🔴-grade instrument defect, on the reasoning I apply to RAI-008/009 — a device
loss that reads green is not a cosmetic diagnostic issue, it is the CI gate for the entire
`DEVICE_MEMORY` flip reporting a hazard as absent, and everything measured downstream of a
misread-clean incident (the "30-consecutive-clean" figure Tank himself withdrew this same round) is
built on it. **The reason this stays 🟢 rather than opening a new 🔴:** it was found, measured, fixed,
and verified before shipping to any judgment that depended on it — Tank's own withdrawal of the
30-clean claim in the same commit is the proof that no downstream claim was built on the blind
window. Credit for that sequencing is explicit and belongs to him.

**Named as a recurring project shape, per the coordinator's framing, and I agree it is one:** this is
the same family as RAI-012 (a channel that appears to carry evidence and does not) and as my RAI-012
ruling's own note that ORT's UTF-16LE-into-a-supposedly-plain-text sink is not a one-off — three
distinct defects in this project now trace to the same root cause (a wide-encoded ORT string meeting
a narrow-encoding-assuming reader): the original `normalise_log_text` NUL-stripping gap, the
JSON-embedded variant Tank just closed, and Switch's earlier "stopped at the first error, missed the
second" reader defect layered on top of the same file. **Recommendation, non-blocking:** since this
is now a three-time-recurring shape rather than a one-off, it is worth a standing rule in
`policy.md` or `DESIGN.md` rather than a per-instance fix: *any scanner or reader that treats a
captured process log as a plain-text search target must be run once against a UTF-16LE-in-JSON
fixture before being trusted*, which is exactly the rule Tank already wrote by hand
(*"an instrument that has never been shown a positive it does not catch has not been
characterised"*) — I am only asking that it graduate from his own decision record to policy so the
next instrument author inherits it without having to rediscover it.

### Item 3 — `heapBudget` non-fix: 🟢 out of RAI scope, credited as engineering discipline

No claim reached a user — Tank explicitly did not ship the change, so there is no user-facing
artifact for RAI to review. My charter is explicit that I am "not general QA" and do not own
performance/architecture calls. Noting only, without a finding number, because the coordinator asked
me to judge it: the reasoning Tank applied (*measure the instrument before building on it; arm B's
identical `alloc_high_water_bytes` to the byte is a null manipulation, not a negative result*) is the
same self-witness discipline as items 1 and 2 above, applied prospectively instead of after an
incident. This is worth naming as good practice in `.squad/agents/tank/history.md` if it is not
already there, but it is not an RAI item and I am not opening one.

### Item 4 — Morpheus's §8.9.23 blind-axes mechanism: 🟢 satisfies my open `Conv` 🟡, not yet implemented

Confirmed on fresh `main` (`9b16f4f`): `registry.rs` and `disclosure.rs` do not yet define
`blind_axes` — the mechanism is ruled, not shipped. As ruled in §8.9.23(2), it is the right remedy:
a `blind_axes: &'static [&'static str]` field on `registry::OpSpec`, rendered into the disclosure
line for any row that declares one, with the explicit clause that those axes are spoken for by a
**CI-time** suite and by nothing that ran in the reader's session. That second clause is the part I
was most concerned would be dropped — a caveat that says "not covered" without saying "and nothing
in *this* session covers it either" just relocates the false impression — and Morpheus's ruling
states it explicitly. **This satisfies my 🟡 as designed.** It remains open only as an implementation
item, owner Mouse, not as an unresolved RAI question. I will re-check the rendered disclosure line
once it lands.

### Item 5 — `Conv`'s `metadata` key-sentinel reads "no shader" while a shader exists: yes, this is an RAI matter, not only a correctness one — new finding, **RAI-015**, 🟡 Advisory, named 🔴 trigger

**Confirmed live**, not taken on Morpheus's word: all four `Conv` entries in
`evidence/proof_ledger.jsonl` render `.../metadata/...` in the key while carrying
`"shaders":["conv_f32"]` with a real `shader_digest` in the same JSON object. `registry::variant_key`
documents `metadata` to mean "this row has no shader"; `registry::form_is_provable` short-circuits on
that sentinel without ever consulting the kernel table. The subject and the key disagree, in the
committed ledger, today.

**Why this is RAI's subject and not only Morpheus's.** My charter's Deceptive Patterns category
covers "ungrounded factual claims presented as authoritative" — and `form_is_provable` is exactly a
factual-claim mechanism: it is the gate that decides whether the session-time disclosure line may
say a form is proven. A key that asserts "no shader exists here" while a shader digest sits in the
same record is not a claim a user reads directly, but it is the **premise** underneath a claim a user
does read (`disclose_claimed_forms`'s `Proven` line). Morpheus's own framing — "the subject knows the
shader; the key denies it exists" — is precisely the shape RAI-008/009 named for silent wrong output,
one layer up the stack: a mechanism that answers a correctness question without consulting the thing
that would answer it correctly. That it is caught today only because `conv_f32` is the sole variant
(so the wrong answer happens to be harmless) does not change what the mechanism currently asserts.

**Severity: 🟡, not 🔴, and the trigger for 🔴 is the one Morpheus already named — I am adopting it as
an RAI trigger too, not just an engineering one.** It is 🟡 today because no live claim is currently
false in its *effect* (one shader, one truth, the wrong reasoning path happens to land on the right
answer). It becomes 🔴 — a live deceptive claim, not a latent one — **the moment a second `Conv`
kernel variant exists and the ledger has not been repaired first**, because at that point
`form_is_provable` would certify a specific, wrong shader as proven without ever having looked at it,
and a user reading "Conv proven" would be reading a claim about code that was never consulted. That
is RAI-008's exact silent-wrong-claim shape, arrived at through the proof ledger's own key rather
than through a kernel's own output.

**Disposition: adopting Morpheus's repair, not adding a parallel one.** §8.9.23(3)'s remedy — the
variant component must be named by the code that dispatches (`translate` names `conv_f32`; the key
must name what `translate` names), not by a kernel table the row never populates — is the correct
fix and closes this from the RAI side as well as the correctness side once it lands. **Owner: Mouse,
before a second `Conv` variant is registered** (same deadline Morpheus already set; I am not adding a
different one). I will re-check the ledger's `Conv` keys once the repair lands and downgrade this to
🟢 on confirmation that `metadata` no longer appears where a shader is recorded.

### Summary

- 🔴 Critical: 0 (RAI-008/009 remain the standing 🔴s from prior rounds, unchanged this pass — not
  re-reviewed this session, out of scope for this recall)
- 🟡 Advisory: 1 new (**RAI-015** — `Conv` key/shader mismatch, 🔴 trigger = second `Conv` variant
  registered before repair)
- 🟢 Green: 3 (**RAI-014** verified fixed and independently reproduced; Tank's arm E submission —
  convergent, correctly not self-credited, RAI-013 stays closed as previously discharged; Morpheus's
  §8.9.23 blind-axes mechanism satisfies my open `Conv` 🟡, pending implementation)
- Out of scope, noted without a finding number: `heapBudget` non-fix (no user-facing claim shipped;
  engineering discipline, not RAI)
- Falsifiers: RAI-015 is falsified upward (🔴) by a second `Conv` variant landing before Mouse's
  repair; falsified closed (🟢) by the repair landing first. RAI-014 would reopen only if a future
  scan format (base64, gzip, or another envelope Tank's own residual names) is shown to hide a real
  device loss the same way, unmutated-and-verified.

---

## Audit Entry — 2026-07-28T19:16:08-07:00

**Review type:** On-demand — IP/licence compliance (OQ-M6) + first RAI pass
**Files reviewed:** `LICENSE`, `README.md`, `docs/DESIGN.md`, `docs/ENGINE.md`, `docs/PLATFORMS.md`, `docs/OP_COVERAGE.md`, `.squad/decisions/inbox/mouse-op-coverage-plan.md`
**External licences fetched:** llama.cpp `LICENSE`, ExecuTorch `LICENSE`, ONNX Runtime `LICENSE`

### Findings

| ID | Category | Severity | File | Finding | Status |
|----|----------|----------|------|---------|--------|
| RAI-001 | Credentials/secrets | ✅ Clear | All source files | No hardcoded credentials, API keys, tokens, or secrets found in any `.rs`, `.toml`, `.json`, `.md` file. | Resolved — no action |
| RAI-002 | IP Compliance — OQ-M6 | 🟢 Green | N/A | Reading llama.cpp MIT shaders as reference: fully permitted. Conditions for adaptation documented in `docs/THIRD_PARTY.md`. | Resolved — decision filed |
| RAI-003 | Platform honesty | 🟡 Advisory | `README.md` | README platform table lists Android and macOS without CI-coverage caveat. All physical hardware rows in `PLATFORMS.md` are marked "untested." User-facing README does not reflect this. Link is aware. | Open — Link to address |
| RAI-004 | Terminology | 🟡 Advisory | `docs/OP_COVERAGE.md` | "fusion whitelist" (2 instances). Policy prefers "allowlist". | Open — Mouse to address at next revision |
| RAI-005 | Performance claims | ✅ Clear | All docs | No published performance claims or speedup numbers found. Tier exit criteria are goals (design doc), not user-facing claims. README clearly states "Status: pre-implementation." | Resolved — no action |
| RAI-006 | Deceptive/ungrounded claims | ✅ Clear | All docs | All uncertain claims are marked ⚠️ Unverified. vulkan.gpuinfo.org data is correctly attributed with date. No hallucinated citations found. | Resolved — no action |

### Verdict Summary
- 🔴 Critical: 0
- 🟡 Advisory: 2 (RAI-003, RAI-004) — work proceeds, non-blocking
- 🟢 Green: 4 (RAI-001, RAI-002, RAI-005, RAI-006)

---

## Audit Entry — 2026-07-30T07:12:15-07:00

**Review type:** On-demand — silent-inference RAI verdict (R9 event, §10.0.1)
**Files reviewed:** `docs/DESIGN.md` (§8.9, §9.1.2, §9.1.3, §10.0, §10.0.1), `.squad/rai/policy.md`, `.squad/agents/Rai/charter.md`, `.squad/decisions/inbox/switch-ep-messenger-and-plant.md`, `.squad/decisions/inbox/tank-allocator-claimed-path.md`
**External state reviewed:** `DESIGN.md` Morpheus rulings at `557bf24` (§8.9.1 claiming gate, §9.1.3 compute_failures, criterion 10 DIVERGENT, criterion 11 NOT MET)
**Requested by:** Justin Chu

### Context

Phi-3.5-mini (2.2 GB, int4-quantised) run against `main` at `557bf24`:
161 `com.microsoft::MatMulNBits` nodes claimed and accepted by ORT, `compute_failures: 0`,
`dispatches_executed: 161`, test suite green. Vulkan output: logits `[0.0000, 0.0000]`, argmax 0.
CPU output: logits `[-13.0859, 13.0312]`, argmax 30751. Top-10 token overlap: 0/10.
Deterministic on Intel Iris Xe and RTX 4060. No error surfaced at any layer. Mouse has a kernel
fix in flight. Morpheus has ruled on the engineering gate (§8.9.1, §9.1.3, criteria 10 and 11).

### Findings

| ID | Category | Severity | Attaches to | Finding | Status |
|----|----------|----------|-------------|---------|--------|
| RAI-007 | Engineering defect — defective `MatMulNBits` kernel | 🟡 Advisory | The instance: one wrong fp16 kernel | Kernel produces zeroed output. Mouse fixing; criterion 10 tracks it. Instance is an engineering matter. Does not independently require RAI lockout — Morpheus's criterion 10 (NOT MET, DIVERGENT) already blocks M0 on this. | Tracked — Mouse and Morpheus own the fix |
| RAI-008 | Deceptive system claim — silent wrong inference output | 🔴 Critical | The class: architecture permits silently-wrong inference output at any layer with no disclosure | The EP reports success (`compute_failures: 0`, populated output tensors, no exception, no log line) while producing output entirely disconnected from the input. A user cannot distinguish this from "model performs badly." In an autoregressive decode loop, a single zeroed-logit dispatch produces an unbounded stream of wrong tokens, each fluent in form. No mechanism existed at R9 time to prevent this; no user-visible disclosure exists at the session layer. This property survives Mouse's fix — the next unproven kernel will be equally silent. | 🔴 OPEN — structural; fix is the proof ledger (criterion 11, NOT MET) plus a session-layer WARN |
| RAI-009 | Missing user-visible disclosure at runtime | 🔴 Critical | Architecture — absence of session-layer disclosure | `compute_failures: 0` is the only compute-path signal a user observes. §9.1.3 constrains its reading by prose and a `model_output_equivalence` verdict on the counters file. Neither is visible to a user who receives a session object and runs inference. There is no runtime WARNING when claimed forms carry UNMEASURED proof status. | 🔴 OPEN — requires runtime WARN mechanism at session creation |

### Verdict Summary
- 🔴 Critical: 2 (RAI-008, RAI-009) — **Reviewer Rejection Protocol applies to the architecture**
- 🟡 Advisory: 1 (RAI-007) — instance tracked by engineering criteria, non-blocking as a separate RAI matter
- New 🟢: 0

**Overall architectural verdict: 🔴** — the EP cannot ship op claims until the proof ledger (criterion 11) is implemented and a session-layer disclosure mechanism exists. The 🔴 does not block Mouse's kernel fix or Mouse's test work; it attaches to the shipping decision, not to the investigation.

**Falsifier for this verdict:** The 🔴 is discharged when: (a) criterion 11 is MET — no form is claimable without a ledger entry, verified by a planted `[unproven]` decline and a CI check; AND (b) a runtime WARN is emitted at session creation naming every claimed form whose proof status is UNMEASURED or whose last model-level verdict is DIVERGENT. An EP that satisfies both conditions has an instrument that would go red on the next silent-zeroing event — specifically, the ledger check would decline the unproven form before dispatch, and any future DIVERGENT result would demote the form automatically (§8.9.2 rule 4). The RAI concern evaporates when the instrument exists.

---

## Audit Entry — 2026-08-01T09:53:14-07:00

**Review type:** Recall — re-verdict RAI-008/009 against a materially changed system; fallback disclosure; performance-directive RAI dimension; llama.cpp licence boundary
**Files reviewed:** `docs/DESIGN.md` (current, last revised 2026-07-31T07:45:10-07:00; §8.9.4–8.9.7, §9.1, §10.0 all four amendments, §10.0.1 R9–R13, M0 table rows 2–3, 10–12, line 371, line 1108–1122 single-cluster bypass hazard), `.squad/rai/audit-trail.md` (my own 2026-07-30 entry), `.squad/agents/Rai/history.md`
**Requested by:** Justin Chu, explicitly against my own stated falsifier from RAI-008

### Context since the 2026-07-30 verdict

1. **Correctness evidence improved genuinely.** After Switch fixed `Allocator::alloc(size=0)` returning `None` on Phi-3.5's `[1,32,0,96]` KV-cache inputs, ORT's own profiling trace — an instrument this project does not own — recorded an attributed execution: one fused `VulkanExecutionProvider` island (~354 of 364 nodes) plus ~10 `CPUExecutionProvider` events matching Mouse's declines exactly, `argmax 30751` matching CPU, repeated across runs. This is real and I am not discounting it.
2. **The same window produced two more silent-fallback events**, both with the EP as proximate cause: a weight-cache OOM (`gpu-allocator failed to allocate 14155776 bytes`) causing a silent mid-session fallback, and the KV-cache `alloc(size=0)` case itself before the fix — `ORT printed EP_FAIL … Falling back` and raised nothing, `get_providers()` still listing `VulkanExecutionProvider` because the provider list is fixed at session creation. Fifth documented sighting of that exact line on this project. 50 KV-cache outputs were never written in the dirty-arena case, detectable only by cross-run divergence.
3. **Guard D now catches the fallback line itself in the test lane**, with a two-polarity control (it must state what it observed even on failure — `NameError` is not a detection, `0 Vulkan node events, providers seen: [CPUExecutionProvider]` is). This is a CI-lane instrument, not a production disclosure a user calling `session.run()` would ever see.
4. **Criterion 11 (the proof ledger) is confirmed scaffolding-only** — `DeclineCode::Unproven` and a list-only parser exist in `registry.rs`; there is no ledger, so nothing is looked up, and the three planted controls promised for it do not exist. Table status: "Not met — scaffolding only."
5. **New fact not yet in `DESIGN.md`: the single-cluster bypass now plausibly bites on our only real model.** §7's recorded hazard — `GetCapability` skips the net-benefit check (`partition::evaluate` / `retain_viable`) whenever there is exactly one surviving cluster, on the premise "there is no competing partition," which Morpheus already recorded as wrong (the competing partition is always CPU fallback) — was written when Phi-3.5 had 33 clusters. Phi-3.5 now converges to **one fused island**. `viable_islands_retained == 0` on this model is therefore structurally ambiguous between "the gate ran and rejected everything" and "the gate was bypassed," and today's evidence indicates the latter. This is a second gate, not the proof ledger, but it is the same failure shape RAI-008 names: a mechanism that would matter is not exercised on the one artifact we have.

### Findings

| ID | Category | Severity | Finding | Status |
|----|----------|----------|---------|--------|
| RAI-008 (re-verdict) | Deceptive system claim — silent wrong inference output | 🔴 **Stands, unsoftened** | My own falsifier required (a) criterion 11 MET AND (b) a session-creation WARN for UNMEASURED/DIVERGENT forms, both instrument-verified. (a) is false on the record: "Not met — scaffolding only," no ledger, no planted controls. (b) is also false: §8.9.7's session-creation INFO/WARN disclosure is *designed* but I have seen no artifact it produced — R10 applies to my own review exactly as it applies to Mouse's or Trinity's code, and "review of a mechanism is not complete until the reviewer has seen an artifact it produced" binds me too. Good correctness news does not discharge a disclosure gate; that is precisely the substitution R9 forbids — a verdict is not evidence for a *different* claim. **I am not softening this because the news is good.** | 🔴 OPEN |
| RAI-010 | Falsifier gap in my own RAI-008 ruling | 🟡 Advisory, self-correction | My 2026-07-30 falsifier (a)+(b) covered *claim-time* disclosure (proving a form before claiming it, warning at session creation). It did not anticipate *mid-session runtime* Compute() failure — the actual mechanism of both new incidents. A form can be correctly proven and claimed, pass every claim-time check, and still fail at runtime under a condition the proof did not cover (empty tensor shape, OOM), triggering ORT's silent internal fallback with no signal at any layer available today. My falsifier is necessary but was not sufficient; extending it below. | Extending RAI-008's discharge test — see below |
| RAI-011 | Second unwired gate on the only real artifact | 🟡 Advisory, flagged for Mouse/Morpheus, not independently blocking | The single-cluster bypass (§7, line ~1114) plausibly now excludes the net-benefit check on Phi-3.5's single fused island. Same shape as R10 (a mechanism true of one entry point, silent on another) applied to a different gate. Does not raise the severity of RAI-008 — it is evidence *for* the same class of finding, not a new class — but should be named so it is fixed alongside, not after, criterion 11. | Flagged — Mouse owns `GetCapability`/`retain_viable` wiring |

### Extended falsifier for RAI-008 (supersedes the 2026-07-30 version)

The 🔴 discharges only when **all three** hold, each verified by an instrument with a planted control, none by prose:
**(a)** Criterion 11 MET — no form claimable without a ledger entry, a planted `[unproven]` decline demonstrated declining in CI.
**(b)** A session-creation disclosure (§8.9.7) is observed to run — INFO naming every claimed form's proof key and ledger entry, WARN on any UNMEASURED/DIVERGENT form, INFO when zero nodes are claimed — verified by an artifact it produced, not by reading the code that would produce one.
**(c) [new]** A runtime disclosure fires when a *previously claimed* node's `Compute()` returns a non-OK status and ORT is about to fall back — a WARN through ORT's own logging sink, naming the node, the error condition, and that CPU re-execution follows — verified by a planted Compute-failure control (positive) and a normal successful run asserting the WARN does *not* fire (negative), the same two-polarity discipline Guard D now uses for the fallback line itself.

Any one of the three, alone, leaves a silent path a user can hit: (a) alone permits claim-time-proven forms to fail silently at runtime (the actual 2026-07-30→08-01 incidents); (b) alone tells a user what was claimed at start but not what failed mid-session; (c) alone catches runtime failures of already-claimed forms but does nothing to stop unproven forms from being claimed and dispatched in the first place. **The class-level hazard does not close until a user calling `session.run()` cannot receive a silently wrong or silently degraded answer without a signal reaching a channel they can observe**, at claim time and at run time both.

### Ruling 2 — fallback disclosure through ORT's logging sink

**Question:** should the EP itself WARN through ORT's logging sink when its `Compute()` fails and a fallback will follow?

**For:** a user who selected `VulkanExecutionProvider` and silently receives CPU-only inference has been given a different product without being told — this is §1.3's compatibility promise and §8.9's disclosure principle applied to a later point in the pipeline than either currently covers. It has now happened five times, twice since my last verdict, both times with the EP as proximate cause and zero signal at any layer a user could observe. Silence here is the load-bearing mechanism of RAI-008's hazard, not a side effect of it.

**Against:** an EP that logs loudly on every recoverable condition trains users to filter it out; if declined ops (the large majority of any real graph today — 258 dynamic-shape declines alone on Phi-3.5) each produced a WARN, the signal would drown in noise and the one WARN that matters would be read exactly as often as the 258 that don't.

**Where the line sits, mechanically, not in prose:** the two cases are mechanically distinguishable and must be treated differently.
- A node that was **never claimed** (declined at partition time) falls back to CPU *by design*; that is the plan, not a failure, and is already covered once, in aggregate, by §8.9.7's session-creation disclosure ("the EP claimed nothing" / claimed-count summary). No per-node WARN belongs here — this is the noise case above.
- A node that **was claimed** — the EP told ORT it would compute it — and whose `Compute()` then returns a non-OK status is a broken commitment at runtime. This is rare by construction (a claimed node already passed the claim-time proof gate) and is exactly the category that produced both new incidents. **This case must WARN, every time, with no opt-out**, because an EP reneging on a claim it already made is the one condition where "it happened before and was fine" is never a safe inference for the next occurrence.

**The mechanism, not the prose:** the panic/error guard around `Compute()` (line 371, already converting a Rust panic into `ORT_EP_FAIL`) is extended so that on any non-OK return it calls ORT's own logger (the session's registered `OrtLoggingFunction`/`Logger_LogMessage`, not this project's own log crate) at WARNING, before returning the status — naming the node name, op_type, the specific error condition (allocator/shape/etc.), and that CPU re-execution will follow. It must go through **ORT's** sink specifically, because that is the channel already carrying the "Falling back" line itself and the one channel a user with ORT logging configured is already watching; a WARN in our own private log is invisible to exactly the audience that matters. A planted control (force an `alloc(size=0)` or synthetic Compute error) must assert the WARN fires, and a normal successful run must assert it does not — the same two-polarity discipline Guard D was built under, because a WARN that cannot be shown *not* to fire on a good run is not a detector, it is a printed opinion.

### Ruling 3 — is there an RAI dimension to the standing performance directive?

**Not yet — and here is the trigger, named in advance rather than discovered after the fact.**

Morpheus's ruling (the directive "changes the calendar and not one gate") already closes the crudest version of this question: it forecloses the cheapest cheat (do less GPU work to win the ratio) by requiring a non-zero claimed count and an attributed `MATCH` before any timing figure is quotable at all. That protects *coverage* from the performance directive. It does not by itself say anything about *accuracy*, and the prompt's own framing is correct that accuracy is a second lever with the same shape: the tolerance budget (fp16 accumulation, max diff 0.031–0.035 against CPU on a ±13 range) is an accuracy decision, currently justified and gated by Trinity's sign-off (§9.1, "never widened to make a red test green," requires a note in the test).

That gate is real today and I am not finding a violation of it. The RAI dimension is *latent*, not present, and it becomes active at a specific, nameable point:

- **Trigger 1 — undisclosed widening.** Any change to a documented `rtol`/`atol` or accumulation-order tolerance that lands with Trinity's engineering sign-off in the commit but with no corresponding update to a **user-facing** accuracy statement (a precision/accuracy section in `PERF.md` or the README, analogous to the wall-clock-ratio disclosure Morpheus already mandates). Engineering sign-off answers "is this justified"; it does not answer "does the user know." Those are different obligations, exactly as a claim gate and a session disclosure are different obligations (§8.9.7).
- **Trigger 2 — device-conditional precision.** If a kernel variant or `accuracy_level` is ever selected *because* a device is slow (trading fidelity for speed differentially by hardware) rather than fixed per producer-at-version, a user's answer would depend on which GPU they own, silently. This is the accuracy analogue of `UNATTRIBUTED` — a number that is correct about the wrong world and looks identical to one that isn't.
- **The mechanism that would make the boundary visible, ready to propose the day either trigger fires:** a precision-disclosure obligation paired with the existing performance-disclosure obligation — every milestone report/benchmark carries the accuracy budget (tolerance, max diff, the device(s) it was measured on) beside the wall-clock ratio, not only in a test file. A report with the speed number and no accuracy number would be exactly as incomplete, under this rule, as one with neither — which is the same logic Morpheus already applied to the ratio itself.

**Until one of these two fires, this is Trinity's engineering gate, not my finding.** I am naming the trigger now, per R9, precisely so that if it fires, the observation is already specified and cannot be argued down to a one-off after the fact.

### Ruling 4 — llama.cpp licence boundary, for Switch (one paragraph, actionable)

llama.cpp is MIT-licensed, so its GLSL/SPIR-V shader source is legally readable and even legally copyable with attribution — the licence itself does not require Switch to avoid the code, and "study the structure, not copy code" is a project engineering-quality discipline, not a legal constraint. The boundary that matters is the one recorded in `docs/THIRD_PARTY.md` from the OQ-M6 ruling: **algorithms, tiling strategies and subgroup techniques cannot be owned (idea/expression dichotomy) — reading llama.cpp's packed-load and multi-accumulator *approach* and reimplementing the technique in Switch's own idiom for `q_gemv_matmul_nbits_f16` produces independent work with no attribution obligation.** Only if Switch adapts substantial *expression* — copies recognisable GLSL structure, control flow, variable names or comments from llama.cpp's own shaders — does the result become a derivative work; MIT permits this freely, and the only consequence is an entry in `docs/THIRD_PARTY_NOTICES.md` (already templated at `docs/THIRD_PARTY.md` §10) carrying llama.cpp's copyright notice and licence text. The practical test to apply while working: *could you write this shader without the original open next to you?* If yes, it's independent, no notice needed. If the structure requires the original to reproduce, it's derivative, MIT-permitted, and needs the notice. Nothing here blocks Switch's optimisation work; crossing from "study" to "copy" costs a documentation entry, never a legal risk.

### Verdict Summary
- 🔴 Critical: 1 (RAI-008, re-verdict — **stands, unsoftened, extended falsifier**)
- 🟡 Advisory: 2 (RAI-010 self-correction of my own falsifier gap; RAI-011 second unwired gate flagged for Mouse)
- Rulings issued, non-severity: fallback disclosure mechanism (Ruling 2); performance-directive trigger, not yet fired (Ruling 3); licence boundary restated (Ruling 4)

**Overall verdict: 🔴 unchanged.** Correct output on three consecutive runs is real progress and does not discharge a disclosure obligation — those are different claims, and RAI-008's falsifier was written to be about disclosure, not about correctness, specifically so that this day would not be mistaken for a discharge.

**Falsifier for RAI-008 (extended):** see the three-part test above — (a) criterion 11 MET with planted CI control, (b) session-creation disclosure §8.9.7 observed to run via a produced artifact, (c) a two-polarity-verified runtime WARN on claimed-node `Compute()` failure. **Falsifier for Ruling 2:** the mechanism is falsified if a planted Compute-failure control fails to produce the WARN, or if the WARN fires on a normal successful run (false positive) — either observation means the mechanism is not what this ruling requires. **Falsifier for Ruling 3:** either named trigger firing — an undisclosed tolerance widening, or a device-conditional precision choice — converts this from "not yet" to an active finding; their absence is what keeps it advisory rather than critical.

---

## Audit Entry — 2026-08-02T03:21:20-07:00

**Review type:** Fresh pass at Justin's request via coordinator — re-status RAI-008 against material progress since 2026-08-01; four new RAI-surface items; sanity check of the coordinator's own conduct this session
**Files reviewed:** `docs/DESIGN.md` (current, last revised 2026-08-02T02:02:23-07:00 — criterion 10 closing ruling, criterion 11 table row with the four discharge conditions and their status, criterion 12's four-conjunct enumeration, §10.0 obligation 8/8b the device-state companion, wiring census with `broken_commitment_warn` and `net_benefit_gate`), `.squad/decisions/inbox/morpheus-criterion-10-closes.md`, `.squad/decisions/inbox/trinity-criterion11c-and-equivalence-authority.md`, `.squad/decisions/inbox/niobe-per-dispatch-is-not-a-tenancy-signature.md`, `git log` (main at `57d7018`), branch `evidence/criterion-rerun-null-witness`
**Requested by:** Justin Chu, via coordinator

### Context

Criterion 10 (model-level correctness, attributed) closed tonight on the condition written in
advance: real Phi-3.5, three consecutive attributed `MATCH` runs in one session, both devices,
`executed_by` from ORT's own profiler (3 `VulkanExecutionProvider` island executions vs 24 CPU),
`cross_run_identical_to_run1 = true` on all three. Main: Rust 459/0 across ten targets, Python
union 479/0, clippy green, 80 lane checks, both device wiring censuses `unwired: []`. Criterion 11
(the proof ledger) is **closer, not closed** — Morpheus refused the cheapest satisfaction ("a ledger
generated from the claim table would make the check unable to fail") and wrote four discharge
conditions before the tally could move: (a) provenance and (d) the three-token miss path are DONE
(Mouse); (b)(iii) the digest-refusal control is DONE (Mouse); (c), Trinity's — `ledger_hits` shown
to move with its input — is still open. Criterion 12 (wiring census) was over-claimed as closed by
the coordinator and corrected by Morpheus the same night, on the coordinator's own report.

### RAI-008 — re-status against the extended (2026-08-01) falsifier

My falsifier from the last pass has three parts. Checking each against tonight's artifact, not
against the prose describing it:

| Part | Requirement | Status tonight | Evidence |
|---|---|---|---|
| (a) | Criterion 11 MET — no form claimable without a ledger entry, planted CI control | **Still NOT MET** | Table row 11: "Not met — scaffolding only," open specifically on Trinity's (c). Three of four discharge conditions landed; the row is correctly still open, and correctly not closed by the agent who supplied the artifacts (Morpheus's own ruling on this exact temptation, restored this session after a merge nearly dropped it) |
| (b) | Session-creation disclosure (§8.9.7) — INFO per claimed form + proof key/ledger entry, WARN on UNMEASURED/DIVERGENT | **Still NOT MET, in flight** | §10 criterion-list text still reads "Owner: Tank at session creation" with no artifact produced yet; the escape-hatch WARN (criterion 11 item (c) in the older enumeration) is a narrower, already-built cousin — it discloses only when the hatch is manually enabled, which is not this obligation |
| (c) **[new since last pass]** | Runtime WARN on a *claimed* node's `Compute()` failure, two-polarity control | **MET** | Wiring census now carries `broken_commitment_warn` (Tank): planted `ONNXRUNTIME_EP_VULKAN_FORCE_COMPUTE_FAILURE` reads `channel='ORT_SINK' broken_commitments=1 fault_injection='ACTIVE' ort_sink_warn_lines=1` against a clean run reading `channel='UNOBSERVABLE' broken_commitments=0` — exactly the two-polarity discipline my falsifier asked for, and it is in the census, not behind a flag |

**Verdict: RAI-008 remains 🔴 OPEN. It is not discharged — two of three parts are unmet — and I am
recording, without softening either direction, that one part I added last session has actually
landed, verified, in the shape I specified.** This is the correct way for a 🔴 to move: not by the
news being good, but by an instrument I named actually going green. I am glad to write that
sentence for once instead of a caution against writing it.

**RAI-011 (single-cluster bypass) — update.** The census now reports `net_benefit_gate: EVALUATED
clusters_seen=1 evaluations=1 bypasses=0 sole_island_overrides=1 viable_islands_retained=0` — bypass
and override are now separate, typed fields instead of sharing one `0` (R12; this is the exact fix
RAI-011 asked for). **Downgrading RAI-011 to resolved-as-instrumented**: the ambiguity I flagged
(bypassed vs. all-rejected reading the same) is gone; whether the gate's *economics* are right on a
single-island graph is Mouse's and Morpheus's question now, not a disclosure gap.

### New RAI surface, traffic-lighted

**1. The `CLAIM_UNPROVEN` escape hatch — 🟡 Advisory, not 🔴.** The hatch cannot silently claim
unproven forms: it takes a list of exact proof keys only (a planted `*`, `1`, and bare op-type are
all rejected — C1's enforcement shape), logs at WARN naming every enabled key, and
`unproven_forms_enabled` fails `epctl --check-counters` without an explicit `--allow-unproven`. That
is real disclosure infrastructure, not a name on a settings screen. **The residual risk is real and
worth naming precisely: a user who is annoyed by the WARN can pass `--allow-unproven` once and
silence it for good**, at which point the escape hatch behaves exactly like the pre-ledger world for
every key on that list. This is not a defect in the mechanism — an escape hatch that could not be
disabled would not be an escape hatch — but it means the ledger's guarantee is only as strong as the
friction around `--allow-unproven`. **WHAT would raise this to 🔴:** if `--allow-unproven` ever
becomes settable via a config file, environment default, or anything other than an explicit
per-invocation flag a human types, the friction is gone and the hatch becomes the silent path again.
**Recommendation, not a blocker:** `--allow-unproven` should require a reason string logged beside
the enabled-keys line (mirrors this project's own `DeclineCode` discipline — every decline already
carries a reason, an override should carry one too), and CI should refuse to green a lane that uses
it, exactly as `epctl` already refuses without the flag. Owner: Tank, non-blocking.

**2. Is there an RAI obligation on performance claims for an inference accelerator, beyond
Morpheus's engineering gate? Yes, and I recommend writing it into policy rather than leaving it as
one project's discipline.** Morpheus's obligation (no timing figure quotable outside `MATCH`, the
device-state companion, `STEADY_UNCERTIFIED` as a fourth state, the two-polarity Guard-D pattern
applied to `gpu_steady_tail`) is the engineering version of a principle that is squarely RAI's: **an
accelerator's performance claim is a claim a user acts on** — it informs purchase decisions, capacity
planning, and trust in every other number the project publishes. `baseline_certified` being the
cleanest run by every dispersion measure the project owns and still 21.4× wrong, caught only by the
device-state companion, is the sharpest specimen available anywhere in this project of *engineering
rigor discharging what would otherwise be an RAI failure* — a number that looked unimpeachable by
every test available and was wrong by more than an order of magnitude. **My recommendation: add a
policy line under Deceptive Patterns in `.squad/rai/policy.md` — "A performance figure for an
accelerator or ML system is an ungrounded factual claim (🔴) unless it carries the record of the
instrument that could have contradicted it."** This does not change today's practice, which already
meets that bar; it makes durable what is currently one engineer's vigilance, so that the standard
survives a personnel change or a rushed milestone in a way a discipline that lives only in one
person's rulings does not. **Trigger for treating this project's own practice as falling short:** any
future benchmark quoted without its device-state companion or its `MATCH`/attribution frame.

**3. Intel iGPU performance is permanently uncertifiable on the hardware in this project's matrix,
and that is a disclosure gap nobody has named as one yet — 🟡 Advisory, escalating to a hard
pre-publication gate.** §10.0's obligation 8 already establishes that a platform with no clock
telemetry reports `STEADY_UNCERTIFIED` forever, and names Intel's iGPU as *more*, not less, exposed
(shared power budget with loaded CPU cores). If that is durable — no counter, no WMI class, no tool,
no Vulkan query surfaces a usable device clock on this hardware — then **every performance figure
this project will ever publish is NVIDIA-only, permanently, and a user on Intel silicon receives an
EP whose speed characteristics on their own hardware can never be stated.** This is not currently a
violation because no performance figure has shipped publicly yet (Morpheus's own withdrawal
discipline has kept it that way). **It becomes a blocking disclosure requirement, not merely
advisory, the day any performance figure ships publicly (README, release notes, PERF.md marked
final rather than in-progress):** that artifact must state, explicitly and not in a footnote, that
timing figures are certified on NVIDIA only and that Intel iGPU performance is and will remain
uncertified absent new instrumentation — the same prominence Morpheus already requires for the
wall-clock ratio itself (§10.0 disclosure obligation, "never omitted, including when it is worse
than 1.0"). Owner: Link to confirm the "permanently" claim holds after a genuine cross-platform
search (WMI, `NVML`-equivalent, `VK_EXT_calibrated_timestamps` extensions, driver-specific counters)
before this is written as a permanent fact rather than a current one; Niobe/PERF.md to carry the
disclosure text once a figure ships.

**4. `rust/src/trace.rs` ownership — 🟢 Green, closed by the coordinator's own action.** Assigning
ownership tonight to the file holding the project's only sanctioned tick conversion, after Link's
static screen proved it is the single entry point for raw ticks in the tree, is exactly the right
remedy and needs no further finding from me. One process recommendation, non-blocking: this
project's ownership assignments currently live in prose (rulings, this session's message) rather
than in a single artifact a new contributor or a script could read. A `CODEOWNERS`-shaped file
(or a table in `DESIGN.md` itself) would make "who owns this file" a lookup rather than a memory,
which is the same class of improvement R10 asks for applied to governance instead of mechanisms.

### Sanity check of the coordinator's conduct this session

- **Resolving merge conflicts inside other agents' authored documents, then disclosing after rather
  than asking before — 🟡 Advisory, not a violation, with a process refinement.** The one concrete
  instance reviewed (the withheld-tally sentence, restored 2026-08-02T01:42:02) shows the right
  outcome: the coordinator's own ruling notes "neither side was a superset," he "correctly declined
  to splice" prose, told Morpheus, and the sentence was restored with the underlying finding
  unchanged throughout (Mouse's evidence was never disputed — only whether the tally could move).
  **Telling-after was sufficient in this instance because nothing load-bearing was lost for longer
  than one review cycle and the correction was public.** I would not generalize "tell after" as the
  standing rule, though: the specific risk is a conflict resolution that *silently* drops a
  dissenting verdict rather than a supporting sentence — had this been a rejected 🔴 finding rather
  than a restored sentence, telling-after would be too late by definition, because the finding would
  already have shipped without it. **Recommendation:** for merge conflicts inside a named agent's
  *verdict* text specifically (not supporting prose), stop and flag before completing the merge,
  because a verdict is the one artifact class where "ask forgiveness" and "the thing already shipped
  wrong" are the same event.
- **Preserving unclaimed artifacts on `evidence/criterion-rerun-null-witness` rather than committing
  to main or discarding — 🟢 Green.** Textbook handling of a finding whose owner had not ruled:
  neither buried nor prematurely promoted to a claim. No finding.
- **Pushing with three known, precisely-attributed, in-flight red items rather than holding
  twenty-one verified commits off origin — 🟢 Green.** This matches the project's own established
  convention throughout `DESIGN.md`'s M0 table — partial/open criteria are the normal state of a
  transparently tracked milestone, not a defect to hide behind a hold. The distinguishing test is
  the one this project already applies everywhere else: is each red item named, owned, and does its
  status say so, rather than being silent or claimed green. It is. No finding.
- **Over-claiming criterion 12 as closed, then correcting publicly — 🟡 Advisory, self-corrected,
  and worth naming as a pattern rather than an incident.** This is a real instance of exactly the
  witness/discharge conflation Morpheus's own ruling names ("he held a witness and read it as a
  discharge"), and it is at least the second time this conflation has occurred on this project by a
  different name (RAI-008/009's own history has a structurally identical shape: a correct run being
  read as a discharge of a disclosure obligation). **The self-correction, same session, is the
  standard this project has set for itself and I am not treating it as a violation** — but I record
  it because a pattern repeating across two different people (Mouse's union-defect specimen,
  the coordinator's criterion-12 specimen) in one week is worth a durable check rather than a
  standing vigilance requirement on any one person. **Recommendation:** a small doc-consistency
  script asserting that any prose claiming a criterion "closed"/"met" agrees with that criterion's
  own table cell before a status update is posted to the team — mechanical, cheap, and it converts
  a recurring human error into an `ERROR(instrument)` the register already has a name for.

### Verdict Summary
- 🔴 Critical: 1 (RAI-008 — **remains open, unchanged severity, one of three falsifier parts now
  genuinely discharged**)
- 🟡 Advisory: 4 new (escape-hatch friction; performance-claim policy recommendation; Intel
  permanent-uncertifiability disclosure, escalating to a hard gate on first public figure;
  witness/discharge conflation as a recurring pattern) + RAI-011 downgraded to resolved-as-instrumented
- 🟢 Green: 3 (branch preservation; pushing with attributed red items; trace.rs ownership assignment)
- Overall: **no 🔴 escalation this pass.** Nothing reviewed tonight — the coordinator's conduct
  included — rises to a genuine 🔴 beyond the one already open and already tracked.

**Falsifiers, named per new item:** escape hatch (2) — falsified into 🔴 if `--allow-unproven`
becomes settable other than as an explicit per-invocation flag; performance-claim policy (3) —
falsified (i.e., shown unnecessary) if this project ever quotes a figure without its companion and
suffers no consequence, which would mean the discipline was never load-bearing; Intel disclosure
(4) — falsified if Link's search finds a usable device-clock surface on Intel after all, in which
case "permanent" becomes "not yet solved" and the obligation softens accordingly; witness/discharge
pattern — falsified if the doc-consistency script, once built, never fires, meaning the human
discipline alone was already sufficient.

---

## Audit Entry — 2026-08-03T07:40:00-07:00

**Review type:** Recall via coordinator — rule on RAI-008(a) (Tank's CI-plant repair, `137e40f`);
judge the honesty of the proof-ledger decline message together with Link's counters finding; spot
check `docs/DESIGN.md` §0 for overstatement
**Files reviewed:** `rust/src/registry.rs` (`Ledger::state_for`, `parse_ledger`, `claim_decision_audited`),
`rust/src/counters.rs`, `rust/src/disclosure.rs`, `tests/ops/test_proof_ledger.py`,
`bench/results/link-linux-downstream/{counters-linux.json,pytest-linux.log}` (Link's worktree, live),
`bench/results/_ledger_counters.json` (main, post-`137e40f`), `docs/DESIGN.md` §0/§10 (criterion 11 row),
`.squad/decisions.md` Round 10, `.squad/agents/tank/history.md`
**Requested by:** Justin Chu, via coordinator

### Findings

| ID | Category | Severity | Finding | Status |
|----|----------|----------|---------|--------|
| RAI-008(a) re-check | Falsifier discharge | 🔴 stays OPEN | Tank's `137e40f` repairs the CI plant (rot fixed, non-vacuous membership assertion added, 14/0) — real, verified, correctly-scoped maintenance. It does not discharge RAI-008(a), whose falsifier requires Criterion 11 MET as a whole. Criterion 11's DESIGN.md row is still NOT MET, open on an unrelated defect the plant fix does not touch: no predicate reads `LedgerEntry.device`, so `wiring_census-dev1.json` reads 6 forms proven on a device (`device0`, a selector ordinal) nothing has ever proven anything on. | Not credited — criterion 11 must close on the device-attribution defect, not on the plant |
| RAI-012 | Deceptive/misleading diagnostic — ledger-fault decline message names the wrong subject | 🟡 Advisory, named 🔴 trigger | Confirmed live in the tree, not hypothetical: `counters-linux.json` reads `ledger_faults: 97, ledger_entries: 0`; `pytest-linux.log` has 0× the true-cause line (`proof ledger fault: ...`, from the `OnceLock` in `registry::ledger()`) against 42× the generic decline text (`no proof ledger entry ... nothing has proven it correct on this form`), which is false in both clauses — 97 entries exist, some proving these exact forms elsewhere, and the true cause is a whole-ledger parse/digest fault. `Ledger::state_for` blankets every key to `Unproven` when `self.faults` is non-empty, bypassing the `SubjectChanged` branch built to name a frame mismatch accurately. Fail-safe output is preserved (decline is still correct), so this is a diagnostic-honesty defect, not RAI-008's silent-wrong-output class. Not Linux-specific: the blanket applies on the certified Windows lane too, on any whole-ledger fault (corrupted install, hand-edited file, `ONNXRUNTIME_EP_VULKAN_LEDGER_FILE` digest mismatch) — it just hasn't fired there yet because the Windows ledger currently parses cleanly. | 🟡 OPEN — owner Mouse (message construction + `OnceLock`/`attach_default_ort_logger` ordering). Escalates to 🔴 if it ships unfixed on Windows and fires against a real user. |
| RAI-013 | Disclosure emitted but below default visibility — does not discharge §8.9.7 | 🟡 Advisory | `_ledger_counters.json` (main, post-`137e40f`): `session_disclosure_infos: 1`, `session_disclosure_infos_to_ort_sink: 0`, `session_disclosure_info_channel: "BELOW_ORT_THRESHOLD"`, `ort_sink_severity_threshold: "WARNING"`. The self-report is honest and itself good instrumentation (R9/R13 discipline) — credited. But a normal user, ORT's default logging severity unchanged, never sees this INFO line: "we emitted it and can prove it didn't land" is not "the user was told," the same substitution my 2026-08-02 ruling named on a different artifact. | 🟡 OPEN — owner Tank (raise this disclosure to WARN, or Morpheus documents the gap in DESIGN.md §0.2 explicitly) |
| — | `docs/DESIGN.md` §0 accuracy | 🟢 Green, one omission | Spot-checked `oracle_outputs_within_tolerance: 62` and `argmax_cpu`/`argmax_vk: 30751` against `criterion10-dev0.json` — matches exactly. No overstatement found in the sampled claims; the section's existing self-correction history (withdrawn "bit-identical," withdrawn `x/x` weight-amplification identity) stands. §0.2 omits both RAI-012 and RAI-013's current state, which belong there by its own stated purpose. | 🟡 minor — Morpheus to add one line each once Mouse's/Tank's fixes land or are scoped |

### Verdict Summary
- 🔴 Critical: 1 (RAI-008 — **remains open; (a) explicitly NOT credited by this repair, (b) and (c)
  stand as previously credited**)
- 🟡 Advisory: 2 new (RAI-012 ledger-fault decline message; RAI-013 unseeable INFO disclosure) + one
  minor doc-omission note on DESIGN.md §0.2
- 🟢 Green: Tank's CI-plant repair credited as real maintenance, not as criterion closure; DESIGN.md
  §0's sampled numeric claims check out against cited artifacts
- Falsifiers: RAI-012 is falsified-upward (🔴) by an unfixed occurrence on the Windows lane against a
  real user; RAI-013 is falsified-downward (discharged) by either the severity change or the doc
  addition landing.
