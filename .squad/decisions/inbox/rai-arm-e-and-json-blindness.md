### 2026-08-04: Rai — ruling on Tank's RAI-013 arm E submission, plus RAI-014 (new) and RAI-015 (new)

**By:** Rai

**RAI-013 stays closed, discharged as previously ruled.** Tank's arm E submission
(`.squad/decisions/inbox/tank-rai-013-arm-e-submitted-not-credited.md`) asked which of two readings
applies to his own repair. I already ruled reading (a) — discharged to the boundary of what a
process can witness about itself, with the residual a **bound**, not a gap — on my own independent
probe (`probe_disclosure_reachability.py`, both devices, both polarities). His arm E is a third
independent instrument converging on the same boundary Morpheus has since named THE SELF-WITNESS
BOUND (§8.9.23(5)). It corroborates; it does not reopen or move any tally. **He was right to submit
rather than claim** — this is the third time (RAI-008(a), RAI-013's original repair, now this) he
has declined to rule on his own work, and I'm recording that as a standing precedent worth crediting
explicitly, not just three isolated correct calls.

**New finding — RAI-014, 🟢 VERIFIED FIXED, independently reproduced by me (would have been 🔴
live).** `check_device_loss.py` was blind to a device loss encoded as UTF-16LE inside a JSON string
(NULs serialize as `\u0000`, invisible to a raw-NUL-stripping scan). I built my own mutation from the
real, unmodified `phi35_kv_chain-ctx4096-BOTH-dev0.json` — not Tank's prepared file — and confirmed
pre-fix `PASS`/exit 0, post-fix `FAIL(condition=device_lost_reported)`/exit 1, signature record
unaffected, negative control 20/20 arms firing including both the PLANTED and REPLAYED arms for this
exact defect. This is a safety-relevant instrument reporting a hazard absent when it occurred — the
sharper failure mode relative to RAI-012 (misattributed true positive vs. this false negative) —
hence its own number rather than folding into RAI-012. Credited 🟢 because it was caught, fixed, and
verified before any downstream claim was built on the blind window (Tank's own withdrawal of the
"30-consecutive-clean" figure in the same commit is the proof of that sequencing).
**Recommendation, non-blocking:** this is now the third defect in this project traceable to the same
root cause (ORT's UTF-16LE output meeting a narrow-encoding-assuming reader) — worth promoting Tank's
own rule ("an instrument that has never been shown a positive it does not catch has not been
characterised") from his decision record into `policy.md`/`DESIGN.md` as a standing requirement for
new scanners.

**New finding — RAI-015, 🟡 Advisory, named 🔴 trigger.** Confirmed live in
`evidence/proof_ledger.jsonl`: all four `Conv` entries render the key's variant component as
`metadata` (documented to mean "no shader") while carrying `"shaders":["conv_f32"]` with a real
digest in the same record — `registry::form_is_provable` never consults the kernel table for this
row. This is RAI's subject, not only Morpheus's: `form_is_provable` is the gate behind the
session-time "proven" claim a user reads, and a key that denies a shader exists while one is recorded
is the same silent-wrong-claim shape as RAI-008/009, arrived at through the proof ledger rather than
through a kernel's output. 🟡 today because the wrong reasoning path currently lands on a harmless
answer (one shader, one truth); **escalates to 🔴 the moment a second `Conv` variant is registered
before the ledger is repaired**, because at that point the mechanism would certify an unconsulted
shader as proven. Adopting Morpheus's §8.9.23(3) remedy as the fix for both the correctness and RAI
readings of this defect — the variant component must be named by the code that dispatches
(`translate`), not by an unpopulated kernel table. Owner: Mouse, same deadline Morpheus already set.

**Morpheus's §8.9.23 blind-axes disclosure mechanism satisfies my earlier `Conv` 🟡.** Confirmed not
yet implemented on fresh `main` — ruled but not shipped. The design is correct: `blind_axes` field on
`OpSpec`, rendered into the disclosure line, with the explicit clause that those axes are covered by
a CI-time suite and by nothing in the reader's own session. That second clause was the part I was
watching for and it is present. Remains open only as an implementation item, owner Mouse.

**`heapBudget` non-fix — out of RAI scope, no finding number.** No user-facing claim shipped (Tank
did not ship the change), so there is nothing for RAI to review. Noted as good self-witness
discipline applied prospectively, same family as items above, but not an RAI item.

**Verification performed this pass:** `cargo test --lib` → 553 passed / 0 failed / 4 ignored
(independently reproduced, matches coordinator's stated baseline). Reproduced Tank's mutation myself
from the real committed record, not his prepared fixture. Ran `ci/negative_control_device_loss.py`
live: 20/20. Read `evidence/proof_ledger.jsonl` directly for the `Conv` keys. Confirmed
`blind_axes` absent from `registry.rs`/`disclosure.rs` on fresh `main`.

**Full detail in `.squad/rai/audit-trail.md`, Audit Entry 2026-08-04T08:30:00-07:00.**
