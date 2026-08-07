"""Criterion 11 / §8.9 — the proof ledger gate, verified by a planted decline.

WHY THIS FILE EXISTS
--------------------
RAI-008(a), re-verdicted unsoftened in RAI-010, requires that "no form is claimable without a
ledger entry" be verified by **a planted `[unproven]` decline and a CI check** — not by reading
`registry.rs`. R10 is the reason: *the falsifier for "X is wired" is an artifact X produced whose
content varies with its input.* This file supplies the input variation.

THE PLANT
---------
`evidence/cases/sub_f16_dyn_unproven.onnx` is a form the EP has a kernel for and the ledger has
no entry for. Its sibling `evidence/cases/sub_f16.onnx` is the same op, same dtype, same graph —
differing in exactly one key component, `shape_class` — and it **is** proven. The pair is the
control:

  * if the runtime-extent arm is claimed, the gate is not gating;
  * if the static arm is declined, the gate is declining unconditionally and is equally useless.

THE CONTROL MOVED ON 2026-08-02, AND IT MOVED BECAUSE IT FIRED
--------------------------------------------------------------
It was `mul_f16_unproven` against `mul_f32`, differing in dtype. Populating the ledger for the op
suite added a `mul_f16` case — the same form as the control — and this file went red in the lane
with *"the gate is not gating"*. It was right, and it was the only thing that noticed.

The control was moved rather than the proof withdrawn, because keeping a whole dtype of a core op
permanently unclaimable to protect one file name trades real coverage for a name. What was added
instead is the cheap check that was missing: `ledger_case_models.PLANTED_CONTROL_KEY` is declared
once, `gen_proof_ledger.py` **refuses** to write an entry for it, and `--check` fails if it
appears. A control whose only guard is that somebody reads the case list is a control that will be
disarmed again.

`shape_class` is a better discriminator than dtype for this pair, incidentally: it is the
component that separates decode from prefill, which is where §8.7 says a path difference lives.

A one-armed version of this test would pass against a gate that returned `false` for everything.
This is Switch's `arms_must_differ` applied to claiming: two arms, and they must disagree.

NOT BEHIND `#[ignore]`, NOT `xfail`
-----------------------------------
Per Rai: *"a control that must be opted into is not in the lane."* Every test here runs in the
default `pytest tests/ops` invocation. The only skip condition is the absence of a Vulkan device,
which is the same skip the rest of the suite uses and is reported as SKIP, never as PASS.

OWNERSHIP NOTE
--------------
`tests/` is Trinity's. Criterion 11's row assigns its planted controls to Mouse, so this file is
written by Mouse and flagged in `.squad/decisions/inbox/mouse-proof-ledger.md`. Trinity owns the
census; nothing here modifies `test_wiring_census.py`.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import subprocess
import sys
from contextlib import redirect_stdout

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
CASES = REPO / "evidence" / "cases"
LEDGER = REPO / "evidence" / "proof_ledger.jsonl"
TOOLS = REPO / "rust" / "tools"

sys.path.insert(0, str(TOOLS))


# ---------------------------------------------------------------------------
# The ledger artifact itself — checks that need no GPU
# ---------------------------------------------------------------------------


def _ledger_lines() -> list[str]:
    return [l for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_ledger_artifact_exists_and_parses():
    """The ledger is a real file with a header and zero or more entries.

    R12 distinguishes UNOBSERVABLE (no ledger in this frame) from UNWIRED (a ledger nothing
    consults). This test establishes which frame we are in: if it fails, every downstream
    reading is UNOBSERVABLE and none of them is a detection.
    """
    assert LEDGER.is_file(), f"no ledger at {LEDGER}; every gate reading below is UNOBSERVABLE"
    lines = _ledger_lines()
    header = json.loads(lines[0])
    assert header.get("__ledger__") == 1, "first line must be the ledger header"
    assert header["entry_count"] == len(lines) - 1, (
        f"header claims {header['entry_count']} entries, file has {len(lines) - 1}"
    )
    for line in lines[1:]:
        entry = json.loads(line)
        for field in ("key", "verdict", "artifact", "artifact_sha256", "device"):
            assert field in entry, f"ledger entry missing {field!r}: {line}"


def test_ledger_admits_only_attributed_matches():
    """§8.9.2 rule 4 — only MATCH proves.

    `DIVERGENT` demotes, `UNATTRIBUTED` (the EP executed nothing) proves nothing, and `ERROR`
    (the instrument, not the kernel) neither proves nor demotes. The generator mechanises
    "demote" as "do not emit", so a non-MATCH verdict must never appear in the file.
    """
    for line in _ledger_lines()[1:]:
        entry = json.loads(line)
        assert entry["verdict"] == "MATCH", (
            f"ledger contains a non-MATCH entry, which would claim a form on evidence that does "
            f"not support it: {line}"
        )


def test_ledger_digest_detects_a_hand_edit():
    """A hand-edited ledger fails its own digest.

    This is a checksum, not a signature: it catches the careless edit, not the determined forger.
    The defence against forgery is `--check` re-running the evidence artifacts by sha256. Stated
    plainly here so no reader mistakes the guarantee for a stronger one.
    """
    import gen_proof_ledger as g

    lines = _ledger_lines()
    body = "".join(l + "\n" for l in lines[1:])
    digest = f"{g.fnv1a64(body.encode('utf-8')):016x}"
    assert digest == json.loads(lines[0])["content_fnv1a64"], (
        "ledger content does not match its header digest — the file has been edited by hand"
    )

    tampered = body.replace("MATCH", "MATCH ", 1) if "MATCH" in body else body + "x\n"
    tampered_digest = f"{g.fnv1a64(tampered.encode('utf-8')):016x}"
    assert tampered_digest != digest, (
        "the digest did not move when the content did — it is not a digest"
    )


def test_ledger_has_no_duplicate_keys():
    """Two entries for one key means one of them was never consulted."""
    keys = [json.loads(l)["key"] for l in _ledger_lines()[1:]]
    assert len(keys) == len(set(keys)), (
        f"duplicate proof keys in the ledger: "
        f"{sorted(k for k in set(keys) if keys.count(k) > 1)}"
    )


def test_no_ledger_key_is_a_wildcard():
    """§8.9.4 rule 1 applies to the ledger, not only to the escape hatch.

    A key that means "everything" is exactly as dangerous when a generator writes it as when an
    operator types it.
    """
    import gen_proof_ledger as g

    for line in _ledger_lines()[1:]:
        key = json.loads(line)["key"]
        assert key.count("/") >= 5 and "::" in key, f"malformed or over-broad ledger key: {key!r}"
        assert key not in ("*", "1", "all", "true", "yes"), f"wildcard key in ledger: {key!r}"


# ---------------------------------------------------------------------------
# The planted decline — needs a device
# ---------------------------------------------------------------------------


def _discover(model: pathlib.Path):
    """Run the model with the claim log armed and return (unlockable_keys, census)."""
    import gen_proof_ledger as g

    return g.discover_keys(str(model))


@pytest.fixture(scope="module")
def _ledger_keys() -> set[str]:
    return {json.loads(l)["key"] for l in _ledger_lines()[1:]}


def test_planted_unproven_form_declines(require_vulkan, _ledger_keys):
    """THE PLANT. A form with no ledger entry must decline with `[unproven]`.

    The plant is **imported, never spelled**: `ledger_case_models.PLANTED_CONTROL_CASE` is the
    single declaration, and this test derives its model path from it. A planted control names a
    condition the team is working to eliminate, so it will rot by design — it already moved once,
    from `mul_f16_unproven` to `sub_f16_dyn_unproven`, and on that day three separate readers were
    still spelling the old name. A literal here would have made this file the fourth.

    The prediction, restated against the current plant rather than the one it replaced: the
    declined key is the runtime-extent f16 `Sub` form, the decline code is `unproven` and nothing
    else, and the model still computes correctly because ORT falls back to the CPU EP.
    """
    from ledger_case_models import PLANTED_CONTROL_CASE, PLANTED_CONTROL_KEY

    model = CASES / f"{PLANTED_CONTROL_CASE}.onnx"
    assert model.is_file(), f"the planted control model is missing: {model}"
    assert PLANTED_CONTROL_KEY not in _ledger_keys, (
        f"the plant {PLANTED_CONTROL_KEY!r} has acquired a ledger entry, so it is no longer "
        "unproven and this arm would pass against a gate that claims everything. Move the plant "
        "rather than deleting this assertion."
    )

    unlockable, census = _discover(model)

    assert census["records"] > 0, (
        "the claim log recorded no decisions — ERROR(instrument), not a detection: the EP never "
        "reached claim_decision, so this run says nothing about the gate"
    )
    assert unlockable, (
        f"the planted unproven form was NOT declined for want of a proof. Census: {census}. "
        f"The gate is not gating."
    )
    for key in unlockable:
        assert key not in _ledger_keys, (
            f"key {key!r} is in the ledger yet was reported as needing a proof — the lookup and "
            f"the artifact disagree"
        )
    assert any(
        "f16" in k and "Sub" in k and "/runtime-extent/" in k for k in unlockable
    ), (
        f"expected the planted runtime-extent f16 Sub form among the declines, got {unlockable}"
    )


def test_proven_form_is_claimed(require_vulkan, _ledger_keys):
    """THE OTHER ARM. A form WITH a ledger entry must be claimed.

    Without this, `test_planted_unproven_form_declines` would pass against a gate that declined
    everything — a perfectly stable, perfectly wrong answer.
    """
    if not _ledger_keys:
        pytest.skip("ledger is empty; there is no proven form to test the other arm with")

    model = CASES / "sub_f16.onnx"
    assert model.is_file(), f"the proven-arm model is missing: {model}"

    unlockable, census = _discover(model)

    assert census["records"] > 0, "claim log empty — ERROR(instrument)"
    assert census["claimed"] > 0, (
        f"a form with a ledger proof was not claimed. Census: {census}. Either the ledger lookup "
        f"is broken or the proof does not cover the form it was written for."
    )
    assert not unlockable, (
        f"a proven form was reported as needing a proof: {unlockable}"
    )


def test_the_two_arms_disagree(require_vulkan, _ledger_keys):
    """R11's `arms_must_differ`, stated as an assertion rather than an assumption.

    Same op, same shape, same graph structure; only the dtype differs, and only one is proven.
    If both arms produce the same claiming outcome, the gate's output does not vary with its
    input and the mechanism is unwired regardless of what its source says.
    """
    if not _ledger_keys:
        pytest.skip("ledger is empty; both arms would decline for the same reason")

    unproven, unproven_census = _discover(CASES / "sub_f16_dyn_unproven.onnx")
    proven, proven_census = _discover(CASES / "sub_f16.onnx")

    assert (len(unproven) > 0) != (len(proven) > 0), (
        f"both arms produced the same claiming outcome — unproven={unproven_census}, "
        f"proven={proven_census}. A gate whose answer does not move with its input is not a gate."
    )

    # And the keys themselves must differ: a static proof can never be returned for a
    # runtime-extent node. Same op, same dtype, same graph — one component apart.
    key_dyn = unproven[0]
    key_static = next(
        iter(k for k in _ledger_keys if "Sub" in k and "f16" in k and "/static/" in k)
    )
    assert key_dyn != key_static, (
        f"the static and runtime-extent forms of Sub f16 produced the same proof key "
        f"{key_dyn!r} — a proof of one would be returned for the other, which is the shape of "
        f"the 2026-07-30 all-zero-logits defect"
    )


def test_the_degeneracy_guard_fires_and_stays_silent(require_vulkan, tmp_path):
    """The case-model degeneracy guard, in both polarities, in the lane.

    Twelve of the ops proved on 2026-08-02 return **bool**. A comparison whose CPU reference is
    constant is vacuous — two constant tensors agree to any tolerance — so `Equal` sampled from
    two independent normals is all-False and would have reported `MATCH worst_rel 0.0` having
    tested nothing. That is the cheapest possible way to "prove" twelve ops, and it is the same
    failure this project has hit seven times.

    Rai's rule is that a control which must be opted into is not in the lane, so this runs here
    rather than as a script: mutate `equal_f32`'s input domain away from `discrete` and the
    generator must return `ERROR(case_model_degenerate_reference)`; leave it as shipped and it
    must return `MATCH`. Both arms, because a guard that always fires is as useless as one that
    never does.
    """
    import gen_proof_ledger as g
    import ledger_case_models as lcm

    case = "equal_f32"
    tolerance = (1e-3, 1e-5)
    model = str(lcm.build(case, tmp_path))

    saved = lcm.INPUT_DOMAIN.pop(case, None)
    assert saved is not None, (
        f"ERROR(instrument): {case} declares no input domain, so the mutation below is a no-op "
        f"and this test would pass without testing anything"
    )
    try:
        keys, _ = g.discover_keys(model, reprove=True)
        mutated_verdict, mutated_detail = g.prove(model, keys, tolerance)
    finally:
        lcm.INPUT_DOMAIN[case] = saved

    keys, _ = g.discover_keys(model, reprove=True)
    shipped_verdict, shipped_detail = g.prove(model, keys, tolerance)

    assert mutated_verdict == "ERROR", (
        f"the guard did not fire on a deliberately degenerate reference; it returned "
        f"{mutated_verdict} {mutated_detail}"
    )
    assert mutated_detail.get("instrument") == "case_model_degenerate_reference", (
        f"the guard fired for some other reason: {mutated_detail}"
    )
    assert shipped_verdict == "MATCH", (
        f"the case as shipped no longer proves; the guard may be firing on everything, which "
        f"reads the same as a guard that works: {shipped_verdict} {shipped_detail}"
    )


# ---------------------------------------------------------------------------
# The CI check — the second half of RAI-008(a)
# ---------------------------------------------------------------------------

def test_check_ledger_passes_on_the_shipped_artifact():
    """`gen_proof_ledger.py --check` is the CI check RAI-008(a) requires.

    It re-derives the digest, re-hashes every evidence artifact, and rejects duplicates,
    malformed keys and non-MATCH verdicts. Run here so the lane fails at the same moment CI
    would, rather than one merge later.
    """
    r = subprocess.run(
        [sys.executable, str(TOOLS / "gen_proof_ledger.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=300,
    )
    # R13: quote the failure text, never the failure count.
    assert r.returncode == 0, f"--check failed:\n{r.stdout}\n{r.stderr}"


def test_check_ledger_fails_on_a_tampered_artifact(tmp_path):
    """The CI check must move when its subject is wrong.

    R9 amendment 5 asks which way a check moves when its subject is wrong. A `--check` that only
    ever passed would move *with* the reader's confidence and could not be repaired by
    tightening it. So we plant a corrupted ledger and require a non-zero exit.
    """
    corrupt = tmp_path / "proof_ledger.jsonl"
    lines = _ledger_lines()
    if len(lines) < 2:
        pytest.skip("ledger has no entries to corrupt")
    entry = json.loads(lines[1])
    entry["key"] = "*"  # the wildcard §8.9.4 forbids
    corrupt.write_text(
        lines[0] + "\n" + json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8"
    )

    r = subprocess.run(
        [sys.executable, str(TOOLS / "gen_proof_ledger.py"), "--check", "--out", str(corrupt)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=300,
    )
    assert r.returncode != 0, (
        f"--check accepted a ledger whose only entry is a wildcard key:\n{r.stdout}\n{r.stderr}"
    )


# ---------------------------------------------------------------------------
# Issue #43 — the stale-but-unmoved frame witness.
#
# Two screens already watch `source_digest` and BOTH were blind to the same event:
#
#   * ci/check_ledger_census.py reads git history and convicts a witness that MOVES in
#     the file with no declaration. A digest that goes stale UNDER an untouched entry
#     never moves in the file, so the census reads clean.
#   * gen_proof_ledger.py --check compares the ledger to the build, but a stale
#     source_digest whose SPIR-V still matches is SOURCE-COSMETIC, which FORGIVES. That
#     verdict is toolchain-dependent, so the same file passes on Windows and, wherever
#     the compiler happens to emit different SPIR-V, reads SUBJECT-CHANGED and declines.
#
# That is exactly how PR #35 shipped. `source_digest_for` hashes the WHOLE text of a
# template, and ew_unary.comp is shared by 42 op selectors, so an attribution header and
# four Asin/Acos-only functions moved the source_digest of all 55 ew_unary entries. 53
# were left stale. Windows said 78 identical + 55 SOURCE-COSMETIC and printed PASS; Linux
# CI found different SPIR-V for five of them — IsInf x3, IsNaN, Not — and with both
# witnesses moved it could no longer tell a moved compiler from a moved kernel, so it
# declined, and test_op_table[IsInf|IsNaN|Not] failed at assert_vulkan_claims.
#
# The screen below asks the one question neither of the others asks, and asks it WITHOUT
# consulting SPIR-V: does every recorded source_digest still equal the one this build
# computes? Being source-only makes it platform-independent — it fails on the author's own
# machine, at the moment the shared template is edited, instead of one merge later on the
# only platform whose compiler disagrees.
# ---------------------------------------------------------------------------


def _source_digest_audit():
    """-> (checked, stale) where stale is [(key, recorded, this_build)].

    Deliberately reads ONLY `source_digest`. Consulting `shader_digest` here would rebuild
    the very forgiveness that hid #35: SPIR-V agreement is what makes a stale source digest
    *survivable* on one toolchain, not what makes it *correct*.
    """
    import gen_proof_ledger as gpl

    lib = gpl._find_lib(os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB", ""))
    if lib is None:
        pytest.skip("no built EP to compare the ledger against")
    checked, stale = 0, []
    for line in _ledger_lines()[1:]:
        entry = json.loads(line)
        recorded = entry.get("source_digest")
        stems = entry.get("shaders") or []
        if not recorded or not stems:
            continue
        built = gpl._shader_subject(lib, stems).get("source_digest")
        if not built:
            # This build has no module for those stems. A real finding, but a different
            # one (`no-module-in-build`), already ruled on by --check. Not ours to claim.
            continue
        checked += 1
        if built != recorded:
            stale.append((entry["key"], recorded, built))
    return checked, stale


def test_no_entry_carries_a_stale_source_digest():
    """Every recorded §8.9.19 source witness still describes the source this build hashed.

    This is the screen that would have failed PR #35 on Windows. It is the standing repair
    for issue #43 and the negative control the issue asks for against collateral edits to a
    SHARED template: because `source_digest_for` hashes the whole file, touching
    ew_unary.comp for one operator moves the witness of all 55 entries derived from it, and
    this test names every one that was not re-witnessed or re-proved afterwards.
    """
    checked, stale = _source_digest_audit()
    assert checked, "no ledger entry could be compared; this test is UNOBSERVABLE, not passing"
    # R13: quote the failures, never just the count.
    assert not stale, (
        f"{len(stale)} of {checked} ledger entr(ies) carry a source_digest this build does not "
        "compute. The recorded witness describes source text that no longer exists, so on any "
        "toolchain that also emits different SPIR-V the EP must decline the op:\n"
        + "\n".join(f"  {k}: recorded {r} != this build's {b}" for k, r, b in sorted(stale)[:20])
        + (f"\n  ... +{len(stale) - 20} more" if len(stale) > 20 else "")
        + "\n\nIf the kernel changed, `--reprove` it. If only the source text moved and the "
        "SPIR-V is identical, `--backfill-frame --rewitness-source` and declare the move in "
        "evidence/proof_rewitness.json."
    )


def test_the_stale_source_digest_screen_can_say_no(tmp_path, monkeypatch):
    """A screen whose red state nobody has seen is not evidence.

    Plants the exact #35 regression — one entry left holding a source witness from a
    withdrawn revision of the template — and requires the audit to convict it. Without this,
    a `_shader_subject` that silently returned the recorded value would make the test above
    pass forever.
    """
    import gen_proof_ledger as gpl

    if gpl._find_lib(os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB", "")) is None:
        pytest.skip("no built EP to compare the ledger against")
    lines = _ledger_lines()
    victim = next(
        (json.loads(l) for l in lines[1:]
         if json.loads(l).get("source_digest") and json.loads(l).get("shaders")),
        None,
    )
    if victim is None:
        pytest.skip("no entry carries a source_digest to falsify")

    real = victim["source_digest"]
    victim["source_digest"] = "dead" + real[4:] if not real.startswith("dead") else "beef" + real[4:]
    planted = tmp_path / "proof_ledger.jsonl"
    planted.write_text(
        lines[0] + "\n" + json.dumps(victim, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(sys.modules[__name__], "LEDGER", planted)
    try:
        checked, stale = _source_digest_audit()
    finally:
        monkeypatch.undo()
    assert checked == 1, f"expected to compare the single planted entry, compared {checked}"
    assert [k for k, _, _ in stale] == [victim["key"]], (
        "the audit did not convict an entry whose source_digest was hand-edited away from "
        f"this build's; it reported {stale!r}. A screen that cannot say no is not a screen."
    )


# ---------------------------------------------------------------------------
# §8.9.24 — mintability. Can a key exist at all?
# ---------------------------------------------------------------------------


def _mint_lib():
    import gen_proof_ledger as gpl

    lib = gpl._find_lib(os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB", ""))
    if lib is None:
        pytest.skip("no built EP; mintability is a question about an artifact")
    return gpl, lib


def test_the_mintability_screen_can_say_no():
    """A screen whose red state is never shown is a screen nobody has seen work.

    Measured 2026-08-04: all 43 retired keys passed `--check` cleanly, because nothing asked
    whether a key could ever be minted. This asserts the positive state on a **real** key —
    every `_i64` module declares `OpCapability Int64` and `ENGINE_ENABLED_CAPABILITIES` does not
    carry it — and asserts a sibling on the same op answers the other way, so the screen is
    non-vacuous by measurement rather than by argument.
    """
    gpl, lib = _mint_lib()
    unmintable = "ai.onnx::Cast/6+/i64>i32/ew_cast_i64_to_i32/static/n1"
    mintable = "ai.onnx::Cast/6+/f32>i32/ew_cast_f32_to_i32/static/n1"
    report, err = gpl._mintability(lib, [unmintable, mintable])
    assert not err, err
    assert report[unmintable]["mintable"] is False, report[unmintable]
    assert report[unmintable]["loadable"] == "no", (
        f"the report has to say why, or the caller is back to guessing: {report[unmintable]}"
    )
    assert report[mintable]["mintable"] is True, (
        f"a screen that can only say `no` is as blind as one that can only say `yes`: "
        f"{report[mintable]}"
    )


def test_the_mintability_screen_fails_a_ledger_that_holds_an_unmintable_key(tmp_path):
    """And the red reaches `--check`'s exit code, not only the report."""
    gpl, lib = _mint_lib()
    lines = _ledger_lines()
    if len(lines) < 2:
        pytest.skip("ledger has no entries to clone")
    header = json.loads(lines[0])
    entries = [json.loads(l) for l in lines[1:]]
    donor = dict(entries[0])
    donor["key"] = "ai.onnx::Cast/6+/i64>i32/ew_cast_i64_to_i32/static/n1"
    entries.append(donor)
    body = "".join(json.dumps(e, sort_keys=True) + "\n" for e in entries)
    header["entry_count"] = len(entries)
    header["content_fnv1a64"] = f"{gpl.fnv1a64(body.encode('utf-8')):016x}"
    dest = tmp_path / "proof_ledger.jsonl"
    dest.write_text(json.dumps(header, sort_keys=True) + "\n" + body, encoding="utf-8")

    buf = io.StringIO()
    with redirect_stdout(buf):
        # A synthetic ledger differs from the baked one by construction; `expect_rebuild` is the
        # same allowance a fresh generation run gets, and without it this would be asserting the
        # baked-vs-disk check instead.
        rc = gpl.check_ledger(dest, lib, expect_rebuild=True)
    out = buf.getvalue()
    assert rc == 1, f"--check accepted a ledger holding an unmintable key:\n{out}"
    assert "NOT MINTABLE" in out and "ew_cast_i64_to_i32" in out, out


def test_the_mintability_arithmetic_line_is_printed_on_a_pass():
    """The PASS line is a summary; the subject arithmetic line is the reading.

    Twice in two days a real loss sat behind a PASS — a `source_digest` revert on 115 of 121
    entries, then 9 more on a merge — and was caught only by reading the arithmetic. So the
    counts print on the green path too, and this asserts they add up to their populations.
    """
    gpl, lib = _mint_lib()
    retired, err = gpl._retired_keys()
    assert not err, err
    keys = {json.loads(l)["key"] for l in _ledger_lines()[1:]}
    fails, notes, merr = gpl.check_mintability(lib, keys, retired)
    assert not merr, merr
    assert not fails, fails
    line = next((n for n in notes if n.startswith("mintability:")), "")
    assert line, f"no arithmetic line among {notes}"
    assert f"{len(keys)} ledger key(s)" in line and f"{len(retired)} retired key(s)" in line, line


def test_a_build_that_cannot_be_asked_is_an_instrument_error_not_a_pass():
    """A missing export is the same finding as a missing artifact: ERROR, never PASS.

    Any pre-2026-08-04 build lacks `OrtEpVulkanGetFormMintability`. Skipped rather than faked
    when no such artifact is on this machine — an unrun control is not a passed one.
    """
    gpl, lib = _mint_lib()
    older = [
        p
        for p in sorted(REPO.parent.glob("ep-vulkan-*/rust/target/release/onnxruntime_vulkan_ep.dll"))
        if p.resolve() != lib.resolve()
    ]
    for cand in older:
        _, err = gpl._mintability(cand, ["ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2"])
        if err and "OrtEpVulkanGetFormMintability" in err:
            return
    pytest.skip("no artifact without the export is available to ask")


# ---------------------------------------------------------------------------
# §8.9.14 — the writer's own refusals, planted rather than read
# ---------------------------------------------------------------------------


def _proof_run(**over):
    """A minimal healthy proof-run record, as `prove()` returns on MATCH.

    `shaders_dispatched_source_digest` is present because a *healthy* run has one — §8.9.19 made
    the frame witness mandatory at the writer, and without it every test below refused for the
    wrong reason (the accepted red `proof_ledger_writer_refuses`, opened 2026-08-03). These
    tests each plant one specific defect and require the writer to name *that* one; a fixture
    that is defective in a second way turns them all into one test of the same refusal.
    """
    run = {
        "worst_rel": 1e-6,
        "claimed_nodes": 1,
        "dispatches_executed": 1,
        "shaders_dispatched": ["ew_binary_add_f32"],
        "shaders_dispatched_digest": "0123456789abcdef",
        "shaders_dispatched_source_digest": "fedcba9876543210",
        "shader_toolchain": "shaderc v2026.2 v2026.2",
        "shaders_dispatched_spec_digest": "abcdef0123456789",
        "compute_calls": 1,
    }
    run.update(over)
    return run


def _write_entry(run):
    import gen_proof_ledger as g

    return g.entry_line(
        "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2",
        "device0",
        "1.20.0",
        "rtol=1e-3,atol=1e-5",
        1e-6,
        "0" * 64,
        "2026-08-02T00:00:00-07:00",
        run,
    )


def test_the_writer_refuses_a_run_with_no_subject_witness():
    """NO-SUBJECT-WITNESS must be reachable *and* refused, not merely nameable.

    Switch's first GQA isolation arm read `calls=0 dispatches=0` — it ran entirely on the CPU EP
    and would have reported "GQA is innocent" from a run GQA never entered. He caught it by
    reading the counters by hand. The question the coordinator asked is whether the harness can
    produce that state, and the honest answer is only worth something as an artifact: so both
    shapes are planted here and the refusal is required.

    Both polarities: the healthy record must still be written, or this test would pass by
    refusing everything.
    """
    healthy = _write_entry(_proof_run())
    assert json.loads(healthy)["verdict"] == "MATCH"

    # (a) no attribution witness — the state Switch's arm was in.
    with pytest.raises(SystemExit) as e:
        _write_entry(_proof_run(claimed_nodes=0, dispatches_executed=0))
    assert "attribution" in str(e.value), str(e.value)

    # (b) attribution present but no subject witness — a run that dispatched nothing nameable.
    with pytest.raises(SystemExit) as e:
        _write_entry(_proof_run(shaders_dispatched=[], shaders_dispatched_digest=""))
    assert "subject witness" in str(e.value), str(e.value)


def test_the_writer_refuses_a_multi_inference_proof_run():
    """§8.9.14 — one inference per arm is what makes an entry immune to input-cache staleness.

    Switch's `(cpu_ptr, byte_size)` cache defect made every inference after the first in a
    session read the first one's inputs, and no shader digest can see it because the shaders do
    not change. Every entry here is immune because each arm is a fresh subprocess with exactly
    one `sess.run`. That is a fact about today's harness; this makes it a field, so a future
    multi-inference case cannot inherit the immunity by looking the same.
    """
    assert json.loads(_write_entry(_proof_run(compute_calls=1)))["compute_calls"] == 1
    with pytest.raises(SystemExit) as e:
        _write_entry(_proof_run(compute_calls=2))
    assert "Compute() calls" in str(e.value), str(e.value)


def test_a_shrinking_write_is_a_failure_not_a_footnote(tmp_path):
    """A destructive write that reports success is the defect this file keeps re-learning.

    `--reprove` without `--append` took the ledger from 74 entries to 1 and printed `PASS`. The
    carry-forward fix makes that unlikely; it does not make it a *detection*. So the writer
    compares against what is on disk and refuses, and the refusal names what would have gone.
    """
    import gen_proof_ledger as g

    p = tmp_path / "proof_ledger.jsonl"
    lines = [json.dumps({"key": f"k{i}"}, sort_keys=True) for i in range(5)]
    assert g.write_ledger(p, lines) == 0

    assert g.write_ledger(p, lines[:2]) != 0, "the writer accepted a silent shrink"
    kept = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()][1:]
    assert len(kept) == 5, "the refused write still touched the file"

    # ...and it is a refusal, not a ban: --rebuild is the deliberate instruction.
    assert g.write_ledger(p, lines[:2], allow_shrink=True) == 0
    kept = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()][1:]
    assert len(kept) == 2


# ── the §8.9.19 declaration register itself ────────────────────────────────────────────────


def _rewitness_doc():
    """The register, parsed by the CANONICAL parser rather than by `json.load`.

    Reading it with `json.load` here would test that the file is JSON, which nothing disputes.
    Reading it through `ci/check_ledger_census.py` tests the thing that matters: that the
    register the lane screens is the register this suite is looking at, and that its schema
    validation is reachable from the test lane and not only from a shell script CI runs.
    """
    sys.path.insert(0, str(REPO / "ci"))
    import check_ledger_census as clc  # noqa: PLC0415

    return clc, clc.load_rewitness(REPO / "evidence" / "proof_rewitness.json")


def test_the_rewitness_register_parses_under_the_canonical_checker():
    clc, doc = _rewitness_doc()
    records = doc.get("rewitness", [])
    assert records, "the register is empty; this test is UNOBSERVABLE, not passing"
    for rec in records:
        # Raises on an unknown schema — the fail-loud path, exercised on the real file.
        assert clc._record_schema(rec) in clc.KNOWN_SCHEMAS


def test_no_content_addressed_record_names_a_revision():
    """The whole point of `rewitness/2` and `/3` is that they have nothing for a squash to erase.

    A `revision` on either would be dead weight at best and, far more likely, the thing a
    future reader trusts — reintroducing by habit the coupling the schema removed. It is not
    merely unused; it must not be there. v3 goes one step further and names no commit at all,
    so for a v3 record this also catches a half-finished migration from v2.
    """
    clc, doc = _rewitness_doc()
    offenders = [
        (i, clc._record_schema(rec), rec.get("revision"))
        for i, rec in enumerate(doc.get("rewitness", []))
        if clc._record_schema(rec) in clc.CONTENT_SCHEMAS and "revision" in rec
    ]
    assert not offenders, (
        f"{len(offenders)} content-addressed record(s) carry a `revision`: {offenders!r}. "
        "Matching is (field, key, old, new); a revision here is a sha a squash will erase."
    )


def test_every_declared_transition_lands_on_the_digest_this_build_computes():
    """A declaration is a claim about this ledger, so it is checkable against this build.

    Each v2 transition says a key moved `old` -> `new`. `new` is what the ledger carries now,
    and `test_no_entry_carries_a_stale_source_digest` already ties the ledger to the build —
    but only in aggregate. This one is per declared row, so a record that enumerates a key
    with a plausible-looking digest nobody computes is convicted HERE, in the file that
    declares it, rather than showing up as a mystery decline on the other platform.

    It also closes the `old == new` loophole from the other side: a row whose `new` is not the
    current value is either a stale declaration or a wrong one, and both are defects.

    Both content-addressed schemas are read, not only v2: a v3 record enumerates the same
    `(key, old, new)` rows and differs only in how it names the CAUSE, so exempting it here
    would have made the migration a quiet loss of coverage.
    """
    clc, doc = _rewitness_doc()
    entries = {json.loads(l)["key"]: json.loads(l) for l in _ledger_lines()[1:]}
    wrong, checked = [], 0
    for rec in doc.get("rewitness", []):
        if clc._record_schema(rec) not in clc.CONTENT_SCHEMAS:
            continue
        field = rec["field"]
        for t in rec["transitions"]:
            entry = entries.get(t["key"])
            if entry is None:
                wrong.append((t["key"], "declared but absent from the ledger", t["new"]))
                continue
            checked += 1
            if entry.get(field) != t["new"]:
                wrong.append((t["key"], entry.get(field), t["new"]))
    if not checked:
        pytest.skip("no content-addressed transitions to check against the ledger")
    assert not wrong, (
        f"{len(wrong)} declared transition(s) do not describe the shipped ledger:\n"
        + "\n".join(f"  {k}: ledger has {a!r}, declaration says new={b!r}" for k, a, b in wrong[:20])
    )


def test_the_declaration_screen_survives_a_squash_of_this_branch():
    """The end-to-end control for the landing-safety claim, run against the real repository.

    `ci/simulate_squash_rewitness.py` clones this repo, lands HEAD's tree onto `origin/main`
    the way a maintainer's chosen button would, screens the result, screens it again one
    unrelated commit later, and then plants two broken registers into the SAME landing and
    requires both to go red. Both polarities, because a simulation that only shows green
    proves the simulation, not the schema.

    This runs the SQUASH landing only, because that is the landing this repository uses and a
    pytest that clones the object store three times is a pytest people learn to skip. The
    other two landings are covered by planted arms in `ci/negative_control_ledger_census.py`
    ("green under a MERGE landing", "green under a REBASE landing", and their v2 contrast),
    and a maintainer can run the full matrix here with no arguments.

    Skipped rather than failed when `origin/main` is not fetched: that is an unobservable
    environment, not a defect in the register.
    """
    sim = REPO / "ci" / "simulate_squash_rewitness.py"
    if not sim.is_file():
        pytest.skip("simulator absent")
    if subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "origin/main^{commit}"],
        cwd=REPO, capture_output=True, text=True,
    ).returncode != 0:
        pytest.skip("origin/main is not present in this clone")
    r = subprocess.run([sys.executable, str(sim), "--landing", "squash"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, f"squash simulation failed:\n{r.stdout}\n{r.stderr}"
    assert "survives every landing" in r.stdout, r.stdout
    # The controls must have FIRED, not merely been run: a matrix of greens is what a screen
    # that stopped reading the register also produces.
    assert "unlanded_rewitness_cause" in r.stdout, r.stdout
    assert "uncorroborated_rewitness_cause" in r.stdout, r.stdout

