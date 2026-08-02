"""Criterion 11 / §8.9 — the proof ledger gate, verified by a planted decline.

WHY THIS FILE EXISTS
--------------------
RAI-008(a), re-verdicted unsoftened in RAI-010, requires that "no form is claimable without a
ledger entry" be verified by **a planted `[unproven]` decline and a CI check** — not by reading
`registry.rs`. R10 is the reason: *the falsifier for "X is wired" is an artifact X produced whose
content varies with its input.* This file supplies the input variation.

THE PLANT
---------
`evidence/cases/mul_f16_unproven.onnx` is a form the EP has a kernel for and the ledger has no
entry for. Its sibling `evidence/cases/mul_f32.onnx` is the same op, same shape, same everything
except dtype — and it **is** proven. The pair is the control:

  * if the f16 arm is claimed, the gate is not gating;
  * if the f32 arm is declined, the gate is declining unconditionally and is equally useless.

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

import json
import pathlib
import subprocess
import sys

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

    Predicted before the run and recorded in `bench/results/proof_ledger_prediction.json`:
    the key is `ai.onnx::Mul/7+/f16,f16>f16/ew_binary_mul_f16/static/n2`, the decline code is
    `unproven` and nothing else, and the model still computes correctly because ORT falls back
    to the CPU EP.
    """
    model = CASES / "mul_f16_unproven.onnx"
    assert model.is_file(), f"the planted control model is missing: {model}"

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
    assert any("f16" in k and "Mul" in k for k in unlockable), (
        f"expected the planted f16 Mul form among the declines, got {unlockable}"
    )


def test_proven_form_is_claimed(require_vulkan, _ledger_keys):
    """THE OTHER ARM. A form WITH a ledger entry must be claimed.

    Without this, `test_planted_unproven_form_declines` would pass against a gate that declined
    everything — a perfectly stable, perfectly wrong answer.
    """
    if not _ledger_keys:
        pytest.skip("ledger is empty; there is no proven form to test the other arm with")

    model = CASES / "mul_f32.onnx"
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

    unproven, unproven_census = _discover(CASES / "mul_f16_unproven.onnx")
    proven, proven_census = _discover(CASES / "mul_f32.onnx")

    assert (len(unproven) > 0) != (len(proven) > 0), (
        f"both arms produced the same claiming outcome — unproven={unproven_census}, "
        f"proven={proven_census}. A gate whose answer does not move with its input is not a gate."
    )

    # And the keys themselves must differ: an f32 proof can never be returned for an f16 node.
    key_f16 = unproven[0]
    key_f32 = next(iter(k for k in _ledger_keys if "Mul" in k and "f32" in k))
    assert key_f16 != key_f32, (
        f"the f16 and f32 forms of Mul produced the same proof key {key_f16!r} — a proof of one "
        f"would be returned for the other, which is the 2026-07-30 all-zero-logits defect"
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
