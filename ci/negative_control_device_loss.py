#!/usr/bin/env python3
"""Negative control for ci/check_device_loss.py — does the screen actually go red?

A screen that has only ever been observed passing is one step from a screen that cannot
fail. This file constructs inputs on which `check_device_loss.py` MUST report a specific
condition, and asserts it reports that condition and not merely "some failure". §10.0.1
R9: evidence scales only with falsifying instruments.

Arm provenance is stated per arm and never blurred, because "we tested it" reads the same
whether the input was real or synthesised and the two are worth very different amounts:

  REPLAYED  the input is a real artifact of a real incident, fed to the screen after the
            fact. Proves the screen would have caught it. Does not prove the screen fires
            during a live run.
  LIVE      the screen found this on a file it was not written against, in a normal run.
  PLANTED   the input was synthesised here to exercise one rule. Proves the rule fires.
            Proves nothing about whether the rule's event ever occurs in reality.

Run: python ci/negative_control_device_loss.py     (exit 0 = every arm fired)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK = REPO_ROOT / "ci" / "check_device_loss.py"
TANK_ARTIFACT = REPO_ROOT / "bench" / "results" / "ctx512_device_lost.txt"
TRINITY_LOG = REPO_ROOT / "bench" / "results" / "trinity-suite-dev1.log"
#: Switch's ctx-4096 resident-lane incident record. Its device-loss witness sits inside a
#: JSON string, in two encodings: ORT's own UTF-16LE line and a plain-ASCII Python
#: traceback. Used by the arm that removes the second and requires the first to be seen.
SWITCH_CTX4096 = REPO_ROOT / "bench" / "results" / "phi35_kv_chain-ctx4096-BOTH-dev0.json"

CLEAN_LOG = """\
[vulkan-ep] INFO: VulkanExecutionProvider: device 0 (NVIDIA GeForce RTX 4060)
[vulkan-ep] INFO: claimed fused subgraph #1 (355 nodes)
2026-08-02 12:00:00 [I:onnxruntime:, session_state.cc] Node placements
2026-08-02 12:00:00 [V:onnxruntime:, session_state.cc]  VulkanExecutionProvider: [Fused]
argmax 30751
all 65 outputs bit-identical to the CPU reference
EXIT = 0
"""


def run(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(CHECK), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return proc.returncode, proc.stdout + proc.stderr


class Arms:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, bool, str]] = []

    def arm(self, name: str, provenance: str, ok: bool, note: str) -> None:
        self.results.append((name, provenance, ok, note))

    def expect_condition(
        self, name: str, provenance: str, args: list[str], condition: str
    ) -> None:
        code, out = run(args)
        want = f"FAIL(condition={condition})"
        ok = code == 1 and want in out
        self.expect_note = ""
        self.arm(
            name,
            provenance,
            ok,
            f"exit={code} " + (f"saw {want}" if ok else f"did NOT see {want}"),
        )

    def expect_reported(
        self, name: str, provenance: str, args: list[str], condition: str
    ) -> None:
        """Red for any condition, with `condition` named among them."""
        code, out = run(args)
        ok = code == 1 and f"[{condition}]" in out
        self.arm(
            name,
            provenance,
            ok,
            f"exit={code} " + (f"reported [{condition}]" if ok else f"no [{condition}]"),
        )

    def expect_absent(
        self, name: str, provenance: str, args: list[str], condition: str
    ) -> None:
        """`condition` is NOT among the findings.

        Deliberately not "the run is green": this artifact is a real device loss and must
        stay red for the conditions that are true of it. Asserting green here would let a
        check that had gone blind to everything satisfy the arm, which is the defect this
        control exists to catch, arriving through the control itself.
        """
        code, out = run(args)
        ok = f"[{condition}]" not in out
        self.arm(
            name,
            provenance,
            ok,
            f"exit={code} "
            + (
                f"no [{condition}] — and still red for the conditions that ARE true of it"
                if ok
                else f"still reports [{condition}], which is no longer true of this file"
            ),
        )

    def expect_foreign_check(
        self, name: str, provenance: str, path: "Path", expect_code: int, why: str
    ) -> None:
        """Assert an exit code from a *different* check, on the same evidence.

        The extents ruling between this check and ci/check_fatal_log.py is only meaningful
        if both sides of it are observed. Reading one and asserting the other in a comment
        is how the two drift into either a gap or a duplicate without anyone seeing it.
        """
        fatal_log = REPO_ROOT / "ci" / "check_fatal_log.py"
        if not fatal_log.exists():
            self.arm(name, provenance, False, f"{fatal_log} is missing — outage, not a pass")
            return
        proc = subprocess.run(
            [sys.executable, str(fatal_log), str(path)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.arm(
            name,
            provenance,
            proc.returncode == expect_code,
            f"check_fatal_log exit={proc.returncode}, wanted {expect_code} ({why})",
        )

    def expect_pass(self, name: str, provenance: str, args: list[str]) -> None:
        code, out = run(args)
        ok = code == 0 and ": PASS" in out
        self.arm(name, provenance, ok, f"exit={code} " + ("PASS" if ok else "not PASS"))

    def expect_error(
        self, name: str, provenance: str, args: list[str], instrument: str
    ) -> None:
        code, out = run(args)
        want = f"ERROR(instrument={instrument})"
        ok = code == 4 and want in out
        self.arm(
            name,
            provenance,
            ok,
            f"exit={code} " + (f"saw {want}" if ok else f"did NOT see {want}"),
        )


def main() -> int:
    arms = Arms()
    tmp = Path(tempfile.mkdtemp(prefix="device-loss-control-"))

    # --- The red arm that matters, on real evidence -------------------------
    if TANK_ARTIFACT.exists():
        arms.expect_condition(
            "real device loss (Tank, ctx 512, vkWaitForFences)",
            "REPLAYED",
            [str(TANK_ARTIFACT)],
            "device_lost_reported",
        )
        arms.expect_reported(
            "the same log's ORT fallback announcement",
            "REPLAYED",
            [str(TANK_ARTIFACT)],
            "runtime_fallback_announced",
        )
        # The arm that used to sit here required `marker_list_misses_real_line` on this
        # artifact. Trinity repaired the shared vocabulary on 2026-08-02 and that finding
        # became FALSE — ci/check_fatal_log.py now goes red on this exact file. The arm
        # went on firing anyway, because my cross-check was comparing a physical line to a
        # matched span by string equality; it had stopped being able to tell repair from
        # breakage while still reading as confident, and it would have turned this control
        # red if anyone had fixed it. Replaced by the two arms below, which assert the
        # repair rather than the defect: pinning a defect is how a control outlives its
        # subject.
        arms.expect_absent(
            "the shared vocabulary is NOT blind to this log any more (Trinity, 2026-08-02)",
            "REPLAYED",
            [str(TANK_ARTIFACT)],
            "marker_list_misses_real_line",
        )
        arms.expect_foreign_check(
            "  ...and check_fatal_log IS red on it, which is what that means",
            "REPLAYED",
            TANK_ARTIFACT,
            expect_code=1,
            why="0 would mean the vocabulary regressed to matching our own sentences",
        )
    else:
        arms.arm(
            "real device loss",
            "REPLAYED",
            False,
            f"{TANK_ARTIFACT} is missing — the red arm cannot be run, which is an "
            "instrument outage in this control, not a pass",
        )

    if TRINITY_LOG.exists():
        arms.expect_condition(
            "second device loss, dev1 vkQueueSubmit, 2026-07-31",
            "LIVE (found by the screen on a file it was not written against)",
            [str(TRINITY_LOG)],
            "device_lost_reported",
        )

    # --- The green arm ------------------------------------------------------
    clean = tmp / "clean.log"
    clean.write_text(CLEAN_LOG, encoding="utf-8")
    arms.expect_pass("a run that did not lose the device", "PLANTED", [str(clean)])

    # --- The structural rule, which needs no text at all --------------------
    short = tmp / "short.json"
    short.write_text(
        json.dumps({"points": [{"iters": 25, "compute_calls": 9}]}), encoding="utf-8"
    )
    arms.expect_condition(
        "declared 25 inferences, observed 9, no log text anywhere",
        "PLANTED",
        [str(short)],
        "observation_ended_early",
    )

    whole = tmp / "whole.json"
    whole.write_text(
        json.dumps({"points": [{"iters": 25, "compute_calls": 25}]}), encoding="utf-8"
    )
    arms.expect_pass("declared 25, observed 25", "PLANTED", [str(whole)])

    inflight = tmp / "inflight.json"
    inflight.write_text(json.dumps({"uploads": 5, "readbacks": 4}), encoding="utf-8")
    arms.expect_condition(
        "uploads == readbacks + 1 (an inference caught in flight)",
        "PLANTED",
        [str(inflight)],
        "observation_ended_early",
    )

    rejected = tmp / "rejected.json"
    rejected.write_text(
        json.dumps({"rejected_points": [{"iters": 25, "compute_calls": 9}]}),
        encoding="utf-8",
    )
    arms.expect_pass(
        "the SAME truncation, but the producer already rejected it", "PLANTED", [str(rejected)]
    )

    # --- The reach arm: the evidence for the extents ruling ------------------
    # A device loss the EP reports and ORT never announces. check_fatal_log reads
    # ORT's announcement, so it is green here; this check is red. That difference is
    # the whole reason both exist, and it is demonstrated rather than argued.
    ep_only = tmp / "ep_only.log"
    ep_only.write_text(
        "[vulkan-ep] ERROR: vkQueueSubmit failed: The logical device has been lost.\n"
        "inference 3 of 25\n",
        encoding="utf-8",
    )
    arms.expect_condition(
        "EP reports the loss, ORT announces nothing (reach beyond check_fatal_log)",
        "PLANTED",
        [str(ep_only)],
        "device_lost_reported",
    )
    fatal_log = REPO_ROOT / "ci" / "check_fatal_log.py"
    if fatal_log.exists():
        proc = subprocess.run(
            [sys.executable, str(fatal_log), str(ep_only)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        arms.arm(
            "  ...and check_fatal_log is NOT red on that same file",
            "PLANTED",
            proc.returncode != 1,
            f"check_fatal_log exit={proc.returncode} (1 would mean it caught it too, and "
            "this check would then have no reach of its own to justify it)",
        )

    # --- The cross-check must still be able to fire ------------------------
    # Removing a false positive is only half the work: a divergence detector that can no
    # longer report anything is quieter and equally useless. These plant the two shapes
    # the repaired comparison has to get right.
    diverge = tmp / "vocabulary_blind.log"
    diverge.write_text(
        "[ORT] Falling back to CPUExecutionProvider because the EP declined\n",
        encoding="utf-8",
    )
    arms.expect_reported(
        "a fallback line MY regex sees and the shared vocabulary does not",
        "PLANTED",
        [str(diverge)],
        "marker_list_misses_real_line",
    )

    # ORT's C++ sink writes UTF-16LE into an otherwise UTF-8 file (Trinity, 2026-08-02).
    # Read as UTF-8 the message arrives NUL-separated, so an un-normalised scanner sees
    # nothing. Before today MY side was un-normalised too — and two blind scanners agree
    # perfectly, so no divergence would have been reported and this would have read as
    # agreement. The arm is the wide form and nothing else.
    wide = tmp / "utf16_only.log"
    wide.write_bytes(
        "[ORT] Falling back to ['CPUExecutionProvider'] and retrying.\n".encode("utf-16-le")
    )
    arms.expect_condition(
        "the announcement in UTF-16LE only — both sides must still see it",
        "PLANTED",
        [str(wide)],
        "runtime_fallback_announced",
    )
    arms.expect_absent(
        "  ...and that is agreement, not two blind scanners agreeing",
        "PLANTED",
        [str(wide)],
        "marker_list_misses_real_line",
    )

    # --- The wide capture, one level further in: inside a JSON string --------
    #
    # MEASURED BLIND, 2026-08-04. A probe that captures a worker's stderr into a JSON
    # field stores ORT's UTF-16LE text as `\u0000`-escaped characters. There are no real
    # NULs in the file, so `normalise_log_text` has nothing to strip and every marker
    # misses. This check reported PASS on the real incident record with its (ASCII) Python
    # traceback removed — it was reading the accident, not the witness.
    #
    # Two arms, because the previous arm above proves only that a *raw* wide log is seen.
    json_wide = tmp / "wide_in_json.json"
    json_wide.write_text(
        json.dumps(
            {
                "lanes": {
                    "resident": {
                        "verdict": "ERROR(instrument)",
                        "stderr_tail": (
                            "vkWaitForFences failed: the Vulkan device was lost "
                            "(VK_ERROR_DEVICE_LOST)\n"
                        ).encode("utf-16-le").decode("utf-8", "replace"),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    arms.expect_condition(
        "a UTF-16LE device loss inside a JSON string value",
        "PLANTED",
        [str(json_wide)],
        "device_lost_reported",
    )

    # And the same shape replayed off the committed incident record rather than built:
    # `phi35_kv_chain-ctx4096-BOTH-dev0.json` is red today only because its `stderr_tail`
    # also carries a plain-ASCII Python traceback. Strip that and keep ORT's own wide line
    # — which is exactly what the artifact looks like when ORT falls back and the process
    # exits 0, the original ctx-512 incident — and the check must still be red.
    if SWITCH_CTX4096.exists():
        try:
            doc = json.loads(SWITCH_CTX4096.read_text(encoding="utf-8"))
            tail = doc["lanes"]["resident"]["stderr_tail"]
            doc["lanes"]["resident"]["stderr_tail"] = tail[: tail.index("Traceback")]
            replay = tmp / "ctx4096_wide_only.json"
            replay.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            arms.expect_condition(
                "the real ctx-4096 record with its ASCII traceback removed",
                "REPLAYED",
                [str(replay)],
                "device_lost_reported",
            )
        except Exception as exc:  # noqa: BLE001
            arms.arm(
                "the real ctx-4096 record with its ASCII traceback removed",
                "REPLAYED",
                False,
                f"could not build the replay from {SWITCH_CTX4096.name}: {exc} — an "
                "instrument outage in this control, not a pass",
            )
    else:
        arms.arm(
            "the real ctx-4096 record with its ASCII traceback removed",
            "REPLAYED",
            False,
            f"{SWITCH_CTX4096} is missing — the arm cannot be run, which is an instrument "
            "outage in this control, not a pass",
        )

    # --- The exclusion list, held to what it excludes (issue #24) -----------
    #
    # WHY THESE ARMS EXIST. Until 2026-08-07 an entry in
    # ci/device_loss_incident_records.json silenced its file WHOLESALE and FOREVER, and
    # said nothing about what it silenced. Six artifacts of real, already-diagnosed losses
    # had landed committed with no entry at all, so the lane screen was red on every run
    # that reached it — the state the list exists to prevent. The obvious repair, naming
    # the six files, would have bought green with an exclusion that ALSO blinds the screen
    # to the next loss recorded in the same file. So an exclusion now declares the
    # finding(s) it accounts for and the check re-reads every excluded file to hold it
    # there. These arms are what make that claim falsifiable.
    #
    # The planted records files use ABSOLUTE paths in `file`. `REPO_ROOT / entry["file"]`
    # yields the absolute path unchanged, which is how a control can plant an exclusion
    # without writing anything into the tree it is diagnosing (Tank, issue #14: a
    # diagnostic must not mutate its subject).
    def plant_records(where: Path, entries: list[dict]) -> Path:
        doc = {"records": entries}
        target = where / "records.json"
        target.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return target

    def excl(path: Path, witness: dict, **over) -> dict:
        entry = {
            "file": str(path),
            "witness": witness,
            "reason": "planted by ci/negative_control_device_loss.py",
            "owner": "link",
            "date": "2026-08-07",
        }
        entry.update(over)
        return entry

    BYSTANDER = "a run that kept its device\nEXIT = 0\n"
    LOSS_LINE = "[vulkan-ep] ERROR: vkWaitForFences failed: VK_ERROR_DEVICE_LOST\n"
    # Two DISTINCT lines. A finding is a line, and the same line seen twice in one file is
    # one finding, not two (scan_artifact de-duplicates); a second loss in a real capture
    # carries its own dispatch count, so distinct is also the faithful shape.
    LOSS_AT = "[vulkan-ep] ERROR: vkWaitForFences failed: VK_ERROR_DEVICE_LOST after {} dispatches\n"

    inert = tmp / "excl_inert"
    inert.mkdir()
    (inert / "bystander.log").write_text(BYSTANDER, encoding="utf-8")
    quiet = inert / "quiet.log"
    quiet.write_text("nothing happened here at all\n", encoding="utf-8")
    arms.expect_condition(
        "an exclusion over a file that says nothing (it can only blind)",
        "PLANTED",
        [
            "--incident-records",
            str(plant_records(inert, [excl(quiet, {"device_lost_reported": 1})])),
            str(inert),
        ],
        "incident_record_covers_nothing",
    )

    widened = tmp / "excl_widened"
    widened.mkdir()
    (widened / "bystander.log").write_text(BYSTANDER, encoding="utf-8")
    two_losses = widened / "loss.log"
    two_losses.write_text(LOSS_AT.format(1775) + LOSS_AT.format(0), encoding="utf-8")
    arms.expect_condition(
        "a second loss appended to an already-forgiven file",
        "PLANTED",
        [
            "--incident-records",
            str(plant_records(widened, [excl(two_losses, {"device_lost_reported": 1})])),
            str(widened),
        ],
        "incident_record_widened",
    )

    # The same exclusion, accounting for what is actually there. This is the GREEN
    # polarity of the pair: without it, "widened fires" is satisfied by an audit that
    # reds on every exclusion, which would delete the exclusion list rather than bound it.
    arms.expect_pass(
        "  ...and the SAME file is green once the exclusion accounts for both",
        "PLANTED",
        [
            "--incident-records",
            str(plant_records(widened, [excl(two_losses, {"device_lost_reported": 2})])),
            str(widened),
        ],
    )

    # The counting rule itself, stated as an arm so it cannot drift silently: a witness
    # counts DISTINCT finding lines. The same line twice is one finding, so a witness of 1
    # over a file that repeats one loss line is accurate, not narrowed.
    repeated = tmp / "excl_repeated_line"
    repeated.mkdir()
    (repeated / "bystander.log").write_text(BYSTANDER, encoding="utf-8")
    same_twice = repeated / "loss.log"
    same_twice.write_text(LOSS_LINE * 2, encoding="utf-8")
    arms.expect_pass(
        "a witness counts distinct finding lines, not repetitions of one line",
        "PLANTED",
        [
            "--incident-records",
            str(plant_records(repeated, [excl(same_twice, {"device_lost_reported": 1})])),
            str(repeated),
        ],
    )

    other_cond = tmp / "excl_other_condition"
    other_cond.mkdir()
    (other_cond / "bystander.log").write_text(BYSTANDER, encoding="utf-8")
    mixed = other_cond / "mixed.json"
    mixed.write_text(
        json.dumps(
            {
                "stderr_tail": "vkQueueSubmit failed: The logical device has been lost\n",
                "points": [{"iters": 25, "compute_calls": 9}],
            }
        ),
        encoding="utf-8",
    )
    arms.expect_condition(
        "a truncation appears in a file forgiven only for its device-loss text",
        "PLANTED",
        [
            "--incident-records",
            str(plant_records(other_cond, [excl(mixed, {"device_lost_reported": 2})])),
            str(other_cond),
        ],
        "incident_record_widened",
    )

    no_witness = tmp / "excl_no_witness"
    no_witness.mkdir()
    (no_witness / "bystander.log").write_text(BYSTANDER, encoding="utf-8")
    unbounded = no_witness / "loss.log"
    unbounded.write_text(LOSS_LINE, encoding="utf-8")
    bad = {
        "file": str(unbounded),
        "reason": "no witness — an exclusion over everything this file will ever say",
        "owner": "link",
        "date": "2026-08-07",
    }
    arms.expect_error(
        "an exclusion with no witness is an outage, not a permissive default",
        "PLANTED",
        ["--incident-records", str(plant_records(no_witness, [bad])), str(no_witness)],
        "incident_record_file_unreadable",
    )

    # REPLAYED, on the shipped record and the shipped artifact: take Tank's arm-A
    # repetition-0 capture and its real accounted count out of
    # ci/device_loss_incident_records.json, append one more real device-lost line to a
    # COPY, and require the audit to see the difference. This is the arm that proves the
    # number in the shipped file is the number the check actually holds it to — a planted
    # pair proves the rule, not that the rule is wired to the real list.
    shipped_records = REPO_ROOT / "ci" / "device_loss_incident_records.json"
    gate_capture = REPO_ROOT / "bench" / "results" / "device_loss_gate" / "armA_rep000.capture.txt"
    try:
        shipped = json.loads(shipped_records.read_text(encoding="utf-8"))
        accounted = next(
            r["witness"]["device_lost_reported"]
            for r in shipped["records"]
            if r["file"] == "bench/results/device_loss_gate/armA_rep000.capture.txt"
        )
        replay_dir = tmp / "excl_shipped_replay"
        replay_dir.mkdir()
        (replay_dir / "bystander.log").write_text(BYSTANDER, encoding="utf-8")
        copy = replay_dir / "armA_rep000.capture.txt"
        copy.write_text(
            gate_capture.read_text(encoding="utf-8", errors="replace")
            + LOSS_AT.format("a further 3141"),
            encoding="utf-8",
        )
        arms.expect_condition(
            f"Tank's real ctx-4096 capture, +1 loss, against its shipped count of {accounted}",
            "REPLAYED",
            [
                "--incident-records",
                str(
                    plant_records(
                        replay_dir, [excl(copy, {"device_lost_reported": accounted})]
                    )
                ),
                str(replay_dir),
            ],
            "incident_record_widened",
        )
    except Exception as exc:  # noqa: BLE001
        arms.arm(
            "Tank's real ctx-4096 capture, +1 loss, against its shipped count",
            "REPLAYED",
            False,
            f"could not build the replay: {exc} — an instrument outage in this control, "
            "not a pass",
        )

    # --- Instrument-outage arms: an outage is never a pass ------------------
    arms.expect_error("no path named at all", "PLANTED", [], "no_paths_given")
    empty = tmp / "empty_dir"
    empty.mkdir()
    arms.expect_error(
        "a directory with nothing readable in it",
        "PLANTED",
        [str(empty)],
        "nothing_scanned",
    )
    arms.expect_error(
        "a path that does not exist",
        "PLANTED",
        [str(tmp / "absent.log")],
        "nothing_scanned",
    )

    # --- Report -------------------------------------------------------------
    width = max(len(n) for n, _, _, _ in arms.results)
    print("\nNEGATIVE CONTROL: ci/check_device_loss.py")
    print("=" * (width + 34))
    failed = 0
    for name, provenance, ok, note in arms.results:
        status = "FIRED" if ok else "DID NOT FIRE"
        print(f"  [{status:>12}] {name:<{width}}  {provenance}")
        print(f"                 {note}")
        if not ok:
            failed += 1
    live = sum(1 for _, p, _, _ in arms.results if p.startswith("LIVE"))
    replayed = sum(1 for _, p, _, _ in arms.results if p.startswith("REPLAYED"))
    planted = sum(1 for _, p, _, _ in arms.results if p.startswith("PLANTED"))
    print(
        f"\n  provenance: {live} LIVE, {replayed} REPLAYED, {planted} PLANTED "
        f"of {len(arms.results)} arms."
    )
    print(
        "  A PLANTED arm proves the rule fires on an input built to make it fire. It does "
        "not evidence that the rule's event occurs in reality; only the LIVE and REPLAYED "
        "arms do that."
    )
    if failed:
        print(f"\nNEGATIVE CONTROL: FAIL — {failed} arm(s) did not fire.")
        print(
            "An arm that does not fire means the screen cannot detect the thing that arm "
            "describes. Quote the arm, not the count."
        )
        return 1
    print("\nNEGATIVE CONTROL: PASS — every arm fired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
