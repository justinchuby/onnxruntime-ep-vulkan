"""One-shot amendment of ci/open_reds_device.json for the ctx-4096 lane-identity resolution.

Written as a script rather than hand-edited JSON because the register is append-only in
`subjects` and validated by ci/check_open_reds.py; a hand edit that drops a subject prints
PASS while ruling on fewer things, which is the exact defect `_check_subjects` exists for.
"""
import json
import pathlib

REG = pathlib.Path(__file__).resolve().parents[2] / "ci" / "open_reds_device.json"
doc = json.loads(REG.read_text(encoding="utf-8"))

LANE_IDENTITY = (
    " LANE IDENTITY RESOLVED 2026-08-04 (switch), because the reopened reason above eliminated "
    "the arena by showing the EP REFUSES under KV_ARENA=1, and that raised the question of what "
    "lane the signature run was actually in. Answer: DEVICE_MEMORY=1 with KV_ARENA OFF. It is "
    "NOT mis-attributed and it is NOT an arena fault. Two independent lines. (a) STRUCTURAL, and "
    "it is the decisive one: the signature record's failure is `vkWaitForFences failed: the "
    "Vulkan device was lost`, which is reached only AFTER a command buffer has been submitted. "
    "The alias sweep in vk/session.rs runs BEFORE any submit and bails the whole Compute. On "
    "Phi-3.5 `present` is declared symbolically (`total_sequence_length`), so "
    "`declared_present_len` is None and EVERY GQA node arenas when the flag is armed "
    "(ops/attention.rs, `let arena = ctx.kv_arena() && declared_present_len.is_none() && ...`) -- "
    "there is no partial arming. A run that reached a fence wait therefore had zero aliased "
    "outputs, therefore did not have the arena armed. The guard predates the loss: it landed in "
    "0a91fda, 2026-08-03 10:14:55, and the loss is 2026-08-03 23:41:42. (b) MEASURED, at ctx "
    "4096 on the merged build with both flags pinned explicitly in every arm: arm A "
    "(DEVICE_MEMORY=1, KV_ARENA=0) runs 710 dispatches with kv_cache_convention=GROWING, 4/4 "
    "clean; arm B (DEVICE_MEMORY=1, KV_ARENA=1) exits 1 at the alias refusal with 0 dispatches, "
    "confirming the refusal is total; arm C (KV_ARENA=1 on probe_kv_arena_phi35.py, which binds "
    "the SAME OrtValue as `past` input and `present` output, arena 4096 / past 4090) runs 710 "
    "dispatches with kv_cache_convention=SHARED, ARENA_TAKEN_AND_BIT_IDENTICAL, 0 losses. Arm C "
    "is the case the brief called `a probe that binds correctly`: the arena executes at this "
    "context and did not lose the device in this observation. One observation is not a rate and "
    "is not offered as one -- the attribution rests on (a). Artifacts: "
    "bench/results/lane-identity-arm{A-arena0,B-arena1,C-arena4096}.json and "
    "lane-identity-armA-screened-r{1,2,3}.json, each carrying `env_pinned` and "
    "`kv_cache_convention` in the record, which the signature record did not."
    " WHAT THIS COSTS THE SCORING, and it is not nothing. The four-gate scoring applied one "
    "discriminator, written down before any run: A GATE BLOCKS THE FLIP ONLY IF ITS FAILURE "
    "SEPARATES THE LANES. That discriminator was never actually applied to this gate, and when "
    "applied it does not hold -- not because the armed lane is clean, but because THE UNARMED "
    "LANE CANNOT BE OBSERVED AT ctx 4096. Measured 2026-08-04, deterministic, 3 of 3: the "
    "DEVICE_MEMORY=0 lane at seed_past 4096 fails inside ORT with `gpu-allocator failed to "
    "allocate 14155776 bytes for 'ep_in_427': Out of memory`, ORT silently re-executes all 355 "
    "nodes on the CPU EP, and the worker exits 0 with no exception and dispatches_executed = 0. "
    "So at this context there is no OFF lane to compare against: the flag cannot be shown to "
    "cause the loss, because the only lane that reaches a dispatch is the armed one. Recorded as "
    "a separate subject, device_memory_ctx4096_shipping_lane_cannot_run. This entry therefore "
    "keeps its id (subjects is append-only) but its CORRECTED NAME is: an UNSEPARATED "
    "device loss at ctx 4096, reachable only on the device-resident lane because it is the only "
    "lane that runs there. It is a real fault and Tank's closes_when stands unchanged."
)

SEPARATION_CLAUSE = (
    " ADDED 2026-08-04 by switch, and it is a SECOND necessary condition, not an alternative: "
    "the fix must also be shown on a lane comparison that EXISTS. Today it cannot be, because "
    "the DEVICE_MEMORY=0 lane at ctx 4096 executes zero EP dispatches. Either "
    "device_memory_ctx4096_shipping_lane_cannot_run closes first, giving this entry a real "
    "control lane, or the demonstration must be made at a context where both lanes run and the "
    "fault is still reachable. A quiet box cannot satisfy either clause, and neither can a lane "
    "that was never running."
)

for c in doc["checks"]:
    if c["id"] == "device_memory_flip_blocker_ctx4096_device_loss":
        c["reason"] += LANE_IDENTITY
        c["closes_when"] += SEPARATION_CLAUSE
        c["lane_identity"] = (
            "DEVICE_MEMORY=1, KV_ARENA=0 (GROWING). Established structurally and by measurement "
            "2026-08-04; see `reason`."
        )

NEW_ID = "device_memory_ctx4096_shipping_lane_cannot_run"
if NEW_ID not in doc["subjects"]:
    doc["subjects"].append(NEW_ID)

doc["checks"].append({
    "id": NEW_ID,
    "cmd": ["python", "ci/check_device_loss.py",
            "bench/results/lane-identity-armA-arena0.json"],
    "expect": "red",
    "signature": "alloc_device failed for input buffer",
    "owner": "switch",
    "opened": "2026-08-04",
    "review_by": "2026-11-01",
    "reason": (
        "THE SHIPPING LANE CANNOT RUN Phi-3.5 AT ctx 4096 AT ALL, AND IT SAYS SO ONLY ON stderr. "
        "With ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=0 and seed_past 4096, the EP claims all 355 "
        "nodes, then its first Compute fails at `gpu-allocator failed to allocate 14155776 bytes "
        "for 'ep_in_427': Out of memory`. ORT re-executes the whole fused subgraph on the CPU EP, "
        "get_providers() still lists VulkanExecutionProvider, and the process EXITS 0 with no "
        "exception. Counters: dispatches_executed = 0, compute_calls = 1, compute_failures = 1, "
        "kv_cache_convention = UNOBSERVABLE. Deterministic: 3 of 3, both with KV_ARENA=0 and "
        "with KV_ARENA=1, so it is not the arena either. MECHANISM, and it is the same one "
        "measured in session 49's `ship_threads` arm: the shipping path allocates a transient "
        "device buffer per input per Compute, so at ctx 4096 the input side alone exceeds what "
        "is left of a 7959 MiB board. The device-resident lane does not, because ORT's own "
        "OrtValues already hold the KV on the device and are bound rather than copied -- 710 "
        "dispatches, alloc_high_water_bytes 5513708168, clean. "
        "WHY THIS IS ITS OWN SUBJECT AND NOT A NOTE ON THE BLOCKER: it is what removes the "
        "blocker's control lane. A device loss observed only on the armed lane, at a context "
        "where the unarmed lane executes nothing, is UNSEPARATED -- the flag cannot be shown to "
        "cause it. That reading is only available once this is written down as a fact about the "
        "OFF lane rather than as an incidental failure in someone else's record. "
        "IT WAS INVISIBLE TO THE DETECTOR UNTIL TODAY, AND THAT WAS MY INSTRUMENT'S FAULT. "
        "probe_kv_chain_phi35.py::_run_lane captured stderr ONLY when the worker's return code "
        "was non-zero -- and this failure's defining property is that the return code is ZERO. "
        "The probe deleted the single artifact ci/check_device_loss.py screens for. Verified by "
        "mutation on the same artifact name and the same arm: PASS before the probe fix, "
        "FAIL(condition=broken_commitment_reported) after, quoting three lines. This is the "
        "third time on this project that a lane reported a state it was not in, and the second "
        "of those three in an instrument of mine."
    ),
    "closes_when": (
        "The DEVICE_MEMORY=0 lane completes a 2-step Phi-3.5 chain at seed_past 4096 with "
        "dispatches_executed > 0 and compute_failures = 0, on the same box and board, with the "
        "allocation change ARGUED -- the shipping path's per-Compute transient input allocation "
        "is a design fact, not a budget accident, so the fix is to stop making the allocation, "
        "not to make the board bigger. A run that passes because something else was not resident "
        "does NOT close this: the entry closes on a counter showing the transient allocation is "
        "gone (alloc_allocations per Compute on the OFF lane falling to the same order as the "
        "resident lane's), not on a green exit. It does NOT close by raising the context down to "
        "one that fits, and it does NOT close by deleting the artifact -- if the artifact is "
        "regenerated it must be regenerated by a run, with stderr captured."
    ),
    "timeout": 900,
})

REG.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"amended: {len(doc['checks'])} checks, {len(doc['subjects'])} subjects")

