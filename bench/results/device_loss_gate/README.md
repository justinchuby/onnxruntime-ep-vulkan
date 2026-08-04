# ctx-4096 device-loss gate — captures, and what each file is

Produced by `rust/tools/device_loss_gate.py` on 2026-08-04 by Tank, investigating
`device_memory_flip_blocker_ctx4096_device_loss` in `ci/open_reds_device.json`.

**Every arm ran the resident lane only**, `--steps 8 --seed-past 4096`, with
`ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` and `ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS=1` pinned
explicitly and `KV_ARENA` off — the arena cannot be exercised through this probe at all
(see `.squad/decisions/inbox/tank-the-arena-was-never-in-the-frame.md`). Every repetition
is screened on `dispatches_executed > 0`, with the screen deliberately **not** firing on a
repetition that recorded a device loss: a loss at Compute #1 records 0 dispatches, and
treating that as vacuous would delete the worst observation from numerator and denominator
together.

The predictions for arms A–E were written **before** the arms ran and are in
`bench/results/device_loss_arms_predictions.json`, together with the two that came back
wrong.

| file | arm | what it is |
|---|---|---|
| `armA_baseline.log` | A | uncapped baseline, 5 repetitions. **3 losses in 5.** The reproduction. |
| `armA_rep000/1/2.capture.txt` | A | the three repetitions that **lost the device**. rep000 lost at Compute #6 after 1775 dispatches; rep001 and rep002 lost at Compute #1 with 0 dispatches. |
| `armA_rep003/4.capture.txt`, `.counters.json` | A | the two clean repetitions, for the side-by-side. Note `alloc_high_water_bytes`. |
| `armB_budget.log`, `armB_rep002.counters.json` | B | `--budget-mb 5600`. 3/3 clean **and worthless**: `alloc_high_water_bytes = 5518426760` is identical to arm A's, so the cap never bound and nothing was refused. A null manipulation. |
| `armD_cotenant.log`, `armD_rep000/1.capture.txt` | D | 2560 MiB held and touched by a foreign process (`rust/examples/vram_occupant.rs`). 2/2 clean. |
| `armD2_cotenant.log`, `armD2_rep000/1.counters.json` | D2 | 3584 MiB held. 3584 + 5262 = 8846 MiB against a 7959 MiB board — a genuine overcommit. 2/2 clean, 8/8 Computes, 2840 dispatches, `alloc_device_attach_failures = 0`. Board occupancy is not the variable. |

Arm C (ctx 2048) was **withdrawn before running**, with the reason recorded: arm B had
already shown the EP's footprint is constant across losing and clean runs, so arm C could
not have discriminated anything. Arm E is not a gate run — it is the `heapBudget`
measurement in the predictions file, and it is the one that stopped a fix shipping.

`rep005.counters.json` is deleted rather than kept: it was a partial capture from the
repetition in flight when arm A was stopped, and a partial is not an observation.

**The one number to carry away:** `alloc_high_water_bytes = 5518426760` in every capture
here — capped, uncapped, losing, clean, with a co-tenant and without. A constant cannot be
the variable.
