# Criterion 10 after `872d739` — the route, the axis, and the run that would fail it

**Recorded:** 2026-08-02T23:xx-07:00 by Trinity, on `squad/trinity` merged to `main` @ `6ef62bb`.
**Binary:** `onnxruntime_vulkan_ep.dll` `7F55C0C1CD68…` → rebuilt `5D0E5726B6EC73DB714EBBD429E640C18B7DEB09A928B997FDF7E647FDB4BAA8`, `rebuilt=True`.
**Control binary:** `872d739^` built in a scratch worktree → `8FC56D895159DB2E8B01F9A2322B19CE05135BE6F503CE417DC12B4BC4F29EAA`.
**Artifact:** `phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx`, one session, 3 consecutive runs, 65 outputs.
**Devices, read off the run (`alloc_device_frame_session_devices`), never off the selector:**
selector 0 → `1=NVIDIA GeForce RTX 4060 Laptop GPU`; selector 1 → `0=Intel(R) Iris(R) Xe Graphics`.
Note the leading digits: the allocator's enumeration index is **not** the selector.

No wall-clock figure is quoted anywhere below. The box is permanently contended.

---

## 1. The brief's premise was false for the run criterion 10 performs

> "Switch has just made those tensors device-resident, so the code path they take has
> changed underneath this criterion."

Measured, both devices, on the merged tree:

| run | `outputs_device_bound` | `outputs_host_resident` | `alloc_device_authority_grants` | `alloc_device_downloads` | route |
|---|---|---|---|---|---|
| criterion-10 lane, default env | 0 | 196 | 0 | 0 | `HOST_STAGING` |
| `DEVICE_MEMORY=1 BIND_OUTPUTS=1` | 196 | 0 | 196 | 196 / 914 752 B | `DEVICE_AUTHORITATIVE` |

`ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS` ships **OFF** (`vk/session.rs`), so the criterion-10
lane's KV outputs still leave the island through the host staging block. The path did not
change underneath the criterion; a **second** path appeared beside it, and the criterion
was measuring only one of the two while saying neither.

That is now in the record. Every criterion-10 artifact carries `kv_writeback_route`, read
off the counters the run emitted and never off the environment variable that requested it —
Step 1c **unbinds on refusal**, so a declined bind would otherwise be recorded as a route
that was taken.

## 2. Both routes, all 65 outputs, per output — and they are identical

`tests/ops/compare_criterion10_records.py`, per-output, no aggregate:

```
dev0 HOST_STAGING vs dev0 DEVICE_AUTHORITATIVE : 0 of 65 outputs moved
dev1 HOST_STAGING vs dev1 DEVICE_AUTHORITATIVE : 0 of 65 outputs moved
   both: 65 compared, 62 within tolerance, 0 degenerate, failing [0, 63, 64]
```

`median/p99/max ULP`, `max_abs_diff` and the cancellation count agree to the digit on every
one of the 65 outputs. The device-authoritative writeback delivers the same bytes as the
staging block.

**A null reading is inadmissible until the same apparatus is shown able to produce a
non-null one** (Switch's rule, 2026-08-02). Two demonstrations, on the same comparison, on
the real artifact:

* **dev0 vs dev1**, same route: **61 of 65** outputs move (in the cancellation-sensitive
  `p99`/`max`; medians and `max_abs_diff` do not). The comparison is not blind.
* **the pre-`872d739` binary with outputs bound** — see §4.

## 3. The ULP work, per layer, across all 32 layers

Prediction on record before any ULP existed
(`bench/results/criterion10-ulp-prediction.md`): **flat at 1–3 across all 32 layers; flat
⇒ no defect, a step ⇒ a located one, and the layer index is the location.**

Measured. Median ULP, `key/value`, in **depth order** — byte-identical on both devices and
on both writeback routes:

```
L0 :0/0  L1 :0/1  L2 :0/1  L3 :1/1  L4 :1/1  L5 :1/1  L6 :1/1  L7 :1/1
L8 :1/1  L9 :1/1  L10:1/1  L11:1/1  L12:1/1  L13:1/1  L14:1/1  L15:0/1
L16:1/1  L17:2/1  L18:1/1  L19:1/1  L20:1/1  L21:1/1  L22:1/1  L23:1/1
L24:2/2  L25:1/1  L26:2/2  L27:2/2  L28:2/2  L29:2/2  L30:3/3  L31:4/4
```

* **Flat, and the prediction holds for the KV cache.** Largest layer-to-layer step is
  **1 ULP**. There is no discontinuity at any layer: the exceedance at layer 31 is the top
  of a smooth climb that begins around layer 24, not a step.
* **Two exceedances, both at layer 31** (key and value, 4 ULP against a predicted ceiling
  of 3). Recorded as exceedances rather than absorbed, because the alternative was to
  widen the band after seeing it. `multiple_of_baseline = 4`, against `12` for the logits.
* **The step in this model is output 0**, the logits, at **12 ULP** — and it is not a
  layer. "Flat at 1–3 across all 32 layers" is true of the layers and false of the model.

## 4. The run that would fail criterion 10 if the claim were false — and it is reachable

Named before criterion 10 is recorded as met, per R9's third generalisation:

> **The run:** the criterion-10 three-run series on Phi-3.5 with
> `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1 ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS=1`, on a build
> whose bound spans are not marked device-authoritative.

**It is reachable, it was run, and it fails**, on the binary from three hours before this
was written (`872d739^`, `8FC56D89…`), with today's harness:

```
device (read off the run) : NVIDIA GeForce RTX 4060 Laptop GPU
KV writeback route        : DEVICE_AUTHORITATIVE (device-bound 196, host-resident 0,
                            authority grants None — the counter does not exist in this build)
oracle                    : NOT_PERFORMED over 0/65 outputs, 65 degenerate  (x3 runs)
cross_run_identical_to_run1: True   <- wrong AND stable, on all three runs
series verdict            : UNMEASURED
```

This is not a plant. It is the project's own previous commit, and it is **the exact defect
shape criterion 10 was reopened for**: every output wrong, every output stable, so
cross-run identity passes it perfectly. The three gates behave as they should:

| gate | on the pre-`872d739` bound run | what it proves |
|---|---|---|
| cross-run identity (all 65) | **passes** — bit-identical every run | determinism only |
| logits-only oracle | DISAGREE (`argmax 0` vs `30751`) | caught this one, by luck of zeros |
| all-output oracle + non-triviality guard | **NOT_PERFORMED → UNMEASURED** | absence of evidence, named as such |

The logits-only comparison happens to catch *this* instance, because zeros move the argmax.
It is still blind to the class: a KV-only defect leaves the logits bit-identical, which is
what `tests/ops/test_criterion10_oracle.py::test_the_old_one_output_comparison_would_have_passed_the_same_plant`
demonstrates directly.

## 5. `atol` was not touched

`within` is still `np.allclose` on the incumbent tolerance.
`tests/ops/test_criterion10_ulp.py::test_the_ulp_statistic_does_not_move_the_pass_fail_decision_yet`
reads the source of `compare_all_outputs_to_cpu` and fails if the gate becomes
ULP-denominated. **The verdict is still `DIVERGENT` on both devices and both routes.**
Criterion 10 stays open. Nothing went green because a number moved.

The tolerance question — an absolute `atol` applied to tensors of growing scale — is
argued in `.squad/decisions/inbox/trinity-criterion10-route-axis-and-tolerance.md` and is
for Morpheus to rule on, not for me to apply.

## 6. The near-miss, which is the part worth reading

I measured the depth curve, read it off the artifact, and concluded **the axis was
wrong** — that the residual peaked at **layer 9** and fell back, making the deepest layers
the quietest in the model. I had the write-up drafted.

It was false. `criterion10-dev*.json` is serialised with `json.dumps(..., sort_keys=True)`,
so `output_coverage.per_output` is stored **alphabetised**, and `present.10` sorts before
`present.2`. Zipping that key order against the ULP curve puts `present.9.value` at index
64 and `present.31.value` at index 52. The curve it produces is smooth, plausible,
reproducible, and identical on both vendors — every property one uses to believe a
reading.

It was caught by asking the **session** for its output order instead of the file:
`sess.get_outputs()` returns depth order, and this model's true order is *not* its own
sort. Fixed three ways:

* the record now carries `output_names` in session order, with `output_names_order` saying
  what it is and what it is not;
* `_kv_depth.assert_names_are_session_order()` refuses a name list equal to its own sort —
  a cheap, positive tell, and the test asserts the tell can only work because the two
  orders differ for this model;
* `tests/ops/test_criterion10_depth.py` pins both readings on the same medians and asserts
  they reach **different conclusions**, so the refusal is tested against a finding rather
  than against a string list.

R12's fourth generalisation, one level out: the frame of a name is the run that produced
it, not the file that stored it. A sorted container had discarded the only property it was
being read for.

## Artifacts

```
bench/results/criterion10_route-dev0-host_staging.json
bench/results/criterion10_route-dev0-device_authoritative.json
bench/results/criterion10_route-dev1-host_staging.json
bench/results/criterion10_route-dev1-device_authoritative.json
bench/results/criterion10_route-dev0-preswitch_control.json      <- the run that fails it
bench/results/criterion10_route_delta-dev0.json
bench/results/criterion10_route_delta-dev1.json
bench/results/criterion10_route_delta-dev0-preswitch_control.json <- the non-null control
```

Reproduce the tables with
`python tests/ops/print_criterion10_table.py bench/results/criterion10_route-dev0-device_authoritative.json`.

## Verification

* `tests/ops/test_criterion10_{depth,route,oracle,ulp}.py` — **61 passed**, GPU-free,
  deterministic, no model required.
* criterion-10 lane, both devices, both routes — records written on the failing path, as
  designed; verdict `DIVERGENT`, `ERROR(instrument): 0` throughout.
* `cargo test --lib` — 492 passed. See the note below.
* `cargo clippy --all-targets` — exit 0, no warnings.

**Reported, not fixed:** `counters::tests::a_variant_is_named_by_its_spec_constants_and_not_by_its_stem`
failed **once** in a full `cargo test --lib` run and then passed in isolation and in four
consecutive full runs (1 failure in 6). `git diff origin/main -- rust/` is empty on this
branch, so it is not this branch's condition. I do not have the failure text — I kept only
the tail — which weakens the report, and it is made anyway: intermittent is a bug report,
not a quarantine reason. Owner: Mouse/Switch.
