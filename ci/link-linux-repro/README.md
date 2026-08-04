# Linux-lane reproducers (Link, 2026-08-02)

These are **not lane checks**. They are the scripts that produced the figures in
`docs/PLATFORMS.md` §7.21, kept so a reader can re-run them rather than take the numbers
on trust. Nothing in CI invokes them.

They run against **WSL Ubuntu** from a Windows checkout, which imposes two constraints
found the hard way and worth stating once:

* `$HOME` inside `wsl -d Ubuntu -- bash -lc` inherits the *Windows* value, so every script
  here sets `HOME=/home/justinchu` and `PATH="$HOME/.cargo/bin:$PATH"` explicitly. A
  non-login `bash` has neither cargo nor, in some invocations, `/usr/bin` on `PATH`.
* `CARGO_TARGET_DIR` must point **inside** WSL's own filesystem. Left unset it lands on
  `/mnt/c/.../rust/target`, where it clobbers the Windows build and races the other
  worktree.

They must be LF-terminated: a CRLF script fails as `cd: $'...\r': No such file or
directory`, which reads like a missing directory rather than a line ending.

| Script | What it answers |
| --- | --- |
| `link_linux_lint.sh` | Do `cargo check --all-targets` and `cargo clippy -- -D warnings` pass on Linux, each reporting its **own** exit status? |
| `link_linux_downstream.sh` | The three cargo steps that sat behind the misnamed clippy step (layering, portability, integration targets). |
| `link_linux_device_steps.sh` | The device steps behind it: loader probe, op-correctness pytest, Criterion 10 gate and both verdict readers, fatal-log check, ledger-portability screen. Deliberately does **not** stop at the first red — halting there is the CI behaviour that hid the rest. |
| `link_linux_failure_map.sh` | Maps each failing `cargo test --lib` name to its panic line, to separate ledger faults from anything else. |
| `link_linux_flake_scan.sh` | Lists failing test names once per run, to separate stable failures from order-dependent ones. |
| `link_linux_repro.sh` | Is the release `.so` byte-identical across forced rebuilds? (On Linux: yes. On Windows the `.dll` is not — see §7.21.3.) |
| `link_linux_subject_frame.sh` | Builds a fresh `.so` and runs `gen_proof_ledger.py --check` on it, capturing the **subject arithmetic line** and a `grep -c toolchain` over the same live output. This is the before-reading for §7.26. Sets `ONNXRUNTIME_VULKAN_EP_LIB`; without it `--check` compares against nothing and PASSes having read nothing. |
| `link_linux_subject_frame_after.sh` | Same reading, taken after `--backfill-frame --rewitness-source`. Kept separate from the `before` script so both readings survive rather than one overwriting the other. |
| `link_linux_counters_abi.sh` | Two arms against the `epctl --check-counters` exit 3 (`ABI 4 vs 8`). Arm A reads the existing snapshot; **Arm B deletes it and re-runs the gate**, which is what refuted the ABI story — no counters file was written at all, so the "ABI 4" was a stale snapshot's age, not a live mismatch. |
| `link_linux_shaderless_probe.sh` | Asks the *loaded* `.so` for its shader subject and `strings`-counts its SPIR-V. Written to explain why a mid-suite artifact reported `built without shaders`; found the shader-less witness had overwritten the shared build via an inherited `CARGO_TARGET_DIR`. |
| `link_linux_device_steps2.sh` | `link_linux_device_steps.sh` re-run after the `_shaderless.py` target-dir pin. Kept as a separate script so the 19-failed and 8-failed logs are both reproducible from a named command. |
