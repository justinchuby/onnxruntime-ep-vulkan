#!/usr/bin/env bash
# Scratch helper: run the Linux-lane steps that sat behind the misnamed clippy step,
# one at a time, reporting each step's own exit status. Deliberately does NOT stop at
# the first red: the question is which of them pass on their own merits, and a runner
# that halts at the first failure is the behaviour that hid them in the first place.
#
# 2026-08-03: each step now TEES its output and the matching productivity floor is run
# against the capture, because `cargo test` exits 0 on `running 0 tests` and the exit
# status this script prints cannot see that. The floors were measured on Windows; this
# is where the claim that Linux agrees is checked rather than assumed.
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
REPO=/mnt/c/Users/justinchu/dev/ep-vulkan-link
OUT="$REPO/bench/results/link-suite-productivity"
mkdir -p "$OUT"
cd "$REPO/rust" || exit 1

step() {
  local label="$1"; local log="$2"; local suite="$3"; shift 3
  echo "=== STEP: $label ==="
  set +e
  "$@" 2>&1 | tee "$OUT/$log" | tail -4
  echo "exit=${PIPESTATUS[0]}"
  (cd "$REPO" && python3 ci/check_suite_productivity.py \
     --suite "$suite" --harness libtest --lane build-test-linux \
     "bench/results/link-suite-productivity/$log" | head -8)
  echo "floor-exit=$?"
  set -e
}

step "layering lint" cargo-test-layering-linux.log "cargo test --test layering" \
  cargo test --test layering
step "portability lint" cargo-test-portability-linux.log "cargo test --test portability" \
  cargo test --test portability --release
step "integration targets" cargo-test-integration-linux.log \
  "cargo test --test cdylib_load,dump_capabilities,host_registration,validation_control" \
  cargo test --release --test cdylib_load --test dump_capabilities --test host_registration --test validation_control