#!/usr/bin/env bash
# Scratch helper: run the Linux-lane steps that sat behind the misnamed clippy step,
# one at a time, reporting each step's own exit status. Deliberately does NOT stop at
# the first red: the question is which of them pass on their own merits, and a runner
# that halts at the first failure is the behaviour that hid them in the first place.
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
cd /mnt/c/Users/justinchu/dev/ep-vulkan-link/rust || exit 1

step() {
  local label="$1"; shift
  echo "=== STEP: $label ==="
  set +e
  "$@" 2>&1 | tail -6
  echo "exit=${PIPESTATUS[0]}"
  set -e
}

step "layering lint"        cargo test --test layering
step "portability lint"     cargo test --test portability
step "integration targets"  cargo test --test cdylib_load --test dump_capabilities --test host_registration --test validation_control
