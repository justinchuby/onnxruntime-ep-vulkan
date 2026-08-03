#!/usr/bin/env bash
# Scratch helper: the Linux lane's clippy step, exactly as CI runs it, reporting its own
# exit status rather than the pipeline's.
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
cd /mnt/c/Users/justinchu/dev/ep-vulkan-link/rust || exit 1

echo "=== STEP: cargo check --all-targets (compile errors, not lints) ==="
cargo check --all-targets 2>&1 | tail -5
echo "check_exit=${PIPESTATUS[0]}"

echo "=== STEP: cargo clippy --all-targets -- -D warnings ==="
cargo clippy --all-targets -- -D warnings 2>&1 | tail -8
echo "clippy_exit=${PIPESTATUS[0]}"
