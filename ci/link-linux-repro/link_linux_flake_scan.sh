#!/usr/bin/env bash
# Scratch helper: list the failing `cargo test --lib` names on Linux, once per run,
# to separate stable failures from order-dependent (flaky) ones.
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
cd /mnt/c/Users/justinchu/dev/ep-vulkan-link/rust || exit 1
runs="${1:-5}"
for i in $(seq 1 "$runs"); do
  echo "--- run $i ---"
  cargo test --lib 2>&1 | sed -n '/^failures:$/,/^test result/p' | grep -E '^    [a-z]' | sort -u
done
