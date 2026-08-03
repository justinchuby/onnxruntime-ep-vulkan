#!/usr/bin/env bash
# Scratch helper: map each failing `cargo test --lib` name to its panic line on Linux.
set -uo pipefail
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
cd /mnt/c/Users/justinchu/dev/ep-vulkan-link/rust
cargo test --lib 2>&1 | awk '
  /^---- / { name = $2 }
  /panicked at/ { getline msg; print name " :: " substr(msg, 1, 100) }
' | sort -u
