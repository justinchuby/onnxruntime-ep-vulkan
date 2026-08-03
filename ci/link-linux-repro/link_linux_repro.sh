#!/usr/bin/env bash
# Scratch helper: is the release cdylib byte-reproducible from identical source?
# Two forced rebuilds, same tree, hashes printed. An unchanged hash is only evidence
# of "same bytes" if the build actually ran — see the Windows result.
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:/usr/bin:/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
cd /mnt/c/Users/justinchu/dev/ep-vulkan-link/rust || exit 1
for i in 1 2; do
  touch src/ep.rs
  cargo build --release > /dev/null 2>&1
  sha256sum "$CARGO_TARGET_DIR/release/libonnxruntime_vulkan_ep.so"
done
