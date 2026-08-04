#!/usr/bin/env bash
# Scratch helper: after re-witnessing source_digest under the §8.9.19 hashing rule,
# rebuild on Linux and read the subject arithmetic again.
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:/usr/bin:/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
REPO=/mnt/c/Users/justinchu/dev/ep-vulkan-link
OUT="$REPO/bench/results/link-linux-subject-frame"
mkdir -p "$OUT"

cd "$REPO/rust" || exit 1
cargo build --release 2>&1 | tail -2
export ONNXRUNTIME_VULKAN_EP_LIB="$CARGO_TARGET_DIR/release/libonnxruntime_vulkan_ep.so"
sha256sum "$ONNXRUNTIME_VULKAN_EP_LIB"

cd "$REPO" || exit 1
python3 rust/tools/gen_proof_ledger.py --check > "$OUT/ledger-check-linux-final.log" 2>&1
echo "LINUX_CHECK_EXIT=$?"
head -8 "$OUT/ledger-check-linux-final.log"
echo "--- 'toolchain' hits in the live Linux --check output: $(grep -c -i toolchain "$OUT/ledger-check-linux-final.log") ---"
grep -n -i toolchain "$OUT/ledger-check-linux-final.log" | cut -c1-200
