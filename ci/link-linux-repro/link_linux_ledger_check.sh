#!/usr/bin/env bash
# `gen_proof_ledger.py --check` against a FRESH Linux .so, with the subject actually set.
# Without ONNXRUNTIME_VULKAN_EP_LIB the check PASSes having read nothing, so the export is
# the load-bearing line here, not the invocation.
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:/usr/bin:/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
REPO=/mnt/c/Users/justinchu/dev/ep-vulkan-link
export ONNXRUNTIME_VULKAN_EP_LIB="$CARGO_TARGET_DIR/release/libonnxruntime_vulkan_ep.so"
cd "$REPO" || exit 1
ls -l "$ONNXRUNTIME_VULKAN_EP_LIB"
python3 rust/tools/gen_proof_ledger.py --check 2>&1 | tee bench/results/link-linux-downstream/ledger-check-linux.log | tail -25
echo "exit=${PIPESTATUS[0]}"
echo "toolchain grep: $(grep -c toolchain bench/results/link-linux-downstream/ledger-check-linux.log)"
