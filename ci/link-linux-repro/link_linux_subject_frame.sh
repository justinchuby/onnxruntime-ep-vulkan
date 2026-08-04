#!/usr/bin/env bash
# Scratch helper: rebuild the release .so from scratch and read the SUBJECT ARITHMETIC
# that gen_proof_ledger.py --check now prints, on Linux.
#
# The row this exists to read is PROVEN-ELSEWHERE{toolchain}: on the 2026-08-03 reading
# it was structurally unreachable (classifier tested spirv_digest first, unconditionally)
# and the toolchain note was computed and discarded before `return 1`. Both are claimed
# repaired by b1dab01. This script re-reads them on a FRESH .so.
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:/usr/bin:/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
REPO=/mnt/c/Users/justinchu/dev/ep-vulkan-link
cd "$REPO/rust" || exit 1

echo "=== STEP: forced fresh release build ==="
touch src/ep.rs
cargo build --release 2>&1 | tail -3
echo "build exit=${PIPESTATUS[0]}"

SO="$CARGO_TARGET_DIR/release/libonnxruntime_vulkan_ep.so"
ls -l "$SO"
sha256sum "$SO"
echo "--- exported symbols (two-digest accessors) ---"
nm -D --defined-only "$SO" | grep -i -E 'digest|toolchain' | head -20

export ONNXRUNTIME_VULKAN_EP_LIB="$SO"
cd "$REPO" || exit 1
OUT=bench/results/link-linux-subject-frame
mkdir -p "$OUT"

echo
echo "=== STEP: gen_proof_ledger.py --check (Linux, fresh .so) ==="
python3 rust/tools/gen_proof_ledger.py --check 2>&1 | tee "$OUT/ledger-check-linux.log" | tail -40
echo "check exit=${PIPESTATUS[0]}"

echo
echo "=== GREP: 'toolchain' in the live Linux --check output ==="
grep -n -i 'toolchain' "$OUT/ledger-check-linux.log" | head -30
echo "toolchain hits=$(grep -c -i toolchain "$OUT/ledger-check-linux.log")"

echo
echo "=== GREP: subject arithmetic line ==="
grep -n -E 'identical|SOURCE-COSMETIC|PROVEN-ELSEWHERE|SUBJECT-CHANGED|SUBJECT-INDETERMINATE|no-module-in-build' "$OUT/ledger-check-linux.log" | head -20
