#!/usr/bin/env bash
# Scratch helper: is `counters ABI 4, but this epctl understands 8` a LIVE writer defect,
# or a STALE FILE the gate spliced into?
#
# Session 18b recorded 13 Linux ERROR(instrument) all downstream of `epctl --check-counters`
# exit 3 with that message, and recorded explicitly that ONE FIX GREENING ALL THIRTEEN was
# not yet demonstrated. This is the demonstration arm: remove the counters file, re-run the
# gate that produces it, and read the abi_version the EP actually writes.
#
# ARM A  the file as the previous run left it  -> abi_version
# ARM B  the file deleted, gate re-run          -> abi_version
# If B is 8, the "ABI 4" was never a writer defect: it was a stale artifact surviving a
# `mkdir -p` that never cleans, and the message named the file's age as an ABI break.
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:/usr/bin:/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
REPO=/mnt/c/Users/justinchu/dev/ep-vulkan-link
ORT_LIB=$(python3 -c 'import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), "capi"))')
export LD_LIBRARY_PATH="$ORT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO" || exit 1
OUT=bench/results/link-linux-downstream
BIN="$CARGO_TARGET_DIR/release"
export ONNXRUNTIME_VULKAN_EP_LIB="$BIN/libonnxruntime_vulkan_ep.so"

echo "=== ARM A: the file as the last run left it ==="
python3 -c "import json;d=json.load(open('$OUT/counters-linux.json'));print('abi_version =', d.get('abi_version'))"
"$BIN/epctl" --check-counters "$OUT/counters-linux.json" --require-dispatches 1 2>&1 | grep -E 'NO REPORT|ABI' | head -3
echo "arm A epctl exit=${PIPESTATUS[0]}"

echo
echo "=== ARM B: delete it, re-run the gate that writes it ==="
rm -f "$OUT/counters-linux.json"
ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE="$PWD/$OUT/counters-linux.json" \
  python3 ci/gate_chain_fp32.py --verdict-out "$OUT/verdict-linux.json" --workdir "$OUT" \
  > "$OUT/gate-armb.log" 2>&1
echo "gate exit=$?"
python3 -c "import json;d=json.load(open('$OUT/counters-linux.json'));print('abi_version =', d.get('abi_version'))"
"$BIN/epctl" --check-counters "$OUT/counters-linux.json" --require-dispatches 1 2>&1 | tail -6
echo "arm B epctl exit=${PIPESTATUS[0]}"
