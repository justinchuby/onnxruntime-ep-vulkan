#!/usr/bin/env bash
# Scratch helper: the Linux-lane DEVICE steps that sat behind the misnamed clippy step.
#
# Each step reports its own exit status and the script does not stop at the first red:
# the question is which of them pass on their own merits, and halting at the first
# failure is precisely the CI behaviour that hid the six behind it.
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
REPO=/mnt/c/Users/justinchu/dev/ep-vulkan-link
ORT_LIB=$(python3 -c 'import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), "capi"))')
export LD_LIBRARY_PATH="$ORT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO" || exit 1

OUT=bench/results/link-linux-downstream
mkdir -p "$OUT"

echo "=== STEP: build release (cargo build --release) ==="
(cd rust && cargo build --release 2>&1 | tail -3)
echo "exit=$?"

BIN="$CARGO_TARGET_DIR/release"
export ONNXRUNTIME_VULKAN_EP_LIB="$BIN/libonnxruntime_vulkan_ep.so"

echo "=== STEP: Probe Vulkan loader (epctl --probe-loader) ==="
"$BIN/epctl" --probe-loader 2>&1 | tee "$OUT/loader-probe.log" | tail -12
echo "exit=${PIPESTATUS[0]}"

echo "=== STEP: Op-correctness + barrier-parity tests (pytest, lavapipe) ==="
touch "$OUT/.lane-reached"
python3 -m pytest tests/ops -q --tb=line -p no:randomly --strict-markers \
  2>&1 | tee "$OUT/pytest-linux.log" | tail -12
echo "exit=${PIPESTATUS[0]}"

echo "=== STEP: Criterion 10 gate artifact (gate_chain_fp32) ==="
ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE="$PWD/$OUT/counters-linux.json" \
  python3 ci/gate_chain_fp32.py \
    --verdict-out "$OUT/verdict-linux.json" \
    --workdir "$OUT" 2>&1 | tail -12
echo "exit=${PIPESTATUS[0]}"

echo "=== STEP: Criterion 10 verdict gate (independent reader) ==="
python3 ci/check_verdict.py "$OUT/verdict-linux.json" 2>&1 | tail -8
echo "exit=${PIPESTATUS[0]}"

echo "=== STEP: Criterion 10 verdict gate (epctl --check-counters) ==="
"$BIN/epctl" --check-counters "$OUT/counters-linux.json" --require-dispatches 1 2>&1 | tail -8
echo "exit=${PIPESTATUS[0]}"

echo "=== STEP: Known-fatal log line (check_fatal_log.py) ==="
python3 ci/check_fatal_log.py --lane-marker="$OUT/.lane-reached" "$OUT/pytest-linux.log" 2>&1 | tail -8
echo "exit=${PIPESTATUS[0]}"

echo "=== STEP: Proof-ledger portability screen ==="
python3 ci/check_ledger_portability.py --device-lane \
  --run-log "$OUT/pytest-linux.log" --loader-artifact "$OUT/loader-probe.log" 2>&1 | tail -12
echo "exit=${PIPESTATUS[0]}"
