#!/usr/bin/env bash
# run_conformance.sh — run the Vulkan EP ONNX conformance suite (one op per subprocess)
#
# Prerequisites: see tests/conformance/README.md
# Usage:
#   ./run_conformance.sh
#   OPS="Add Mul" MAX_EXAMPLES=50 SEED=42 ./run_conformance.sh
#   PROFILE=1 ./run_conformance.sh   # → per-op attr_<Op>.json attribution files
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# ---------------------------------------------------------------------------
# Configuration (environment variable overrides)
# ---------------------------------------------------------------------------
VULKAN_EP_LIB="${VULKAN_EP_LIB:-$REPO_ROOT/rust/target/release/libonnxruntime_vulkan_ep.so}"
ONNX_TESTS_DIR="${ONNX_TESTS_DIR:-$REPO_ROOT/../onnx-tests}"
PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
ORT_LIB_DIR="${ORT_LIB_DIR:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-20}"
SEED="${SEED:-0}"
PROFILE="${PROFILE:-0}"

# ---------------------------------------------------------------------------
# Resolve claimed ops
# ---------------------------------------------------------------------------
if [ -n "${OPS:-}" ]; then
  IFS=' ' read -r -a CLAIM_OPS <<< "$OPS"
else
  mapfile -t CLAIM_OPS < <(grep -v '^#' "$HERE/claimed_ops.txt" | grep -v '^\s*$')
fi

echo "=== Vulkan EP conformance run ==="
echo "EP lib: $VULKAN_EP_LIB"
echo "onnx-tests: $ONNX_TESTS_DIR"
echo "Ops under test: ${CLAIM_OPS[*]}"
echo "MAX_EXAMPLES=$MAX_EXAMPLES SEED=$SEED PROFILE=$PROFILE"
echo

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [ ! -f "$VULKAN_EP_LIB" ]; then
  echo "ERROR: VULKAN_EP_LIB not found: $VULKAN_EP_LIB"
  echo "Build the EP crate first: cd rust && cargo build --release"
  exit 1
fi
if [ ! -d "$ONNX_TESTS_DIR" ]; then
  echo "ERROR: ONNX_TESTS_DIR not found: $ONNX_TESTS_DIR"
  echo "Clone onnx-tests: git clone https://github.com/cbourjau/onnx-tests ../../onnx-tests"
  exit 1
fi
if [ ! -x "$PIXI" ]; then
  echo "ERROR: pixi not found at $PIXI"
  echo "Install pixi: curl -fsSL https://pixi.sh/install.sh | bash"
  exit 1
fi

# ---------------------------------------------------------------------------
# Per-op subprocess loop
# ---------------------------------------------------------------------------
RESULTS_CSV="$HERE/results.csv"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"
echo "op,status,exit_code" > "$RESULTS_CSV"

pass=0
fail=0
crash=0
skip=0

for op in "${CLAIM_OPS[@]}"; do
  log_file="$LOG_DIR/${op}.log"
  echo -n "  $op ... "

  env_args=(
    "RUN_CANDIDATE=vulkan_runtime_wrapper.run_vulkan"
    "VULKAN_EP_LIB=$VULKAN_EP_LIB"
    "VULKAN_EP_NAME=${VULKAN_EP_NAME:-VulkanExecutionProvider}"
    "PYTHONPATH=$HERE${PYTHONPATH:+:$PYTHONPATH}"
    "MAX_EXAMPLES=$MAX_EXAMPLES"
    "SEED=$SEED"
  )
  if [ -n "$ORT_LIB_DIR" ]; then
    env_args+=("LD_LIBRARY_PATH=$ORT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
  fi
  if [ -n "${VK_ICD_FILENAMES:-}" ]; then
    env_args+=("VK_ICD_FILENAMES=$VK_ICD_FILENAMES")
  fi
  if [ "$PROFILE" = "1" ]; then
    env_args+=("VULKAN_EP_PROFILE=1" "VULKAN_EP_ATTR_OUT=$HERE/attr_${op}.json")
  fi

  set +e
  env "${env_args[@]}" "$PIXI" run python -m pytest \
    "$ONNX_TESTS_DIR/tests" \
    -k "test_${op}_" \
    --hypothesis-max-examples="$MAX_EXAMPLES" \
    --hypothesis-seed="$SEED" \
    -p no:cacheprovider \
    -q > "$log_file" 2>&1
  rc=$?
  set -e

  if [ $rc -eq 5 ]; then
    echo "SKIP (no tests collected)"
    echo "$op,SKIP,$rc" >> "$RESULTS_CSV"
    ((skip++)) || true
  elif [ $rc -ge 128 ] || [ $rc -eq 134 ] || [ $rc -eq 139 ]; then
    echo "CRASH (rc=$rc)"
    echo "$op,CRASH,$rc" >> "$RESULTS_CSV"
    ((crash++)) || true
  elif [ $rc -ne 0 ]; then
    echo "FAIL (rc=$rc)"
    echo "$op,FAIL,$rc" >> "$RESULTS_CSV"
    ((fail++)) || true
  else
    echo "PASS"
    echo "$op,PASS,$rc" >> "$RESULTS_CSV"
    ((pass++)) || true
  fi
done

echo
echo "=== Summary: PASS=$pass FAIL=$fail CRASH=$crash SKIP=$skip ==="
echo "Results: $RESULTS_CSV"
echo "Logs:    $LOG_DIR/"

if [ $fail -gt 0 ] || [ $crash -gt 0 ]; then
  exit 1
fi
