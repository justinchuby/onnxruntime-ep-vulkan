#!/usr/bin/env bash
# Twelve identical runs of the same suite at the same commit, each captured separately.
#
# `vk::barrier::tests::backend_probe_writes_legacy_token` was observed failing about one
# run in nine on this box: the `backend_probe_*` env vars are a PROCESS global and the
# tests race for them. That makes it the one intermittent in this repo whose second
# observation is cheap enough to buy, which is exactly what ci/check_flake_witness.py
# needs in order to be demonstrated in its positive state rather than reasoned about.
#
# Nothing here plants anything. If the flake does not show up in twelve runs, the arm
# does not fire and that is a fact about the flake, not a fact about the check.
set -o pipefail

export HOME=/home/justinchu
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.cargo/bin"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target

REPO=/mnt/c/Users/justinchu/dev/ep-vulkan-link
OUT="$REPO/bench/results/link-flake-witness"
RUNS="${1:-12}"

rm -rf "$OUT/runs"
mkdir -p "$OUT/runs"

cd "$REPO/rust" || exit 2

for i in $(seq 1 "$RUNS"); do
  cargo test --lib > "$OUT/runs/lib-linux-run$i.log" 2>&1
  rc=$?
  echo "run $i: cargo exit=$rc  $(grep -c '\.\.\. FAILED' "$OUT/runs/lib-linux-run$i.log") named failure(s)"
done

echo "==== any run that named a failure ===="
grep -l '\.\.\. FAILED' "$OUT/runs"/lib-linux-run*.log || echo "(none in $RUNS runs)"
grep -h '\.\.\. FAILED' "$OUT/runs"/lib-linux-run*.log | sort | uniq -c || true
