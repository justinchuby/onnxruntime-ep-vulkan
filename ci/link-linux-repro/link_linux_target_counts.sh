#!/usr/bin/env bash
# Count what each integration target actually executes on Linux.
#
# The provenance strings in ci/suite_floor.json were measured on Windows. A floor
# is a claim about work, and a claim measured on one platform and asserted on two
# is a claim about one platform wearing the label of two. This script produces the
# Linux number so the provenance can say which platform it came from.
set -o pipefail

export HOME=/home/justinchu
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.cargo/bin"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target

REPO=/mnt/c/Users/justinchu/dev/ep-vulkan-link
OUT="$REPO/bench/results/link-suite-productivity"
mkdir -p "$OUT"

cd "$REPO/rust" || exit 2

cargo test --test portability 2>&1 | tee "$OUT/cargo-test-portability-linux.log"
echo "portability cargo exit=${PIPESTATUS[0]}"

cargo test --test cdylib_load --test dump_capabilities \
           --test host_registration --test validation_control 2>&1 \
  | tee "$OUT/cargo-test-integration-linux.log"
echo "integration cargo exit=${PIPESTATUS[0]}"

echo "==== summary ===="
grep -E "^(Running|running [0-9]+ tests|test result:)" \
  "$OUT/cargo-test-portability-linux.log" "$OUT/cargo-test-integration-linux.log"
