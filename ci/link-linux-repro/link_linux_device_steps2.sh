#!/usr/bin/env bash
# Re-run of link_linux_device_steps.sh AFTER pinning CARGO_TARGET_DIR in the criterion-5
# shader-less witness. The previous run's op-suite number was taken against an artifact that
# witness had overwritten with a shader-less build partway through, so it is not a reading
# about the shipped EP at all.
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:/usr/bin:/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
REPO=/mnt/c/Users/justinchu/dev/ep-vulkan-link
cd "$REPO/rust" || exit 1
touch src/ep.rs
cargo build --release 2>&1 | tail -2
SO=$CARGO_TARGET_DIR/release/libonnxruntime_vulkan_ep.so
sha256sum "$SO"
echo "shader modules embedded: $(strings -a "$SO" | grep -c '^ew_binary_add_f32$')"
cd "$REPO" || exit 1
bash ci/link-linux-repro/link_linux_device_steps.sh