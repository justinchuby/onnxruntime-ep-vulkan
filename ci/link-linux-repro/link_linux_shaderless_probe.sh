#!/usr/bin/env bash
# Scratch helper: does the Linux .so the gate registers actually carry shader modules?
# `probe_devices` logged "built without shaders (ALLOW_MISSING_GLSLC build)" for a binary
# whose OrtEpVulkanGetShaderSubject answers with 215 real, Linux-specific SPIR-V digests.
# Both cannot be true of the same file, so find out which file each side saw.
export HOME=/home/justinchu
export PATH="$HOME/.cargo/bin:/usr/bin:/bin:$PATH"
export CARGO_TARGET_DIR=/home/justinchu/link-linux-target
REPO=/mnt/c/Users/justinchu/dev/ep-vulkan-link
cd "$REPO" || exit 1
SO="$CARGO_TARGET_DIR/release/libonnxruntime_vulkan_ep.so"
echo "so      : $SO"
sha256sum "$SO"
echo "glslc   : $(command -v glslc) $(glslc --version 2>&1 | head -1)"
echo "env ALLOW: '${ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC:-<unset>}'"
echo "--- .spv sections / embedded module count ---"
strings -a "$SO" | grep -c '^ew_binary_add_f32$'
echo "--- ask the artifact for one module's subject ---"
ONNXRUNTIME_VULKAN_EP_LIB="$SO" python3 -c "
import sys, pathlib
sys.path.insert(0, 'rust/tools')
import gen_proof_ledger as G
print(G._shader_subject(pathlib.Path('$SO'), ['ew_binary_add_f32']))
"
echo "--- epctl (same target dir) probe ---"
"$CARGO_TARGET_DIR/release/epctl" --probe-loader 2>&1 | tail -4
echo "--- register the SAME path through ORT and read the warning ---"
ONNXRUNTIME_VULKAN_EP_LIB="$SO" python3 -c "
import onnxruntime as ort, pathlib
p = pathlib.Path('$SO').resolve()
print('registering', p)
ort.register_execution_provider_library('VulkanExecutionProvider', str(p))
print('devices:', [ (d.device.type, d.ep_name) for d in ort.get_ep_devices() if 'ulkan' in d.ep_name ])
" 2>&1 | tail -12
