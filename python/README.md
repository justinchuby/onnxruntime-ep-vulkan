# `onnxruntime-ep-vulkan` (Python)

The consumer-facing shim for the [Vulkan execution
provider](https://github.com/justinchuby/onnxruntime-ep-vulkan). It loads a stock,
unmodified ONNX Runtime's plugin-EP library and gets out of the way.

```python
import onnxruntime as ort
import onnxruntime_ep_vulkan

onnxruntime_ep_vulkan.register_execution_provider_library()
sess = ort.InferenceSession(model, providers=onnxruntime_ep_vulkan.providers())
onnxruntime_ep_vulkan.assert_ep_selected(sess)   # ORT will not raise for you
```

## Why a package, when the API it wraps is one call

`ort.register_execution_provider_library(name, path)` is the whole ORT API. Four measured
properties of it are what this package exists for — see
`bench/results/consumption_surface_dev0.json` in the repository, six cases, each in its own
subprocess because plugin registration is process-global state:

| Measured | Consequence |
|---|---|
| A relative path resolves against ORT's own `capi` directory, not the caller's CWD | the absolute path is mandatory, and nothing says so |
| The registration name is never checked against the library | the name at registration and the name in `providers=[...]` must agree, enforced by nobody |
| Registering the same name twice raises | the documented call is not safe to run twice in one process |
| **A session asking for an unregistered EP name does not raise** — it warns, falls back to CPU, and returns correct numbers | the natural failure mode is a session that silently never touches the GPU |

The last one is why `assert_ep_selected` exists and why it is in the three-line example.
It asserts the EP was *selected for the session*; it does not assert that any node was
claimed or that any dispatch executed.

## Where the library comes from

In priority order: an explicit `path=` argument; `$ONNXRUNTIME_VULKAN_EP_LIB`; the artifact
bundled in the wheel; the cargo `target/release` (then `target/debug`) directory of a
source checkout. A failure names every path it tried.

## Inspecting an installation

```
python -m onnxruntime_ep_vulkan            # what is installed, and its provenance
python -m onnxruntime_ep_vulkan --check    # register, run a trivial Add, prove it works
```

## Building the wheel

```
python python/build_wheel.py
```

Runs `cargo build --release`, stages the platform-correct cdylib into the package with a
`_provenance.json` (commit, dirty flag, artifact sha256, a digest over every shader source,
toolchain versions), and produces a `py3-none-<platform>` wheel. It refuses to package an
artifact containing zero SPIR-V modules, which is how `DESIGN.md` §7.8 condition 4 ("no
release artifact from an escape-hatch build") is enforced against the bytes rather than
against the environment. The staged artifact is gitignored: no binary enters the tree.
