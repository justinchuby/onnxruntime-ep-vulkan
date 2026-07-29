# Vendored ONNX Runtime C-API headers

## What is here

`include/` contains three headers copied verbatim from the ONNX Runtime source tree:

| File | Upstream path |
|---|---|
| `onnxruntime_c_api.h` | `include/onnxruntime/core/session/onnxruntime_c_api.h` |
| `onnxruntime_ep_c_api.h` | `include/onnxruntime/core/session/onnxruntime_ep_c_api.h` |
| `onnxruntime_error_code.h` | `include/onnxruntime/core/session/onnxruntime_error_code.h` |

## Provenance

- **Upstream:** https://github.com/microsoft/onnxruntime
- **Tag:** `v1.28.0`
- **Commit:** `da9b5e364c465de65c49d91e696cd6485270757f`
- **Released:** 2026-07-24
- **`ORT_API_VERSION` declared by these headers:** `28`
- **Vendored on:** 2026-07-28T19:16:08-07:00 by Tank

## Licence

ONNX Runtime is licensed under the MIT Licence. The upstream licence text is reproduced verbatim
in `LICENSE` next to this file. These headers are redistributed under that licence; the copyright
notice is retained in each file.

## Why vendored rather than resolved from an ORT install

`rust/build.rs` runs `bindgen` over these headers to generate the plugin-EP C ABI bindings.
Vendoring means:

1. **The bindings are byte-reproducible.** Every developer machine and every CI runner generates
   the same `sys` module from the same bytes. A build cannot silently pick up a different ORT
   version's headers from `$ORT_HOME` and produce a struct with a different field order — which is
   silent UB, not a compile error, because the vtable is a bag of function pointers.
2. **The build has no network and no ORT install requirement.** Android and cross-compile lanes do
   not need an ORT release tarball unpacked just to compile a Rust crate that never links
   `libonnxruntime`.
3. **An ORT API bump is a visible, reviewable diff.** Bumping ORT means replacing these files in a
   PR; the header diff and the resulting binding diff are both in review.

`rust/build.rs` still honours `ORT_INCLUDE_DIR` as an override for anyone who deliberately wants
to build against a different ORT checkout — but the vendored copy is the default and the one CI
uses.

## How to update

```powershell
$ver = 'v1.29.0'
$base = "https://raw.githubusercontent.com/microsoft/onnxruntime/$ver/include/onnxruntime/core/session/"
foreach ($f in 'onnxruntime_c_api.h','onnxruntime_ep_c_api.h','onnxruntime_error_code.h') {
    Invoke-WebRequest -Uri ($base + $f) -OutFile "third_party\onnxruntime\include\$f" -UseBasicParsing
}
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/microsoft/onnxruntime/$ver/LICENSE" `
    -OutFile 'third_party\onnxruntime\LICENSE' -UseBasicParsing
```

Then update this file (tag, commit, date, `ORT_API_VERSION`), update
`ORT_API_VERSION_EXPECTED` in `rust/src/sys.rs`, and bump `version` in `rust/Cargo.toml` to
`0.<new ORT_API_VERSION>.0`. `rust/src/sys.rs` carries a compile-time assertion that the
generated `ORT_API_VERSION` equals `ORT_API_VERSION_EXPECTED`, so forgetting the second step is a
build failure, not a runtime surprise.

Two more things to check on a bump, neither of which the compiler can catch for you:

* **`ORT_API_VERSION_MIN`** (currently 24) is a *separate* decision — the oldest host we run
  against, not the newest we build against. Raise it only when we deliberately drop support, not
  as a side effect of upgrading.
* **`sys::since::*`** must gain a constant for any newly-depended-upon entry point, and
  `sys::importer_seam` must still compile — it names the external-resource-importer types
  explicitly so that an upstream rename becomes a build failure here rather than a surprise
  whenever someone gets round to implementing zero-copy IO binding.
