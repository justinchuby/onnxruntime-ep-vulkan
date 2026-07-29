# Third-Party Notices and Compliance Guide

**Owner:** Rai (RAI Reviewer)
**Date:** 2026-07-28T19:16:08-07:00
**Scope:** Licences of all external sources this project studies, adapts, or vendors; the obligations attached to each; and practical rules for engineers writing shaders and Rust code.

> **Disclaimer:** I am not a lawyer. This document gives the team a workable rule they can apply today, grounded in the plain text of the relevant licences and standard industry practice. It is not legal advice. If the project ever ships commercially at scale, get counsel to review before shipping.

---

## 1. This Project's Licence

`LICENSE` at the repo root is **MIT**, Copyright (c) 2026 Justin Chu.

MIT permits any use, modification, distribution, and sublicensing, with one condition: the copyright notice and licence text must be included in all copies or substantial portions. That condition applies to *our code* when others copy it — it does not restrict what we can build from.

---

## 2. Licences of Sources the Team Studies or Adapts

| Source | Licence | Copyright | Verified |
|--------|---------|-----------|---------|
| **llama.cpp** (`ggml/src/ggml-vulkan/vulkan-shaders/`) | MIT | Copyright (c) 2023–2026 The ggml authors | ✅ fetched from `ggerganov/llama.cpp` `LICENSE` |
| **ExecuTorch** (`backends/vulkan/`) | BSD 3-Clause | Copyright (c) Meta Platforms, Inc. and affiliates; Arm Ltd.; Qualcomm Innovation Center, Inc.; Apple Inc.; MediaTek Inc.; NXP; Samsung; Intel | ✅ fetched from `pytorch/executorch` `LICENSE` |
| **ONNX Runtime** (headers, C API, examples) | MIT | Copyright (c) Microsoft Corporation | ✅ fetched from `microsoft/onnxruntime` `LICENSE` |
| **vulkan.gpuinfo.org data** (cited in `PLATFORMS.md`) | CC-BY 4.0 | Sascha Willems | Already attributed in `PLATFORMS.md` |

---

## 3. The Distinction That Actually Matters

### 3.1 Reading code to learn — always free

Reading a codebase (llama.cpp shaders, ExecuTorch Vulkan backend, ORT examples) to understand an algorithm, a tiling strategy, a subgroup technique, or a quantization approach **requires no attribution and creates no obligations**. Copyright protects the specific expressive form of code, not the ideas, algorithms, or mathematical techniques it implements. This is the idea/expression dichotomy and it is settled law.

**"Clean" means:** you understand the technique from reading the source, then you write your own implementation from scratch using your understanding. Your shader is yours.

### 3.2 Copying or substantially adapting source — MIT/BSD conditions attach

If you take a shader body, a tiling loop, a table of constants, or any other substantial expression and adapt it into our codebase — even with variable renaming or structural reshuffling — you have created a **derivative work**. Both MIT licences (llama.cpp, ORT) and the BSD-3 licence (ExecuTorch) permit this, but with conditions:

**MIT conditions (llama.cpp, ORT):**
1. The original copyright notice must appear in our codebase (in the adapted file's header and/or in `THIRD_PARTY_NOTICES.md`, see §5).
2. The MIT licence text (or a pointer to it) must appear in the distribution.

**BSD 3-Clause conditions (ExecuTorch), additionally:**
1. The copyright notice must appear in redistributions of source code.
2. The copyright notice must appear in binary redistributions **in the documentation and/or other materials provided with the distribution** — i.e., in our `THIRD_PARTY_NOTICES.md` or equivalent, which must be distributed alongside the `.so`/`.dll`/`.dylib`.
3. Neither "Meta" nor any other contributor name listed in the ExecuTorch `LICENSE` may be used to endorse or promote this project without prior written permission.

### 3.3 The grey zone — concrete rule of thumb

**The test question:** Could you write the same code independently, after understanding the algorithm from reading the original, without needing to refer back to the original to produce the structure and sequence of the code?

| Scenario | Classification | What to do |
|----------|---------------|-----------|
| You read a shader, understood the tiling math, closed the tab, and wrote your own shader implementing the same algorithm | **Independent work** | No attribution needed |
| You translated a GLSL shader to our GLSL, renaming variables and adapting for our push-constant layout, but the operation sequence and structure track the original | **Derivative work** | Add attribution header + THIRD_PARTY_NOTICES.md entry |
| You ported a non-trivial algorithmic structure (e.g., the two-pass quantized GEMV/GEMM switch, the specific subgroup reduction tree) that would require consulting the original to reproduce | **Derivative work** | Add attribution header + THIRD_PARTY_NOTICES.md entry |
| You used a published algorithm description (e.g., the flash-attention paper) implemented by llama.cpp, and wrote your own implementation of the algorithm | **Independent work** | No attribution to llama.cpp; cite the paper |
| You copied a block of 20+ lines of shader code nearly verbatim | **Clear derivative** | Attribution required; must use the full file header format in §4 |

**One-line rule:** If the specific phrasing of the code, not just the idea, came from the source — attribute. If only the idea came from the source — no obligation, but a comment citing the source is good engineering practice.

---

## 4. Compliant File Header Format

When a shader file in `shaders/glsl/` contains code substantially adapted from a third-party source, the file header must include:

```glsl
// SPDX-License-Identifier: MIT
//
// Portions of this file are adapted from llama.cpp
// Copyright (c) 2023-2026 The ggml authors
// Licensed under the MIT License.
// Source: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/
//
// The full MIT License text is reproduced in docs/THIRD_PARTY_NOTICES.md.
```

Or for ExecuTorch:

```glsl
// SPDX-License-Identifier: MIT
//
// Portions of this file are adapted from ExecuTorch
// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the BSD 3-Clause License.
// Source: https://github.com/pytorch/executorch/tree/main/backends/vulkan/
//
// The full BSD 3-Clause License text is reproduced in docs/THIRD_PARTY_NOTICES.md.
```

A new file that is purely original work needs no such header — but adding `// SPDX-License-Identifier: MIT` is encouraged for clarity.

---

## 5. SPIR-V Compiled from Adapted GLSL

SPIR-V compiled from adapted GLSL is a **derivative work** of the original GLSL. The compiled `.spv` (or the `&'static [u8]` blob in `build.rs`'s output) carries the same licence obligations as the source.

**MIT condition for compiled distribution:** The copyright notice and licence text must appear in "copies or substantial portions of the Software." For a compiled binary (our `.so`/`.dll`), the industry-standard and legally sufficient practice is to include a `THIRD_PARTY_NOTICES.md` (or `NOTICES`) file in the same distribution package — you do not need to embed text inside the SPIR-V binary or the Rust binary. Any distribution channel (crate, GitHub Release, pip wheel) that ships the binary must also ship the notices file.

**Concretely:** If `shaders/glsl/gemm_q4.glsl` is adapted from `llama.cpp`, then `libonnxruntime_vulkan_ep.so` (which embeds that shader's SPIR-V) is a derivative work, and `docs/THIRD_PARTY_NOTICES.md` must be included in any release archive or package.

---

## 6. Attribution in Commit Messages and Docs

- **Commit messages**: If a commit introduces an adapted shader, add a line: `Adapted from llama.cpp <URL-to-file> (MIT)` in the commit body. This creates a forensic trail, not just the file header.
- **`docs/` references**: When citing another project's design or technique in documentation (as Switch has done in `ENGINE.md`), cite the source URL. No licence obligation for documentary references, but good engineering practice.
- **`THIRD_PARTY_NOTICES.md`**: See §7. Must be distributed with any binary release that includes adapted code.

---

## 7. Per-Source Ruling Table

| Source | What is permitted | Obligations if code is adapted | Red lines |
|--------|------------------|-------------------------------|-----------|
| **llama.cpp shaders (MIT)** | Read freely. Adapt freely. Embed SPIR-V derived from it. | Copyright notice + MIT licence text in `docs/THIRD_PARTY_NOTICES.md`. File header in adapted shader (§4). Commit message note. | None — MIT is maximally permissive. |
| **ExecuTorch Vulkan backend (BSD-3)** | Read freely. Adapt freely. Embed derived SPIR-V. | Copyright notice + BSD-3 licence text in `docs/THIRD_PARTY_NOTICES.md`. File header in adapted file. Do NOT use "Meta", "Arm", "Qualcomm" etc. names to endorse this project. | Endorsement clause: don't use contributor names in project promotion without permission. |
| **ORT C API headers (MIT)** | Use the public C ABI freely — calling a public API is not copying code, and creates zero licence obligations. If Tank vendors ORT header files directly into `rust/sys/` (copying header text into our tree), treat those files as MIT-attributed inclusions and ensure their copyright notices are preserved. | If headers are vendored: copyright notices in the vendored files must be preserved. No separate NOTICES entry needed unless substantial body text is adapted. | Do not claim to be part of ORT or Microsoft. |
| **ORT examples (`nv_vulkan_test.cc`, etc.) (MIT)** | Read freely. No `.cc` files were found vendored in this repo as of this review — Switch studied them online. If any example code is adapted, standard MIT attribution applies. | If adapted: file header + THIRD_PARTY_NOTICES.md entry. | None. |
| **vulkan.gpuinfo.org data (CC-BY 4.0)** | Use the data in documentation. Already correctly attributed in `PLATFORMS.md` with source URL, author, and date. CC-BY 4.0 data does not affect code licence for this project. | Attribution already present in `PLATFORMS.md`. Ensure it stays on any page that uses the data. | Do not embed gpuinfo.org data in a way that claims it as ours. |

---

## 8. Compatibility Check: MIT + MIT + BSD-3

| Our licence | Source licence | Compatible for distribution? |
|-------------|---------------|------------------------------|
| MIT | MIT (llama.cpp, ORT) | ✅ Yes. Both MIT. No conflict. |
| MIT | BSD-3 (ExecuTorch) | ✅ Yes. BSD-3 is compatible with MIT for this direction. Attribution conditions satisfied by THIRD_PARTY_NOTICES.md. |
| MIT | CC-BY 4.0 (gpuinfo.org) | ✅ Yes for documentation data — CC-BY applies to the data, not the software. |

None of the sources the team is drawing on impose a copyleft (GPL/LGPL/AGPL) condition. There is no licence conflict with our MIT repository.

---

## 9. OQ-M6 Verdict

🟢 **Green.** Mouse's proposal — reading llama.cpp's MIT-licensed Vulkan shaders as a reference for GroupQueryAttention, MatMulNBits, and LinearAttention — is **fully permitted with no obligation**. Pure reading creates no copyright issue.

The conditions below apply only when implementation moves from learning to adapting:

| Condition | Trigger |
|-----------|---------|
| Add file header (§4) | Any shader that substantially adapts llama.cpp shader code |
| Add THIRD_PARTY_NOTICES.md entry (§5 + below) | Same trigger |
| Commit message note | Same trigger |
| Distribute THIRD_PARTY_NOTICES.md alongside binary releases | Any release containing SPIR-V compiled from adapted shaders |

If engineers write their own shaders from scratch after understanding the algorithms from reading llama.cpp, none of these conditions trigger.

---

## 10. When to Create THIRD_PARTY_NOTICES.md

`docs/THIRD_PARTY_NOTICES.md` **does not need to exist today** — no code has been copied yet (project is pre-implementation). It must be created before the first binary release that embeds SPIR-V compiled from adapted third-party shader source.

The template for when it is needed:

```markdown
# Third-Party Notices

This software includes components adapted from the following third-party projects:

---

## llama.cpp

Source: https://github.com/ggml-org/llama.cpp
Files adapted: [list of our shader files that adapt llama.cpp code]

MIT License
Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

When an adapted ExecuTorch file is added, append a section with the full BSD-3 notice and the full list of copyright holders from ExecuTorch's `LICENSE` file.
