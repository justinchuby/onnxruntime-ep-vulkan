# Third-Party Notices

This software includes components adapted from the following third-party projects.

**This file must be distributed alongside any binary release of this project** — the `.dll`,
`.so` or `.dylib` embeds SPIR-V compiled from the adapted shader source listed below, and that
SPIR-V is a derivative work of it. See `docs/THIRD_PARTY.md` §5 and §10.

---

## Cephes Mathematical Library

**Source:** <https://netlib.org/cephes/> — single-precision `asinf.c`, distributed in `single.tgz`

**Files adapted:** `rust/shaders/glsl/templates/ew_unary.comp` (the `ew_asin_core` polynomial
coefficients and the inverse-trigonometric range reduction they are fitted for, used by the
`Asin` and `Acos` op selectors)

```
Cephes Math Library Release 2.2:  June, 1992
Copyright 1984, 1987, 1992 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
```

This software is derived from the Cephes Math Library and is incorporated herein by permission of
the author.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions
  and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of
  conditions and the following disclaimer in the documentation and/or other materials provided
  with the distribution.
* Neither the name of the organization nor the names of its contributors may be used to endorse
  or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### Provenance of the permission above — read this before changing it

Cephes carries **no SPDX identifier assigned by its author**, and the wording above was assembled
rather than copied from a `LICENSE` file, because Cephes has never shipped one. Recording exactly
where each part came from, so a later reader can re-check it instead of trusting this file:

**1. What the canonical distribution says.** The whole of the licence-bearing text at
<https://netlib.org/cephes/readme> is:

> Some software in this archive may be from the book _Methods and Programs for Mathematical
> Functions_ (Prentice-Hall or Simon & Schuster International, 1989) or from the Cephes
> Mathematical Library, a commercial product. In either event, it is copyrighted by the author.
> What you see here may be used freely but it comes with no support or guarantee.
>
> — Stephen L. Moshier

That is a permissive statement but **not** a standard licence: it does not name modification or
redistribution rights, and it calls the library a commercial product. On its own it would not
support the BSD grant above.

**2. Where the BSD grant actually comes from.** Stephen Moshier granted it by email to the Debian
project on 28 December 2004, archived at
<https://lists.debian.org/debian-legal/2004/12/msg00295.html>. He wrote:

> I think you do not have to dictate the terms under which I distribute the material through other
> channels. You probably need only a permission that allows you to distribute it under your
> particular terms. Here is an example boilerplate that worked for BSD. A recent go-around with
> the FSF resulted in something a bit more wordy but similar in spirit.
>
> ```
>  /*
>   * Cephes Math Library Release 2.8:  June, 2000
>   * Copyright 1984, 1995, 2000 by Stephen L. Moshier
>   *
>   * This software is derived from the Cephes Math Library and is
>   * incorporated herein by permission of the author.
>   *
>   * [standard BSD license here]
>   */
> ```

The `[standard BSD license here]` placeholder is his; the BSD 3-Clause text reproduced above is
what fills it, following the same reading SciPy uses.

**3. Who else reads it this way.** SciPy vendors Cephes and records it as `BSD-3-Clause`, citing
that same Debian message as its authority — see
<https://raw.githubusercontent.com/scipy/xsf/main/LICENSES_bundled.txt>, which reproduces
Moshier's boilerplate with the full BSD 3-Clause text inserted. This project follows SciPy's
reading rather than inventing its own.

**4. Two honest caveats.**

* Moshier's 2004 message is addressed to a specific redistributor and says "you probably need only
  a permission that allows *you* to distribute it under *your* particular terms". Whether that is
  a general public relicence or a per-redistributor permission is genuinely ambiguous. It is
  nonetheless a direct, unambiguous grant from the copyright holder that BSD-style redistribution
  is acceptable, and it has been relied on in this form by SciPy and Debian for two decades.
* His boilerplate names **Release 2.8: June, 2000 / Copyright 1984, 1995, 2000**. The file this
  project actually adapted, single-precision `asinf.c`, carries **Release 2.2: June, 1992 /
  Copyright 1984, 1987, 1992**, and that is the notice reproduced above, because it is the notice
  on the code that was used. Same author, same permission, different release.

Neither caveat is a reason to weaken the attribution; both are reasons to keep this section, so
that the next person to touch it inherits the evidence rather than the conclusion.

**Not legal advice.** `docs/THIRD_PARTY.md`'s disclaimer applies here unchanged.
