"""Which #[test] fns touch a process-global without holding the test lock?

Round 36 diagnosis: `cargo test --lib counters` failed 8 times in 20, in four different
tests at four different assertion sites, because `counters::reset()` and the `record_*`
writers act on process-global statics that libtest runs concurrently on its thread pool.
The convention already exists -- `let _g = crate::allocator::ledger::test_lock();` -- it
was simply not applied everywhere, and three unguarded tests were enough to clobber every
guarded one.

Round 37 extends the extent to a **third** global family with the same hazard and none of
the same syntax: the **process environment**. `backend_probe_*` in `vk/barrier.rs` shares
`ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE` across three tests; measured 6 failures in 40 runs of
`cargo test --lib backend_probe`, and the pre-Round-37 auditor was green on all of it,
because the environment is not a static and `std::env::set_var` is not a `record_*` call.
The per-test unique file path (`PROBE_COUNTER`) that was already there is what makes the
diagnosis unambiguous: the *paths* were de-conflicted, so a token crossing between two
tests can only have crossed through the one thing they still shared.

Three things this checks, because a guard that is the *wrong mutex* would look identical to
a guard that works:

  1. every #[test] that reads or writes the global counters holds a `test_lock()`;
  2. every #[test] that touches a *contended* environment variable holds one too;
  3. every file that holds one imports it from `crate::allocator::ledger`, so all of them
     are the same lock.

Round 38 is (3) again, in the shape the tool was already able to see but had never been
*shown* to see: `rust/src/vk/instance.rs` guarded `ONNXRUNTIME_EP_VULKAN_DEVICE_SELECTOR`
with two disjoint mutexes -- three tests on `crate::allocator::ledger::test_lock()` and four
on a module-private `serial_env()` with the same signature and the same call-site spelling.
Measured: 14 reds in 25 reps of `contention_gate.py --pool env:device_selector` on Windows.
The two specimens added for it (`ENV_PRIVATE_LOOKALIKE_SPECIMEN`,
`ENV_FORWARDING_ALIAS_SPECIMEN`) differ by exactly one thing -- which mutex the local guard
fn reaches -- so the selftest now demonstrates the discrimination rather than asserting it,
and demonstrates it in *both* directions: the private lookalike is a finding and the
`ep.rs::serialize()`-style forwarding helper is not.

**Contended** is the load-bearing word in (2), and it is measured from the tree, not
declared: a variable is contended when two or more tests in the concurrently-scheduled
population touch it and at least one of them writes. That definition is also the diagnostic
handed back to a human -- `--pairs` prints, per variable, the tests that share it, so a red
run naming two test functions can be read straight back to the one global they contend on.

Run:  python rust/tools/audit_counter_test_lock.py [--check] [--selftest] [--pairs] [--json]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RUST_SRC = Path(__file__).resolve().parents[1] / "src"

#: Every Rust source, not a hand-written list. Round 36: the first version of this tool
#: named `counters.rs` and `disclosure.rs` explicitly, went green, and the very next full
#: run failed in `ep.rs` -- a third module with the same defect that the list could not see.
def rust_sources():
    return sorted(RUST_SRC.rglob("*.rs"))


#: Anything that reads or writes the process-global counter statics.
#: Outside `counters.rs` the call must be qualified, or generic names like `reset()` and
#: `snapshot()` belonging to other types would be counted as touches.
TOUCHES_QUALIFIED = re.compile(
    r"\bcounters::(reset\(\)|record_[a-z_]+\(|snapshot\(\)|note_[a-z_]+\()"
    r"|\b(attach|detach)_ort_logger\("
)
TOUCHES_BARE = re.compile(
    r"\b(reset\(\)|record_[a-z_]+\(|snapshot\(\)|note_[a-z_]+\()"
    r"|\b(attach|detach)_ort_logger\("
)
HOLDS_LOCK = re.compile(r"\btest_lock\(\)")
LOCK_IMPORT = re.compile(r"crate::allocator::ledger::(test_lock|\{[^}]*\btest_lock\b)")


def _brace_body(text: str, start: int):
    """Return (body, end_index) for the block whose opening brace follows `start`."""
    i = text.index("{", start)
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return text[i : j + 1], j + 1


def test_bodies(text: str, whole_file_is_test_mod: bool = False):
    """Yield (name, line_no, body, attrs) for every fn inside a test module.

    Extent: this follows *bodies*, not call graphs. A #[test] that reaches the globals
    only through a helper defined elsewhere is invisible here -- which is why helpers
    inside the test module are enumerated too, and why the empirical repeat harness
    (`rust/tools/contention_gate.py`) remains the check of record.

    `whole_file_is_test_mod` covers the second structural blind spot found in Round 37:
    `vk/dispatch_integration.rs` has its `#[test]` fns at file top level because the file
    *is* the test module -- declared `#[cfg(test)] mod dispatch_integration;` in a
    different file. Scanning only for `mod ... {` blocks inside each file cannot see it,
    for the same reason the Round-36 hand-written two-file list could not see `ep.rs`.
    """
    scopes = []
    if whole_file_is_test_mod:
        scopes.append((text, 0))
    for mod_m in re.finditer(r"\bmod\s+([a-z0-9_]+)\s*\{", text):
        if "test" not in mod_m.group(1):
            # Only test modules; production code is allowed to touch its own globals.
            head = text[max(0, mod_m.start() - 200) : mod_m.start()]
            if "#[cfg(test)]" not in head:
                continue
        body, _ = _brace_body(text, mod_m.end() - 1)
        scopes.append((body, mod_m.start()))
    seen = set()
    for body_text, base in scopes:
        for m in re.finditer(r"\bfn\s+([a-z0-9_]+)\s*[(<]", body_text):
            name = m.group(1)
            try:
                fn_body, _ = _brace_body(body_text, m.end())
            except ValueError:
                continue
            line_no = text.count("\n", 0, base + m.start()) + 1
            if (name, line_no) in seen:
                continue
            seen.add((name, line_no))
            yield name, line_no, fn_body, _attrs_before(body_text, m.start())


def _attrs_before(text: str, pos: int) -> str:
    """The attribute/doc block immediately preceding a fn item.

    Sliced back to the previous `}` or `;` so the preceding fn's body cannot leak in.
    Doc prose mentioning `#[ignore]` would be misread; stated, not defended against --
    the cost is a test wrongly excluded from the contended population, which is visible
    in the `--pairs` census rather than silent.
    """
    head = text[:pos]
    cut = max(head.rfind("}"), head.rfind(";"))
    return head[cut + 1 :] if cut >= 0 else head


# ──────────────────────────────────────────────────────────────────────────────
# Family 3: the process environment
# ──────────────────────────────────────────────────────────────────────────────

#: `std::env::set_var(NAME, ..)` / `remove_var(NAME)` / `var(NAME)` / `var_os(NAME)`,
#: with or without the `std::` prefix. NAME is captured as written -- a string literal or
#: a const path -- and resolved to its literal below, because two files naming the same
#: variable through different consts contend on one global and must compare equal.
ENV_CALL = re.compile(
    r"\b(?:std::)?env::(set_var|remove_var|var|var_os)\s*\(\s*(&?[A-Za-z0-9_:]+|\"[^\"]*\")"
)
ENV_CONST = re.compile(
    r"(?:pub(?:\s*\([^)]*\))?\s+)?(?:static|const)\s+([A-Z0-9_]+)\s*:\s*&(?:'static\s+)?str\s*=\s*\"([^\"]*)\""
)
ENV_MUTATORS = ("set_var", "remove_var")


def env_consts(sources) -> "dict[str, str]":
    """`ENV_QUARANTINE_SPANS` -> `"ONNXRUNTIME_EP_VULKAN_QUARANTINE_SPANS"`, tree-wide.

    Keyed on the last path segment, so `crate::factory::ENV_DEVICE_MEMORY` and a bare
    `ENV_DEVICE_MEMORY` resolve to the same variable. A duplicate ident bound to two
    different literals is recorded as ambiguous rather than resolved to one of them.
    """
    out: dict[str, str] = {}
    for path, text in sources:
        for m in ENV_CONST.finditer(text):
            ident, literal = m.group(1), m.group(2)
            if ident in out and out[ident] != literal:
                out[ident] = "AMBIGUOUS:" + ident
            else:
                out.setdefault(ident, literal)
    return out


def env_touches(body: str, consts: "dict[str, str]"):
    """Yield (var_name, kind) for each env access in a fn body."""
    for m in ENV_CALL.finditer(body):
        kind, tok = m.group(1), m.group(2).lstrip("&")
        if tok.startswith('"'):
            name = tok[1:-1]
        else:
            seg = tok.split("::")[-1]
            name = consts.get(seg, f"UNRESOLVED:{tok}")
        yield name, kind


def env_audit(sources, consts):
    """Return {var: [record, ...]} over the concurrently-scheduled test population.

    A record is a dict: file, test, line, kinds, guarded, ignored.
    `#[ignore]` tests are excluded from the population *and* recorded, because "it is
    ignored" is the whole of the reason `dispatch_integration.rs` is not a finding, and a
    reason that is not written down gets re-litigated every round.
    """
    census: dict[str, list] = {}
    cfg_test_files = cfg_test_module_files(sources)
    for path, text in sources:
        aliases = lock_aliases(text)
        holds = re.compile(
            r"\btest_lock\(\)" + ("|" + "|".join(rf"\b{a}\(\)" for a in aliases) if aliases else "")
        )
        whole = path in cfg_test_files
        for name, line_no, body, attrs in test_bodies(text, whole_file_is_test_mod=whole):
            if "#[test]" not in attrs:
                continue
            touched: dict[str, set] = {}
            for var, kind in env_touches(body, consts):
                touched.setdefault(var, set()).add(kind)
            for var, kinds in touched.items():
                census.setdefault(var, []).append(
                    {
                        "file": path,
                        "test": name,
                        "line": line_no,
                        "kinds": sorted(kinds),
                        "guarded": bool(holds.search(body)),
                        "ignored": "#[ignore" in attrs,
                    }
                )
    return census


def env_findings(census):
    """(findings, contended, sole) -- only *contended* variables gate.

    Contended := two or more tests in the scheduled population touch it and at least one
    writes. One toucher is not a test-against-test race, and this tool's stated extent is
    bodies, not call graphs, so it cannot see a production reader on another thread; that
    case is printed as `SOLE-MUTATOR` and deliberately does not fail `--check`.

    There is no verdict token here whose only possible value is the empty set: on this tree
    at `d46327b` `findings` is non-empty (three `backend_probe_*` tests) and so is `sole`
    (`ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY`, `..._BUDGET_MB`, `..._LEDGER_FILE`). Both states
    of the contended/sole distinction occur in the tree, not only in the selftest specimens.
    """
    findings, contended, sole = [], {}, {}
    for var, recs in sorted(census.items()):
        live = [r for r in recs if not r["ignored"]]
        writers = [r for r in live if any(k in ENV_MUTATORS for k in r["kinds"])]
        if len(live) >= 2 and writers:
            contended[var] = live
            for r in live:
                if not r["guarded"]:
                    findings.append((r["file"], r["test"], r["line"], var))
        elif writers:
            sole[var] = live
    return findings, contended, sole


def cfg_test_module_files(sources) -> "set[str]":
    """Files that *are* a test module because someone else declared them one."""
    out = set()
    by_rel = {path for path, _ in sources}
    for path, text in sources:
        parent = Path(path).parent
        for m in re.finditer(r"#\[cfg\(test\)\]\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([a-z0-9_]+)\s*;", text):
            child = m.group(1)
            for cand in ((parent / f"{child}.rs").as_posix(), (parent / child / "mod.rs").as_posix()):
                cand = cand.lstrip("./")
                if cand in by_rel:
                    out.add(cand)
    return out


def lock_aliases(text: str) -> "set[str]":
    """Local zero-arg guard fns that simply forward to `ledger::test_lock()`.

    `ep.rs` defines `serialize()` this way. Treating it as a guard is safe *only* because
    its body is checked to reach the same mutex -- an alias that forwarded somewhere else
    is exactly the wrong-lock defect this tool exists to catch.
    """
    aliases = set()
    for m in re.finditer(r"\bfn\s+([a-z0-9_]+)\s*\(\s*\)\s*->\s*[^{]*MutexGuard", text):
        try:
            body, _ = _brace_body(text, m.end())
        except ValueError:
            continue
        if "ledger::test_lock()" in body:
            aliases.add(m.group(1))
    return aliases


def audit_text(fname: str, text: str, whole_file_is_test_mod: bool = False):
    """Return (unguarded, wrong_lock) findings for one file's source."""
    touches = TOUCHES_BARE if Path(fname).name == "counters.rs" else TOUCHES_QUALIFIED
    aliases = lock_aliases(text)
    holds = re.compile(
        r"\btest_lock\(\)" + ("|" + "|".join(rf"\b{a}\(\)" for a in aliases) if aliases else "")
    )
    unguarded, wrong_lock = [], []
    uses_bare = False
    for name, line_no, body, _attrs in test_bodies(text, whole_file_is_test_mod):
        if name in aliases:
            continue
        if not touches.search(body):
            continue
        if not holds.search(body):
            unguarded.append((fname, name, line_no))
        elif "ledger::test_lock()" not in body and not aliases:
            uses_bare = True
    if uses_bare and not LOCK_IMPORT.search(text):
        wrong_lock.append(
            (fname, "bare test_lock() with no import from crate::allocator::ledger")
        )
    return unguarded, wrong_lock


def load_sources():
    return [
        (p.relative_to(RUST_SRC).as_posix(), p.read_text(encoding="utf-8"))
        for p in rust_sources()
    ]


def audit():
    sources = load_sources()
    cfg_test_files = cfg_test_module_files(sources)
    unguarded, wrong_lock = [], []
    for rel, text in sources:
        u, w = audit_text(rel, text, whole_file_is_test_mod=rel in cfg_test_files)
        unguarded += u
        wrong_lock += w
    consts = env_consts(sources)
    census = env_audit(sources, consts)
    env_unguarded, contended, sole = env_findings(census)
    return unguarded, wrong_lock, env_unguarded, contended, sole


def contended_test_names():
    """The test fns that touch any process-global family -- the gate's filter, derived.

    `contention_gate.py` imports this rather than carrying a list of module names. A
    hand-written list is the exact defect this file was written to stop being blind to.
    """
    sources = load_sources()
    cfg_test_files = cfg_test_module_files(sources)
    consts = env_consts(sources)
    names: dict[str, set] = {}
    for rel, text in sources:
        touches = TOUCHES_BARE if Path(rel).name == "counters.rs" else TOUCHES_QUALIFIED
        aliases = lock_aliases(text)
        whole = rel in cfg_test_files
        for name, _line, body, attrs in test_bodies(text, whole_file_is_test_mod=whole):
            if "#[test]" not in attrs or "#[ignore" in attrs or name in aliases:
                continue
            fams = set()
            if touches.search(body):
                fams.add("counters")
            if next(env_touches(body, consts), None):
                fams.add("env")
            if fams:
                names.setdefault(name, set()).update(fams)
    return names


GUARDED_SPECIMEN = """
mod tests {
    use crate::allocator::ledger::test_lock;
    #[test]
    fn a_guarded_test() {
        let _g = test_lock();
        reset();
    }
}
"""

UNGUARDED_SPECIMEN = """
mod tests {
    #[test]
    fn an_unguarded_test() {
        reset();
        record_compute_call();
    }
}
"""

FOREIGN_LOCK_SPECIMEN = """
mod tests {
    use crate::somewhere_else::test_lock;
    #[test]
    fn a_test_holding_the_wrong_mutex() {
        let _g = test_lock();
        reset();
    }
}
"""

#: Two tests sharing one variable, one of them writing: contended, both unguarded.
ENV_CONTENDED_SPECIMEN = """
const ENV_THING: &str = "ONNXRUNTIME_EP_VULKAN_SPECIMEN_THING";
mod tests {
    use super::*;
    #[test]
    fn writer_without_the_lock() {
        unsafe { std::env::set_var(ENV_THING, "1") };
        unsafe { std::env::remove_var(ENV_THING) };
    }
    #[test]
    fn reader_without_the_lock() {
        let _v = std::env::var("ONNXRUNTIME_EP_VULKAN_SPECIMEN_THING");
    }
}
"""

#: The same pair, guarded. Green -- otherwise the check above fails on everything.
ENV_GUARDED_SPECIMEN = """
const ENV_THING: &str = "ONNXRUNTIME_EP_VULKAN_SPECIMEN_THING";
mod tests {
    use crate::allocator::ledger::test_lock;
    #[test]
    fn writer_with_the_lock() {
        let _g = test_lock();
        unsafe { std::env::set_var(ENV_THING, "1") };
    }
    #[test]
    fn reader_with_the_lock() {
        let _g = test_lock();
        let _v = std::env::var(ENV_THING);
    }
}
"""

#: One writer, no second toucher: reported, never gated. Present so the distinction
#: between "contended" and "sole" is demonstrated in both of its states, not asserted.
ENV_SOLE_SPECIMEN = """
mod tests {
    #[test]
    fn the_only_toucher() {
        unsafe { std::env::set_var("ONNXRUNTIME_EP_VULKAN_SPECIMEN_SOLO", "1") };
    }
}
"""

#: An `#[ignore]`d writer sharing a variable with one live reader: the live population is
#: one, so it is not contended. This is `dispatch_integration.rs`'s situation, in miniature.
ENV_IGNORED_SPECIMEN = """
mod tests {
    #[test]
    #[ignore = "positive control, run in isolation"]
    fn the_ignored_writer() {
        unsafe { std::env::set_var("ONNXRUNTIME_EP_VULKAN_SPECIMEN_IGN", "1") };
    }
    #[test]
    fn a_live_reader() {
        let _v = std::env::var("ONNXRUNTIME_EP_VULKAN_SPECIMEN_IGN");
    }
}
"""

#: A `#[test]` at file top level -- the file *is* the test module. Invisible to the
#: pre-Round-37 scanner, which only looked inside `mod ... {` blocks.
ENV_WHOLE_FILE_SPECIMEN = """
#[test]
fn top_level_writer() {
    unsafe { std::env::set_var("ONNXRUNTIME_EP_VULKAN_SPECIMEN_TOP", "1") };
}

#[test]
fn top_level_reader() {
    let _v = std::env::var("ONNXRUNTIME_EP_VULKAN_SPECIMEN_TOP");
}
"""

#: Round 38. A module-private mutex that *looks* exactly like the sanctioned guard: same
#: signature, same `unwrap_or_else(into_inner)` shape, same `let _g = ..()` call site --
#: and a different lock. This is PR #54's blocker in miniature: four
#: `select_device_strict_*` tests held a private `serial_env()` while three tests over the
#: same variable held `crate::allocator::ledger::test_lock()`. Two disjoint mutexes over one
#: process-global is not mutual exclusion, and a reviewer reading either test alone sees a
#: guard. Measured pre-fix: 14 reds in 25 reps of the `env:device_selector` pool.
#:
#: The writer must be flagged and the reader must not, because "some tests here are
#: guarded" is precisely the state that made this survive three reviews.
ENV_PRIVATE_LOOKALIKE_SPECIMEN = """
const ENV_THING: &str = "ONNXRUNTIME_EP_VULKAN_SPECIMEN_LOOKALIKE";
mod tests {
    use super::*;

    /// Looks like the sanctioned guard. Is a different mutex.
    fn serial_env() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn writer_under_the_private_mutex() {
        let _g = serial_env();
        unsafe { std::env::set_var(ENV_THING, "1") };
    }

    #[test]
    fn reader_under_the_project_lock() {
        let _g = crate::allocator::ledger::test_lock();
        let _v = std::env::var(ENV_THING);
    }
}
"""

#: Round 38, the other half. The sanctioned *forwarding* helper -- a different name for the
#: same mutex, which `ep.rs::serialize()` has used since Round 36. It must stay green, or
#: the only remedy for the lookalike above would be to ban local guard fns outright, and an
#: auditor that fires on the accepted precedent gets suppressed rather than obeyed.
#:
#: This pair and the pair above differ by exactly one thing -- where the body's lock comes
#: from -- so a green here plus a red there is evidence the tool discriminates on the lock
#: identity and not on the spelling of the call site.
ENV_FORWARDING_ALIAS_SPECIMEN = """
const ENV_THING: &str = "ONNXRUNTIME_EP_VULKAN_SPECIMEN_FORWARD";
mod tests {
    use super::*;

    /// Same shape as the lookalike above, forwarding to the one project-wide lock.
    fn serialize() -> std::sync::MutexGuard<'static, ()> {
        crate::allocator::ledger::test_lock()
    }

    #[test]
    fn writer_through_the_alias() {
        let _g = serialize();
        unsafe { std::env::set_var(ENV_THING, "1") };
    }

    #[test]
    fn reader_through_the_alias() {
        let _g = serialize();
        let _v = std::env::var(ENV_THING);
    }
}
"""


def _env_specimen(text: str, whole=False):
    sources = [("specimen.rs", text)]
    consts = env_consts(sources)
    census: dict[str, list] = {}
    aliases = lock_aliases(text)
    holds = re.compile(
        r"\btest_lock\(\)" + ("|" + "|".join(rf"\b{a}\(\)" for a in aliases) if aliases else "")
    )
    for name, line_no, body, attrs in test_bodies(text, whole_file_is_test_mod=whole):
        if "#[test]" not in attrs:
            continue
        touched: dict[str, set] = {}
        for var, kind in env_touches(body, consts):
            touched.setdefault(var, set()).add(kind)
        for var, kinds in touched.items():
            census.setdefault(var, []).append(
                {
                    "file": "specimen.rs",
                    "test": name,
                    "line": line_no,
                    "kinds": sorted(kinds),
                    "guarded": bool(holds.search(body)),
                    "ignored": "#[ignore" in attrs,
                }
            )
    return env_findings(census)


def selftest() -> int:
    """An auditor never shown to fire is indistinguishable from a blind one."""
    failures = []
    u, w = audit_text("counters.rs", GUARDED_SPECIMEN)
    if u or w:
        failures.append(f"clean specimen was flagged: {u} {w}")
    u, _ = audit_text("counters.rs", UNGUARDED_SPECIMEN)
    if not u:
        failures.append("unguarded specimen was NOT flagged -- the auditor is blind")
    _, w = audit_text("counters.rs", FOREIGN_LOCK_SPECIMEN)
    if not w:
        failures.append("foreign-lock specimen NOT flagged -- a different mutex reads as a guard")

    f, c, s = _env_specimen(ENV_CONTENDED_SPECIMEN)
    if len(f) != 2 or not c:
        failures.append(f"env contended specimen: expected 2 findings, got {f}")
    if "ONNXRUNTIME_EP_VULKAN_SPECIMEN_THING" not in c:
        failures.append("env const was not resolved to its literal -- two spellings, two globals")
    f, c, s = _env_specimen(ENV_GUARDED_SPECIMEN)
    if f:
        failures.append(f"guarded env specimen was flagged: {f}")
    if not c:
        failures.append("guarded env specimen was not even seen as contended")
    f, c, s = _env_specimen(ENV_SOLE_SPECIMEN)
    if f or c:
        failures.append(f"sole env writer must not gate: findings={f} contended={c}")
    if not s:
        failures.append("sole env writer was not reported at all -- silently dropped")
    f, c, s = _env_specimen(ENV_IGNORED_SPECIMEN)
    if f or c:
        failures.append(f"#[ignore]d writer must not make a variable contended: {f} {c}")
    f, c, s = _env_specimen(ENV_WHOLE_FILE_SPECIMEN, whole=True)
    if len(f) != 2:
        failures.append(f"top-level #[test] fns not seen in a whole-file test module: {f}")
    f2, _, _ = _env_specimen(ENV_WHOLE_FILE_SPECIMEN, whole=False)
    if f2:
        failures.append("whole-file flag is inert -- it found them without being told")

    # Round 38: the lock's *identity*, not the call site's spelling. These two specimens are
    # identical except for which mutex the local guard fn reaches, so they only both hold if
    # the tool is discriminating on the thing that actually decides mutual exclusion.
    f, c, _s = _env_specimen(ENV_PRIVATE_LOOKALIKE_SPECIMEN)
    if [name for _f, name, _l, _v in f] != ["writer_under_the_private_mutex"]:
        failures.append(
            "a module-private mutex read as a guard (or the project-locked reader was flagged): "
            f"{f} -- this is PR #54's blocker and it must be exactly one finding"
        )
    if "ONNXRUNTIME_EP_VULKAN_SPECIMEN_LOOKALIKE" not in c:
        failures.append("lookalike specimen was not even seen as contended")
    f, c, _s = _env_specimen(ENV_FORWARDING_ALIAS_SPECIMEN)
    if f:
        failures.append(
            f"a helper forwarding to ledger::test_lock() was flagged: {f} -- an auditor that "
            "fires on the accepted `ep.rs::serialize()` precedent gets suppressed, not obeyed"
        )
    if "ONNXRUNTIME_EP_VULKAN_SPECIMEN_FORWARD" not in c:
        failures.append("forwarding-alias specimen was not seen as contended -- vacuously green")

    for line in failures:
        print(f"SELFTEST FAIL: {line}")
    if not failures:
        print(
            "selftest: 11/11 (counters: clean green, unguarded red, foreign-lock red; "
            "env: contended red, guarded green, sole ungated-but-reported, ignored ungated, "
            "whole-file red and inert without the flag; private-lookalike mutex red on the "
            "writer only, forwarding alias green and non-vacuously contended)"
        )
    return 1 if failures else 0


def main() -> int:
    if "--selftest" in sys.argv:
        rc = selftest()
        if rc:
            return rc
    unguarded, wrong_lock, env_unguarded, contended, sole = audit()
    for fname, name, line_no in unguarded:
        print(f"UNGUARDED  {fname}:{line_no}  {name}")
    for fname, why in wrong_lock:
        print(f"WRONG-LOCK {fname}  {why}")
    for fname, name, line_no, var in env_unguarded:
        print(f"ENV-UNGUARDED {fname}:{line_no}  {name}  [{var}]")

    if "--pairs" in sys.argv:
        print("\n-- contended environment variables (>=2 live tests, >=1 writer) --")
        for var, recs in sorted(contended.items()):
            print(f"  {var}")
            for r in recs:
                mark = "ok " if r["guarded"] else "RACE"
                print(f"    {mark} {r['file']}:{r['line']} {r['test']} ({','.join(r['kinds'])})")
        print("\n-- sole writers (reported, NOT gated: no second test touches them) --")
        for var, recs in sorted(sole.items()):
            for r in recs:
                print(f"  {var}: {r['file']}:{r['line']} {r['test']} ({','.join(r['kinds'])})")

    if "--json" in sys.argv:
        print(
            json.dumps(
                {
                    "unguarded": unguarded,
                    "wrong_lock": wrong_lock,
                    "env_unguarded": env_unguarded,
                    "contended": contended,
                    "sole_writers": sole,
                },
                indent=2,
            )
        )

    print(
        f"\n{len(unguarded)} unguarded counter test(s), {len(wrong_lock)} wrong-lock finding(s), "
        f"{len(env_unguarded)} unguarded env test(s) over {len(contended)} contended variable(s) "
        f"({len(sole)} sole writer(s) reported, not gated) "
        f"across {len(rust_sources())} rust source(s)"
    )
    if "--check" in sys.argv:
        return 1 if (unguarded or wrong_lock or env_unguarded) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
