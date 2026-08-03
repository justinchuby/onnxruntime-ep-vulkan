"""Which #[test] fns touch the process-global counters without holding the test lock?

Round 36 diagnosis: `cargo test --lib counters` failed 8 times in 20, in four different
tests at four different assertion sites, because `counters::reset()` and the `record_*`
writers act on process-global statics that libtest runs concurrently on its thread pool.
The convention already exists -- `let _g = crate::allocator::ledger::test_lock();` -- it
was simply not applied everywhere, and three unguarded tests were enough to clobber every
guarded one.

Two things this checks, because a guard that is the *wrong mutex* would look identical to
a guard that works:

  1. every #[test] that reads or writes the global counters holds a `test_lock()`;
  2. every file that holds one imports it from `crate::allocator::ledger`, so all of them
     are the same lock.

Run:  python rust/tools/audit_counter_test_lock.py [--check] [--selftest]
"""
from __future__ import annotations

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


def test_bodies(text: str):
    """Yield (name, line_no, body) for every fn inside a `mod tests`/`#[cfg(test)] mod`.

    Extent: this follows *bodies*, not call graphs. A #[test] that reaches the globals
    only through a helper defined elsewhere is invisible here -- which is why helpers
    inside the test module are enumerated too, and why the empirical repeat harness
    (`rust/tools/repeat_counters_tests.ps1`) remains the check of record.
    """
    for mod_m in re.finditer(r"\bmod\s+([a-z0-9_]+)\s*\{", text):
        if "test" not in mod_m.group(1):
            # Only test modules; production code is allowed to touch its own globals.
            head = text[max(0, mod_m.start() - 200) : mod_m.start()]
            if "#[cfg(test)]" not in head:
                continue
        body, _ = _brace_body(text, mod_m.end() - 1)
        base = mod_m.start()
        for m in re.finditer(r"\bfn\s+([a-z0-9_]+)\s*[(<]", body):
            name = m.group(1)
            try:
                fn_body, _ = _brace_body(body, m.end())
            except ValueError:
                continue
            line_no = text.count("\n", 0, base + m.start()) + 1
            yield name, line_no, fn_body


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


def audit_text(fname: str, text: str):
    """Return (unguarded, wrong_lock) findings for one file's source."""
    touches = TOUCHES_BARE if Path(fname).name == "counters.rs" else TOUCHES_QUALIFIED
    aliases = lock_aliases(text)
    holds = re.compile(
        r"\btest_lock\(\)" + ("|" + "|".join(rf"\b{a}\(\)" for a in aliases) if aliases else "")
    )
    unguarded, wrong_lock = [], []
    uses_bare = False
    for name, line_no, body in test_bodies(text):
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


def audit():
    unguarded, wrong_lock = [], []
    for path in rust_sources():
        rel = path.relative_to(RUST_SRC).as_posix()
        u, w = audit_text(rel, path.read_text(encoding="utf-8"))
        unguarded += u
        wrong_lock += w
    return unguarded, wrong_lock


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
    for line in failures:
        print(f"SELFTEST FAIL: {line}")
    if not failures:
        print("selftest: 3/3 (clean green, unguarded red, foreign-lock red)")
    return 1 if failures else 0


def main() -> int:
    if "--selftest" in sys.argv:
        rc = selftest()
        if rc:
            return rc
    unguarded, wrong_lock = audit()
    for fname, name, line_no in unguarded:
        print(f"UNGUARDED  {fname}:{line_no}  {name}")
    for fname, why in wrong_lock:
        print(f"WRONG-LOCK {fname}  {why}")
    print(
        f"\n{len(unguarded)} unguarded test(s), {len(wrong_lock)} wrong-lock finding(s) "
        f"across {len(rust_sources())} rust source(s)"
    )
    if "--check" in sys.argv:
        return 1 if (unguarded or wrong_lock) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
