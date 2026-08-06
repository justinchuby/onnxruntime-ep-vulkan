#!/usr/bin/env python3
"""``phi35_identity_audit`` — which files produce Phi-3.5 evidence, and do they name the
model they produced it from?  Answered from the **abstract syntax tree**, not from grep.

THE DEFECT THIS EXISTS FOR
==========================
Issue #19 gave every archival Phi-3.5 probe a ``_result_identity()`` stamp — the resolved
``onnx_file`` and its exact ``onnx_sha256`` — so a ``PHI35_MODEL`` override, or a stale file
silently re-downloaded to the historical default path, can never be absorbed into the
evidence unnoticed.  The contract was then policed by a pair of regexes::

    _SUBPROCESS_SPAWNS_A_SCRIPT   = re.compile(r'subprocess\\.run\\(\\s*\\[\\s*sys\\.executable')
    _SUBPROCESS_INHERITS_FULL_ENV = re.compile(r'dict\\(os\\.environ\\)')

Both are **source-text** screens, and a source-text screen over Python has three failures
that are not incidental to how it was written:

1.  IT SEES ONLY ONE SPELLING.  The regex above matches an argv written *inline inside the
    call*.  The two files this module was written for do not write it that way::

        cmd = [sys.executable, str(PROBE), "--worker", ...]      # rust/tools/device_loss_gate.py
        proc = subprocess.run(cmd, env=env, capture_output=True)

        cmd = [sys.executable, str(NIOBE_PROBE), "--out", str(out)]  # bench/results/probe_device_memory_kv.py
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True)

    Identical semantics, one local variable, invisible to the screen.  Both spawned a
    PHI35_MODEL-reading probe with an inherited environment and then wrote their own JSON
    record — ``device_loss_gate.json`` and ``device_memory_kv_lanes.json`` — with no model
    identity in it at all.  ``probe_device_memory_kv.py`` went further and *read* the
    child's record, which carries ``onnx_file``/``onnx_sha256``, and dropped both fields on
    the floor while writing its own.  The contract said every producer names its model; two
    producers did not; the screen said PASS.

2.  IT MATCHES ITS OWN SOURCE.  A screen written as a string that describes the shape it
    rejects contains that shape.  ``ci/test_lane_checks.py`` spawns real checks with
    ``subprocess.run([sys.executable, ...])`` and hands them ``dict(os.environ)``, and it
    names probe module stems in its assertions — so the discovery regex matched the screen
    itself, and the only reason it did not report the screen as a violation is that the
    screen does not happen to write a JSON record.  A guard whose non-firing depends on an
    incidental property of the file it is looking at is not a guard.

3.  IT CANNOT TELL CODE FROM PROSE.  A docstring quoting ``subprocess.run([sys.executable``
    to explain the defect is a match; a planted fixture written as a triple-quoted string in
    a test is a match.  Every such match is a false positive that has to be suppressed by
    hand, and each hand-suppression is another place the real thing can hide.

WHAT THIS MODULE DOES INSTEAD
=============================
It parses each file with :mod:`ast` and answers four questions structurally:

*   **Does it read ``PHI35_MODEL``?**  ``os.environ.get("PHI35_MODEL")``,
    ``os.environ["PHI35_MODEL"]``, ``os.getenv("PHI35_MODEL")``, and the same four spelled
    through ``import os as _os``, ``from os import environ``, ``from os import environ as
    env``, ``from os import getenv``.  A string ``"PHI35_MODEL"`` sitting in a docstring or
    in a planted test fixture is not an environment read and is not counted.

*   **Does it spawn a Python script, and with whose environment?**  ``subprocess.run``,
    ``.Popen``, ``.call``, ``.check_call``, ``.check_output`` — including
    ``from subprocess import run`` and ``import subprocess as sp`` — with the argv given
    either inline *or* as a local variable assigned a list literal.  Simple
    assignment-tracing, one level, no reaching-definitions analysis: that is enough for both
    real cases and for the shapes anyone writes by hand, and it is stated here rather than
    left to be discovered.  The environment is traced the same way: ``env=dict(os.environ)``,
    ``env=os.environ.copy()``, ``env={**os.environ}``, or any of those bound to a name first
    and mutated before the call — which is exactly what both real cases do.

*   **Does it write a JSON record?**  ``json.dump(...)``, or a ``.write_text``/``.write``
    whose argument subtree contains ``json.dumps(...)``.  A ``json.dumps`` that only reaches
    ``print()`` is not a record and does not count — the previous ``json\\.dumps?\\(`` regex
    could not tell those apart.

*   **Does it name the model?**  A ``def _result_identity``, an import of one, or a literal
    use of the ``onnx_file``/``onnx_sha256`` field names in code (a propagator that lifts the
    fields out of a child's record rather than re-hashing).

A file is an IDENTITY-BEARING PRODUCER when it writes a JSON record **and** the model
reaches it — either because it reads ``PHI35_MODEL`` itself, or because it spawns, with an
inherited environment, a script that does.  That second relation is closed transitively, so
a wrapper of a wrapper is caught by construction rather than by someone remembering to add
a third case.  A producer that does not name its model is the violation this module reports.

WHY IT DOES NOT MATCH ITSELF
============================
Two independent reasons, because one of them is not enough:

*   **Structural.**  Nothing in this file or in its tests is a *call*; the shapes they quote
    live in docstrings and in fixture strings, and an AST walk never reaches them.  This is
    the load-bearing reason.

*   **Declared.**  :data:`NOT_A_PRODUCER` lists the screens themselves — this module, its
    lane tests, the negative controls — with the reason: a screen that spawns a checker and
    writes a temporary file is not producing evidence about Phi-3.5.  The list is short,
    explicit, and asserted upon: ``ci/test_lane_checks.py`` plants the same shapes into a
    ``tmp_path`` fixture and requires them to be REPORTED there, so the exclusion is known to
    be a declaration about those files and not a hole in the analysis.

VERDICTS
--------
    0  PHI35-IDENTITY: PASS  — every producer names the model it consumed
    1  PHI35-IDENTITY: FAIL  — a producer writes a record with no model identity; each named
    4  PHI35-IDENTITY: ERROR(instrument=...) — the audit could not read or parse the tree

An unparseable file is an ERROR, never a silent skip: "I could not look at it" and "I looked
and it was clean" are different answers and must not share an exit code.

STATED LIMITS
=============
Written down because an unstated limit reads as a claim:

*   **Record consumption is not a relation here.**  A file that reads another producer's
    JSON *record* off disk and derives a new record from it is not discovered by this
    module; the relations are direct environment read and subprocess spawn.  The one such
    reader in the tree today — ``bench/results/probe_lane_logits_identity.py``, which grades
    the device-loss gate's leftover logits — was found by hand and propagates the gate
    record's identity explicitly.  Adding the relation would mean matching output filenames
    to input filenames across argparse defaults, which is guesswork wearing an AST.

*   **Assignment tracing is flat and last-write-wins.**  A name rebound in a branch, or an
    argv assembled by ``append`` in a loop, resolves to whatever the last module- or
    function-level assignment was.  Both real cases, and every hand-written spawn in this
    tree, assign once.

*   **Dynamic dispatch is out of reach.**  ``getattr(subprocess, name)(...)``, a script path
    computed at runtime from data, or an environment built by a helper in another module is
    not traced.  Nothing in the tree does this; if something starts to, this module will go
    quiet about it rather than fail, which is why the LIVE arm is paired with planted arms
    that prove the rule still fires.

Run:  python ci/phi35_identity_audit.py [--root PATH] [--list]
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

TAG = "PHI35-IDENTITY"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 4

#: The environment variable that selects which Phi-3.5 artifact a probe measures.
MODEL_ENV = "PHI35_MODEL"

#: The two fields a record must carry to name the bytes it was computed from.
IDENTITY_FIELDS = ("onnx_file", "onnx_sha256")

#: Never walked: not our source, or build output.  Same list as
#: ``ci/check_hardcoded_foundry_paths.py`` so the two screens cannot disagree about what
#: "the tree" means.
EXCLUDED_DIRS = (".git", ".venv", "venv", "target", "node_modules", "__pycache__", ".squad")

#: Screens, not producers.  Each of these really does spawn a Python script with an
#: inherited environment and really does write JSON — into a temporary directory, about a
#: *check*, never about a model.  Requiring them to stamp a model identity would be
#: requiring them to invent one.  This is a declaration about these five files; the analysis
#: itself is structural, and ci/test_lane_checks.py plants the identical shapes elsewhere in
#: the tree and requires them to be reported.
NOT_A_PRODUCER = (
    "ci/phi35_identity_audit.py",
    "ci/test_lane_checks.py",
    "ci/negative_control_hardcoded_foundry_paths.py",
    "ci/check_hardcoded_foundry_paths.py",
    "ci/check_open_reds.py",
)

_SUBPROCESS_SPAWNERS = ("run", "Popen", "call", "check_call", "check_output")
_JSON_WRITE_SINKS = ("write_text", "write", "writelines")


@dataclass(frozen=True)
class Spawn:
    """One subprocess call that starts a Python script."""

    lineno: int
    argv_origin: str            # "inline" or "variable:<name>"
    scripts: tuple[str, ...]    # basenames of *.py named anywhere in the argv
    inherits_env: bool
    env_origin: str             # "inline", "variable:<name>", "os.environ" or "none"


@dataclass
class FileFacts:
    rel: str
    reads_model_env: bool = False
    model_env_lines: tuple[int, ...] = ()
    spawns: tuple[Spawn, ...] = ()
    writes_json_record: bool = False
    json_record_lines: tuple[int, ...] = ()
    defines_result_identity: bool = False
    imports_result_identity: bool = False
    uses_identity_fields: bool = False
    reads_json: bool = False

    @property
    def names_the_model(self) -> bool:
        return (
            self.defines_result_identity
            or self.imports_result_identity
            or self.uses_identity_fields
        )

    @property
    def script_stem(self) -> str:
        return self.rel.rsplit("/", 1)[-1]


@dataclass
class Violation:
    rel: str
    why: str
    reached_via: str
    json_lines: tuple[int, ...]


class AuditError(RuntimeError):
    """The audit could not read or parse a file.  Never a silent skip."""


# ---------------------------------------------------------------------------
# AST analysis
# ---------------------------------------------------------------------------


class _Analyzer(ast.NodeVisitor):
    """One pass over one module.

    Name resolution is deliberately flat: a single ``self.assigned`` map from a bare name to
    the last expression assigned to it anywhere in the file.  That is not sound
    reaching-definitions analysis and is not meant to be — the shapes in question
    (``cmd = [...]`` then ``subprocess.run(cmd, ...)``, ``env = dict(os.environ)`` then
    mutate then pass) are local and unambiguous, and a name bound twice to two different
    argv lists still yields the union of the scripts named, which errs toward *reporting*
    rather than toward silence.
    """

    def __init__(self) -> None:
        self.os_aliases: set[str] = set()
        self.environ_aliases: set[str] = set()
        self.getenv_aliases: set[str] = set()
        self.subprocess_aliases: set[str] = set()
        self.spawner_aliases: dict[str, str] = {}
        self.json_aliases: set[str] = set()
        self.json_dump_aliases: set[str] = set()
        self.json_dumps_aliases: set[str] = set()
        self.json_load_aliases: set[str] = set()

        self.assigned: dict[str, ast.expr] = {}
        self.env_names: set[str] = set()

        self.facts_reads_model = False
        self.model_lines: list[int] = []
        self.spawns: list[Spawn] = []
        self.json_record_lines: list[int] = []
        self.defines_result_identity = False
        self.imports_result_identity = False
        self.uses_identity_fields = False
        self.reads_json = False

    # -- imports ------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            if alias.name == "os" or alias.name.startswith("os."):
                self.os_aliases.add(bound)
            elif alias.name == "subprocess":
                self.subprocess_aliases.add(bound)
            elif alias.name == "json":
                self.json_aliases.add(bound)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            bound = alias.asname or alias.name
            if mod == "os":
                if alias.name == "environ":
                    self.environ_aliases.add(bound)
                elif alias.name == "getenv":
                    self.getenv_aliases.add(bound)
            elif mod == "subprocess":
                if alias.name in _SUBPROCESS_SPAWNERS:
                    self.spawner_aliases[bound] = alias.name
            elif mod == "json":
                if alias.name == "dump":
                    self.json_dump_aliases.add(bound)
                elif alias.name == "dumps":
                    self.json_dumps_aliases.add(bound)
                elif alias.name in ("load", "loads"):
                    self.json_load_aliases.add(bound)
            if alias.name == "_result_identity":
                self.imports_result_identity = True
        self.generic_visit(node)

    # -- definitions --------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "_result_identity":
            self.defines_result_identity = True
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name == "_result_identity":
            self.defines_result_identity = True
        self.generic_visit(node)

    # -- assignments --------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned[target.id] = node.value
                if self._is_environ_copy(node.value):
                    self.env_names.add(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            self.assigned[node.target.id] = node.value
            if self._is_environ_copy(node.value):
                self.env_names.add(node.target.id)
        self.generic_visit(node)

    # -- expressions --------------------------------------------------------

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self._is_environ(node.value) and _const_str(node.slice) == MODEL_ENV:
            self.facts_reads_model = True
            self.model_lines.append(node.lineno)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # An identity FIELD NAME used as a real constant in code — a propagator lifting
        # `onnx_file`/`onnx_sha256` out of a child's record rather than re-hashing.  A
        # docstring mentioning the words is an ast.Expr statement, not one of these: it is
        # skipped in `visit_Expr` below before this can see it.
        if isinstance(node.value, str) and node.value in IDENTITY_FIELDS:
            self.uses_identity_fields = True
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        # A bare string statement is a docstring/comment-by-string.  Its contents are prose,
        # never code, and must not be able to satisfy — or trip — any part of this audit.
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._check_env_read(node)
        self._check_spawn(node)
        self._check_json(node)
        self.generic_visit(node)

    # -- helpers ------------------------------------------------------------

    def _is_environ(self, node: ast.expr) -> bool:
        """``os.environ`` under any alias, or a bare name bound to it by ``from os import``."""
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            return isinstance(node.value, ast.Name) and node.value.id in self.os_aliases
        return isinstance(node, ast.Name) and node.id in self.environ_aliases

    def _is_environ_copy(self, node: ast.expr) -> bool:
        """An expression that produces a *mutable copy of the whole parent environment*.

        ``dict(os.environ)``, ``os.environ.copy()``, ``{**os.environ}``, and the same three
        under any alias.  This is the shape that hands ``PHI35_MODEL`` to a child without a
        single line in the parent ever naming it.
        """
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "dict" and node.args:
                return self._is_environ(node.args[0])
            if isinstance(func, ast.Attribute) and func.attr == "copy":
                return self._is_environ(func.value)
        if isinstance(node, ast.Dict):
            return any(
                key is None and self._is_environ(value)
                for key, value in zip(node.keys, node.values)
            )
        if isinstance(node, ast.Name):
            return node.id in self.env_names
        return False

    def _check_env_read(self, node: ast.Call) -> None:
        func = node.func
        name = None
        if isinstance(func, ast.Attribute):
            if func.attr in ("get", "getenv") and (
                self._is_environ(func.value)
                or (isinstance(func.value, ast.Name) and func.value.id in self.os_aliases)
            ):
                name = func.attr
        elif isinstance(func, ast.Name) and func.id in self.getenv_aliases:
            name = "getenv"
        if name is None:
            return
        if node.args and _const_str(node.args[0]) == MODEL_ENV:
            self.facts_reads_model = True
            self.model_lines.append(node.lineno)

    def _resolve(self, node: ast.expr) -> ast.expr:
        seen = 0
        while isinstance(node, ast.Name) and node.id in self.assigned and seen < 8:
            node = self.assigned[node.id]
            seen += 1
        return node

    def _scripts_in(self, node: ast.expr, depth: int = 0) -> set[str]:
        """Every ``*.py`` basename this expression can name, following variables.

        The literal almost never sits in the argv list itself.  Both real cases spell the
        target as a module-level constant::

            PROBE = REPO / "bench" / "results" / "probe_kv_chain_phi35.py"
            cmd = [sys.executable, str(PROBE), "--worker", ...]

        so an analysis that only reads string constants *inside the call* sees an argv with
        no script in it — which is precisely how a regex over the call site missed both.
        Every ``ast.Name`` reachable from the argv is therefore resolved through the
        assignment map and searched too, to a bounded depth.
        """
        out: set[str] = set()
        if depth > 6:
            return out
        for sub in ast.walk(node):
            text = _const_str(sub)
            if text and text.endswith(".py"):
                out.add(text.replace("\\", "/").rsplit("/", 1)[-1])
            elif isinstance(sub, ast.Name) and sub.id in self.assigned:
                out |= self._scripts_in(self.assigned[sub.id], depth + 1)
        return out

    def _mentions_sys_executable(self, node: ast.expr, depth: int = 0) -> bool:
        if depth > 6:
            return False
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr == "executable"
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "sys"
            ):
                return True
            if isinstance(sub, ast.Name) and sub.id in self.assigned:
                if self._mentions_sys_executable(self.assigned[sub.id], depth + 1):
                    return True
        return False

    def _check_spawn(self, node: ast.Call) -> None:
        func = node.func
        is_spawn = False
        if isinstance(func, ast.Attribute) and func.attr in _SUBPROCESS_SPAWNERS:
            is_spawn = isinstance(func.value, ast.Name) and func.value.id in self.subprocess_aliases
        elif isinstance(func, ast.Name) and func.id in self.spawner_aliases:
            is_spawn = True
        if not is_spawn or not node.args:
            return

        argv = node.args[0]
        origin = f"variable:{argv.id}" if isinstance(argv, ast.Name) else "inline"
        resolved = self._resolve(argv)
        scripts = tuple(sorted(self._scripts_in(resolved)))
        if not scripts and not self._mentions_sys_executable(resolved):
            return

        inherits, env_origin = False, "none"
        for kw in node.keywords:
            if kw.arg != "env" or kw.value is None:
                continue
            env_origin = (
                f"variable:{kw.value.id}" if isinstance(kw.value, ast.Name) else "inline"
            )
            if self._is_environ(kw.value):
                inherits, env_origin = True, "os.environ"
            elif self._is_environ_copy(self._resolve(kw.value)) or (
                isinstance(kw.value, ast.Name) and kw.value.id in self.env_names
            ):
                inherits = True
        if not any(kw.arg == "env" for kw in node.keywords):
            # No `env=` at all: the child inherits this process's environment wholesale.
            # That is the *strongest* form of inheritance, not the absence of it.
            inherits, env_origin = True, "inherited (no env= given)"

        self.spawns.append(
            Spawn(
                lineno=node.lineno,
                argv_origin=origin,
                scripts=scripts,
                inherits_env=inherits,
                env_origin=env_origin,
            )
        )

    def _is_json_dumps(self, node: ast.expr) -> bool:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            if isinstance(f, ast.Attribute) and f.attr == "dumps":
                if isinstance(f.value, ast.Name) and f.value.id in self.json_aliases:
                    return True
            if isinstance(f, ast.Name) and f.id in self.json_dumps_aliases:
                return True
        return False

    def _check_json(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in ("load", "loads") and (
                isinstance(func.value, ast.Name) and func.value.id in self.json_aliases
            ):
                self.reads_json = True
            if func.attr == "dump" and (
                isinstance(func.value, ast.Name) and func.value.id in self.json_aliases
            ):
                self.json_record_lines.append(node.lineno)
                return
            if func.attr in _JSON_WRITE_SINKS:
                if any(self._is_json_dumps(a) for a in node.args):
                    self.json_record_lines.append(node.lineno)
                return
        if isinstance(func, ast.Name):
            if func.id in self.json_load_aliases:
                self.reads_json = True
            elif func.id in self.json_dump_aliases:
                self.json_record_lines.append(node.lineno)


def _const_str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


# ---------------------------------------------------------------------------
# Repository-level audit
# ---------------------------------------------------------------------------


def analyze_source(rel: str, source: str) -> FileFacts:
    """Facts about one module, from its AST.  Raises :class:`AuditError` if it will not parse."""
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:  # noqa: PERF203
        raise AuditError(f"{rel}: {exc}") from exc
    an = _Analyzer()
    an.visit(tree)
    return FileFacts(
        rel=rel,
        reads_model_env=an.facts_reads_model,
        model_env_lines=tuple(sorted(set(an.model_lines))),
        spawns=tuple(an.spawns),
        writes_json_record=bool(an.json_record_lines),
        json_record_lines=tuple(sorted(set(an.json_record_lines))),
        defines_result_identity=an.defines_result_identity,
        imports_result_identity=an.imports_result_identity,
        uses_identity_fields=an.uses_identity_fields,
        reads_json=an.reads_json,
    )


def iter_python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        yield path, rel.as_posix()


def analyze_tree(root: Path) -> dict[str, FileFacts]:
    facts: dict[str, FileFacts] = {}
    for path, rel in iter_python_files(root):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise AuditError(f"{rel}: unreadable ({exc})") from exc
        facts[rel] = analyze_source(rel, source)
    return facts


def model_bearing_scripts(facts: dict[str, FileFacts]) -> dict[str, str]:
    """Basename → rel path for every script the model reaches.

    Seeded with the direct ``PHI35_MODEL`` readers and closed under "spawns it with an
    inherited environment", so a wrapper of a wrapper is included by construction.  Fixed
    point rather than a fixed number of hops: the previous screen hardcoded one hop and the
    two files it missed were one hop away, which is the coincidence that made the bug look
    like a scoping choice.
    """
    reached: dict[str, str] = {
        f.script_stem: rel for rel, f in facts.items() if f.reads_model_env
    }
    changed = True
    while changed:
        changed = False
        for rel, f in facts.items():
            if f.script_stem in reached:
                continue
            for spawn in f.spawns:
                if spawn.inherits_env and any(s in reached for s in spawn.scripts):
                    reached[f.script_stem] = rel
                    changed = True
                    break
    return reached


def producers(facts: dict[str, FileFacts]) -> dict[str, str]:
    """Every file that writes a JSON record the model reaches → how the model reached it."""
    reached = model_bearing_scripts(facts)
    out: dict[str, str] = {}
    for rel, f in facts.items():
        if rel in NOT_A_PRODUCER or not f.writes_json_record:
            continue
        if f.reads_model_env:
            out[rel] = f"reads {MODEL_ENV} directly (line(s) {list(f.model_env_lines)})"
            continue
        for spawn in f.spawns:
            named = [s for s in spawn.scripts if s in reached and reached[s] != rel]
            if spawn.inherits_env and named:
                out[rel] = (
                    f"line {spawn.lineno}: spawns {', '.join(sorted(named))} with argv from "
                    f"{spawn.argv_origin} and env from {spawn.env_origin}"
                )
                break
    return out


def violations_in(facts: dict[str, FileFacts]) -> tuple[list[Violation], dict[str, str]]:
    """The verdict rule itself, over facts from any source.

    Split out from :func:`audit` so the tests can drive the *production* rule over planted
    sources instead of restating it — a control that reimplements the rule it is checking
    proves only that two copies agree.
    """
    found = producers(facts)
    violations = [
        Violation(
            rel=rel,
            why=(
                "writes a JSON record but never names the model: no `_result_identity`, no "
                "import of one, and no use of the onnx_file/onnx_sha256 fields"
            ),
            reached_via=how,
            json_lines=facts[rel].json_record_lines,
        )
        for rel, how in sorted(found.items())
        if not facts[rel].names_the_model
    ]
    return violations, found


def audit(root: Path) -> tuple[list[Violation], dict[str, str], dict[str, FileFacts]]:
    facts = analyze_tree(root)
    violations, found = violations_in(facts)
    return violations, found, facts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO, help="repository root to audit")
    ap.add_argument("--list", action="store_true", help="print every producer, not only reds")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"{TAG}: ERROR(instrument=root_absent) {args.root}")
        return EXIT_ERROR

    try:
        violations, found, facts = audit(args.root)
    except AuditError as exc:
        print(f"{TAG}: ERROR(instrument=unparseable_source) {exc}")
        return EXIT_ERROR

    if args.list:
        for rel, how in sorted(found.items()):
            mark = "names-model" if facts[rel].names_the_model else "NO IDENTITY"
            print(f"  {mark:<12} {rel}  [{how}]")

    if violations:
        print(f"\n{TAG}: FAIL — {len(violations)} Phi-3.5 evidence producer(s) write a record "
              f"that does not name the model it was computed from:")
        for v in violations:
            print(f"  {v.rel}: {v.why}")
            print(f"      reached because {v.reached_via}")
            print(f"      writes JSON at line(s) {list(v.json_lines)}")
        print(
            "\nA record that does not name its model cannot be replayed and cannot be "
            "falsified: a PHI35_MODEL override, or a different file arriving at the same "
            "path, changes what the number measured with nothing in the record to say so. "
            "Stamp `_result_identity()` (rust/tools/model_provenance.py:sha256_of), or "
            "propagate the child record's onnx_file/onnx_sha256 — and record an explicit "
            "identity error rather than writing a blank on the success path."
        )
        return EXIT_FAIL

    print(
        f"{TAG}: PASS — {len(found)} Phi-3.5 evidence producer(s) discovered by AST, "
        f"every one names the model it consumed"
    )
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
