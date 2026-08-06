#!/usr/bin/env python3
"""``check_gh_auth`` — every API-using `gh` invocation in these workflows has a token path.

WHY THIS EXISTS
===============
PR #13 run 31052604259: the test-lock auditor succeeded, and the very next step —
`Open-reds negative control` — exited 4 at `gh run list` with `gh: To use GitHub CLI in
a GitHub Actions workflow, set the GH_TOKEN environment variable`. `ci/check_main_is_green
.py` did exactly what it was built to do (ERROR(instrument=github_unreachable), never a
false PASS); the defect was one level up, in a workflow step that called it with no token
in scope. Main's run before #13 never reached that step at all, so the gap could not
surface until #6 made the lane observable and #13 made the step before it stop eating the
failure.

This is the general form, screened statically so the NEXT missing token is a pull-request
diff rather than a run log: **which workflow steps reach the GitHub API through `gh`, and
does every one of them have GH_TOKEN or GITHUB_TOKEN declared somewhere it can see?**

THE TWO WAYS A STEP REACHES `gh`
=================================
DIRECT   its own script contains a `gh <subcommand>` where the subcommand talks to the
         API — `run`, `api`, `pr`, `issue`, `release`, `workflow`, `repo`, `gist`, `org`,
         `project`, `search`, `secret`, `variable`, `cache`, `auth status`. NOT `--version`,
         `help`, `completion`, `config`, `extension` — none of those touch the network and
         flagging them would teach a reader to ignore this screen's output.

INDIRECT its script invokes a `ci/*.py` tool that itself shells out to `gh` (detected the
         same way — the head of a subprocess argv is `"gh"`), or is one of the two scripts
         independently verified to run `ci/open_reds.json`'s REAL, default register end to
         end while that register still has a live entry pointing at a `gh`-shelling
         script. This is exactly the PR #13 chain, two steps removed from the literal
         string `gh`: `negative_control_open_reds.py` runs `check_open_reds.py` against
         the real register, whose `main_is_green` entry runs `check_main_is_green.py`,
         which calls `gh run list`. See `KNOWN_REAL_REGISTER_RUNNERS` below for why this
         is a short, named, hand-verified list rather than a generic "does file A's text
         mention file B's name" graph: `ci/test_lane_checks.py` also spawns
         `check_open_reds.py` as a subprocess, but always against a synthetic register it
         builds in a tmp dir, so it never reaches `gh` — and a generic textual graph
         cannot tell those two shapes apart without producing that false positive.

WHAT "has a token path" MEANS
==============================
`GH_TOKEN` or `GITHUB_TOKEN` is declared as a key in an `env:` block whose scope covers
the step: the step's own `env:`, its job's `env:`, or the workflow's top-level `env:`.
Declared, not valued — this screen reads key NAMES out of `env:` blocks and never reads,
prints, or evaluates a value, so it cannot leak a token even if one were hardcoded (which
would be its own, different, defect). Both supported YAML shapes for that block are read
the same way, semantically, not by line-shape guessing:

    env:                              env: {GH_TOKEN: ${{ github.token }}}
      GH_TOKEN: ${{ github.token }}

The block form spans multiple lines under `env:`; the inline (flow-mapping) form puts
`{key: value, ...}` on the `env:` line itself — or across SEVERAL lines, since YAML flow
mappings may wrap. Issue #21: a step remediated with the inline form was being FALSELY
CONVICTED of having no token path, because the block-only line-shape check saw `env:
{...}` and, since nothing followed on the NEXT line the way a block mapping requires,
read the step as having declared no keys at all — the exact remediation text read as if
it were the defect it fixes.

Issue #25 (adversarial review of #22's fix): a line-oriented "is this line more indented
than X" check cannot tell a REAL `env:` apart from two other shapes that merely look like
one:

* text resembling `env:`/`GH_TOKEN:` sitting inside a `run: |`/`run: >` BLOCK SCALAR — a
  step's own shell script, echoed prose, or a heredoc — which is executed text, not YAML
  structure, no matter how deeply it happens to be indented under the step;
* a real YAML mapping named `env` that is nested under something else entirely —
  `services.<id>.env` (a service CONTAINER's environment) or `with: env:` (an action
  INPUT that happens to be called `env`) — neither of which is the step/job/workflow
  execution environment `gh` would see.

The parser below is therefore YAML-STRUCTURAL rather than line-oriented: it walks a real
stack of open mapping/sequence frames, tracked by indentation the way YAML's own grammar
requires, and recognises `env:` at exactly three ancestor shapes — the document root, one
`jobs.<id>`, one `steps:` sequence item — never merely "however many spaces more indented
than something else". A block scalar's body is walked as literal text and never
re-examined as YAML at all, so it can neither convict nor acquit a step. Two keys
declared in one `env:` mapping (block or flow, quoted or not) is an unsupported/ambiguous
construct this parser will not silently resolve one way or the other; it raises instead
(`ERROR(instrument=unsupported_yaml_construct)`, never a guess that happens to come out
green). See `parse_workflow()` for the mechanism and `_env_scope_for()` for the exact
three shapes.

PR #27 REVIEW FIXES (issue #25, five more blind spots in the structural parser itself)
=======================================================================================
R1  A `steps:` frame only mints a new step under the EXACT ancestor path
    `jobs.<job>.steps` — not any frame whose leaf key happens to be `steps`. A legal
    action input named `steps` (`with: steps: [...]`, e.g. a matrix-generating action)
    used to reset/orphan the real step and silently drop whatever ran after it (a
    `run: gh api ...` sibling included) from body capture — a false PASS by omission,
    not a wrong token judgement. See `_Frame.is_real_steps` / `_is_job_steps_ancestor()`.

R2  A scalar value this parser does not structurally understand (`&anchor`, `!!tag`,
    `*alias`) now raises `YamlStructureError` if the NEXT content line is more
    indented — instead of silently letting that more-indented content (e.g. a nested
    `env:`) be attributed to the grandparent frame, which could read a step-level
    `env:` as satisfied by a mapping actually nested inside an anchored/tagged value.

R3  Two more shapes now raise rather than guess:
    (a) a multi-document file (`---` separator) — this screen reads one YAML document
        per file and will not guess which document a file's steps/env belong to;
    (b) a nested flow collection as an `env: {...}` value (`{FOO: {GH_TOKEN: x}}`, the
        inner key is FOO's own value, not this mapping's sibling) and a quoted flow
        value containing a literal `{`/`}`/`[`/`]` (`{FOO: '{"GH_TOKEN": "1"}'}`) — both
        previously (correctly, but silently) risked being misread rather than refused;
        a `${{ github.token }}` GitHub Actions expression is NOT this shape (its
        internally-balanced braces are recognised and skipped as opaque, exactly the
        remediation text this screen's own FAIL message recommends) and still passes.

N1  A duplicate SIBLING `env:` key at the same scope (two `env:` blocks under one job,
    in either YAML form) is now `ERROR(instrument=unsupported_yaml_construct)`, not
    silently unioned via `set.update()` the way two prior, separate `env:` records for
    the same scope used to be.

N2  A dash-inline `- env:` block now computes the REAL column its content starts at
    (from the actual whitespace after the dash) instead of a synthetic `dash_indent +
    1` offset, which used to undershoot `env:`'s true column and let a later TRUE
    SIBLING field at that real column be wrongly swallowed as one of `env:`'s own
    children.

Each of the above is unsupported/ambiguous input, not a silent guess in either
direction — every one raises `YamlStructureError`, surfaced as `ERROR(instrument=
unsupported_yaml_construct)`, rather than resolving to a PASS or FAIL a human did not
actually write.

ZERO SUBJECTS
=============
If nothing across the screened files reaches the GitHub API through `gh`, that is either
an honestly `gh`-free set of workflows, or this screen has been pointed at the wrong
scope — a subdirectory, a stale path, the wrong working directory — and is reading
nothing. Issue #21: `PASS — 0 gh-reaching step(s)` is indistinguishable, on the page, from
"this screen was never actually run", so a zero-subject frame is `ERROR(instrument=
zero_gh_reaching_subjects)` by default, not a PASS. Pass `--allow-empty-frame` if an empty
frame is genuinely the intended one (e.g. deliberately screening a workflow directory
known to contain no `gh` calls yet, as a forward-looking regression barrier) — that mode
is opt-in and named on the command line, never the silent default.

A directory may be given in place of individual files: every `*.yml`/`*.yaml` file under
it (recursively) is screened, so a caller that names a whole `.github/workflows/`
directory keeps covering every workflow in it as files are added, rather than only the
ones somebody remembered to list by name — the concrete way a check "over a subdirectory"
would otherwise pass vacuously by construction.

WHAT IT DELIBERATELY DOES NOT DO
=================================
It does not run `gh`, and it does not need network access or a real token to answer its
question — it is a read of the YAML and the `ci/` tree, nothing else. It does not check
that the token has enough SCOPE for the call (`actions: read` vs `contents: write` etc.);
that is a human judgement about least privilege, made once per step and reviewed in the
diff, not a thing this screen can derive from the text. It does not follow `uses:` steps
(third-party actions) — those authenticate however their own inputs say to, which is a
different surface.

Terminal states (R13):

    0  GH-AUTH: PASS
    1  GH-AUTH: FAIL(condition=missing_token_path)
    2  usage
    4  GH-AUTH: ERROR(instrument=...)   workflow_not_found | empty_workflow_directory |
                                        no_steps_parsed | zero_gh_reaching_subjects |
                                        unsupported_yaml_construct

USAGE
    python ci/check_gh_auth.py .github/workflows/ci.yml .github/workflows/conformance.yml
    python ci/check_gh_auth.py .github/workflows
    python ci/check_gh_auth.py --allow-empty-frame .github/workflows/docs-only.yml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_DIR = REPO_ROOT / "ci"
DEFAULT_REGISTER = CI_DIR / "open_reds.json"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

TOKEN_NAMES = ("GH_TOKEN", "GITHUB_TOKEN")

#: `gh <subcommand>` invocations that talk to the GitHub API. Word-boundary on `gh` so this
#: does not fire on `ghost`, `high`, or a path containing the letters. Deliberately excludes
#: --version/help/completion/config/extension: those are local, and flagging them would
#: teach a reader that this screen's FAILs are noise.
_GH_API_SUBCOMMAND_RE = re.compile(
    r"""(?<![\w./-])gh\s+
        (run|api|pr|issue|release|workflow|repo|gist|org|project|search|
         secret|variable|cache|label|ruleset|attestation|codespace)\b
        (?!\s+(?:--version|help))
    """,
    re.VERBOSE,
)
#: `gh auth status` reaches the API to validate the token; `gh auth login`/`gh auth setup-git`
#: do not read repository data the way this screen cares about, so only `status` is included.
_GH_AUTH_STATUS_RE = re.compile(r"(?<![\w./-])gh\s+auth\s+status\b")

_PY_SCRIPT_RUN_RE = re.compile(r"(?:^|[\s;&|])python3?\s+(?P<path>ci/[A-Za-z0-9_./-]+\.py)")

#: A mapping key at the start of a (de-indented) logical line: `key:`, `key: value`,
#: `"key": value`, or `'key': value`. `rest` is everything after the colon, RAW (not yet
#: comment-stripped or trimmed) — "" if nothing follows. Job ids and step field names
#: this repository uses are `[A-Za-z0-9_.-]+`; env var names are the stricter POSIX
#: shape, checked separately where it matters (TOKEN_NAMES membership).
_MAP_KEY_NOINDENT_RE = re.compile(
    r"""^(?:(?P<q>['"])(?P<qkey>[A-Za-z_][A-Za-z0-9_.-]*)(?P=q)|(?P<key>[A-Za-z_][A-Za-z0-9_.-]*))
        \s*:(?P<rest>.*)$""",
    re.VERBOSE,
)
#: A block-sequence item: `- ` (or bare `-`) starting a logical line. `rest` is whatever
#: follows the dash and its one required space, e.g. `name: build` for `- name: build`.
_SEQ_ITEM_RE = re.compile(r"^-(?:\s(?P<rest>.*))?$")
#: A block-scalar indicator as a key's entire (comment-stripped, trimmed) value: `|`,
#: `>`, with an optional chomping indicator (`-`/`+`) and an optional explicit
#: indentation indicator digit — `run: |`, `run: |-`, `run: >2`, etc. Everything more
#: indented than the key itself is then literal text, never YAML structure (#25: a
#: `run: |` body that happens to contain the substring `env:` must never be read as one).
_BLOCK_SCALAR_RE = re.compile(r"^[|>][+-]?[0-9]?$")
#: Key names inside a YAML flow mapping's `{...}` text (single- or multi-line, already
#: joined). A key starts right after the opening brace or a top-level comma — the two
#: positions flow-style permits — optionally quoted. Values are never inspected (this
#: screen's own non-disclosure rule): they may contain `${{ ... }}` expressions, which
#: this repository's usage never puts a bare `identifier:` inside, so scanning for that
#: shape finds keys without needing a real YAML flow-mapping parser.
_FLOW_MAPPING_KEY_RE = re.compile(r"""(?:\{|,)\s*(['"]?)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\1\s*:""")


class YamlStructureError(ValueError):
    """A YAML construct this constrained parser will not guess about (issue #25): two
    keys declared in one `env:` mapping, or a multi-line flow collection that never
    closes. Raised rather than silently resolved one way or the other — an ambiguous
    input is an ERROR(instrument), never a guess that happens to come out green."""


def _split_comment(text: str) -> tuple[str, str]:
    """Split `text` into (code, comment) by YAML's own rule: `#` starts a comment when
    it is the first character or is immediately preceded by whitespace, and never
    inside a single- or double-quoted scalar. Returns the comment WITH its `#`, "" if
    there is none. Needed so a step's own remediation — `env: {GH_TOKEN: ...}  # ci`
    trailing a real declaration — is not read as having declared nothing (#25)."""
    quote = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or text[i - 1].isspace()):
            return text[:i], text[i:]
    return text, ""


def _bracket_delta(code: str) -> int:
    """Net change in flow-collection nesting depth contributed by comment-stripped
    `code`, counting `{`/`}`/`[`/`]` outside quoted scalars. `${{ ... }}` expressions
    are internally balanced (equal `{` and `}`) so they never perturb the count of the
    flow mapping/sequence that encloses them."""
    depth = 0
    quote = None
    for ch in code:
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return depth


def _flow_mapping_key_list(inline: str, file_str: str, line_no: int) -> list[str]:
    """Key names declared DIRECTLY in a YAML flow mapping's own top level, IN ORDER,
    WITH duplicates — ``{GH_TOKEN: ${{ github.token }}}`` -> ``["GH_TOKEN"]``. `inline`
    may be the single-line text or several physical lines already joined; the regex
    only cares about a key's adjacency to `{`/`,`, not physical line boundaries, so a
    multi-line flow mapping is read exactly the same way as a single-line one once its
    lines are concatenated (#25).

    Before running that regex, this walks `inline` once, tracking quote state and flow-
    collection nesting depth, and REFUSES two shapes the regex alone cannot safely
    resolve (#25 R3) rather than guessing at either of them:

    * a NESTED flow collection as a value — ``{FOO: {GH_TOKEN: x}}`` — once depth goes
      past 1, everything inside belongs to FOO's own value, not to this mapping, and a
      regex blind to nesting would misreport GH_TOKEN as this mapping's OWN sibling key
      rather than a key buried inside FOO's value;
    * a quoted value that itself contains a literal `{`/`}`/`[`/`]` —
      ``{FOO: '{"GH_TOKEN": "1"}'}`` — the quote-tracking below would in fact treat
      that text as opaque and not misextract GH_TOKEN from it, but this parser is
      deliberately conservative about a shape it cannot independently confirm is inert
      text rather than a YAML author's typo, and refuses rather than silently trusting
      its own reading either way.
    """
    depth = 0
    quote: str | None = None
    i = 0
    length = len(inline)
    while i < length:
        ch = inline[i]
        if quote is not None:
            if ch in "{}[]":
                raise YamlStructureError(
                    f"{file_str}:{line_no}: a quoted value inside this `env: {{...}}` "
                    f"mapping contains a literal {ch!r} — this screen cannot safely "
                    "tell opaque text apart from YAML structure here and will not "
                    "guess which one it is."
                )
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if inline.startswith("${{", i):
            # A GitHub Actions expression, not YAML flow-collection nesting -- its
            # `{`/`}` are balanced by construction and every ordinary GH_TOKEN
            # declaration is exactly this shape (`${{ github.token }}`). Skip the
            # whole span opaquely so its internal braces never perturb `depth`;
            # treating them as real nesting would misfire on correct workflows,
            # not just adversarial ones.
            end = inline.find("}}", i + 3)
            i = length if end == -1 else end + 2
            continue
        if ch in "{[":
            depth += 1
            if depth > 1:
                raise YamlStructureError(
                    f"{file_str}:{line_no}: this `env: {{...}}` mapping has a nested "
                    "flow collection as a value — this screen only reads the "
                    "mapping's own direct keys and will not guess which of a nested "
                    "collection's keys, if any, are meant to be seen as its own."
                )
        elif ch in "}]":
            depth -= 1
        i += 1
    return [m.group("key") for m in _FLOW_MAPPING_KEY_RE.finditer(inline)]


def _flow_mapping_keys(inline: str, file_str: str, line_no: int) -> set[str]:
    """Key names declared in a YAML flow mapping, e.g. ``{GH_TOKEN: ${{ github.token }}}``."""
    return set(_flow_mapping_key_list(inline, file_str, line_no))


def _first_duplicate(keys: list[str]) -> str | None:
    """The first key that occurs twice in `keys`, in insertion order — None if every
    key is unique."""
    seen: set[str] = set()
    for k in keys:
        if k in seen:
            return k
        seen.add(k)
    return None


@dataclass
class Step:
    name: str
    job: str
    file: str
    line: int
    body: str = ""
    #: Token names visible to this step: its own env:, its job's env:, the workflow's env:.
    token_names: set[str] = field(default_factory=set)

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line} [{self.job}] {self.name!r}"

    @property
    def code_body(self) -> str:
        """`body` with comment-only lines removed.

        Most steps in this repository carry multi-line `#`-prefixed prose directly under
        `- name:`, before `env:`/`run:` — pure YAML-level documentation, never executed.
        This screen's own comments quote example commands (including `gh run list`
        itself, as prose), so matching against the raw body would let a step's
        *description* of another step's defect trip its *own* detector. A shell `#`
        comment inside a real `run:` block does not execute `gh` either, so excluding
        comment-only lines is correct on both sides of that boundary.
        """
        return "\n".join(
            line for line in self.body.splitlines() if not line.strip().startswith("#")
        )


@dataclass
class _Frame:
    """One open mapping/sequence-item context, on a stack kept in step with the file's
    actual indentation — YAML nesting IS indentation, so a generic stack of these
    (rather than two hand-picked thresholds) is enough to tell a real `env:` apart from
    one nested under `services.<id>:` or `with:` (#25), without a full YAML grammar."""

    indent: int
    #: The mapping key that opened this frame ("jobs", a job id, "steps", "env",
    #: "with", "services", a service id, ...). None for a bare sequence-item frame
    #: (e.g. one `- ...` entry of `steps:`) — the item has no key of its own, only a
    #: position in its parent sequence.
    key: str | None = None
    is_seq_item: bool = False
    #: True only for a frame opened by a block-scalar indicator (`run: |`, `run: >`,
    #: ...). While this frame is on top of the stack, every more-indented line is
    #: literal scalar text — never re-interpreted as YAML structure, so a `run: |` body
    #: that happens to contain the substring `env:` is never read as a real one (#25).
    is_block_scalar: bool = False
    #: Set only when `key == "env"` and this frame was opened by the BLOCK form (the
    #: `env:` line's own value is empty; keys follow, indented, below). Computed once,
    #: at open time, from the ancestor shape: "workflow" | "job" | "step" | None (this
    #: `env:` is nested under something else entirely and is not a real token scope —
    #: `services.<id>.env`, `with: env:`, `container: env:`, ...).
    env_scope: str | None = None
    #: key -> the line it was first declared on, for THIS ONE `env:` mapping only.
    #: Discarded with the frame on dedent, so it can never confuse a different `env:`
    #: elsewhere in the file. A repeat is a duplicate YAML mapping key — ambiguous,
    #: never silently resolved (#25).
    env_seen: dict[str, int] = field(default_factory=dict)
    #: True only for a frame opened by the `steps:` key whose ancestor stack, at open
    #: time, was the EXACT shape `jobs.<job_id>` — this repository's only source of
    #: real job steps. A same-named `steps:` key anywhere else (an action `with:
    #: steps:` input, a matrix key, ...) gets an ordinary frame with this False, so a
    #: sequence item under it is never minted as a Step (#25 R1).
    is_real_steps: bool = False
    #: True once this frame has had a DIRECT `env:` child, in either YAML form. A
    #: second `env:` key at the same scope is a duplicate mapping key — the same YAML
    #: defect as a duplicate key inside one `env:` mapping, one level up — and is
    #: refused rather than having its keys silently unioned into the first (#25 N1).
    has_env_child: bool = False


def _env_scope_for(stack: list[_Frame]) -> str | None:
    """Which frame an `env:` mapping belongs to, by its EXACT ancestor shape — not mere
    indentation. "workflow" at the document root, "job" directly under `jobs.<id>`,
    "step" directly under one `steps:` sequence item. Anything else — nested one level
    deeper under `services.<id>:`, `with:`, `container:`, `strategy:`, or any other
    intervening key — is None: a real YAML mapping, but not an execution-scope env for
    any step this screen tracks (#25's two named blind spots, and the general case)."""
    if not stack:
        return "workflow"
    if len(stack) == 2 and stack[0].key == "jobs" and not stack[0].is_seq_item and not stack[1].is_seq_item:
        return "job"
    if (
        len(stack) == 4
        and stack[0].key == "jobs"
        and not stack[0].is_seq_item
        and not stack[1].is_seq_item
        and stack[2].key == "steps"
        and not stack[2].is_seq_item
        and stack[3].is_seq_item
    ):
        return "step"
    return None


def _is_job_steps_ancestor(stack: list[_Frame]) -> bool:
    """True only when `stack` — the frames already open, NOT including the `steps:`
    frame about to be pushed for this key — is the exact ancestor shape `jobs.<job_id>`.
    That is this repository's only source of REAL job steps; a same-named `steps:` key
    anywhere else (a legal action `with: steps:` list input, a matrix key, ...) is a
    same-named key at the wrong ancestor shape, not a second source of steps (#25 R1).
    """
    return (
        len(stack) == 2
        and stack[0].key == "jobs"
        and not stack[0].is_seq_item
        and not stack[1].is_seq_item
    )


def parse_workflow(path: Path) -> list[Step]:
    """Extract every step with its `run:` body and its VISIBLE token names, structurally.

    No PyYAML: this runs in the lane-checks job, which installs pytest/onnx/numpy and
    nothing else (same reasoning as ci/check_build_precondition.py), and a screen
    skipped on the runner because an import failed is a screen that does not exist.
    PyYAML's default loader is also silent about duplicate mapping keys (last one wins,
    with no error), which is the opposite of what #25 asks for — the constrained parser
    below raises on exactly that shape instead.

    This is a real, if small, YAML skeleton: a stack of open mapping/sequence-item
    frames, tracked by indentation the way YAML's own grammar requires. `env:` is
    recognised at three EXACT ancestor shapes only — workflow root, `jobs.<id>`, one
    `steps:` sequence item — never merely "deeper than the job" or "deeper than the
    step", so `services.<id>.env` and `with: env:` (#25's two named review findings) are
    seen as the unrelated nested maps they are, not as a token declaration. A `run: |`/
    `run: >` block scalar's body is walked as literal text and never re-parsed as YAML
    at all, so text that merely resembles `env:` inside a shell script cannot convict or
    acquit a step either way. A YAML flow mapping (`{...}`) may span multiple physical
    lines; a duplicate key declared twice in one `env:` mapping, in either shape, is an
    unsupported/ambiguous construct — this parser will not guess which one wins, and
    raises `YamlStructureError` instead (caught by `screen()`, reported as
    ERROR(instrument=unsupported_yaml_construct), never a silent green).
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    n = len(lines)
    file_str = str(path).replace("\\", "/")

    steps: list[Step] = []
    workflow_env: set[str] = set()
    job_env: dict[str, set[str]] = {}
    #: True once a top-level (document-root) `env:` key has been seen, in either YAML
    #: form. There is no _Frame for the document root (`stack` is simply empty there),
    #: so this mirrors `_Frame.has_env_child` for the one scope that has no frame of
    #: its own (#25 N1).
    workflow_env_seen = False

    job = "<top-level>"
    current: Step | None = None
    current_indent: int = 0
    buf: list[str] = []

    stack: list[_Frame] = []

    # Set only while a multi-line flow collection (`{`/`[`) is open and has not yet
    # balanced back to depth 0. Lines while this is set are NOT run through the normal
    # frame-stack dispatch at all — they are raw continuation text of one scalar value,
    # regardless of their own indentation (a flow collection's closing bracket may sit
    # at ANY column, including back at the opening key's own indent or less).
    pending_flow: dict | None = None

    def _peek_next_indent(after_idx: int) -> int | None:
        """The indentation of the next physical line, scanning forward from 0-based
        `after_idx`, that is neither blank nor a full-line comment — None if no such
        line exists before EOF. Blank lines and comment-only lines carry no YAML
        structure of their own and must never be mistaken for the indentation of the
        node that structurally follows (#25 R2)."""
        j = after_idx
        while j < n:
            raw = lines[j]
            s = raw.strip()
            if s and not s.startswith("#"):
                return len(raw) - len(raw.lstrip())
            j += 1
        return None

    def flush() -> None:
        nonlocal current, buf
        if current is not None:
            current.body = "\n".join(buf)
            steps.append(current)
        current = None
        buf = []

    def record(scope: str, keys: set[str]) -> None:
        if scope == "workflow":
            workflow_env.update(keys)
        elif scope == "job":
            job_env[job].update(keys)
        elif scope == "step" and current is not None:
            current.token_names.update(keys)

    def dispatch_key_line(indent: int, text_after_indent: str, line_no: int) -> None:
        """Interpret one logical `key: ...` line — a real physical line, or the inline
        remainder right after a sequence dash (`- key: value`), which is structurally
        identical and processed the same way."""
        nonlocal job, pending_flow, current, workflow_env_seen
        m = _MAP_KEY_NOINDENT_RE.match(text_after_indent)
        if not m:
            return  # not a recognisable key -- an opaque scalar/leaf; nothing to track.
        key = m.group("qkey") or m.group("key")
        code, _comment = _split_comment(m.group("rest"))
        code = code.strip()

        def claim_env_slot() -> None:
            """Raise if `env:` has already been declared once, directly, at this exact
            scope (workflow root, this job, or this step) — a second `env:` key,
            block or flow form on either side, is the SAME YAML defect as a duplicate
            key inside one mapping, one level up: forbidden, and silently unioning the
            two mappings' keys together (#25 N1) is exactly the guess this parser will
            not make."""
            nonlocal workflow_env_seen
            if stack:
                parent = stack[-1]
                if parent.has_env_child:
                    raise YamlStructureError(
                        f"{file_str}:{line_no}: a second `env:` key at this scope — "
                        "YAML forbids duplicate mapping keys and this screen will not "
                        "silently union them."
                    )
                parent.has_env_child = True
            else:
                if workflow_env_seen:
                    raise YamlStructureError(
                        f"{file_str}:{line_no}: a second top-level `env:` key — YAML "
                        "forbids duplicate mapping keys and this screen will not "
                        "silently union them."
                    )
                workflow_env_seen = True

        # A job id is a direct child of `jobs:` — the one place this repo's usage lets
        # an arbitrary identifier introduce a whole new scope.
        if len(stack) == 1 and stack[0].key == "jobs" and not stack[0].is_seq_item:
            job = key
            job_env.setdefault(job, set())

        # A step's own `name:`/`uses:` field, for a readable `where` — cosmetic only,
        # never affects scope classification.
        if (
            current is not None
            and key in ("name", "uses")
            and stack
            and stack[-1].is_seq_item
            and stack[-1].indent == current_indent
        ):
            current.name = code.strip("\"'") if key == "name" else f"<uses {code}>"

        # A leaf key directly inside an open BLOCK-form `env:` mapping.
        if stack and stack[-1].key == "env" and not stack[-1].is_block_scalar:
            frame = stack[-1]
            if key in frame.env_seen:
                raise YamlStructureError(
                    f"{file_str}:{line_no}: key {key!r} declared twice in the `env:` "
                    f"mapping opened at line {frame.env_seen[key]} — YAML forbids "
                    "duplicate keys in one mapping and this screen will not guess "
                    "which one wins."
                )
            frame.env_seen[key] = line_no
            if frame.env_scope is not None and key in TOKEN_NAMES:
                record(frame.env_scope, {key})

        if code == "":
            # Block form: this key's children follow, indented, on later lines.
            if key == "env":
                claim_env_slot()
            frame = _Frame(indent=indent, key=key)
            if key == "env":
                frame.env_scope = _env_scope_for(stack)
            elif key == "steps":
                frame.is_real_steps = _is_job_steps_ancestor(stack)
            stack.append(frame)
            return

        if _BLOCK_SCALAR_RE.match(code):
            stack.append(_Frame(indent=indent, key=key, is_block_scalar=True))
            return

        if code[0] == "{":
            if key == "env":
                claim_env_slot()
            scope = _env_scope_for(stack) if key == "env" else None
            depth = _bracket_delta(code)
            if depth == 0:
                if scope is not None:
                    keys = _flow_mapping_key_list(code, file_str, line_no)
                    dup = _first_duplicate(keys)
                    if dup:
                        raise YamlStructureError(
                            f"{file_str}:{line_no}: key {dup!r} declared twice in one "
                            "`env: {...}` flow mapping — YAML forbids duplicate keys "
                            "in one mapping and this screen will not guess which one "
                            "wins."
                        )
                    record(scope, set(keys) & set(TOKEN_NAMES))
                return
            if depth < 0:
                raise YamlStructureError(
                    f"{file_str}:{line_no}: `{key}:` closes more `}}`/`]` than it opens "
                    "on its own line — not a construct this screen will guess about."
                )
            pending_flow = {
                "text": code, "depth": depth, "key": key, "scope": scope, "line": line_no,
            }
            return

        if code[0] == "[":
            depth = _bracket_delta(code)
            if depth > 0:
                pending_flow = {
                    "text": code, "depth": depth, "key": None, "scope": None, "line": line_no,
                }
            elif depth < 0:
                raise YamlStructureError(
                    f"{file_str}:{line_no}: `{key}:` closes more `}}`/`]` than it opens "
                    "on its own line — not a construct this screen will guess about."
                )
            return

        # A node property/reference indicator — an anchor (`&name`), a tag (`!!type`
        # or `!type`), or an alias (`*name`) — is not a construct this constrained
        # parser structurally understands. If nothing more indented follows, treating
        # the whole thing as an opaque scalar leaf is safe: there is nothing here to
        # misattribute. If MORE indented content DOES follow, that content is the
        # anchored/tagged node's real payload, and attributing it to whatever frame
        # happens to already be open (its grandparent, since no frame was pushed for
        # this key) would be exactly the guess #25 R2 exists to refuse.
        if code[:1] in ("&", "!", "*"):
            next_indent = _peek_next_indent(idx)
            if next_indent is not None and next_indent > indent:
                raise YamlStructureError(
                    f"{file_str}:{line_no}: `{key}: {code}` is followed by more "
                    "indented content — an anchor, tag, or alias node this screen "
                    "does not structurally understand — and this screen will not "
                    "guess which scope that content belongs to."
                )
            return

        # A plain scalar leaf (`name: build`, `uses: actions/checkout@v4`, a one-line
        # `run:` command, ...). Nothing further to track structurally.

    idx = 0
    while idx < n:
        idx += 1
        line = lines[idx - 1]

        if pending_flow is not None:
            code, _comment = _split_comment(line)
            pending_flow["text"] += "\n" + code
            pending_flow["depth"] += _bracket_delta(code)
            if pending_flow["depth"] > 0:
                continue
            if pending_flow["depth"] < 0:
                raise YamlStructureError(
                    f"{file_str}:{pending_flow['line']}: flow collection closes more "
                    "`}`/`]` than it opens across its multiple lines — not a "
                    "construct this screen will guess about."
                )
            if pending_flow["key"] == "env" and pending_flow["scope"] is not None:
                keys = _flow_mapping_key_list(pending_flow["text"], file_str, pending_flow["line"])
                dup = _first_duplicate(keys)
                if dup:
                    raise YamlStructureError(
                        f"{file_str}:{pending_flow['line']}: key {dup!r} declared "
                        "twice in one multi-line `env: {...}` flow mapping — YAML "
                        "forbids duplicate keys in one mapping and this screen will "
                        "not guess which one wins."
                    )
                record(pending_flow["scope"], set(keys) & set(TOKEN_NAMES))
            pending_flow = None
            continue

        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())

        if re.match(r"^-{3}(\s|$)", stripped):
            raise YamlStructureError(
                f"{file_str}:{idx}: a `---` document-separator line — this screen "
                "reads one YAML document per file and will not guess which document "
                "a multi-document file's steps/env belong to."
            )

        if stripped.startswith("#"):
            if current is not None and indent > current_indent:
                buf.append(line)
            continue

        if stack and stack[-1].is_block_scalar and indent > stack[-1].indent:
            # Literal scalar text: never structure, however much it may resemble an
            # `env:` declaration or a `gh` command (the latter IS still relevant to
            # step.body below, just not to YAML structure).
            if current is not None and indent > current_indent:
                buf.append(line)
            continue

        while stack and indent <= stack[-1].indent:
            stack.pop()

        if current is not None and indent > current_indent:
            buf.append(line)

        m = _SEQ_ITEM_RE.match(stripped)
        if m:
            parent_is_steps = (
                bool(stack)
                and stack[-1].key == "steps"
                and not stack[-1].is_seq_item
                and stack[-1].is_real_steps
            )
            stack.append(_Frame(indent=indent, is_seq_item=True))
            if parent_is_steps:
                flush()
                current = Step(name="<step>", job=job, file=file_str, line=idx)
                current_indent = indent
            rest = m.group("rest")
            if rest:
                # The REAL column where `rest` begins in the ORIGINAL line -- not a
                # fixed `indent + 1` offset. `_SEQ_ITEM_RE` requires exactly one
                # whitespace character between the dash and `rest`'s first non-blank
                # run, but this repo's YAML (like YAML generally) permits more than
                # one; a fixed offset that undershoots this key's TRUE column would
                # make every later TRUE SIBLING field (necessarily indented at least
                # as deep as this key's own real column) look like a CHILD of
                # whatever frame this key opens, silently swallowing it (#25 N2).
                after_dash = line[indent + 1 :]
                rest_real_indent = indent + 1 + (len(after_dash) - len(after_dash.lstrip()))
                dispatch_key_line(rest_real_indent, rest, idx)
            continue

        dispatch_key_line(indent, line[indent:], idx)

    flush()

    if pending_flow is not None:
        raise YamlStructureError(
            f"{file_str}:{pending_flow['line']}: a flow collection opened here never "
            "closes before the file ends — not a construct this screen will guess about."
        )

    for step in steps:
        step.token_names |= workflow_env
        step.token_names |= job_env.get(step.job, set())
    return steps


def gh_direct_scripts(ci_dir: Path) -> dict[str, str]:
    """``ci/*.py`` files whose source shells out to the literal `gh` binary."""
    found: dict[str, str] = {}
    pat = re.compile(r'''\[\s*(?:"gh"|'gh')\s*,''')
    if not ci_dir.is_dir():
        return found
    for f in sorted(ci_dir.glob("*.py")):
        text = f.read_text(encoding="utf-8", errors="replace")
        m = pat.search(text)
        if m:
            line_no = text.count("\n", 0, m.start()) + 1
            found[f.name] = f"{f.name}:{line_no} shells out to `gh` directly"
    return found


#: A register `cmd` that carries one of these flags is not going to call `gh` for real no
#: matter what script it names — `--from-json` (check_main_is_green.py) and `--assert-
#: known-limit` (a fixed prose branch, no subprocess at all) are the two bypasses that
#: exist in this tree today. A cmd with neither is a LIVE invocation.
_REGISTER_CMD_BYPASS_FLAGS = ("--from-json", "--assert-known-limit")

#: Scripts independently verified — by reading them, not by a generic call-graph walk —
#: to run ci/open_reds.json's REAL, default register end to end, as opposed to a synthetic
#: one built in a tmp dir. This is the exact chain PR #13 run 31052604259 hit:
#: `negative_control_open_reds.py`'s LIVE arm calls `check_open_reds.py` against the real
#: file, whose `main_is_green` entry runs `check_main_is_green.py`, which calls `gh run
#: list`. It is a closed, hand-verified list rather than "does this file's text mention
#: that filename" for a concrete reason: `ci/test_lane_checks.py` ALSO spawns
#: `check_open_reds.py` as a subprocess (see its `_open_reds()` helper), but every one of
#: its call sites builds a fresh synthetic register in a tmp dir first and passes
#: `--register <tmpfile>`, so it never reaches the real `main_is_green` entry and never
#: calls `gh` — a textual-mention graph cannot tell those two shapes apart; a human
#: reading both files can.
KNOWN_REAL_REGISTER_RUNNERS: dict[str, str] = {
    "check_open_reds.py": (
        "loads --register (default ci/open_reds.json, the real file) and runs every "
        "entry's cmd via subprocess — see its run_entry()."
    ),
    "negative_control_open_reds.py": (
        'its LIVE arm calls run_screen(REGISTER) where REGISTER = HERE / "open_reds.json"'
        " — the real file, no --register override — specifically to prove the shipped "
        "register is the colour it declares."
    ),
}


def register_live_gh_entries(register_path: Path, direct: dict[str, str]) -> list[str]:
    """Entries in `register_path` whose `cmd` really calls a `direct` (gh-shelling) script
    — i.e. carries none of `_REGISTER_CMD_BYPASS_FLAGS`. Empty if the register does not
    exist, is unreadable, or every such entry has been given a bypass flag."""
    if not register_path.exists():
        return []
    try:
        doc = json.loads(register_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    hits = []
    for entry in list(doc.get("checks", [])) + list(doc.get("known_limits", [])):
        cmd = [str(t) for t in entry.get("cmd", [])]
        names = {Path(t).name for t in cmd}
        if names & set(direct) and not any(flag in cmd for flag in _REGISTER_CMD_BYPASS_FLAGS):
            hits.append(
                f"{register_path.name}: entry {entry.get('id', '?')!r} runs a LIVE "
                f"{', '.join(sorted(names & set(direct)))}"
            )
    return hits


def scripts_needing_gh(ci_dir: Path, register_path: Path) -> dict[str, str]:
    """Every ``ci/*.py`` filename that reaches `gh` for real, with a one-line reason:
    scripts that shell out to `gh` directly, plus the closed set of scripts independently
    verified to run the real open-reds register end to end while it still has a live
    entry pointing at one of those direct scripts."""
    direct = gh_direct_scripts(ci_dir)
    reasons: dict[str, str] = dict(direct)

    live_entries = register_live_gh_entries(register_path, direct)
    if live_entries:
        for name, why in KNOWN_REAL_REGISTER_RUNNERS.items():
            if (ci_dir / name).is_file():
                reasons[name] = f"{why} ({'; '.join(live_entries)})"
    return reasons


def step_gh_reason(step: Step, needing: dict[str, str]) -> str | None:
    """Why (if at all) this step reaches the GitHub API through `gh`."""
    code = step.code_body
    m = _GH_API_SUBCOMMAND_RE.search(code)
    if m:
        return f"runs `gh {m.group(1)}` directly"
    if _GH_AUTH_STATUS_RE.search(code):
        return "runs `gh auth status` directly"
    for m in _PY_SCRIPT_RUN_RE.finditer(code):
        name = Path(m.group("path")).name
        if name in needing:
            return f"runs {m.group('path')}, which {needing[name]}"
    return None


def _fail(condition: str, *lines: str) -> int:
    print(f"GH-AUTH: FAIL(condition={condition})", flush=True)
    for line in lines:
        print(line, flush=True)
    return EXIT_FAIL_CONDITION


def _error(instrument: str, *lines: str) -> int:
    print(f"GH-AUTH: ERROR(instrument={instrument})", flush=True)
    for line in lines:
        print(line, flush=True)
    return EXIT_ERROR_INSTRUMENT


def _workflow_files_under(dir_path: Path) -> list[Path]:
    """Every ``*.yml``/``*.yaml`` file under `dir_path`, recursively, sorted.

    Lets a caller name a whole workflows directory instead of listing files one at a
    time, so a workflow added later is screened from the day it exists rather than the
    day someone remembers to add its name to a command line — the concrete way a screen
    "over a subdirectory" (issue #21) would otherwise keep passing on a shrinking view
    of the tree without anybody changing a line of YAML.
    """
    return sorted({*dir_path.rglob("*.yml"), *dir_path.rglob("*.yaml")})


def screen(
    paths: list[Path], ci_dir: Path, register_path: Path, *, allow_empty_frame: bool = False
) -> int:
    missing = [p for p in paths if not p.exists()]
    if missing:
        return _error(
            "workflow_not_found",
            f"{[str(p) for p in missing]} do(es) not exist. A screen over files that "
            "are not there would pass, and a pass from a screen that read nothing is "
            "not an observation.",
        )

    expanded: list[Path] = []
    for p in paths:
        if p.is_dir():
            files = _workflow_files_under(p)
            if not files:
                return _error(
                    "empty_workflow_directory",
                    f"{p} is a directory with no `*.yml`/`*.yaml` file under it. "
                    "Screening a scope that contains no workflow files at all is the "
                    "wrong-working-directory/wrong-scope failure this screen exists to "
                    "surface, not a clean tree.",
                )
            expanded.extend(files)
        else:
            expanded.append(p)
    paths = sorted(set(expanded))

    all_steps: list[Step] = []
    for p in paths:
        try:
            all_steps.extend(parse_workflow(p))
        except YamlStructureError as exc:
            return _error(
                "unsupported_yaml_construct",
                str(exc),
                "A YAML construct this constrained parser will not guess about — a "
                "duplicate key in one `env:` mapping, or a flow collection that never "
                "closes. Ambiguous input is an ERROR(instrument), never a guess that "
                "happens to come out green (#25).",
            )

    if not all_steps:
        return _error(
            "no_steps_parsed",
            f"No `- name:`/`- uses:` steps were found in {[str(p) for p in paths]}. "
            "Either these are not workflow files or the block structure this screen "
            "relies on has changed. UNOBSERVABLE is not zero.",
        )

    needing = scripts_needing_gh(ci_dir, register_path)

    offenders: list[str] = []
    checked = 0
    for step in all_steps:
        reason = step_gh_reason(step, needing)
        if reason is None:
            continue
        checked += 1
        if not (step.token_names & set(TOKEN_NAMES)):
            offenders.append(
                f"  {step.where}\n"
                f"    {reason}, and neither GH_TOKEN nor GITHUB_TOKEN is declared in an "
                f"`env:` block this step can see (its own, its job's, or the workflow's)."
            )

    if offenders:
        return _fail(
            "missing_token_path",
            "",
            "A step that reaches the GitHub API through `gh` with no token in scope. "
            "Without one, `gh` refuses locally with \"set the GH_TOKEN environment "
            "variable\" — the exact failure in PR #13 run 31052604259 — and whatever "
            "the step wraps (ci/check_main_is_green.py included) correctly reports "
            "ERROR(instrument), which then fails for a reason that has nothing to do "
            "with the rule it was checking.",
            *offenders,
            "",
            "Fix: add `env: {GH_TOKEN: ${{ github.token }}}` to the step (or its job), "
            "scoped to the step/job that actually calls `gh` — not to the whole "
            "workflow, which would hand every job in this file a token none of the "
            "others need.",
        )

    if checked == 0 and not allow_empty_frame:
        return _error(
            "zero_gh_reaching_subjects",
            f"0 `gh`-reaching step(s) across {len(paths)} workflow file(s): "
            f"{[str(p) for p in paths]}.",
            "That is either an honestly `gh`-free set of workflows, or this screen has "
            "been pointed at the wrong scope — a subdirectory, a stale path, the wrong "
            "working directory — and is reading nothing. `PASS — 0 gh-reaching "
            "step(s)` would be indistinguishable, on the page, from a screen that was "
            "never actually run (issue #21); a zero-subject frame is therefore an "
            "instrument error by default, not a pass.",
            "If an empty frame is genuinely the intended one, pass --allow-empty-frame "
            "to say so explicitly on the command line — never the silent default.",
        )

    print(
        f"GH-AUTH: PASS — {checked} `gh`-reaching step(s) across {len(paths)} workflow "
        f"file(s), every one with GH_TOKEN or GITHUB_TOKEN declared in scope. "
        f"{len(needing)} ci/*.py script(s) classified as reaching `gh` (directly or "
        "through ci/open_reds.json)."
        + (
            " (--allow-empty-frame: 0 gh-reaching subjects explicitly accepted.)"
            if checked == 0
            else ""
        ),
        flush=True,
    )
    return EXIT_PASS


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "workflows", nargs="*", help="workflow YAML files (or directories) to screen"
    )
    ap.add_argument("--ci-dir", default=str(CI_DIR))
    ap.add_argument("--register", default=str(DEFAULT_REGISTER))
    ap.add_argument(
        "--allow-empty-frame",
        action="store_true",
        help=(
            "accept 0 gh-reaching step(s) as PASS instead of ERROR(instrument="
            "zero_gh_reaching_subjects). Documented opt-in only — the default refuses "
            "an empty frame because it cannot be told apart from this screen having "
            "been pointed at the wrong scope (issue #21)."
        ),
    )
    if not argv:
        print(__doc__, flush=True)
        return EXIT_USAGE
    args = ap.parse_args(argv)
    if not args.workflows:
        print(__doc__, flush=True)
        return EXIT_USAGE
    return screen(
        [Path(p) for p in args.workflows],
        Path(args.ci_dir),
        Path(args.register),
        allow_empty_frame=args.allow_empty_frame,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
