"""Whether a decode host-attribution record may carry **numbers** — one gate, fail closed.

WHY THIS FILE EXISTS
====================
Issue #88 asks for the host cost inside the ORT ``Compute`` callback to stop being
unattributed. ``bench/phases.py::compute_call_attribution`` measures it. This file decides
whether the measurement may be *published*, and it exists because the first attempt at #88
(PR #94, rejected) published an artifact whose numbers survived every one of the following:

* **the weights were never hashed.** The record carried the 26 MB graph's sha256, the graph's
  byte count and the *size* of the 2.29 GB external tensor file. A digest of the topology plus
  a byte count of the numbers admits a same-sized substitution of every weight in the model —
  the record identifies which *shape* of model ran, not which model.
* **the frame was missing.** No device, no driver version, no Vulkan API version, no ORT
  version, no source commit, no build profile, no evidence the GPU was held exclusively. The
  ``taken_at`` field was a hard-coded string, so the artifact could not even say *when* it was
  taken without being believed.
* **equivalence was checked at one point and quoted at four.** ``past=128`` was compared, on
  output 0 (``logits``) only. ``past=512``, ``1024`` and ``2048`` were quoted, and the ``present``
  KV outputs — a decode step's entire reason for existing — were never compared at any length.
  A decode that emits a correct first token from a wrong cache produces a correct *token* and a
  wrong *sequence*.
* **refusal degraded instead of removing.** A record with counters missing, with ABI v8, with
  ``record_path_wired`` false, with an unknown phase or with a required phase absent was still
  graded quotable, and a raw record that *was* refused kept its share table — so the numbers
  read as complete while the verdict beside them said they were not.
* **the denominators were implicit.** ``cmd_upload`` was reported as "16.4% of record" when
  16.396 is its share of the **whole** ``Compute`` callback and its share **of record** is
  95.16%. Two denominators, one label, a 5.8× difference.
* **a new probe read the counters itself** and accepted ABI v8, bypassing
  ``rust/tools/counters_abi.py`` — the reader that exists precisely because three hand-written
  mirrors read the wrong fields twice in one day.
* **the path screen was a denylist of known roots.** ``C:\\Users\\<the author>\\...`` was
  redacted; ``D:\\other-user\\...`` was published verbatim.

Every one of those is a defect of the *artifact*, not of the instrument, and every one of them
is invisible to a reader who is handed the finished table. So the gate is a function, it is
shared, and it is the only supported way to turn an attribution measurement into a public
number.

FAIL CLOSED MEANS THE NUMBERS ARE NOT THERE
===========================================
:func:`publish` does not annotate. On any blocker it returns a dict that **has no attribution
in it** — not a withheld one, not a null-valued one, not one behind a ``quotable: false`` flag.
The refused result is built from scratch and the record is never copied into it, which is why
``test_attribution_gate.py`` can assert the absence structurally rather than by inspection.

The distinction that makes this stricter than it sounds: a *refusal diagnostic* may name a
duration ("2.700 ms of siblings in a 1.000 ms total") because that is evidence of the defect.
A *share* may not appear at all, because a share is the thing that gets copied out of the
artifact and pasted into a status report with its verdict left behind.

WHAT THIS FILE DOES NOT DO
==========================
It does not measure anything and it does not produce a record. It reads one and grades it. It
cannot tell whether the ``gpu_lock`` witness a record carries is true — only that the record
carries one, which is the difference between an unchecked claim and no claim at all, and is
stated here rather than implied.
"""

from __future__ import annotations

import datetime as _dt
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

if str(_HERE) not in sys.path:  # pragma: no cover - import bookkeeping
    sys.path.insert(0, str(_HERE))

import phases  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------------------------

#: Returned **by identity** from :func:`public_path_screen` when nothing in the record looks
#: like an absolute filesystem path.
#:
#: A constant rather than a freshly built empty set, for the reason ``_polarity.selects``
#: exists: two empty sets compare equal, so a screen that silently stopped looking and a screen
#: that looked and found nothing are indistinguishable under ``==``.
NO_PRIVATE_PATHS = frozenset()

#: Witnesses a public attribution record must carry, as ``(section, key)`` paths into it.
#:
#: These are the fields that answer "what produced this number". Absent any one of them the
#: record is a measurement of an unidentified thing on an unidentified machine, which is not a
#: weaker claim than a framed one — it is a different kind of object.
REQUIRED_ENVIRONMENT_WITNESSES = (
    ("device", "name"),
    ("device", "driver_version"),
    ("device", "api_version"),
    ("host", "onnxruntime"),
    ("build", "sha256"),
    ("build", "profile"),
    ("source_commit", "commit"),
    ("gpu_lock", "mechanism"),
)

#: Counters a public attribution record must carry. Not a coverage claim about the counter set —
#: these are the four that say the EP *ran* and that the submissions it counted completed.
REQUIRED_COUNTERS = (
    "compute_calls",
    "dispatches_executed",
    "queue_submits_completed",
)

#: Sibling phases every admissible ``Compute`` call must contain before its decomposition may be
#: published.
#:
#: Deliberately three, and deliberately not all of :data:`phases.HOST_PHASES`: ``compile`` and
#: ``prepack`` are cold-path phases that a warm decode step legitimately does not enter, and a
#: gate that demanded them would be red on every correct record. These three are what a
#: dispatch that reached the GPU must have done. A call missing one of them did something else,
#: and its residual is not the quantity #88 asked for.
REQUIRED_PHASES = ("record", "submit", "fence_wait")

#: The verdict token an equivalence entry must carry. Vocabulary is ``tests/ops/_verdict.py``'s;
#: this names the one value that admits a number, and every other token — including
#: ``UNATTRIBUTED`` and ``UNMEASURED`` — refuses.
EQUIVALENCE_MATCH = "MATCH"

#: Absolute-path shapes, generalised.
#:
#: **Not a list of known roots.** PR #94's screen knew this desk's home directory and redacted
#: it; an artifact produced under ``D:\\other-user\\models\\...`` published the whole path. The
#: rule here is about the *shape* of an absolute path rather than its prefix, so it does not
#: need to be updated when someone runs the harness from a drive nobody anticipated:
#:
#: * ``C:\\`` / ``D:/`` — a drive-letter root, either separator
#: * ``\\\\server\\share`` — a UNC root
#: * ``/a/b`` — POSIX-absolute with at least one more segment
#:
#: The token must start the string or follow whitespace or a quote/bracket/separator, which is
#: what keeps ``https://host/path`` (the ``//`` follows ``:``) and ``1/2`` (follows a digit) out.
_ABS_PATH = re.compile(
    r"""(?:^|(?<=[\s"'(\[=,;:]))"""
    r"""(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/]|/(?=[^\s/]+/))"""
)


# ---------------------------------------------------------------------------------------------
# Helpers — private on purpose. A `_` name is not an instrument, and these render no verdict.
# ---------------------------------------------------------------------------------------------


def _walk_strings(value, path="$"):
    """Yield ``(json_path, string)`` for every string *value* in a JSON-ish object.

    Dict **keys** are structure, not content, and are not yielded: a record keyed by a phase
    name is not publishing a path, and screening keys would make the screen fire on its own
    vocabulary.
    """
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _walk_strings(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _walk_strings(v, f"{path}[{i}]")


def _get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_ts(value):
    """ISO-8601 → aware/naive ``datetime``, or ``None``.

    Accepts the ``%z`` spelling ``+0000`` that ``time.strftime`` produces as well as the
    ``+00:00`` spelling ``fromisoformat`` wants on older interpreters.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    m = re.search(r"([+-]\d{2})(\d{2})$", text)
    if m:
        text = f"{text[: m.start()]}{m.group(1)}:{m.group(2)}"
    try:
        return _dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def _expected_abi():
    """The ABI version **the shared reader** derives from ``rust/src/counters.rs``.

    Imported here, scoped, rather than at module import: ``bench/`` must not leave a
    ``rust/tools`` entry on ``sys.path`` (``test_import_isolation.py``), and a gate that cannot
    be imported without the Rust checkout present is a gate nobody runs in CI.

    Returns ``(version, note)``. ``version is None`` is a refusal, never a default — the one
    thing this must not do is fall back to a hard-coded number, because "the probe read the
    counters itself and accepted v8" is the defect it exists for.
    """
    tools = _REPO / "rust" / "tools"
    before = list(sys.path)
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    try:
        import counters_abi

        return int(counters_abi.abi_version()), f"{counters_abi.COUNTERS_RS.name} via counters_abi"
    except Exception as exc:  # noqa: BLE001 - any failure to read the ABI is a refusal
        return None, f"the shared ABI reader could not be used: {exc!r}"
    finally:
        sys.path[:] = before


# ---------------------------------------------------------------------------------------------
# The gates. Every one is a TOTAL instrument: `(value, why)` on accept, `(None, why)` on refuse.
# ---------------------------------------------------------------------------------------------


def public_path_screen(record) -> "tuple[frozenset | None, str]":
    """Refuse a record that publishes an **absolute filesystem path**, from any root.

    Returns :data:`NO_PRIVATE_PATHS` by identity on accept.

    A public artifact identifies a model by **digest** and a file by **repo-relative path**.
    An absolute path discloses the machine's directory layout and, in the shapes this project
    actually produces, its operator's account name — and it does so in a record whose whole
    purpose is to be pasted somewhere else.

    The screen is on the *shape* of an absolute path, not on a set of known roots. That is the
    entire correction: PR #94's screen knew one home directory, so it redacted
    ``C:\\Users\\<author>\\...`` and published ``D:\\other-user\\models\\phi-3.5\\model.onnx``
    untouched, which is the same disclosure from a root nobody had thought of.

    What it deliberately does not fire on: repo-relative paths (``bench/results/x.json``),
    URLs (``https://host/path`` — the ``//`` follows a colon), version fragments and ratios
    (``1/2`` — the ``/`` follows a digit), and dotted names (``phases.py::publish``).
    """
    hits = []
    for where, text in _walk_strings(record):
        m = _ABS_PATH.search(text)
        if m:
            hits.append((where, text[m.start(): m.start() + 60]))
    if hits:
        shown = "; ".join(f"{w} → {t!r}" for w, t in hits[:4])
        more = f" (+{len(hits) - 4} more)" if len(hits) > 4 else ""
        return None, (
            f"refused: {len(hits)} field(s) publish an absolute filesystem path: {shown}{more}. "
            f"A public record identifies a model by digest and a file by repo-relative path. "
            f"This screen is on the SHAPE of an absolute path and not on a list of known roots, "
            f"because a screen that knows one home directory publishes every other one."
        )
    return NO_PRIVATE_PATHS, (
        f"no absolute filesystem path in any of the "
        f"{sum(1 for _ in _walk_strings(record))} string field(s) screened"
    )


def weight_digest_binding(model) -> "tuple[dict | None, str]":
    """Bind the record to the **bytes the model multiplies**, not just to its topology.

    The Foundry Phi-3.5 graph is 26 MB; its weights are 2.29 GB in a sibling external-data file.
    A provenance block carrying the ``.onnx`` sha256, the graph's byte count and the weight
    file's *size* pins the topology and leaves every number in the model free: swap the blob for
    a same-sized one and the record is unchanged. That is not a hypothetical for a quantised
    model shipped by a downloader that re-materialises weights independently of the graph.

    Accepts either shape, and says which:

    * **external weights** — every file the graph's own ``external_data`` locations name is
      present and hashed. The file list comes from the graph, not from a ``.data`` suffix guess,
      so a model that names its blob something else is still covered.
    * **self-contained** — the graph declares no external initializers, so the ``.onnx`` digest
      already covers the weights, and the record says so rather than being silently in the first
      category with an empty list.
    """
    if not isinstance(model, dict):
        return None, f"refused: no model provenance block (got {type(model).__name__})"
    graph = model.get("sha256")
    if not graph:
        return None, "refused: the model provenance carries no sha256 for the graph file itself"

    ext = model.get("external_data")
    if not isinstance(ext, dict) or not ext.get("scanned"):
        reason = _get(ext, "reason") if isinstance(ext, dict) else None
        return None, (
            f"refused: the external weight tensors were never scanned "
            f"({reason or 'no external_data block'}). The graph digest covers the topology; the "
            f"weights are UNHASHED, and a graph hash plus a weight *size* permits a same-sized "
            f"substitution of every number in the model."
        )

    files = ext.get("files") or []
    if not files:
        return {
            "binding": "self-contained",
            "graph_sha256": graph,
            "weight_files": [],
            "weight_bytes": 0,
        }, (
            "the graph declares no external initializers, so its own sha256 covers the weights: "
            f"{ext.get('reason') or 'stated by the provenance block'}"
        )

    unbound = [
        f"{f.get('location')!r} ("
        + ("missing from disk" if f.get("missing") else
           "no sha256" if not f.get("sha256") else
           "zero bytes")
        + ")"
        for f in files
        if f.get("missing") or not f.get("sha256") or not f.get("bytes")
    ]
    if unbound:
        return None, (
            f"refused: {len(unbound)} of {len(files)} external weight file(s) are not bound to a "
            f"digest: {', '.join(unbound)}. The record would identify the graph and leave "
            f"{sum(int(f.get('bytes') or 0) for f in files) / 1e9:.2f} GB of weights "
            f"unidentified."
        )
    return {
        "binding": "graph+external-bytes",
        "graph_sha256": graph,
        "weight_files": [
            {"location": f["location"], "bytes": int(f["bytes"]), "sha256": f["sha256"]}
            for f in files
        ],
        "weight_bytes": sum(int(f["bytes"]) for f in files),
    }, (
        f"graph sha256 plus a full byte digest of {len(files)} external weight file(s) "
        f"({sum(int(f['bytes']) for f in files) / 1e9:.2f} GB); a same-sized substitution of the "
        f"weights would change the record"
    )


def environment_witnesses(record) -> "tuple[dict | None, str]":
    """Refuse a record that cannot say **what ran it, where, from which source, and when**.

    Every field in :data:`REQUIRED_ENVIRONMENT_WITNESSES` must be present and non-empty, the
    build profile must not be ``unknown``, the source tree must not have been dirty, the GPU
    lock witness must be held, and ``machine_quiescence`` must be ``QUIET``.

    **The timestamp is checked, not read.** ``taken_at`` must lie inside ``[started_at,
    finished_at]`` and the run must have taken non-zero time. A hard-coded ``taken_at`` — which
    is what PR #94 shipped — fails this the moment the run happens on any other day, and that
    is the point: a field nothing can contradict is not evidence. ``bench/environment.py``
    derives all three from the clock.
    """
    if not isinstance(record, dict):
        return None, f"refused: no environment record (got {type(record).__name__})"

    missing = [
        ".".join(path)
        for path in REQUIRED_ENVIRONMENT_WITNESSES
        if not _get(record, *path)
    ]
    if missing:
        return None, (
            f"refused: {len(missing)} required environment witness(es) absent or empty: "
            f"{', '.join(missing)}. A duration with no device, driver, API version, runtime "
            f"version, binary digest, source commit or exclusivity evidence is a number about "
            f"an unidentified thing on an unidentified machine."
        )

    profile = str(_get(record, "build", "profile"))
    if profile.lower() != "release":
        return None, (
            f"refused: build.profile is {profile!r}, not 'release'. The build flags are part of "
            f"the frame, and a host-cost decomposition from a debug build is not a noisier "
            f"version of the release one — unoptimised host code moves `record` and leaves "
            f"`fence_wait` alone, so it changes the *shape* of the split and not just its scale. "
            f"'unknown' is refused for the same reason under a different name: nothing checked."
        )
    if _get(record, "source_commit", "dirty") is not False:
        return None, (
            f"refused: source_commit.dirty is "
            f"{_get(record, 'source_commit', 'dirty')!r}. The commit named in the record does "
            f"not describe the code that ran unless the tree was clean, and 'unknown' is not a "
            f"pass."
        )
    if _get(record, "gpu_lock", "held") is not True:
        return None, (
            f"refused: gpu_lock.held is {_get(record, 'gpu_lock', 'held')!r}. Host attribution "
            f"is a wall-clock decomposition; another client on the queue moves fence_wait and "
            f"every share computed against it."
        )
    quiescence = _get(record, "machine_quiescence", "verdict") or _get(record, "machine_quiescence")
    if str(quiescence).upper() != "QUIET":
        return None, (
            f"refused: machine_quiescence is {quiescence!r}. The same device, build and test "
            f"measured 9.5x apart on host contention alone (docs/PERF.md §10)."
        )

    started = _parse_ts(record.get("started_at"))
    taken = _parse_ts(record.get("taken_at"))
    finished = _parse_ts(record.get("finished_at"))
    if started is None or taken is None or finished is None:
        return None, (
            f"refused: the record's timestamps are absent or unparseable "
            f"(started_at={record.get('started_at')!r}, taken_at={record.get('taken_at')!r}, "
            f"finished_at={record.get('finished_at')!r}). All three are needed to show "
            f"`taken_at` was derived rather than typed."
        )
    if finished <= started:
        return None, (
            f"refused: finished_at ({record.get('finished_at')!r}) does not follow started_at "
            f"({record.get('started_at')!r}); the run window is empty, so nothing can be checked "
            f"against it."
        )
    if not (started <= taken <= finished):
        return None, (
            f"refused: taken_at ({record.get('taken_at')!r}) is outside the run window "
            f"[{record.get('started_at')!r}, {record.get('finished_at')!r}]. This is the shape a "
            f"HARD-CODED timestamp takes: a field that no clock produced and that nothing in the "
            f"record can contradict."
        )
    return {
        "device": dict(record["device"]),
        "onnxruntime": _get(record, "host", "onnxruntime"),
        "build": {"sha256": _get(record, "build", "sha256"), "profile": profile},
        "source_commit": _get(record, "source_commit", "commit"),
        "gpu_lock": dict(record["gpu_lock"]),
        "window": {
            "started_at": record["started_at"],
            "taken_at": record["taken_at"],
            "finished_at": record["finished_at"],
            "seconds": (finished - started).total_seconds(),
        },
    }, (
        f"framed: {_get(record, 'device', 'name')} driver "
        f"{_get(record, 'device', 'driver_version')} Vulkan "
        f"{_get(record, 'device', 'api_version')}, ORT "
        f"{_get(record, 'host', 'onnxruntime')}, {profile} build "
        f"{str(_get(record, 'build', 'sha256'))[:12]}, source "
        f"{str(_get(record, 'source_commit', 'commit'))[:12]} (clean), GPU held, machine QUIET, "
        f"taken_at inside a {(finished - started).total_seconds():.1f} s window"
    )


def equivalence_coverage(record) -> "tuple[dict | None, str]":
    """Every point that gets quoted must have been checked, on **every output**.

    PR #94 compared ``past=128`` on output 0 and quoted ``past=512``, ``1024`` and ``2048``. The
    ``present`` KV outputs were not compared at any length. That is the failure mode that looks
    most like success: a decode step whose logits agree and whose cache is wrong emits a correct
    first token and a wrong sequence, so a single-step, single-output comparison returns MATCH
    on a model that is broken from token two onward.

    Refuses when a quoted point has no entry, when an entry's verdict is not ``MATCH``, or when
    an entry compared fewer outputs than the model produced. ``outputs_compared`` and
    ``outputs_total`` are separate fields for exactly this reason: "we compared the outputs" and
    "we compared *an* output" are indistinguishable when only the verdict is stored.
    """
    if not isinstance(record, dict):
        return None, f"refused: no record to check equivalence coverage on ({type(record).__name__})"
    quoted = record.get("quotable_points")
    if not quoted:
        return None, (
            "refused: the record does not declare which points it quotes "
            "(`quotable_points`), so 'every quoted point was checked' is unfalsifiable."
        )
    entries = record.get("equivalence") or []
    by_point = {}
    for e in entries:
        if isinstance(e, dict):
            by_point[(str(e.get("case")), int(e.get("repeat", 0)))] = e

    uncovered, wrong, partial = [], [], []
    for p in quoted:
        key = (str(_get(p, "case", default=p if isinstance(p, str) else None)),
               int(p.get("repeat", 0)) if isinstance(p, dict) else 0)
        e = by_point.get(key)
        if e is None:
            uncovered.append(key)
            continue
        if str(e.get("verdict", "")).upper() != EQUIVALENCE_MATCH:
            wrong.append((key, e.get("verdict")))
            continue
        total = e.get("outputs_total")
        compared = e.get("outputs_compared")
        if not isinstance(total, int) or not isinstance(compared, int) or total < 1:
            partial.append((key, f"outputs_compared={compared!r} of outputs_total={total!r}"))
        elif compared < total:
            partial.append((key, f"{compared} of {total} outputs compared"))

    if uncovered:
        return None, (
            f"refused: {len(uncovered)} quoted point(s) have NO equivalence entry: "
            f"{', '.join(f'{c}/repeat {r}' for c, r in uncovered[:6])}. A record may not quote a "
            f"context length it never checked; the correctness of past=128 is not evidence about "
            f"past=2048, where the KV cache is sixteen times larger."
        )
    if wrong:
        return None, (
            f"refused: {len(wrong)} quoted point(s) are not {EQUIVALENCE_MATCH}: "
            f"{', '.join(f'{c}/repeat {r} = {v}' for (c, r), v in wrong[:6])}. No performance "
            f"number may be published beside a non-MATCH verdict."
        )
    if partial:
        return None, (
            f"refused: {len(partial)} quoted point(s) compared only SOME outputs: "
            f"{', '.join(f'{c}/repeat {r}: {d}' for (c, r), d in partial[:6])}. Comparing logits "
            f"and skipping the `present` KV outputs returns MATCH on a decode step that emits a "
            f"correct first token from a wrong cache."
        )
    return {
        "points": len(quoted),
        "outputs_compared": sum(int(by_point[k]["outputs_compared"]) for k in by_point),
        "verdict": EQUIVALENCE_MATCH,
    }, (
        f"all {len(quoted)} quoted point(s) carry a {EQUIVALENCE_MATCH} over every output the "
        f"model produced"
    )


def counters_witness(record) -> "tuple[dict | None, str]":
    """The counters must be present, read at the **current** ABI, and show the paths were wired.

    Three separate refusals that PR #94 graded ADMISSIBLE:

    * **counters absent.** A record with no counters cannot show the EP executed anything.
    * **ABI drift.** The expected version comes from ``rust/tools/counters_abi.py``, which
      parses ``rust/src/counters.rs`` — the one reader, because three hand-written ctypes
      mirrors read the wrong fields twice in one day (``tests/ops/test_counters_abi_singleton.py``
      exists for that). A probe that carries its own idea of the version accepts a stale
      struct and misattributes every field after the first inserted one. There is no fallback
      constant here: if the shared reader cannot be used, this refuses.
    * **record path not wired.** ``record_paths`` is the ``first_record``/``replay``/``rerecord``
      breakdown on ``vulkan.session_summary``. All-zero means ``Tracer::record_path`` was never
      called on this run, so the record cannot say whether ``vulkan.record`` was a first record
      or a re-record — and those cost different amounts.
    """
    counters = record.get("counters") if isinstance(record, dict) else None
    if not isinstance(counters, dict) or not counters:
        return None, (
            "refused: the record carries no counters block, so it cannot show this EP executed "
            "anything at all. An attribution table over a run that may have been CPU fallback is "
            "a decomposition of the wrong process."
        )

    expected, note = _expected_abi()
    if expected is None:
        return None, f"refused: {note}. The ABI version may not be assumed or defaulted."
    got = counters.get("abi_version")
    if not isinstance(got, int):
        return None, (
            f"refused: counters.abi_version is {got!r}. Without it the field offsets are "
            f"unverifiable and every counter read is a guess about layout."
        )
    if got != expected:
        return None, (
            f"refused: counters were produced at ABI v{got}; this checkout's "
            f"`{note}` declares v{expected}. Fields appended since v{got} read as absent and "
            f"fields moved since v{got} read as each other — silently, and plausibly."
        )

    absent = [c for c in REQUIRED_COUNTERS if not isinstance(counters.get(c), int)]
    if absent:
        return None, (
            f"refused: required counter(s) absent from the record: {', '.join(absent)}. "
            f"`queue_submits_completed` in particular is the only one that distinguishes a "
            f"submission the GPU executed from a submission that failed to create."
        )
    if not counters.get("compute_calls"):
        return None, (
            f"refused: compute_calls is {counters.get('compute_calls')!r}. The ORT Compute "
            f"callback was never entered, so there is no total to decompose."
        )

    paths = record.get("record_paths")
    if not isinstance(paths, dict) or sum(int(paths.get(k, 0) or 0)
                                          for k in ("first_record", "replay", "rerecord")) == 0:
        return None, (
            f"refused: the record-path breakdown is not wired (record_paths={paths!r}). "
            f"`vulkan.record` is a first record, a replay or a re-record and those are different "
            f"costs; a run that cannot say which one it paid cannot have its `record` share read "
            f"as a property of the engine."
        )
    return {
        "abi_version": got,
        "abi_source": note,
        "counters": {c: int(counters[c]) for c in REQUIRED_COUNTERS},
        "record_paths": {k: int(paths.get(k, 0) or 0)
                         for k in ("first_record", "replay", "rerecord")},
    }, (
        f"counters at ABI v{got} (matches {note}), {counters['compute_calls']} Compute call(s), "
        f"{counters['queue_submits_completed']} completed submission(s), record path wired"
    )


def attribution_shares(row, children=None) -> "tuple[dict | None, str]":
    """Turn one decomposed call into percentages that **carry their own denominator**.

    The correction this exists for, in the exact numbers the Fact Checker recomputed: PR #94
    reported ``cmd_upload`` as "16.4% of record". 16.396 is its share of the **whole** ``Compute``
    callback. Its share **of record** is 95.16%. One label, two denominators, a 5.8× difference,
    and no way for a reader to tell which was meant.

    So no percentage leaves this function without the number it was divided by, in the same
    dict, spelled out:

    * a **sibling** gets ``percent_of_compute_call`` with ``denominator_ms``
    * a **child** gets *both* ``percent_of_parent`` and ``percent_of_compute_call``, each with
      its own denominator and the parent named

    ``percent_sum`` is disclosed rather than reconciled. Per-call percentages summing to 97% or
    103% is arithmetically ordinary — a median of ratios is not a ratio of medians — and the one
    thing that must not happen is a residual row invented to absorb the difference.
    """
    if not isinstance(row, dict):
        return None, f"refused: not a decomposed call row ({type(row).__name__})"
    total = row.get("total_ms")
    if not isinstance(total, (int, float)) or total <= 0:
        return None, (
            f"refused: total_ms is {total!r}. Every percentage in this table is divided by it, "
            f"so a non-positive total makes each of them undefined rather than large."
        )
    phase_ms = row.get("phases") or {}
    if not phase_ms:
        return None, (
            "refused: the row names no phases, so there is nothing to take a share of and the "
            "residual would be 100% of the call by construction."
        )

    siblings = {}
    for name, ms in sorted(phase_ms.items()):
        siblings[name] = {
            "ms": ms,
            "percent_of_compute_call": 100.0 * ms / total,
            "denominator": "compute_call",
            "denominator_ms": total,
        }

    child_rows = {}
    for name, spec in sorted((children or {}).items()):
        parent = spec.get("parent")
        parent_ms = phase_ms.get(parent)
        ms = spec.get("ms")
        if not isinstance(ms, (int, float)) or parent is None:
            return None, (
                f"refused: child phase {name!r} declares parent={parent!r} ms={ms!r}; a child "
                f"share needs both, because its two denominators are the parent and the total."
            )
        if not isinstance(parent_ms, (int, float)) or parent_ms <= 0:
            return None, (
                f"refused: child phase {name!r} names parent {parent!r}, which is not a sibling "
                f"phase of this call ({sorted(phase_ms)}). Its share of a parent that is not in "
                f"the table cannot be computed, and reporting only its share of the total is how "
                f"'95% of record' becomes '16% of record'."
            )
        if ms > parent_ms:
            return None, (
                f"refused: child phase {name!r} is {ms} ms inside a {parent_ms} ms {parent!r}. A "
                f"nested span cannot outlast its parent; this is a nesting defect, not a share."
            )
        child_rows[name] = {
            "ms": ms,
            "parent": parent,
            "percent_of_parent": 100.0 * ms / parent_ms,
            "denominator_parent_ms": parent_ms,
            "percent_of_compute_call": 100.0 * ms / total,
            "denominator_compute_call_ms": total,
        }

    residual = row.get("residual_ms")
    residual_percent = (
        (100.0 * residual / total) if isinstance(residual, (int, float)) else None
    )
    percent_sum = sum(s["percent_of_compute_call"] for s in siblings.values()) + (
        residual_percent or 0.0
    )
    # A SINGLE call's parts must sum to that call: the residual is defined as
    # `total - Σ sibling`, so anything else means a share was rescaled, a sibling was
    # double-counted, or the residual was chosen rather than computed. This is the arithmetic
    # self-check; the reading rule about MEDIANS not summing to 100% is a different statement
    # and is disclosed below rather than enforced here.
    if residual_percent is None:
        return None, (
            "refused: the row carries no residual_ms, so the unattributed host cost — the whole "
            "quantity issue #88 asked for — would be absent from a table that otherwise looks "
            "complete."
        )
    if abs(percent_sum - 100.0) > 0.01:
        return None, (
            f"refused: this call's shares sum to {percent_sum:.4f}%, not 100%. Within ONE call "
            f"the residual is defined as total - Σ disjoint siblings, so the parts sum to the "
            f"whole by construction; a deviation means a share was rescaled, a nested span was "
            f"counted as a sibling, or the residual was chosen instead of computed."
        )
    out = {
        "total_ms": total,
        "siblings": siblings,
        "children": child_rows,
        "residual": {
            "ms": residual,
            "percent_of_compute_call": residual_percent,
            "denominator": "compute_call",
            "denominator_ms": total,
            "meaning": ("host cost inside the ORT Compute callback that no phase span names — "
                        "computed as total - Σ disjoint siblings, never assumed zero"),
        },
        "percent_sum": percent_sum,
        "denominator_note": (
            "Every percentage above names the value it was divided by. A child phase carries BOTH "
            "its share of its parent and its share of the whole Compute callback, because those "
            "differ by the parent's own share and quoting one under the other's name is a "
            "several-fold error. Within one call these sum to 100% by construction. A table of "
            "per-call MEDIANS need not: a median of ratios is not a ratio of medians, and a "
            "median column that sums to 97% or 103% must be disclosed as such, never reconciled "
            "by inventing a residual row to absorb the difference."
        ),
    }
    return out, (
        f"{len(siblings)} sibling share(s) and {len(child_rows)} child share(s) against a "
        f"{total} ms compute_call, each carrying its denominator; percentages sum to "
        f"{out['percent_sum']:.2f}%"
    )


def publish(record, events=None, slack=phases.CONTAINMENT_SLACK) -> "tuple[dict | None, str]":
    """The one gate. ``(published, why)`` on accept, ``(None, why)`` on refusal.

    **The refusal is the literal absence of the attribution.** Not a withheld one, not a
    null-valued one, not one behind a ``quotable: false`` flag — ``None``. The refused branch
    computes no share, formats no percentage and never copies the record, so there is nothing
    for a reader to lift out of it. That is the correction to what PR #94 shipped, where a
    refused raw record kept a complete-looking share table beside a verdict saying it should not
    be read; the table is what got quoted.

    ``why`` on the refusal path enumerates **every** blocker, not the first: an artifact that
    fails five ways and is repaired one way per round is five rounds of review for one artifact.
    A blocker sentence may name a duration as *evidence of the defect* ("2.700 ms of siblings in
    a 1.000 ms total"); that is diagnosis. It never contains a share, because a share is the
    thing that travels.
    """
    blockers: "list[str]" = []
    witnesses: "dict[str, object]" = {}

    model = record.get("model") if isinstance(record, dict) else None
    for name, (value, gate_why) in (
        ("public_paths", public_path_screen(record)),
        ("weights", weight_digest_binding(model)),
        ("environment", environment_witnesses(record)),
        ("equivalence", equivalence_coverage(record)),
        ("counters", counters_witness(record)),
    ):
        if value is None:
            blockers.append(f"{name}: {gate_why}")
        else:
            witnesses[name] = gate_why

    rows = None
    if events is None:
        blockers.append(
            "attribution: no trace events were supplied, so `compute_call_attribution` never "
            "ran. A record quoting a decomposition it did not compute is prose."
        )
    else:
        rows, attribution_why = phases.compute_call_attribution(events, slack=slack)
        if rows is None:
            blockers.append(f"attribution: {attribution_why}")
        else:
            absent = sorted({
                p for r in rows for p in REQUIRED_PHASES if p not in (r.get("phases") or {})
            })
            if absent:
                blockers.append(
                    f"attribution: required phase(s) absent from at least one admissible call: "
                    f"{', '.join(absent)}. A dispatch that reached the GPU recorded, submitted "
                    f"and waited; a call missing one of those did something else, and its "
                    f"residual is not the quantity issue #88 asked for."
                )
                rows = None
            else:
                witnesses["attribution"] = attribution_why

    if rows is not None:
        children = record.get("children") or {}
        shares = []
        for row in rows:
            spec = children.get(row["index"], children.get(str(row["index"])))
            table, share_why = attribution_shares(row, spec)
            if table is None:
                blockers.append(f"shares: call {row['index']}: {share_why}")
                shares = None
                break
            shares.append(table)
    else:
        shares = None

    if blockers or shares is None:
        return None, (
            f"REFUSED, {len(blockers)} blocker(s) — no attribution share, percentage or "
            f"per-phase figure is published: "
            + " | ".join(blockers)
        )

    return {
        "calls": len(shares),
        "shares": shares,
        "witnesses": witnesses,
    }, (
        f"{len(shares)} admissible Compute call(s) published with weight-digest binding, a full "
        f"environment frame, full-output equivalence at every quoted point, counters at the "
        f"current ABI, and every percentage carrying its denominator"
    )
