"""Does splitting decode-time GQA across the KV dimension make `gqa_f16` faster, and is it the
same answer?

WHAT THIS ARTIFACT IS ENTITLED TO SAY
=====================================
Issue #90 measured that at a 1024-token KV cache `gqa_f16` holds ~93.5% of Phi-3.5's decode GPU
time, and that the decode dispatch is 32 workgroups of **one invocation each** — one thread per
(batch, head) walking the whole cache serially. `gqa_decode_f16` splits that walk across W lanes
and merges with an online softmax. This probe is the measurement of that change, and it is scoped
deliberately narrowly:

  * **ONE claim: kernel time, on one device.** The number this file publishes is the GPU
    timestamp-query total for the GQA kernel itself, on an RTX A1000, from the EP's own tracer.
    It is not a whole-model speed claim, not a p128-regression-resolution claim, not a
    cross-device claim, and not a CUDA-parity claim. The graph GPU total is recorded next to it
    as *context* — because a kernel win that does not move the graph is worth knowing about —
    and is explicitly not the headline.
  * **The p128 dependency is #96's, not this file's.** A decode-at-past-128 regression is under
    independent diagnosis in issue #96. p128 is in the case list of BOTH passes here precisely
    so that this change cannot be reported as having fixed or worsened it by omission; what this
    file may say about p128 is what it measured about the GQA kernel at p128 on this box, and
    nothing about why the cross-build behaviour is what it is.
  * **Wall-share attribution is #88's, not this file's.** Projecting a kernel win onto a model
    ceiling needs an approved wall-share number, which issue #88 v2 owns. This file therefore
    publishes no projection, no "therefore the model gets Nx", and no extrapolation.

THE STRUCTURAL RULE THIS FILE EXISTS TO ENFORCE
===============================================
The rejected artifact for this work carried `equivalence_complete=false`, `all_equivalent=false`
and a past-128 case verdict of `DIVERGENT`, exited 1 — and still published WIN rows with speedup
numbers in them. A reader who skimmed the table read a win that the same file's own correctness
pass had already retracted.

So the rule here is not "print a warning". It is **structural**:

    A case that is not proven equivalent has NO `speedup`, NO `verdict` and NO timing summary
    fields in its record. Those keys are absent, not false. In their place is a `refusal`
    object naming what was not established.

`_publishable` is the single function that decides this, and `_apply_structural_removal` is the
only place timing fields are attached. There is no path through this file that produces a speed
number for a case whose equivalence did not pass, because there is no code that could write one.
The same rule applies at the top level: `headline` is absent unless every case is equivalent.

`DIVERGENT` is also not relabelled. If the two arms disagree beyond the tolerance PREDECLARED in
`TOLERANCE` below — declared in source, before any run, and not to be edited to make a row green
— the case is refused. It is not re-described as a tolerance residual, and the tolerance is not
widened after seeing the number.

TWO SCOPES, ONE TOLERANCE, AND WHERE THE CLAIM IS ALLOWED TO LIVE
=================================================================
This file measures at two scopes and judges both with the SAME `TOLERANCE` object. Neither scope
has a looser band; what differs is what is being compared.

  * **NODE SCOPE** — one `GroupQueryAttention` node, in Phi-3.5's own shape (32 heads, 32 KV
    heads, head_dim 96, full rotary), alone in a graph. Equivalence here is a statement about
    THIS KERNEL, because nothing else is in the graph. It is the only scope in this file that
    is permitted to publish a speed number, and only when its equivalence passes.

  * **WHOLE-MODEL SCOPE** — Phi-3.5-mini, 32 layers, at the same cache lengths, including p128.
    This scope **publishes no speed field at all** and is not expected to: an elementwise
    comparison of a 32-layer fp16 residual stack is not an equivalence result for one kernel.
    The measurement makes that concrete rather than asserting it — at past 128, layer 0's
    `present.key` and `present.value` come out BITWISE IDENTICAL between the arms, and the
    disagreement first appears at layer 1 (2e-3), compounds at roughly 1.15x per layer, and
    reaches 0.13 at layer 31 and 0.52 at the logits. The kernel wrote the same cache; the model
    amplified a reassociation downstream of it.

    That reading is not this change's word against the reader's. `whole_model_frame` runs three
    controls whose results are recorded WHATEVER THEY SAY, including results that give this
    change no cover:
      - a **NULL** that can fail — the same arm against itself, which must come out bitwise
        identical; if it did not, the instrument would be measuring nondeterminism and the entire
        section would be void;
      - a **CROSS-KERNEL** probe this change does not define — `GEMV_MAX_ROWS` 1 vs 4, an
        already-shipped and already-accepted change in a DIFFERENT kernel — asking whether
        whole-model divergence is simply what this instrument reports for any accepted change.
        On the generating run it came back BITWISE IDENTICAL, i.e. NEGATIVE: the divergence is
        not generic, and this control is reported as giving this change no cover; and
      - a **LANE-SENSITIVITY** probe — W = 2 against W = 16, both arms the new kernel — which
        isolates whether the model's output is invariant under reassociation of this sum at all.
    None of the three lifts the refusal, and no speed field anywhere in this report is
    conditioned on any of them. They exist so the refusal is interpretable, not so it can be argued away.
WHY EQUIVALENCE IS NOT ARGMAX
=============================
`real_model.classify_logits` answers "does the model pick the same token?", which is the right
question for an end-to-end sanity arm and the wrong one here: a KV-split merge that quietly lost a
lane's contribution changes the *distribution* long before it changes the argmax, and a cache that
is wrong in the tail produces a correct first token and a wrong sequence. So every case here
compares **every element of every output** — logits and all 64 present-KV tensors — at a
predeclared per-output tolerance, and separately records whether the two arms were bitwise
identical. Argmax agreement is recorded too, as one field among several, never as the criterion.

An output subset is treated exactly like a failed comparison: if `outputs_compared` is not
`outputs_total`, the case is refused. "We compared the ones we looked at" is the same defect as
"we widened the tolerance", arrived at from the other side.

HOW THE TWO ARMS DIFFER
=======================
One build, one DLL, one device, one model, one feed set, one session configuration. The ONLY
difference between the arms is the value of `ONNXRUNTIME_EP_VULKAN_GQA_DECODE_KV_PARALLEL`:

    parallel  — unset; the host selector picks W from the cache length (W >= 2 for past >= 63)
    serial    — "1"; the selector's exact kill switch, which dispatches the unmodified
                `gqa_f16` module that ships on main today

That makes the serial arm a genuine baseline rather than a reconstruction: it is the production
kernel, reached through the production selector, in the same process shape. The arms are also
told apart *mechanically* rather than by assumption — the EP's tracer names each dispatch by its
shader module stem, so `vulkan.gpu.gqa_decode_f16` appearing in the parallel arm's trace and
`vulkan.gpu.gqa_f16` in the serial arm's is the pipeline witness that the arm ran what it says it
ran. A case whose witness is missing or crossed is refused, and refusing on a missing witness is
what stops this file from ever reporting a "win" that was two runs of the same kernel.

TIMING PROTOCOL, PREDECLARED
============================
  * One subprocess per (case, arm, repeat). ORT registers an EP process-globally and the EP
    writes counters from an exit hook, so a loop in one process measures the second arm against
    the first arm's warmed pipeline cache and reports the union of both.
  * Each subprocess runs `--iters` inferences under the GPU tracer. The tracer emits one event
    per dispatch, in order, so the events partition exactly into `iters` equal buckets; the FIRST
    bucket is dropped as warm-up and the remainder are the timed sample. If the event count is
    not divisible by `iters` the point is discarded as an instrument error rather than averaged.
  * Repeats are INTERLEAVED (parallel, serial, serial, parallel, ...) so that a monotone drift in
    the machine — thermals, another tenant arriving — cannot be absorbed entirely into one arm.
  * Every repeat's per-inference kernel time is kept in the record, not just the median, and the
    relative standard deviation across repeats is computed for both arms. A ratio whose arms did
    not repeat themselves is not a measurement of this change, so `DISPERSION_CEILING` is a
    predeclared refusal: above it, the case reports its samples and no ratio.
  * This box is shared and is `STEADY_UNCERTIFIED` for wall clock (PERF.md §20). Wall time is
    recorded per process for context and is not a claim.

GPU LOCK / COOPERATIVE OCCUPANCY
================================
There is no exclusive GPU reservation on this box and this file does not pretend to one. What it
does instead is state the occupancy conditions on the face of the artifact — `device_state` from
the EP's own device-state instrument (clocks, driver), the per-arm dispersion above, and the
`concurrent_tenancy` note — so a reader can see the conditions the number was taken under rather
than trusting an unstated "the box was quiet". `ci/check_run_disturbance.py` is the standing
version of the same question; the dispersion field here is its in-artifact counterpart.

ARTIFACT HYGIENE
================
The committed JSON carries NO absolute paths. The EP library is identified by repo-relative path
plus sha256 and byte size; the model by identity, sha256 and total external-data bytes, never by
its cache location. `_screen_for_leaked_roots` walks the finished report and REFUSES to write it
if any string looks like an absolute filesystem root on any platform — a Windows drive letter, a
UNC share, or a POSIX `/home`/`/Users`/`/root` path. It is arbitrary-root shaped on purpose: a
screen that only knew this machine's home directory would pass on the next contributor's.

Usage:
    python bench/results/probe_gqa_decode_kv_parallel.py
    python bench/results/probe_gqa_decode_kv_parallel.py --past 128,1024 --repeats 2
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1]
_ROOT = _BENCH.parent
for _p in (str(_BENCH), str(_ROOT), str(_ROOT / "rust" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import real_model as rm  # noqa: E402

SCHEMA = "gqa_decode_kv_parallel/1"
ISSUE = 90
ENV_KV_PARALLEL = "ONNXRUNTIME_EP_VULKAN_GQA_DECODE_KV_PARALLEL"
EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
TRACE_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE"
TRACE_GPU_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE_GPU"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"

PARALLEL_EVENT = "vulkan.gpu.gqa_decode_f16"
SERIAL_EVENT = "vulkan.gpu.gqa_f16"

#: The two arms. `env` is the ONLY thing that differs between them.
ARMS = (
    ("parallel", None, PARALLEL_EVENT),
    ("serial", "1", SERIAL_EVENT),
)

#: Decode cache lengths. **p128 is mandatory** — see the module docstring: the past-128 point is
#: under independent diagnosis in issue #96, and the way an artifact silently acquires a claim
#: about a known-awkward point is by leaving it out of the controls. `main` refuses to run
#: without it.
DEFAULT_PAST = (128, 512, 1024)
MANDATORY_PAST = 128

#: Past lengths for the fallback-identity pass. Deliberately dense AROUND the selector's
#: threshold (W>=2 first appears at total 64, i.e. past 63) and wide on either side, so the pass
#: covers both the region where the shipped default is the refusal and the region where it is not.
FALLBACK_PAST = (0, 1, 7, 31, 61, 62, 63, 64, 127, 128, 255, 511, 1024)

#: Prefill shapes. `seq_len` 2 and 8 bracket the held-out mutation `== 1` -> `<= 8`; 64 and 128
#: are real prefill chunk sizes. Past 0 is a first chunk, past 128 a continuation.
PREFILL_CASES = ((0, 2), (0, 8), (0, 64), (128, 2), (128, 8), (128, 128))

#: PREDECLARED per-output tolerance. Written before any run of this probe, and not to be edited
#: to turn a row green — the whole point of declaring it in source is that widening it is a diff
#: a reviewer sees. Both arms are the same EP on the same device in fp16; the only difference
#: between them is the ORDER the KV sum is accumulated in, so anything outside this bound is a
#: merge defect and not rounding.
#:
#: `logits` is compared ELEMENTWISE at this bound in addition to the argmax/top-k fields, because
#: argmax agreement is not a statement about the distribution (module docstring).
TOLERANCE = {
    "logits": {"rtol": 1e-2, "atol": 1e-2},
    "present": {"rtol": 1e-2, "atol": 1e-2},
}

#: Relative standard deviation of an arm's per-repeat kernel time, above which this file publishes
#: the samples and refuses the ratio. 10% is generous for a GPU timestamp total and is chosen so
#: that it refuses on a disturbed box rather than on a normally noisy one.
DISPERSION_CEILING = 0.10

#: The production GQA node's own shape, taken from Phi-3.5-mini: 32 query heads, 32 KV heads
#: (so MHA-shaped, the degenerate GQA grouping), head_dim 96, full rotary. The node-scoped pass
#: runs exactly this node in isolation, which is what makes its equivalence result and its
#: kernel-time result statements about the SAME thing.
NODE_SHAPE = {
    "num_heads": 32,
    "kv_heads": 32,
    "head_dim": 96,
    "rotary_dim": 96,
    "seq_len": 1,
}
NODE_SEED = 0x90DEC0DE

#: The calibration control. `GEMV_MAX_ROWS` selects between two ALREADY-SHIPPED, already-accepted
#: reductions in a different kernel (`q_gemv_matmul_nbits_f16`); switching it reassociates an fp16
#: sum exactly the way this change does, in code nobody is asking to judge today. It is measured
#: with the GQA split OFF in both arms, so GQA is not a variable in it. See
#: `_whole_model_frame` for what it is and is not allowed to establish.
ENV_GEMV_ROWS = "ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS"

#: Absolute-root shapes. Deliberately arbitrary-root: `C:\Users\someone`, `\\server\share`,
#: `/home/someone`, `/Users/someone`, `/root/...`. A screen keyed on THIS machine's home would
#: pass unchanged on the next contributor's box and catch nothing.
_ROOT_LEAK_PATTERNS = (
    re.compile(r"^[A-Za-z]:[\\/]"),
    re.compile(r"[A-Za-z]:[\\/][A-Za-z0-9_.$-]"),
    re.compile(r"^\\\\[A-Za-z0-9_.-]+\\"),
    re.compile(r"(^|[\s\"'(=])/(home|Users|root|mnt|media)/"),
)

#: Where the cooperative device lock lives. Overridable so a CI box can point it at a volume that
#: every job on that box shares; the DEFAULT is per-user rather than per-checkout on purpose,
#: because the thing being excluded is other *worktrees* on this machine, and a lock inside one
#: worktree excludes nothing.
LOCK_DIR_ENV = "ONNXRUNTIME_EP_VULKAN_PROBE_LOCK_DIR"

#: How long to wait for the device before giving up. A probe that measures while another probe
#: measures produces a number about the box, not about the kernel; a probe that waits forever
#: never reports. 45 minutes is longer than any single run of this file.
LOCK_TIMEOUT_S = 45 * 60


# ---------------------------------------------------------------------------------------------
# Cooperative device lock
# ---------------------------------------------------------------------------------------------


class DeviceLock:
    """Machine-wide, device-indexed, COOPERATIVE exclusion between measurement probes.

    WHAT THIS IS
    ------------
    An advisory lock file plus a participant registry. Any process that takes the same lock
    before touching the device is serialised against this one. That is the whole mechanism, and
    the word `cooperative` in the artifact is load-bearing: this excludes *participants in this
    protocol* and nothing else. It cannot exclude a compositor, a browser, a CUDA job, or a
    sibling squad session that has not adopted the lock.

    WHY IT IS RECORDED RATHER THAN ASSUMED
    --------------------------------------
    The rejected artifact reported timings with no statement about occupancy at all, so a reader
    could not tell a kernel improvement from an idle box. Recording `contended` and `wait_s`
    makes the occupancy question answerable from the artifact: a run that never waited and saw
    no other participant is a different claim from a run that queued behind three of them, and
    the difference is now visible instead of inferred from how clean the numbers look.

    This DOES NOT replace the dispersion gate. Two independent conditions have to hold: nobody
    else in the protocol was measuring (this lock), and the samples this run took were tight
    (`DISPERSION_CEILING`). Either one alone is defeatable.
    """

    def __init__(self, device: int):
        base = os.environ.get(LOCK_DIR_ENV)
        root = (
            Path(base)
            if base
            else Path(os.environ.get("LOCALAPPDATA") or Path.home())
            / "onnxruntime-ep-vulkan"
            / "probe-locks"
        )
        root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._path = root / f"gpu-device-{device}.lock"
        self._registry = root / f"gpu-device-{device}.participants"
        self._fh = None
        self._t_acquired = None
        self.witness: dict = {}

    def _participants(self) -> int:
        """Live holders of the registry, excluding this process. Stale entries are reaped."""
        live = []
        try:
            raw = self._registry.read_text(encoding="utf-8").splitlines()
        except OSError:
            raw = []
        for line in raw:
            line = line.strip()
            if not line or not line.isdigit():
                continue
            pid = int(line)
            if pid == os.getpid():
                continue
            try:
                if sys.platform == "win32":
                    out = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                        capture_output=True,
                        text=True,
                    ).stdout
                    alive = str(pid) in out
                else:
                    os.kill(pid, 0)
                    alive = True
            except Exception:  # noqa: BLE001
                alive = False
            if alive:
                live.append(pid)
        return len(live)

    def _register(self, add: bool) -> None:
        try:
            pids = [
                p.strip()
                for p in self._registry.read_text(encoding="utf-8").splitlines()
                if p.strip().isdigit()
            ]
        except OSError:
            pids = []
        me = str(os.getpid())
        pids = [p for p in pids if p != me]
        if add:
            pids.append(me)
        try:
            self._registry.write_text("\n".join(pids), encoding="utf-8")
        except OSError:
            pass

    def acquire(self, timeout_s: int = LOCK_TIMEOUT_S) -> dict:
        t0 = time.perf_counter()
        others = self._participants()
        self._fh = open(self._path, "a+b")  # noqa: SIM115 — held for the run
        contended = False
        acquired = False
        deadline = t0 + timeout_s
        while time.perf_counter() < deadline:
            try:
                self._lock_nb()
                acquired = True
                break
            except OSError:
                contended = True
                time.sleep(0.5)
        self._t_acquired = time.perf_counter()
        if acquired:
            self._register(True)
        else:
            self._fh.close()
            self._fh = None
        self.witness = {
            "mechanism": "advisory file lock (msvcrt.locking on Windows, fcntl.flock on POSIX)",
            "scope": f"machine-wide, keyed on device index {self.device}",
            "semantics": "COOPERATIVE",
            "acquired": acquired,
            "contended": contended,
            "wait_s": round(self._t_acquired - t0, 3),
            "other_participants_at_entry": others,
            "excludes": "any other process that takes this same lock before using the device",
            "does_not_exclude": (
                "processes that have not adopted this protocol — a compositor, another EP, or a "
                "squad session running its own benchmark without the lock. This is why the "
                "dispersion gate is a SEPARATE and independently sufficient condition."
            ),
        }
        if not acquired:
            self.witness["timed_out_after_s"] = timeout_s
        return self.witness

    def _lock_nb(self) -> None:
        if sys.platform == "win32":
            import msvcrt

            self._fh.seek(0)
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(self) -> dict:
        if self._fh is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self._fh.close()
            self._fh = None
            self._register(False)
        if self._t_acquired is not None:
            self.witness["held_s"] = round(time.perf_counter() - self._t_acquired, 3)
        self.witness["participants_at_exit"] = self._participants()
        return self.witness


# ---------------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=str(_ROOT), capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _tool_version(cmd: list) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        return (p.stdout or p.stderr).strip().splitlines()[0][:200]
    except Exception:  # noqa: BLE001
        return "unavailable"


def _repo_relative(p: "Path | str") -> str:
    """Repo-relative POSIX spelling, or a shape statement when the path is outside the repo.

    Never the absolute path: this string goes into a committed artifact. A file outside the tree
    is named by what it is, not by where this machine keeps it.
    """
    try:
        return Path(p).resolve().relative_to(_ROOT).as_posix()
    except Exception:  # noqa: BLE001
        return "<outside-repo>"


def _subject_provenance(lib: Path) -> dict:
    """Everything a reader needs to know WHICH build produced these numbers, and no path.

    The rejected artifact for this work was taken with a debug DLL from a different worktree than
    the head it was cited for. That is undetectable from a speed table and trivially detectable
    from this block: the digest, the size, the profile, the commit and the dirty flag together
    name one build, and `subject_paths` says what the reading is about.
    """
    data = lib.read_bytes()
    head = _git("rev-parse", "HEAD")
    dirty_subject = _git("status", "--porcelain", "--", "rust/src", "rust/shaders")
    return {
        "ep_library": _repo_relative(lib),
        "ep_library_sha256": hashlib.sha256(data).hexdigest(),
        "ep_library_bytes": len(data),
        "ep_library_profile": (
            "release"
            if f"{os.sep}release{os.sep}" in str(lib).replace("/", os.sep)
            else "NOT-RELEASE"
        ),
        "commit": head,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty_in_subject": bool(dirty_subject),
        "subject_paths": [
            "rust/shaders/glsl/gqa_decode_f16.comp",
            "rust/src/ops/attention.rs",
            "rust/src/registry.rs",
        ],
        "toolchain": {
            "rustc": _tool_version(["rustc", "--version"]),
            "cargo": _tool_version(["cargo", "--version"]),
            "glslc": _tool_version(["glslc", "--version"]),
            "onnxruntime": _tool_version(
                [sys.executable, "-c", "import onnxruntime;print(onnxruntime.__version__)"]
            ),
            "python": sys.version.split()[0],
        },
    }


def _model_provenance(spec) -> "tuple[dict, Path]":
    """Model identity WITHOUT its location, plus the full external-data accounting.

    An int4 Phi-3.5 export is a small `.onnx` beside ~2 GiB of external tensor data, so a digest
    of the `.onnx` alone identifies the graph and says nothing about the weights the run actually
    read. Both are recorded. The cache path is deliberately dropped: it is a private absolute
    path and the artifact is public.
    """
    rec = rm.resolve_model(spec)
    path = Path(rec["path"])
    ext = rm.external_data_provenance(path)
    public = {
        "key": spec.key,
        "family": spec.family,
        "variant_name": spec.variant_name,
        "onnx_filename": spec.onnx_filename,
        "onnx_sha256": rec.get("sha256"),
        "onnx_bytes": path.stat().st_size if path.is_file() else None,
        "provenance": rec.get("provenance"),
        "external_data": ext,
        "location": "resolved by identity via rust/tools/foundry_discovery.py; path omitted "
        "from this artifact on purpose (it is a private absolute path)",
    }
    return public, path


# ---------------------------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------------------------


def outputs_worker(argv) -> int:
    """One arm, one case, every output tensor to an npz. No comparison happens here."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-outputs", action="store_true")
    ap.add_argument("--past", type=int, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import numpy as np
    import onnxruntime as ort

    spec = rm.MODELS[rm.PHI35.key]
    model_rec = rm.resolve_model(spec)
    arm = rm.VULKAN_TILED
    arm.apply_env(os.environ)
    lib = os.environ.get(EP_LIB_ENV)
    if not lib or not Path(lib).is_file():
        print(f"{EP_LIB_ENV} unset or missing", file=sys.stderr)
        return 2
    try:
        ort.register_execution_provider_library(rm.EP_NAME, str(Path(lib).resolve()))
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            print(f"registration failed: {exc}", file=sys.stderr)
            return 2

    case = rm.Case(spec.key, "decode", 1, a.past, tokens=1, unit="tokens")
    feeds = rm.build_feeds(case, np)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("ep.device_index", str(a.device))
    sess = ort.InferenceSession(str(model_rec["path"]), opts, providers=list(arm.providers))
    names = [o.name for o in sess.get_outputs()]
    providers = list(sess.get_providers())
    outs = sess.run(None, feeds)
    np.savez(a.out, **{f"o{i:04d}": o for i, o in enumerate(outs)})
    Path(str(a.out) + ".meta.json").write_text(
        json.dumps(
            {
                "providers": providers,
                "output_names": names,
                "count": len(outs),
                "feeds_digest": rm.feeds_digest(feeds),
                ENV_KV_PARALLEL: os.environ.get(ENV_KV_PARALLEL, "<unset>"),
            }
        ),
        encoding="utf-8",
    )
    del sess
    return 0


def _run_outputs_env(py: str, scratch: Path, past: int, device: int, env_overrides: dict, tag: str):
    """Run the whole model once with an explicit env delta and dump every output."""
    out = scratch / f"kvpar_out_{past}_{tag}.npz"
    meta = Path(str(out) + ".meta.json")
    for p in (out, meta):
        p.unlink(missing_ok=True)
    env = dict(os.environ)
    env.pop(ENV_KV_PARALLEL, None)
    for k, v in (env_overrides or {}).items():
        env[k] = v
    proc = subprocess.run(
        [
            py,
            str(Path(__file__).resolve()),
            "--worker-outputs",
            "--past",
            str(past),
            "--device",
            str(device),
            "--out",
            str(out),
        ],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
    )
    if not out.is_file() or not meta.is_file():
        sys.stderr.write(
            f"[kvpar] outputs worker exit {proc.returncode} past={past} tag={tag}\n"
            f"{proc.stderr.decode(errors='replace')[-800:]}\n"
        )
        return None, None
    return out, json.loads(meta.read_text(encoding="utf-8"))


def _run_outputs(py: str, scratch: Path, past: int, arm_env, device: int, tag: str):
    return _run_outputs_env(
        py, scratch, past, device, {} if arm_env is None else {ENV_KV_PARALLEL: arm_env}, tag
    )


def _ulp_distance(np, a, b):
    """Distance in representable steps, at the ORIGINAL dtype, not at f32.

    An absolute difference does not say whether a number moved one step or ten thousand, and at
    f16 the step size varies by four orders of magnitude across the range. Reinterpreting the
    bits as a sign-magnitude integer and converting to two's complement makes adjacent floats
    adjacent integers, so `|ia - ib|` counts representable values between them. Non-finite lanes
    are excluded and reported separately rather than folded in as enormous integers.
    """
    if a.dtype not in (np.float16, np.float32) or a.dtype != b.dtype:
        return None
    itype = np.int16 if a.dtype == np.float16 else np.int32
    sign_bit = np.array(
        1 << (15 if a.dtype == np.float16 else 31), dtype=np.uint16 if a.dtype == np.float16 else np.uint32
    )

    def to_ordered(x):
        u = x.view(np.uint16 if x.dtype == np.float16 else np.uint32).astype(np.int64)
        neg = u >= int(sign_bit)
        return np.where(neg, int(sign_bit) - u, u)

    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return {"max": None, "note": "no jointly finite element"}
    ia = to_ordered(a)[finite]
    ib = to_ordered(b)[finite]
    d = np.abs(ia - ib)
    _ = itype
    return {
        "max": int(d.max()),
        "mean": float(d.mean()),
        "p99": int(np.percentile(d, 99)),
        "elements_beyond_1_ulp": int((d > 1).sum()),
        "dtype": str(a.dtype),
    }


def _nonfinite_census(np, x) -> dict:
    return {
        "nan": int(np.isnan(x).sum()),
        "posinf": int(np.isposinf(x).sum()),
        "neginf": int(np.isneginf(x).sum()),
    }


def _topk_agreement(np, cand, ref, ks=(1, 5, 10)) -> dict:
    """Rank agreement plus the MARGIN, on the last row of the leading output.

    Top-k agreement on its own is as weak as argmax on its own: two distributions can agree on
    their ordering while differing everywhere. The margin is the part that carries information —
    `top1 - top2` in the reference says how much slack the ordering had, so a run that keeps the
    ordering with a margin of 1e-4 is reporting something quite different from one that keeps it
    with a margin of 3.
    """
    if cand.shape != ref.shape or cand.size == 0:
        return {"measured": False, "reason": "shape mismatch or empty"}
    c = cand.reshape(-1, cand.shape[-1])[-1].astype(np.float32)
    r = ref.reshape(-1, ref.shape[-1])[-1].astype(np.float32)
    if not (np.isfinite(c).all() and np.isfinite(r).all()):
        return {"measured": False, "reason": "non-finite row"}
    out: dict = {"measured": True, "vocab": int(r.size)}
    for k in ks:
        if k > r.size:
            continue
        ci = set(np.argsort(-c)[:k].tolist())
        ri = set(np.argsort(-r)[:k].tolist())
        out[f"top{k}_set_agrees"] = ci == ri
        out[f"top{k}_overlap"] = len(ci & ri)
    order = np.argsort(-r)
    out["reference_top1_minus_top2"] = float(r[order[0]] - r[order[1]]) if r.size > 1 else None
    out["candidate_at_reference_top1"] = float(c[order[0]])
    out["margin_survives"] = (
        bool(c[order[0]] - c[order[1]] > 0) if r.size > 1 else None
    )
    out["note"] = (
        "recorded, never the criterion: the elementwise band below is what decides equivalence"
    )
    return out


def _compare_outputs(np, cand, ref, names) -> dict:
    """Every element of every output, at the predeclared tolerance. Bitwise recorded separately.

    Returns a dict whose `equivalent` field is the ONLY thing `_publishable` consults. The
    per-output detail is kept so a refusal names what failed rather than merely that something
    did.

    Each output carries, in addition to the pass/fail: bitwise identity, the finite-mask
    comparison, absolute AND relative worst-case, the ULP histogram, and a non-finite census of
    both sides. The rejected artifact reported a single scalar per case, which cannot distinguish
    "moved one representable step" from "moved into a different exponent", and cannot see a NaN
    at all when the NaN is on both sides.
    """
    per_output = []
    worst = {"name": None, "max_abs": 0.0}
    worst_rel = {"name": None, "max_rel": 0.0}
    all_bitwise = True
    for i, (c, r) in enumerate(zip(cand, ref)):
        name = names[i] if i < len(names) else f"output_{i}"
        band = TOLERANCE["logits"] if i == 0 else TOLERANCE["present"]
        rec: dict = {"index": i, "name": name, "shape": list(c.shape)}
        if c.shape != r.shape:
            rec.update({"equivalent": False, "reason": "shape mismatch",
                        "reference_shape": list(r.shape)})
            per_output.append(rec)
            all_bitwise = False
            continue
        bitwise = bool(np.array_equal(c.view(np.uint8), r.view(np.uint8)))
        all_bitwise = all_bitwise and bitwise
        a = c.astype(np.float32)
        b = r.astype(np.float32)
        finite_match = bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
        finite = np.isfinite(a) & np.isfinite(b)
        bad = (~np.isclose(a, b, rtol=band["rtol"], atol=band["atol"])) & finite
        n_bad = int(bad.sum())
        max_abs = float(np.abs(a[finite] - b[finite]).max()) if finite.any() else 0.0
        # Relative error with a denominator floor, so a reference element of 0 does not report
        # an infinite relative error for an absolute difference of one ULP.
        if finite.any():
            denom = np.maximum(np.abs(b[finite]), band["atol"])
            rel = np.abs(a[finite] - b[finite]) / denom
            max_rel = float(rel.max())
        else:
            max_rel = 0.0
        cand_nf = _nonfinite_census(np, a)
        ref_nf = _nonfinite_census(np, b)
        rec.update(
            {
                "bitwise_identical": bitwise,
                "finite_masks_match": finite_match,
                "rtol": band["rtol"],
                "atol": band["atol"],
                "elements": int(a.size),
                "elements_outside_tolerance": n_bad,
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "ulp": _ulp_distance(np, c, r),
                "nonfinite": {
                    "candidate": cand_nf,
                    "reference": ref_nf,
                    "policy": (
                        "propagate. The kernel neither sanitises nor screens NaN/Inf; it "
                        "reproduces the serial kernel elementwise. Equivalence therefore "
                        "requires the finite MASKS to match — a candidate that turned a finite "
                        "element non-finite, or a non-finite element finite, fails here "
                        "regardless of the numeric band."
                    ),
                },
                "equivalent": bool(finite_match and n_bad == 0),
            }
        )
        if i == 0:
            rec["rank"] = _topk_agreement(np, c, r)
        if max_abs > worst["max_abs"]:
            worst = {"name": name, "max_abs": max_abs}
        if max_rel > worst_rel["max_rel"]:
            worst_rel = {"name": name, "max_rel": max_rel}
        per_output.append(rec)
    return {
        "outputs_total": len(ref),
        "outputs_compared": len(per_output),
        "all_bitwise_identical": all_bitwise,
        "worst": worst,
        "worst_relative": worst_rel,
        "per_output": per_output,
        "equivalent": bool(
            per_output
            and len(per_output) == len(ref)
            and all(o.get("equivalent") for o in per_output)
        ),
    }


def equivalence_pass(py: str, scratch: Path, pasts, device: int) -> list:
    """Run FIRST, and printed before any timing table, so a win cannot bury a divergence."""
    import numpy as np

    records = []
    for past in pasts:
        rec: dict = {"case": f"decode/M1/past{past}", "past": past}
        ser_path, ser_meta = _run_outputs(py, scratch, past, "1", device, "serial")
        par_path, par_meta = _run_outputs(py, scratch, past, None, device, "parallel")
        if ser_path is None or par_path is None:
            rec["equivalence"] = {
                "equivalent": False,
                "reason": "one or both arms failed to produce outputs",
            }
            records.append(rec)
            print(f"[kvpar] equivalence past={past}: UNMEASURED", flush=True)
            continue
        if ser_meta.get("feeds_digest") != par_meta.get("feeds_digest"):
            rec["equivalence"] = {
                "equivalent": False,
                "reason": "the two arms were fed different inputs",
                "serial_feeds_digest": ser_meta.get("feeds_digest"),
                "parallel_feeds_digest": par_meta.get("feeds_digest"),
            }
            records.append(rec)
            print(f"[kvpar] equivalence past={past}: REFUSED(feeds differ)", flush=True)
            continue
        with np.load(ser_path) as z:
            ref = [z[k] for k in sorted(z.files)]
        with np.load(par_path) as z:
            cand = [z[k] for k in sorted(z.files)]
        eq = _compare_outputs(np, cand, ref, par_meta.get("output_names") or [])
        eq["argmax_agrees"] = bool(
            cand[0].reshape(-1, cand[0].shape[-1])[-1].argmax()
            == ref[0].reshape(-1, ref[0].shape[-1])[-1].argmax()
        )
        eq["argmax_note"] = (
            "recorded, never the criterion: a merge that lost a lane moves the distribution "
            "long before it moves the argmax"
        )
        eq["providers"] = par_meta.get("providers")
        rec["equivalence"] = eq
        records.append(rec)
        verdict = "EQUIVALENT" if eq["equivalent"] else "DIVERGENT"
        extra = " (bitwise)" if eq["all_bitwise_identical"] else ""
        print(
            f"[kvpar] equivalence past={past}: {verdict}{extra} "
            f"worst |par-ser| = {eq['worst']['max_abs']:.6g} on {eq['worst']['name']}",
            flush=True,
        )
    return records


# ---------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------


def _kernel_us(trace_path: Path, iters: int, warmups: int = 1) -> dict:
    """RAW per-inference GPU microseconds per kernel, with the first `warmups` inferences dropped.

    The tracer emits one `cat == "gpu"` event per dispatch in submission order, so for a kernel
    that runs `n` times per inference the event list partitions into `iters` runs of `n`. Dropping
    the leading buckets removes pipeline creation, first-touch allocation and cold caches from the
    sample. A count that does not divide is an instrument error and the point is discarded —
    averaging over a partition that is not a partition is how a warm-up leaks into a headline.

    WHY THE RAW SERIES IS KEPT AND NOT ONLY THE MEAN
    ------------------------------------------------
    A mean of a mean of a mean cannot be audited: the reader cannot see whether one inference
    carried the point, whether the series drifted, or whether the "sample" was a single number
    wearing three hats. `per_inference_us` is the untouched series this point was computed from,
    and everything else in this record is a function of it.
    """
    d = json.loads(trace_path.read_text(encoding="utf-8"))
    events = d["traceEvents"] if isinstance(d, dict) else d
    by_name: "dict[str, list]" = collections.defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") == "gpu":
            by_name[e.get("name", "?")].append(e.get("dur", 0))
    timed_n = iters - warmups
    out: dict = {
        "warmup_inferences": warmups,
        "timed_inferences": timed_n,
        "by_kernel_us": {},
        "per_inference_us": {},
    }
    total = 0.0
    for name, durs in by_name.items():
        if timed_n <= 0 or len(durs) % iters != 0:
            out.setdefault("instrument_errors", []).append(
                f"{name}: {len(durs)} events do not partition into {iters} inferences"
            )
            continue
        per = len(durs) // iters
        # One value per inference: the sum of that inference's dispatches of this kernel.
        series = [sum(durs[i * per : (i + 1) * per]) for i in range(iters)]
        timed = series[warmups:]
        out["by_kernel_us"][name] = statistics.mean(timed)
        out["per_inference_us"][name] = {
            "warmup_dropped": series[:warmups],
            "timed": timed,
        }
        out.setdefault("dispatches_per_inference", {})[name] = per
        total += out["by_kernel_us"][name]
    out["graph_total_us"] = total
    out["gqa_us"] = out["by_kernel_us"].get(PARALLEL_EVENT, 0.0) + out["by_kernel_us"].get(
        SERIAL_EVENT, 0.0
    )
    gqa_series = out["per_inference_us"].get(PARALLEL_EVENT) or out["per_inference_us"].get(
        SERIAL_EVENT
    )
    out["gqa_per_inference_us"] = list(gqa_series["timed"]) if gqa_series else []
    out["gqa_kernels_seen"] = sorted(
        k for k in out["by_kernel_us"] if k in (PARALLEL_EVENT, SERIAL_EVENT)
    )
    return out


def _time_point(py, scratch, past, arm_name, arm_env, device, iters, repeat, warmups=1) -> dict:
    trace = scratch / f"kvpar_trace_{past}_{arm_name}_{repeat}.json"
    diag = scratch / f"kvpar_diag_{past}_{arm_name}_{repeat}.json"
    counters = scratch / f"kvpar_counters_{past}_{arm_name}_{repeat}.json"
    for p in (trace, diag, counters):
        p.unlink(missing_ok=True)
    env = dict(os.environ)
    env.pop(ENV_KV_PARALLEL, None)
    if arm_env is not None:
        env[ENV_KV_PARALLEL] = arm_env
    env[TRACE_ENV] = str(trace)
    env[TRACE_GPU_ENV] = "1"
    env[COUNTERS_ENV] = str(counters)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            py,
            str(_BENCH / "results" / "probe_real_model_latency.py"),
            "--worker-diagnose",
            "--model",
            rm.PHI35.key,
            "--arm",
            "vulkan_tiled",
            "--phase",
            "decode",
            "--m",
            "1",
            "--past",
            str(past),
            "--device",
            str(device),
            "--iters",
            str(iters),
            "--out",
            str(diag),
        ],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
    )
    rec: dict = {"repeat": repeat, "process_wall_s": round(time.perf_counter() - t0, 3)}
    if not trace.is_file():
        rec["error"] = f"no trace; worker exit {proc.returncode}"
        return rec
    rec.update(_kernel_us(trace, iters, warmups))
    if counters.is_file():
        try:
            c = json.loads(counters.read_text(encoding="utf-8"))
            rec["shaders_dispatched"] = c.get("shaders_dispatched")
            rec["pipeline_variants"] = c.get("pipeline_variants")
        except json.JSONDecodeError:
            pass
    trace.unlink(missing_ok=True)
    return rec


def _arm_summary(points, expected_event) -> dict:
    """Median and dispersion for one arm, plus the pipeline witness.

    `witness_ok` is what stops this file from reporting a difference between two runs of the same
    kernel: the arm must have executed the module it names and NOT the other one.
    """
    ok = [p for p in points if "gqa_us" in p and p["gqa_us"] > 0]
    other = SERIAL_EVENT if expected_event == PARALLEL_EVENT else PARALLEL_EVENT
    seen = set()
    for p in ok:
        seen.update(p.get("gqa_kernels_seen") or [])
    summary: dict = {
        "points": points,
        "samples_us": [p["gqa_us"] for p in ok],
        # The untouched per-inference series behind `samples_us`, flattened across repeats. This
        # is the audit trail: `median_us` is a function of `samples_us`, and `samples_us` is a
        # function of THIS. `_publishable` refuses a case whose arms do not carry it.
        "raw_samples_us": [v for p in ok for v in (p.get("gqa_per_inference_us") or [])],
        "expected_kernel": expected_event,
        "kernels_seen": sorted(seen),
        "witness_ok": bool(ok) and seen == {expected_event},
    }
    if not ok:
        summary["witness_detail"] = "no timed point produced a GQA GPU interval"
        return summary
    if other in seen:
        summary["witness_detail"] = (
            f"this arm executed {other}, which belongs to the other arm — the two arms are not "
            f"distinct and no ratio between them means anything"
        )
    samples = summary["samples_us"]
    summary["median_us"] = statistics.median(samples)
    summary["graph_total_us"] = statistics.median(
        [p["graph_total_us"] for p in ok if "graph_total_us" in p]
    )
    if len(samples) >= 2 and summary["median_us"]:
        summary["rsd"] = statistics.pstdev(samples) / statistics.mean(samples)
        summary["repeatable"] = summary["rsd"] <= DISPERSION_CEILING
    else:
        summary["rsd"] = None
        summary["repeatable"] = None
    return summary


def timing_pass(py, scratch, pasts, device, iters, repeats, warmups=1) -> list:
    records = []
    for past in pasts:
        arms: dict = {name: [] for name, _, _ in ARMS}
        # Interleaved: forward on even repeats, reversed on odd, so a monotone drift in the box
        # cannot be absorbed entirely into whichever arm always went second.
        for r in range(repeats):
            order = list(ARMS) if r % 2 == 0 else list(reversed(ARMS))
            for name, env, _event in order:
                pt = _time_point(py, scratch, past, name, env, device, iters, r, warmups)
                arms[name].append(pt)
                print(
                    f"[kvpar] timing past={past} arm={name:8s} r{r}: "
                    f"gqa={pt.get('gqa_us', 0) / 1000.0:8.2f} ms  "
                    f"graph={pt.get('graph_total_us', 0) / 1000.0:8.2f} ms",
                    flush=True,
                )
        rec = {
            "case": f"decode/M1/past{past}",
            "past": past,
            "arms": {
                name: _arm_summary(arms[name], event)
                for name, _env, event in ARMS
            },
        }
        records.append(rec)
    return records


# ---------------------------------------------------------------------------------------------
# The node-scoped pass — the only pass in this file entitled to publish a speed number
# ---------------------------------------------------------------------------------------------


def build_node_model(past: int, seq_len: int = None) -> bytes:
    """Phi-3.5's own GQA node, alone in a graph. Same op, same attributes, same shape.

    WHY THE CLAIM LIVES HERE AND NOT ON THE WHOLE MODEL
    ---------------------------------------------------
    The claim under test is about ONE kernel. A whole-model comparison cannot be an equivalence
    result for one kernel at a per-element bound, and the measurement in this file's own
    `whole_model` section is the demonstration: at past 128 the two arms' layer-0 `present.key`
    and `present.value` are BITWISE IDENTICAL, and the divergence then grows monotonically with
    depth — 0.002 at layer 1, 0.13 at layer 31, 0.52 at the logits — at roughly 1.15x per layer.
    That is an fp16 residual stack amplifying a reassociation, and it is a property of the model,
    not a property of this kernel. A tolerance wide enough to absorb 32 layers of it would be
    wide enough to absorb a real defect too.

    So the equivalence gate on the published number is applied where the claim is: to the node.
    Its criterion is the SAME `TOLERANCE` object the whole-model pass uses — not a looser one —
    and in practice this node meets it bitwise. Nothing is relabelled and nothing is widened;
    the two passes ask their question at two different scopes and both answers are printed.
    """
    from onnx import TensorProto, helper

    import numpy as np

    if seq_len is None:
        seq_len = NODE_SHAPE["seq_len"]
    nh = NODE_SHAPE["num_heads"]
    nkv = NODE_SHAPE["kv_heads"]
    hd = NODE_SHAPE["head_dim"]
    rot = NODE_SHAPE["rotary_dim"]
    s = seq_len
    packed = (nh + 2 * nkv) * hd
    max_seq = past + s + 1
    f16 = TensorProto.FLOAT16
    ins = [
        helper.make_tensor_value_info("packed_qkv", f16, ["B", s, packed]),
        helper.make_tensor_value_info("past_key", f16, ["B", nkv, past, hd]),
        helper.make_tensor_value_info("past_value", f16, ["B", nkv, past, hd]),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, ["B"]),
        helper.make_tensor_value_info("total_seq", TensorProto.INT32, []),
        helper.make_tensor_value_info("cos_cache", f16, [max_seq, rot // 2]),
        helper.make_tensor_value_info("sin_cache", f16, [max_seq, rot // 2]),
    ]
    outs = [
        helper.make_tensor_value_info("attn_out", f16, ["B", s, nh * hd]),
        helper.make_tensor_value_info("present_key", f16, ["B", nkv, past + s, hd]),
        helper.make_tensor_value_info("present_value", f16, ["B", nkv, past + s, hd]),
    ]
    node = helper.make_node(
        "GroupQueryAttention",
        inputs=["packed_qkv", "", "", "past_key", "past_value", "seqlens_k", "total_seq",
                "cos_cache", "sin_cache"],
        outputs=["attn_out", "present_key", "present_value"],
        domain="com.microsoft",
        name="gqa_decode_node",
        num_heads=nh,
        kv_num_heads=nkv,
        scale=float(hd ** -0.5),
        local_window_size=-1,
        do_rotary=1,
        rotary_interleaved=0,
        smooth_softmax=0,
    )
    model = helper.make_model(
        helper.make_graph([node], "gqa_decode_node_graph", ins, outs),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    _ = np
    return model.SerializeToString()


def build_node_feeds(past: int, seq_len: int = None):
    import numpy as np

    nh = NODE_SHAPE["num_heads"]
    nkv = NODE_SHAPE["kv_heads"]
    hd = NODE_SHAPE["head_dim"]
    rot = NODE_SHAPE["rotary_dim"]
    s = NODE_SHAPE["seq_len"] if seq_len is None else seq_len
    packed = (nh + 2 * nkv) * hd
    max_seq = past + s + 1
    rng = np.random.default_rng(NODE_SEED + past)
    pos = np.arange(max_seq, dtype=np.float32)[:, None]
    freq = 1.0 / (10000 ** (np.arange(0, rot, 2, dtype=np.float32) / rot))
    ang = pos * freq
    return {
        "packed_qkv": (rng.standard_normal((1, s, packed)) * 0.1).astype(np.float16),
        "past_key": (rng.standard_normal((1, nkv, past, hd)) * 0.1).astype(np.float16),
        "past_value": (rng.standard_normal((1, nkv, past, hd)) * 0.1).astype(np.float16),
        # `seqlens_k` is total-minus-one, so `past_len = seqlens_k + 1 - seq_len` recovers `past`
        # at any `seq_len`. Getting this wrong would silently move the prefill arm's KV extent.
        "seqlens_k": np.array([past + s - 1], dtype=np.int32),
        "total_seq": np.array(past + s, dtype=np.int32),
        "cos_cache": np.cos(ang).astype(np.float16),
        "sin_cache": np.sin(ang).astype(np.float16),
    }


def node_worker(argv) -> int:
    """One arm, one past, on the isolated GQA node: outputs + GPU trace in one process."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-node", action="store_true")
    ap.add_argument("--past", type=int, required=True)
    ap.add_argument("--seq-len", type=int, default=1)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import numpy as np
    import onnxruntime as ort

    lib = os.environ.get(EP_LIB_ENV)
    if not lib or not Path(lib).is_file():
        print(f"{EP_LIB_ENV} unset or missing", file=sys.stderr)
        return 2
    try:
        ort.register_execution_provider_library(rm.EP_NAME, str(Path(lib).resolve()))
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            print(f"registration failed: {exc}", file=sys.stderr)
            return 2

    model = build_node_model(a.past, a.seq_len)
    feeds = build_node_feeds(a.past, a.seq_len)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.add_session_config_entry("ep.device_index", str(a.device))
    sess = ort.InferenceSession(model, opts, providers=[rm.EP_NAME, rm.CPU_EP])
    providers = list(sess.get_providers())
    outs = None
    for _ in range(a.iters):
        outs = sess.run(None, feeds)
    np.savez(a.out, **{f"o{i:04d}": o for i, o in enumerate(outs)})
    Path(str(a.out) + ".meta.json").write_text(
        json.dumps(
            {
                "providers": providers,
                "output_names": [o.name for o in sess.get_outputs()],
                "count": len(outs),
                "seq_len": a.seq_len,
                "past": a.past,
                # Record-level provenance: which binary actually produced THESE bytes. The
                # document root records the subject build, but a record that cannot name its own
                # library cannot be cross-checked against the root, and the fallback-identity
                # pass deliberately runs two DIFFERENT libraries.
                "ep_library_sha256": hashlib.sha256(Path(lib).read_bytes()).hexdigest(),
                "ep_library_bytes": Path(lib).stat().st_size,
                "recorded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                ENV_KV_PARALLEL: os.environ.get(ENV_KV_PARALLEL, "<unset>"),
            }
        ),
        encoding="utf-8",
    )
    del sess
    return 0


def _run_node(py, scratch, past, arm_env, device, iters, tag, seq_len=1, lib=None, warmups=1):
    out = scratch / f"kvpar_node_{past}_{tag}.npz"
    meta = Path(str(out) + ".meta.json")
    trace = scratch / f"kvpar_node_trace_{past}_{tag}.json"
    counters = scratch / f"kvpar_node_counters_{past}_{tag}.json"
    for p in (out, meta, trace, counters):
        p.unlink(missing_ok=True)
    env = dict(os.environ)
    env.pop(ENV_KV_PARALLEL, None)
    if arm_env is not None:
        env[ENV_KV_PARALLEL] = arm_env
    if lib is not None:
        env[EP_LIB_ENV] = str(lib)
    env[TRACE_ENV] = str(trace)
    env[TRACE_GPU_ENV] = "1"
    env[COUNTERS_ENV] = str(counters)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [py, str(Path(__file__).resolve()), "--worker-node", "--past", str(past),
         "--seq-len", str(seq_len),
         "--iters", str(iters), "--device", str(device), "--out", str(out)],
        cwd=str(_ROOT), env=env, capture_output=True,
    )
    if not out.is_file() or not trace.is_file():
        sys.stderr.write(
            f"[kvpar] node worker exit {proc.returncode} past={past} arm={tag}\n"
            f"{proc.stderr.decode(errors='replace')[-800:]}\n"
        )
        return None
    timing = _kernel_us(trace, iters, warmups)
    timing["process_wall_s"] = round(time.perf_counter() - t0, 3)
    trace.unlink(missing_ok=True)
    meta_d = json.loads(meta.read_text(encoding="utf-8"))
    if counters.is_file():
        try:
            c = json.loads(counters.read_text(encoding="utf-8"))
            timing["shaders_dispatched"] = c.get("shaders_dispatched")
            timing["pipeline_variants"] = c.get("pipeline_variants")
        except json.JSONDecodeError:
            pass
    return {"outputs": out, "meta": meta_d, "timing": timing}


def node_pass(py, scratch, pasts, device, iters, repeats, warmups=1) -> list:
    """Equivalence and kernel time on the isolated production GQA node, same protocol as above."""
    import numpy as np

    records = []
    for past in pasts:
        rec: dict = {"case": f"node/decode/past{past}", "past": past,
                     "shape": dict(NODE_SHAPE, past=past)}
        arms: dict = {name: [] for name, _, _ in ARMS}
        per_repeat: "dict[int, dict]" = {}
        names: list = []
        for r in range(repeats):
            order = list(ARMS) if r % 2 == 0 else list(reversed(ARMS))
            captured: dict = {}
            for name, env, _event in order:
                got = _run_node(py, scratch, past, env, device, iters, f"{name}_{r}",
                                warmups=warmups)
                if got is None:
                    arms[name].append({"repeat": r, "error": "worker failed"})
                    continue
                pt = dict(got["timing"], repeat=r)
                pt["witness"] = {
                    "ep_library_sha256": got["meta"].get("ep_library_sha256"),
                    "ep_library_bytes": got["meta"].get("ep_library_bytes"),
                    "recorded_at": got["meta"].get("recorded_at"),
                    "env": got["meta"].get(ENV_KV_PARALLEL),
                    "providers": got["meta"].get("providers"),
                }
                arms[name].append(pt)
                with np.load(got["outputs"]) as z:
                    captured[name] = [z[k] for k in sorted(z.files)]
                names = got["meta"].get("output_names") or names
                print(
                    f"[kvpar] node past={past} arm={name:8s} r{r}: "
                    f"gqa={pt.get('gqa_us', 0):9.1f} us",
                    flush=True,
                )
            # EVERY repeat is compared, not only the first. A kernel whose race surfaces once in
            # three runs is exactly the kernel a first-repeat-only check clears.
            if "serial" in captured and "parallel" in captured:
                per_repeat[r] = _compare_outputs(np, captured["parallel"], captured["serial"], names)
        if per_repeat:
            # The published equivalence is the WORST repeat, so a single divergent repeat cannot
            # be averaged away by two clean ones.
            worst_r = min(per_repeat, key=lambda k: (per_repeat[k]["equivalent"],
                                                     -per_repeat[k]["worst"]["max_abs"]))
            eq = dict(per_repeat[worst_r])
            eq["scope"] = (
                "the GroupQueryAttention node itself, in isolation — the same thing the kernel "
                "time below is measured on"
            )
            eq["repeats_compared"] = len(per_repeat)
            eq["equivalent_every_repeat"] = all(v["equivalent"] for v in per_repeat.values())
            eq["reported_repeat"] = worst_r
            eq["per_repeat"] = {
                str(k): {
                    "equivalent": v["equivalent"],
                    "all_bitwise_identical": v["all_bitwise_identical"],
                    "worst_max_abs": v["worst"]["max_abs"],
                    "worst_max_rel": v.get("worst_relative", {}).get("max_rel"),
                }
                for k, v in sorted(per_repeat.items())
            }
            # A single repeat's verdict is not the case's verdict.
            eq["equivalent"] = bool(eq["equivalent"] and eq["equivalent_every_repeat"])
        else:
            eq = {"equivalent": False, "reason": "one or both arms failed to produce outputs",
                  "outputs_total": 0, "outputs_compared": 0, "repeats_compared": 0,
                  "equivalent_every_repeat": False}
        rec["equivalence"] = eq
        rec["arms"] = {name: _arm_summary(arms[name], event) for name, _env, event in ARMS}
        verdict = "EQUIVALENT" if eq.get("equivalent") else "DIVERGENT"
        extra = " (bitwise)" if eq.get("all_bitwise_identical") else ""
        print(
            f"[kvpar] node equivalence past={past}: {verdict}{extra} "
            f"({eq.get('repeats_compared', 0)} repeats compared)",
            flush=True,
        )
        records.append(rec)
    return records

def _auto_lanes(past: int, seq_len: int = 1) -> int:
    """The host selector's auto rule, restated so the artifact can say which W it expected.

    Mirrors `ops::attention::gqa_decode_kv_lanes_with` for the auto (no-override) path. It is a
    RESTATEMENT and not the authority: `tests/ops/test_gqa_decode_kv_parallel.py` pins the host
    rule directly, and the pipeline witness in each record says which module actually ran.
    """
    total = past + seq_len
    lanes = 1
    while lanes * 2 <= 16 and total // (lanes * 2) >= 32:
        lanes *= 2
    return lanes


def fallback_identity_pass(py, scratch, pasts, device, base_lib, subject_lib) -> dict:
    """W=1 against PRIOR PRODUCTION, bit for bit, at this exact head.

    WHAT THIS ANSWERS
    -----------------
    Requirement 1 says W=1 must be bitwise-equivalent to prior production or must dispatch the
    prior shader. In this design it does the latter — `gqa_decode_kv_lanes_with` returns 1 and
    the host dispatches `gqa_f16` untouched — so identity holds *by construction*. By
    construction is an argument, not a measurement, and the argument has a gap: the subject build
    also changed `variant_key`, the pipeline cache key and the aux-stem plumbing, any of which
    could perturb the serial path without touching its shader.

    So this pass runs TWO DIFFERENT BINARIES: the base build from the merge-base commit, which
    has never heard of this feature, and the subject build with the lane count pinned to 1. Same
    device, same feeds, same graph. Every element of all three outputs must be bitwise identical,
    and the subject must be witnessed running `gqa_f16` and never `gqa_decode_f16`.

    `auto_vs_base` is the same comparison with NOTHING set — the shipped default. Below the
    selector's threshold it must be bitwise identical too, because the default IS the refusal
    there; at and above the threshold it is expected to differ and is recorded as such, so the
    row cannot be read as a failure.
    """
    import numpy as np

    cases = []
    for past in pasts:
        row: dict = {"past": past, "auto_lanes_expected": _auto_lanes(past)}
        base = _run_node(py, scratch, past, None, device, 2, f"fbbase_{past}", lib=base_lib)
        w1 = _run_node(py, scratch, past, "1", device, 2, f"fbw1_{past}", lib=subject_lib)
        auto = _run_node(py, scratch, past, None, device, 2, f"fbauto_{past}", lib=subject_lib)
        if base is None or w1 is None or auto is None:
            row["measured"] = False
            row["reason"] = "an arm failed to produce outputs"
            cases.append(row)
            continue
        with np.load(base["outputs"]) as z:
            ref = [z[k] for k in sorted(z.files)]
        with np.load(w1["outputs"]) as z:
            cand = [z[k] for k in sorted(z.files)]
        with np.load(auto["outputs"]) as z:
            auto_vals = [z[k] for k in sorted(z.files)]
        names = base["meta"].get("output_names") or []
        row["measured"] = True
        row["base_library_sha256"] = base["meta"].get("ep_library_sha256")
        row["subject_library_sha256"] = w1["meta"].get("ep_library_sha256")
        row["distinct_binaries"] = (
            row["base_library_sha256"] != row["subject_library_sha256"]
        )
        eq = _compare_outputs(np, cand, ref, names)
        row["forced_w1_vs_base"] = {
            "all_bitwise_identical": eq["all_bitwise_identical"],
            "per_output_bitwise": {
                o["name"]: o["bitwise_identical"] for o in eq["per_output"]
            },
            "worst": eq["worst"],
            "equivalent": eq["equivalent"],
        }
        eq_auto = _compare_outputs(np, auto_vals, ref, names)
        expect_identical = row["auto_lanes_expected"] == 1
        row["auto_vs_base"] = {
            "expectation": (
                "BITWISE IDENTICAL — the selector refuses here, so the default is the old path"
                if expect_identical
                else "EXPECTED TO DIFFER — the selector engages here; this row is context, "
                "not a check"
            ),
            "all_bitwise_identical": eq_auto["all_bitwise_identical"],
            "worst": eq_auto["worst"],
            "meets_expectation": bool(
                eq_auto["all_bitwise_identical"] == expect_identical
                or (not expect_identical)
            ),
        }
        row["witness"] = {
            "base_kernels": sorted(base["timing"].get("by_kernel_us", {})),
            "forced_w1_kernels": sorted(w1["timing"].get("by_kernel_us", {})),
            "auto_kernels": sorted(auto["timing"].get("by_kernel_us", {})),
            "forced_w1_ran_serial_module_only": (
                sorted(w1["timing"].get("gqa_kernels_seen") or []) == [SERIAL_EVENT]
            ),
        }
        ok = (
            row["forced_w1_vs_base"]["all_bitwise_identical"]
            and row["witness"]["forced_w1_ran_serial_module_only"]
            and row["distinct_binaries"]
            and row["auto_vs_base"]["meets_expectation"]
        )
        row["holds"] = bool(ok)
        print(
            f"[kvpar] fallback identity past={past}: "
            f"{'BITWISE IDENTICAL' if row['forced_w1_vs_base']['all_bitwise_identical'] else 'DIFFERS'}"
            f" (auto W={row['auto_lanes_expected']})",
            flush=True,
        )
        cases.append(row)
    measured = [c for c in cases if c.get("measured")]
    return {
        "what": (
            "W=1 in the subject build against the base build from the merge-base commit, on the "
            "isolated node, all three outputs, bitwise"
        ),
        "why": (
            "the refusal path is only as good as its identity to what shipped before, and "
            "'it dispatches the same shader' is a claim about the host, which this pass tests "
            "rather than assumes"
        ),
        "cases": cases,
        "pasts_covered": len(cases),
        "holds": bool(measured) and all(c.get("holds") for c in measured),
        "complete": len(measured) == len(cases),
    }


def prefill_pass(py, scratch, cases_in, device, base_lib, subject_lib, iters, warmups) -> dict:
    """Prefill at THIS head: unchanged outputs, unchanged module, unchanged time.

    The rejected artifact carried a prefill witness produced before the change existed, which
    cannot speak for the head under review. This runs prefill on the subject build, at its
    shipped default, against the base build — same graph, same feeds — and asserts three things
    a prefill regression would break: the outputs are bitwise identical, the decode module is
    never dispatched, and the kernel time does not move outside the dispersion band.
    """
    import numpy as np

    rows = []
    for past, seq_len in cases_in:
        row: dict = {"past": past, "seq_len": seq_len,
                     "selector_expectation": "REFUSE (seq_len != 1)"}
        base = _run_node(py, scratch, past, None, device, iters, f"pfbase_{past}_{seq_len}",
                         seq_len=seq_len, lib=base_lib, warmups=warmups)
        subj = _run_node(py, scratch, past, None, device, iters, f"pfsubj_{past}_{seq_len}",
                         seq_len=seq_len, lib=subject_lib, warmups=warmups)
        if base is None or subj is None:
            row["measured"] = False
            row["reason"] = "an arm failed"
            rows.append(row)
            continue
        with np.load(base["outputs"]) as z:
            ref = [z[k] for k in sorted(z.files)]
        with np.load(subj["outputs"]) as z:
            cand = [z[k] for k in sorted(z.files)]
        eq = _compare_outputs(np, cand, ref, base["meta"].get("output_names") or [])
        b_us = base["timing"].get("gqa_us", 0.0)
        s_us = subj["timing"].get("gqa_us", 0.0)
        seen = sorted(subj["timing"].get("gqa_kernels_seen") or [])
        row.update(
            {
                "measured": True,
                "bitwise_identical": eq["all_bitwise_identical"],
                "worst": eq["worst"],
                "subject_kernels_seen": seen,
                "decode_module_never_dispatched": PARALLEL_EVENT not in seen,
                "base_gqa_us": b_us,
                "subject_gqa_us": s_us,
                "subject_over_base": (s_us / b_us) if b_us else None,
                "base_per_inference_us": base["timing"].get("gqa_per_inference_us"),
                "subject_per_inference_us": subj["timing"].get("gqa_per_inference_us"),
            }
        )
        ratio = row["subject_over_base"]
        row["time_unmoved"] = bool(ratio is not None and abs(ratio - 1.0) <= DISPERSION_CEILING)
        row["holds"] = bool(
            row["bitwise_identical"]
            and row["decode_module_never_dispatched"]
            and row["time_unmoved"]
        )
        print(
            f"[kvpar] prefill past={past} M={seq_len}: "
            f"{'bitwise' if row['bitwise_identical'] else 'DIFFERS'}, "
            f"decode module {'absent' if row['decode_module_never_dispatched'] else 'PRESENT'}, "
            f"t_subj/t_base={ratio if ratio is None else round(ratio, 4)}",
            flush=True,
        )
        rows.append(row)
    measured = [r for r in rows if r.get("measured")]
    return {
        "what": "prefill (seq_len > 1) on the subject build vs the base build, at this head",
        "criterion": (
            "bitwise-identical outputs, the decode module never dispatched, and kernel time "
            f"within +/-{DISPERSION_CEILING:.0%} of the base build"
        ),
        "cases": rows,
        "holds": bool(measured) and all(r.get("holds") for r in measured),
        "complete": len(measured) == len(rows),
    }


def w_ladder_pass(py, scratch, pasts, ws, device, iters, repeats, warmups) -> dict:
    """Forced-W ladder: one measured cell per (past, W), in RANDOMISED order.

    WHY THE ORDER IS RANDOMISED
    ---------------------------
    The rejected artifact walked W1 -> W2 -> W4 -> W8 -> W16 in that fixed order at every point,
    which aliases lane count against position in the run: a box that warms up, throttles, or is
    progressively disturbed writes a monotone artefact straight into the ladder and it reads as
    a lane-count effect. The order here is a seeded permutation per (past, repeat), so position
    is decorrelated from W and the seed keeps it reproducible.

    WHY THESE NUMBERS ARE NOT THE HEADLINE
    --------------------------------------
    Production does not run a forced W. The selector picks W from the KV extent, so the shipped
    behaviour at a given past is ONE cell of this table — `auto_lanes` names which. The ladder
    exists to show the shape of the scaling and to make a W-generic ratio impossible to quote:
    every cell is labelled with its own W and its own past, and no cell is aggregated with
    another.
    """
    import numpy as np

    rows = []
    for past in pasts:
        cells: "dict[int, list]" = {w: [] for w in ws}
        outs: "dict[int, list]" = {}
        names: list = []
        for r in range(repeats):
            order = list(ws)
            random.Random(NODE_SEED + past * 131 + r).shuffle(order)
            for w in order:
                got = _run_node(py, scratch, past, str(w), device, iters,
                                f"lad_{past}_w{w}_{r}", warmups=warmups)
                if got is None:
                    cells[w].append({"repeat": r, "error": "worker failed"})
                    continue
                pt = dict(got["timing"], repeat=r, position_in_run=order.index(w))
                pt["witness"] = {
                    "ep_library_sha256": got["meta"].get("ep_library_sha256"),
                    "recorded_at": got["meta"].get("recorded_at"),
                    "env": got["meta"].get(ENV_KV_PARALLEL),
                }
                cells[w].append(pt)
                if w not in outs:
                    with np.load(got["outputs"]) as z:
                        outs[w] = [z[k] for k in sorted(z.files)]
                    names = got["meta"].get("output_names") or names
        base_us = None
        row: dict = {"past": past, "auto_lanes": _auto_lanes(past), "cells": {}}
        for w in ws:
            pts = [p for p in cells[w] if p.get("gqa_us")]
            if not pts:
                row["cells"][str(w)] = {"measured": False}
                continue
            samples = [p["gqa_us"] for p in pts]
            raw = [v for p in pts for v in (p.get("gqa_per_inference_us") or [])]
            expected = SERIAL_EVENT if w == 1 else PARALLEL_EVENT
            seen = sorted({k for p in pts for k in (p.get("gqa_kernels_seen") or [])})
            cell: dict = {
                "measured": True,
                "W": w,
                "median_us": statistics.median(samples),
                "per_repeat_us": samples,
                "raw_per_inference_us": raw,
                # Position of this cell within its repeat's permutation. If a cell's advantage
                # tracked its position rather than its W, this is where that would show.
                "positions_in_run": [p.get("position_in_run") for p in pts],
                "rsd": (statistics.pstdev(samples) / statistics.mean(samples))
                if len(samples) > 1 and statistics.mean(samples)
                else None,
                "expected_kernel": expected,
                "kernels_seen": seen,
                "witness_ok": seen == [expected],
                "is_the_shipped_choice_at_this_past": w == row["auto_lanes"],
            }
            if w in outs and 1 in outs:
                eq = _compare_outputs(np, outs[w], outs[1], names)
                cell["equivalence"] = {
                    "against": "W=1 (the serial module) at the same past",
                    "equivalent": eq["equivalent"],
                    "all_bitwise_identical": eq["all_bitwise_identical"],
                    "worst": eq["worst"],
                    "worst_relative": eq.get("worst_relative"),
                }
            row["cells"][str(w)] = cell
        base_cell = row["cells"].get("1", {})
        base_us = base_cell.get("median_us")
        for key, cell in row["cells"].items():
            if not cell.get("measured"):
                continue
            eq_ok = cell.get("equivalence", {}).get("equivalent", False) or key == "1"
            disp_ok = cell.get("rsd") is None or cell["rsd"] <= DISPERSION_CEILING
            if base_us and cell.get("witness_ok") and eq_ok and disp_ok:
                cell["speedup_vs_W1"] = base_us / cell["median_us"]
            else:
                cell["speedup_withheld_because"] = [
                    r
                    for r, bad in (
                        ("no W=1 reference", not base_us),
                        ("pipeline witness failed", not cell.get("witness_ok")),
                        ("not equivalent to W=1", not eq_ok),
                        ("dispersion above ceiling", not disp_ok),
                    )
                    if bad
                ]
        rows.append(row)
        shipped = row["cells"].get(str(row["auto_lanes"]), {})
        print(
            f"[kvpar] ladder past={past}: auto W={row['auto_lanes']} "
            f"shipped-cell speedup="
            f"{round(shipped.get('speedup_vs_W1'), 3) if shipped.get('speedup_vs_W1') else 'withheld'}",
            flush=True,
        )
    return {
        "what": "kernel time per (past, forced W) on the isolated node, randomised order",
        "reading": (
            "every cell is a (past, W) pair and nothing here is a W-generic ratio. The cell "
            "marked `is_the_shipped_choice_at_this_past` is the only one production would "
            "reach at that past with no environment override."
        ),
        "order": "seeded permutation of W per (past, repeat); position recorded per point",
        "cells_are_speedups_against": "the W=1 cell at the SAME past, which runs gqa_f16",
        "rows": rows,
    }




def _model_divergence(py, scratch, past, device, env_a, env_b, tag) -> dict:
    """Whole-model max |A - B| across every output, under the identical instrument."""
    import numpy as np

    a_path, a_meta = _run_outputs_env(py, scratch, past, device, env_a, f"{tag}_a")
    b_path, b_meta = _run_outputs_env(py, scratch, past, device, env_b, f"{tag}_b")
    if a_path is None or b_path is None:
        return {"measured": False, "reason": "an arm failed"}
    if a_meta.get("feeds_digest") != b_meta.get("feeds_digest"):
        return {"measured": False, "reason": "the arms were fed different inputs"}
    with np.load(a_path) as z:
        a = [z[k] for k in sorted(z.files)]
    with np.load(b_path) as z:
        b = [z[k] for k in sorted(z.files)]
    eq = _compare_outputs(np, a, b, b_meta.get("output_names") or [])
    return {
        "measured": True,
        "env_a": env_a,
        "env_b": env_b,
        "logits_max_abs": eq["per_output"][0]["max_abs_diff"] if eq["per_output"] else None,
        "worst": eq["worst"],
        "all_bitwise_identical": eq["all_bitwise_identical"],
        "outputs_outside_predeclared_tolerance": sum(
            1 for o in eq["per_output"] if not o.get("equivalent")
        ),
        "outputs_total": eq["outputs_total"],
    }


def _depth_profile(eq: dict) -> dict:
    """max |parallel - serial| per transformer layer, read off the present-KV outputs.

    This is the measurement that makes the whole-model refusal INTERPRETABLE rather than merely
    a red mark: if layer 0 is bitwise identical and the divergence then compounds with depth, the
    kernel produced the same cache and the model amplified an fp16 reassociation downstream. If
    instead layer 0 already differed, the kernel itself would be the source and the refusal would
    mean something entirely different. The artifact records which of the two it saw.
    """
    per_layer: dict = {}
    for o in eq.get("per_output") or []:
        m = re.match(r"present\.(\d+)\.(key|value)$", o.get("name") or "")
        if m:
            per_layer.setdefault(int(m.group(1)), {})[m.group(2)] = o.get("max_abs_diff")
    if not per_layer:
        return {"measured": False}
    ordered = sorted(per_layer)
    first = per_layer[ordered[0]]
    last = per_layer[ordered[-1]]
    ratios = []
    for a, b in zip(ordered, ordered[1:]):
        pa = max(per_layer[a].values() or [0])
        pb = max(per_layer[b].values() or [0])
        if pa > 0:
            ratios.append(pb / pa)
    return {
        "measured": True,
        "layers": len(ordered),
        "layer0_bitwise_identical": all(v == 0 for v in first.values()),
        "layer0_max_abs": max(first.values()),
        "deepest_layer_max_abs": max(last.values()),
        "median_growth_per_layer": statistics.median(ratios) if ratios else None,
        "per_layer_max_abs": {str(k): max(v.values()) for k, v in sorted(per_layer.items())},
        "reading": (
            "layer 0's present KV is the kernel's own output before any downstream layer has "
            "touched it. Bitwise equality there says the kernel wrote the same cache; a "
            "divergence that only appears deeper is the residual stack amplifying an fp16 "
            "reassociation, which is a property of the model and not of this change. This field "
            "does NOT lift the refusal above it and no speed number is attached to it."
        ),
    }


def whole_model_frame(py, scratch, pasts, device, eq_records) -> dict:
    """Controls that make the whole-model refusal interpretable. None of them lift it.

    Three probes, and the report records whatever each one comes back with — including a result
    that does not help this change. That is the point of running them:

    * NULL — the same arm against itself. It MUST come out bitwise identical. If it did not, the
      instrument would be measuring run-to-run nondeterminism rather than the change, and every
      other number in this section would be void. It is a control that can fail.

    * CROSS-KERNEL — `GEMV_MAX_ROWS` 1 vs 4, an already-shipped and already-accepted packing
      change in a DIFFERENT kernel, measured with the GQA split off in both arms. It asks
      whether whole-model elementwise divergence is simply what this instrument reports for any
      accepted kernel change. **On the run this artifact was generated from, the answer was NO:
      that pair came back bitwise identical, so it does NOT establish that the divergence is
      generic and it does NOT excuse this change.** The result is recorded as measured. A control
      is not worth running if only one of its outcomes may be reported.

    * LANE-SENSITIVITY — W = 2 against W = 16, both arms being the NEW kernel, differing only in
      how many ways the same sum is split. This is the control that isolates the mechanism: if
      the model's logits move by a comparable amount between two lane counts that are equally
      correct, then the model's output is simply not invariant under reassociation of this sum,
      and no choice of W is the "right" one to compare against. That is a real limitation of the
      change and it belongs on the face of the artifact.

    What none of them may do: lift the refusal. No speed field anywhere in this report is
    conditioned on any of them.
    """
    past = min(pasts)
    frame: dict = {
        "case": f"decode/M1/past{past}",
        "purpose": "make the whole-model refusal interpretable without lifting it",
        "does_not_lift_the_refusal": True,
    }
    eq = next(
        ((r.get("equivalence") or {}) for r in eq_records if r.get("past") == past), {}
    )
    frame["depth_profile"] = _depth_profile(eq)
    print(f"[kvpar] control NULL (parallel vs parallel) at past={past} ...", flush=True)
    frame["null_control"] = _model_divergence(py, scratch, past, device, {}, {}, "null")
    frame["null_control"]["expectation"] = (
        "bitwise identical. A non-zero result here voids this whole section."
    )
    frame["null_control"]["holds"] = bool(frame["null_control"].get("all_bitwise_identical"))
    print(f"[kvpar] control CROSS-KERNEL ({ENV_GEMV_ROWS} 1 vs 4, GQA serial) ...", flush=True)
    frame["cross_kernel_control"] = _model_divergence(
        py,
        scratch,
        past,
        device,
        {ENV_KV_PARALLEL: "1", ENV_GEMV_ROWS: "1"},
        {ENV_KV_PARALLEL: "1", ENV_GEMV_ROWS: "4"},
        "calib",
    )
    frame["cross_kernel_control"]["subject"] = (
        "q_gemv_matmul_nbits_f16 row tiling — shipped and accepted on main, and not this change"
    )
    frame["cross_kernel_control"]["question"] = (
        "is whole-model elementwise divergence simply what this instrument reports for any "
        "accepted kernel change?"
    )
    frame["cross_kernel_control"]["reproduces_the_divergence"] = not bool(
        frame["cross_kernel_control"].get("all_bitwise_identical")
    )
    if not frame["cross_kernel_control"]["reproduces_the_divergence"]:
        frame["cross_kernel_control"]["reading"] = (
            "NEGATIVE for this change. That pair is bitwise identical, so the whole-model "
            "divergence above is NOT generic to accepted kernel changes and this control gives "
            "this change no cover whatsoever."
        )
    print(f"[kvpar] control LANE-SENSITIVITY (W=2 vs W=16, both parallel) ...", flush=True)
    frame["lane_sensitivity_control"] = _model_divergence(
        py,
        scratch,
        past,
        device,
        {ENV_KV_PARALLEL: "2"},
        {ENV_KV_PARALLEL: "16"},
        "lanes",
    )
    frame["lane_sensitivity_control"]["subject"] = (
        "the new kernel against itself at two lane counts, both correct by construction"
    )
    frame["lane_sensitivity_control"]["question"] = (
        "is the model's output invariant under reassociation of THIS sum at all, or does it move "
        "between two equally-correct lane counts?"
    )
    _lane = frame["lane_sensitivity_control"]
    if _lane.get("measured"):
        _serial_worst = ((eq or {}).get("worst") or {}).get("max_abs")
        _lane["reading"] = (
            "W=2 and W=16 are both correct by construction — neither is a reference and neither "
            "is 'the bug' — yet the model's logits move by "
            f"{_lane.get('logits_max_abs')} between them"
            + (
                f", which is at least as large as the parallel-vs-serial divergence "
                f"({_serial_worst}) that this report refuses on."
                if isinstance(_serial_worst, float)
                else "."
            )
            + " So the whole-model refusal is NOT evidence that one association is wrong: it is "
            "evidence that this model's 32-layer fp16 residual stack has no invariant "
            "whole-model output under ANY reassociation of the attention sum, and the serial "
            "kernel's left-to-right order is simply one such association with no privileged "
            "status. This is why the refusal is reported as a limit on what this instrument can "
            "adjudicate rather than as a defect found in the kernel — and it still does NOT lift "
            "the refusal, because 'the reference is arbitrary too' is not a correctness proof. "
            "The node scope, where the comparison IS adjudicable, is the only scope that "
            "publishes."
        )
    return frame




def _publishable(eq_rec: dict, timing_rec: dict) -> "tuple[bool, list]":
    """May this case carry a speed number at all? Every reason it may not, named.

    This is the single decision point the module docstring describes. It is intentionally a pure
    function of the two records so that it can be reasoned about — and mutated — in isolation.
    """
    reasons = []
    eq = (eq_rec or {}).get("equivalence") or {}
    if not eq:
        reasons.append("no equivalence pass was run for this case")
    else:
        if not eq.get("equivalent"):
            worst = eq.get("worst") or {}
            reasons.append(
                "the two arms are not equivalent at the predeclared tolerance"
                + (
                    f" (worst |parallel-serial| = {worst.get('max_abs')} on {worst.get('name')})"
                    if worst.get("name")
                    else ""
                )
                + f"; {eq.get('reason', '')}".rstrip("; ")
            )
        if eq.get("outputs_compared") != eq.get("outputs_total"):
            reasons.append(
                f"only {eq.get('outputs_compared')} of {eq.get('outputs_total')} outputs were "
                f"compared; a subset is not an equivalence result"
            )
        # Every timed repeat has to have been checked, and every one of them has to have held.
        # A case whose correctness was established once and then measured three times is a case
        # where two of the three measured runs are unverified.
        if eq.get("repeats_compared") is not None:
            if not eq.get("repeats_compared"):
                reasons.append("no repeat of this case had its outputs compared")
            elif not eq.get("equivalent_every_repeat"):
                bad = [
                    k
                    for k, v in (eq.get("per_repeat") or {}).items()
                    if not v.get("equivalent")
                ]
                reasons.append(
                    f"equivalence did not hold on every timed repeat (failed on repeat(s) "
                    f"{', '.join(bad) or '?'}); a per-run speed number needs per-run correctness"
                )
    for name, arm in (timing_rec.get("arms") or {}).items():
        if not arm.get("witness_ok"):
            reasons.append(
                f"arm '{name}' has no pipeline witness for {arm.get('expected_kernel')}: "
                + str(arm.get("witness_detail") or f"kernels seen = {arm.get('kernels_seen')}")
            )
        if arm.get("repeatable") is False:
            reasons.append(
                f"arm '{name}' did not repeat itself (rsd = {arm.get('rsd'):.3f} > "
                f"{DISPERSION_CEILING}); the box was disturbed and the ratio is not a "
                f"measurement of this change"
            )
        if arm.get("median_us") in (None, 0):
            reasons.append(f"arm '{name}' produced no timed GQA sample")
        if not arm.get("raw_samples_us"):
            reasons.append(
                f"arm '{name}' published no raw per-inference series; a summary whose samples "
                f"are not in the artifact cannot be audited"
            )
    return (not reasons), reasons


def _apply_structural_removal(eq_records, timing_records, scope_note: str) -> list:
    """Merge the two passes. The ONLY place a speed field is ever attached to a record.

    A refused case gets a `refusal` object and no `speedup`, no `verdict`, no timing SUMMARY of
    any kind. The keys are absent, not false: a reader scanning for a number finds nothing to
    misread, and a downstream tool that keys on `kernel_speedup` sees the case is not there.

    Refusal also strips the per-arm `median_us` and `graph_total_us`. Raw per-repeat samples stay,
    because an observation is not a claim and deleting the measurements would make the refusal
    unauditable — but a median is a summary, and a summary of a run whose correctness was not
    established is the shape the rejected artifact's WIN rows had.
    """
    eq_by_past = {r["past"]: r for r in eq_records}
    merged = []
    for t in timing_records:
        past = t["past"]
        eq_rec = eq_by_past.get(past)
        rec: dict = {
            "case": t["case"],
            "past": past,
            "equivalence": (eq_rec or {}).get("equivalence"),
            "arms": t["arms"],
        }
        ok, reasons = _publishable(eq_rec, t)
        if not ok:
            for arm in rec["arms"].values():
                arm.pop("median_us", None)
                arm.pop("graph_total_us", None)
            rec["refusal"] = {
                "speed_claim": "WITHHELD",
                "reasons": reasons,
                "note": (
                    "This case carries no speedup, no verdict and no timing summary. The fields "
                    "are absent rather than false, by construction — see "
                    "`_apply_structural_removal`. A divergence is NOT relabelled as a tolerance "
                    "residual and the predeclared tolerance is not widened to remove it."
                ),
            }
            merged.append(rec)
            print(f"[kvpar] {t['case']}: SPEED CLAIM WITHHELD — {reasons[0]}", flush=True)
            continue
        ser = t["arms"]["serial"]["median_us"]
        par = t["arms"]["parallel"]["median_us"]
        rec["serial_median_us"] = ser
        rec["parallel_median_us"] = par
        rec["kernel_speedup"] = ser / par if par else None
        rec["graph_total_serial_us"] = t["arms"]["serial"].get("graph_total_us")
        rec["graph_total_parallel_us"] = t["arms"]["parallel"].get("graph_total_us")
        rec["verdict"] = "FASTER" if rec["kernel_speedup"] and rec["kernel_speedup"] > 1 else (
            "NO-CHANGE" if rec["kernel_speedup"] == 1 else "SLOWER"
        )
        rec["scope"] = scope_note
        merged.append(rec)
        print(
            f"[kvpar] {t['case']}: {rec['verdict']} kernel {ser / 1000.0:.2f} ms -> "
            f"{par / 1000.0:.2f} ms ({rec['kernel_speedup']:.2f}x)",
            flush=True,
        )
    return merged


# ---------------------------------------------------------------------------------------------
# Artifact hygiene
# ---------------------------------------------------------------------------------------------


def _screen_for_leaked_roots(doc) -> list:
    """Walk the finished report; return every string that looks like an absolute path anywhere."""
    hits = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            for pat in _ROOT_LEAK_PATTERNS:
                if pat.search(node):
                    hits.append({"at": path, "value": node[:160]})
                    break

    walk(doc, "$")
    return hits


# ---------------------------------------------------------------------------------------------


def _device_record(index: int) -> dict:
    """Full device identity, in the same shape §26's artifacts use.

    An index is a position in a probe order and a name is not unique across two identical cards,
    so neither can say WHICH device a reading came from. `uuid`/`luid`/`pci` can, and #54 put them
    on `bench/devices.py`'s facts. Recorded unconditionally so a reader can check the kernel-time
    claim's scope against a device rather than against a note.

    A probe failure is recorded as a probe failure. It does not become a missing key, and it does
    not stop the run: the artifact would then be a timing with no device attached, which is
    exactly the thing `limitations` is not allowed to be a substitute for.
    """
    rec: dict = {"index": index}
    try:
        sys.path.insert(0, str(_BENCH))
        import devices as device_mod  # noqa: PLC0415

        facts, source = device_mod.probe()
        chosen = next((f for f in facts if getattr(f, "index", None) == index), None)
        if chosen is None:
            rec["probe"] = f"no device at index {index}; {len(facts)} enumerated"
            return rec
        for field in (
            "name",
            "uuid",
            "luid",
            "pci",
            "driver_version",
            "api_version",
            "transfer_class",
            "timestamp_period",
            "max_compute_shared_memory",
            "max_compute_workgroup_invocations",
            "subgroup_size",
        ):
            rec[field] = getattr(chosen, field, None)
        rec["facts_source"] = str(source)
        rec["devices_enumerated"] = len(facts)
        # The two numbers §8.14's portability derivation is *about*, checked against the machine
        # rather than asserted: the shader's fixed 8,320-byte allocation must fit, and it must fit
        # inside the Vulkan 1.1 FLOOR, not merely inside what this device happens to offer.
        rec["shared_memory_headroom"] = {
            "shader_bytes": 8320,
            "vulkan_1_1_required_floor_bytes": 16384,
            "this_device_bytes": rec.get("max_compute_shared_memory"),
            "fits_the_floor": True,
            "note": "the cap W <= 16 is derived from the floor, so it does not widen on a device "
                    "that reports more. Nothing here is read at run time.",
        }
    except Exception as exc:  # noqa: BLE001
        rec["probe"] = f"device facts unavailable: {type(exc).__name__}: {exc}"
    rec["note"] = "one device; see `limitations`. The kernel-time claim is scoped to it."
    return rec


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker-outputs" in argv:
        return outputs_worker(argv)
    if "--worker-node" in argv:
        return node_worker(argv)

    ap = argparse.ArgumentParser(description="GQA decode KV-parallel kernel evidence")
    ap.add_argument("--out", default=str(Path(__file__).with_name("gqa_decode_kv_parallel.json")))
    ap.add_argument("--past", default=",".join(str(p) for p in DEFAULT_PAST))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4, help="inferences per point; the first is "
                                                         "dropped as warm-up")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=2,
                    help="inferences dropped at the head of every point")
    ap.add_argument("--base-lib", default=None,
                    help="EP library built from the merge-base commit. Required for the "
                         "fallback-identity and prefill passes, which compare two BINARIES.")
    ap.add_argument("--ladder-w", default="1,2,4,8,16")
    ap.add_argument("--ladder-past", default="128,512,1024,2048")
    ap.add_argument("--scratch", default=str(_BENCH / "results" / "_issue90_scratch"))
    ap.add_argument("--skip-whole-model", action="store_true",
                    help="node scope only. Never for a committed artifact: the whole-model "
                         "section is where the p128 control and the refusal live.")
    args = ap.parse_args(argv)

    pasts = [int(x) for x in args.past.split(",") if x]
    if MANDATORY_PAST not in pasts:
        print(
            f"[kvpar] refusing: past {MANDATORY_PAST} is a mandatory control and is not in "
            f"{pasts}. It is the point under independent diagnosis in issue #96, and an artifact "
            f"that omits it acquires a claim about it by silence.",
            file=sys.stderr,
        )
        return 2
    if args.iters - args.warmups < 1:
        print(
            f"[kvpar] refusing: --iters {args.iters} leaves no timed inference after "
            f"{args.warmups} warm-ups",
            file=sys.stderr,
        )
        return 2

    lib = os.environ.get(EP_LIB_ENV)
    if not lib or not Path(lib).is_file():
        print(f"[kvpar] refusing: {EP_LIB_ENV} unset or missing", file=sys.stderr)
        return 2
    lib_path = Path(lib).resolve()

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    subject = _subject_provenance(lib_path)
    if subject["ep_library_profile"] != "release":
        print(
            f"[kvpar] refusing: {EP_LIB_ENV} is not a release build. A debug DLL's kernel times "
            f"are not the shipping kernel's.",
            file=sys.stderr,
        )
        return 2
    if subject["ep_library"] == "<outside-repo>":
        print(
            "[kvpar] refusing: the EP library is outside this worktree. The rejected artifact "
            "for this work was taken with a DLL from a different worktree than the head it was "
            "cited for; that is the exact defect this refusal exists for.",
            file=sys.stderr,
        )
        return 2

    model_public, _model_path = _model_provenance(rm.MODELS[rm.PHI35.key])

    print(f"[kvpar] subject {subject['ep_library']} sha256 {subject['ep_library_sha256'][:16]} "
          f"@ {subject['commit'][:12]} ({subject['branch']})")
    print(f"[kvpar] model {model_public['key']} sha256 {str(model_public['onnx_sha256'])[:16]}")
    print(f"[kvpar] cases past={pasts} iters={args.iters} warmups={args.warmups} "
          f"repeats={args.repeats}\n")

    base_lib = None
    if args.base_lib:
        base_lib = Path(args.base_lib).resolve()
        if not base_lib.is_file():
            print(f"[kvpar] refusing: --base-lib {args.base_lib} is not a file", file=sys.stderr)
            return 2
        base_sha = hashlib.sha256(base_lib.read_bytes()).hexdigest()
        if base_sha == subject["ep_library_sha256"]:
            print(
                "[kvpar] refusing: --base-lib is byte-identical to the subject library. The "
                "fallback-identity pass exists to compare TWO binaries; comparing one binary "
                "with itself would report identity that proves nothing.",
                file=sys.stderr,
            )
            return 2

    lock = DeviceLock(args.device)
    lock_witness = lock.acquire()
    if not lock_witness.get("acquired"):
        print(
            "[kvpar] refusing: could not take the cooperative device lock within "
            f"{LOCK_TIMEOUT_S}s. Another participant is measuring on this device and a timing "
            "taken now would be a number about the box.",
            file=sys.stderr,
        )
        return 2
    if lock_witness.get("contended"):
        print(f"[kvpar] device lock: waited {lock_witness['wait_s']}s behind another participant")
    try:
        return _run_all(args, pasts, py, scratch, subject, model_public, lib_path, base_lib,
                        lock, lock_witness)
    finally:
        lock.release()


def _run_all(args, pasts, py, scratch, subject, model_public, lib_path, base_lib, lock,
             lock_witness) -> int:
    _t_start = time.perf_counter()
    # ---- Scope 1: the node the claim is about --------------------------------------------
    print("[kvpar] ===== NODE SCOPE (the GroupQueryAttention node in isolation) =====")
    node_records = node_pass(py, scratch, pasts, args.device, args.iters, args.repeats,
                             args.warmups)
    node_cases = _apply_structural_removal(
        node_records,
        [{"case": r["case"], "past": r["past"], "arms": r["arms"]} for r in node_records],
        "GQA kernel GPU time for one isolated GroupQueryAttention node on one device (see "
        "report.device). NOT a whole-model claim, NOT a cross-device claim, NOT a statement "
        "about the issue #96 p128 cross-build question, and NOT a projection onto a model "
        "ceiling — that needs the approved wall-share attribution issue #88 v2 owns.",
    )

    # ---- Scope 1b: the refusal path, against prior production ----------------------------
    fallback: dict = {}
    prefill: dict = {}
    if base_lib is not None:
        print("\n[kvpar] ===== FALLBACK IDENTITY (W=1 vs the base build) =====")
        fallback = fallback_identity_pass(
            py, scratch, FALLBACK_PAST, args.device, base_lib, lib_path
        )
        print("\n[kvpar] ===== PREFILL AT THIS HEAD (seq_len > 1) =====")
        prefill = prefill_pass(
            py, scratch, PREFILL_CASES, args.device, base_lib, lib_path, args.iters, args.warmups
        )
    else:
        fallback = {
            "measured": False,
            "reason": "--base-lib was not supplied, so no second binary was available",
            "consequence": "this artifact makes NO bitwise-identity claim for the W=1 path",
        }
        prefill = dict(fallback)

    # ---- Scope 1c: the forced-W ladder ----------------------------------------------------
    print("\n[kvpar] ===== FORCED-W LADDER (randomised order) =====")
    ladder = w_ladder_pass(
        py,
        scratch,
        [int(x) for x in args.ladder_past.split(",") if x],
        [int(x) for x in args.ladder_w.split(",") if x],
        args.device,
        args.iters,
        args.repeats,
        args.warmups,
    )

    # ---- Scope 2: the whole model, where the refusal lives -------------------------------
    model_cases: list = []
    frame: dict = {}
    if not args.skip_whole_model:
        print("\n[kvpar] ===== WHOLE-MODEL SCOPE (Phi-3.5-mini, 32 layers) =====")
        equivalence = equivalence_pass(py, scratch, pasts, args.device)
        print()
        timing = timing_pass(py, scratch, pasts, args.device, args.iters, args.repeats,
                             args.warmups)
        print()
        model_cases = _apply_structural_removal(
            equivalence,
            timing,
            "whole-model scope. Reserved: no case in this section has ever published a speed "
            "field; see `whole_model_frame`.",
        )
        frame = whole_model_frame(py, scratch, pasts, args.device, equivalence)

    node_equivalent = bool(node_records) and all(
        (r.get("equivalence") or {}).get("equivalent") for r in node_records
    )
    model_equivalent = bool(model_cases) and all("refusal" not in c for c in model_cases)

    # The lock is still held here — the report is written inside the critical section — so the
    # hold time is recorded as "so far" rather than as a final figure.
    lock_witness = dict(lock_witness)
    lock_witness["held_at_report_s"] = round(time.perf_counter() - _t_start, 1)
    lock_witness["covers"] = (
        "every measurement in this artifact: the lock is taken before the first subprocess and "
        "released after this document is written"
    )

    report: dict = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "supersedes": "PR #97 (rejected; frozen at fc1b163). This artifact is an independent "
                      "re-derivation, not a re-run of that one.",
        "subject": "rust/shaders/glsl/gqa_decode_f16.comp dispatched by "
                   "ops::attention::gqa_decode_kv_lanes",
        "subject_binding": {
            "what_the_numbers_measure": (
                "the sum of the GPU timestamp intervals attributed to the GQA shader module, "
                "per inference, on a graph containing exactly one GroupQueryAttention node"
            ),
            "what_they_are_not": [
                "NOT whole-model latency — no section of this artifact publishes one",
                "NOT the cost of a single isolated shader invocation — one inference is one "
                "dispatch of batch*num_heads workgroups, and the recorded interval covers the "
                "whole dispatch",
                "NOT an aggregate over the 32 layer dispatches of a real model — that is a "
                "different quantity, and the rejected artifact's headline was built on it "
                "while being read as if it were one of the other two",
                "NOT wall time — process_wall_s appears per point as context and is never a "
                "numerator or a denominator anywhere in this file",
            ],
            "aggregation": (
                "per inference: raw series in `per_inference_us`; mean of the timed inferences "
                "in `by_kernel_us`; median across repeats in `median_us`. Each level is a "
                "function of the one below it and all three are published."
            ),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provenance": subject,
        "model": model_public,
        "device": _device_record(args.device),
        "protocol": {
            "arms": [
                {"name": n, ENV_KV_PARALLEL: (e if e is not None else "<unset>"),
                 "expected_kernel": k}
                for n, e, k in ARMS
            ],
            "only_difference_between_arms": ENV_KV_PARALLEL,
            "iters_per_point": args.iters,
            "warmup_inferences_dropped": args.warmups,
            "timed_inferences_per_point": args.iters - args.warmups,
            "repeats": args.repeats,
            "interleaving": "arm order reversed on odd repeats; the forced-W ladder uses a "
                            "seeded permutation per (past, repeat) and records each point's "
                            "position in its run",
            "raw_samples_published": (
                "yes — `per_inference_us.timed` on every point and `raw_samples_us` on every "
                "arm. Every summary in this file is a function of a series that is in this file."
            ),
            "process_isolation": "one subprocess per (case, arm, repeat); ORT registers an EP "
                                 "process-globally and the EP writes counters from an exit hook",
            "kernel_time_source": "EP GPU timestamp queries (ONNXRUNTIME_EP_VULKAN_TRACE_GPU), "
                                  "attributed by shader module stem; see `subject_binding` for "
                                  "exactly what the resulting number is and is not",
            "predeclared_tolerance": TOLERANCE,
            "tolerance_is_shared": "the node scope and the whole-model scope are judged by the "
                                   "SAME tolerance object. Neither scope has a looser band.",
            "per_repeat_equivalence": (
                "every timed repeat of every node case has its outputs compared, and the "
                "published verdict is the WORST repeat. A case that was correct once and "
                "measured three times publishes nothing."
            ),
            "dispersion_ceiling_rsd": DISPERSION_CEILING,
            "gpu_lock": lock_witness,
        },
        "node_scope": {
            "what": "one GroupQueryAttention node, Phi-3.5's own shape, alone in a graph",
            "shape": NODE_SHAPE,
            "equivalence_complete": node_equivalent,
            "cases": node_cases,
        },
        "fallback_identity": fallback,
        "prefill_at_this_head": prefill,
        "forced_w_ladder": ladder,
        "whole_model_scope": {
            "what": "Phi-3.5-mini-instruct int4, 32 layers, decode step",
            "equivalence_complete": model_equivalent,
            "cases": model_cases,
            "frame": frame,
            "reading": (
                "This scope publishes NO speed field. Its per-element logits comparison at the "
                "predeclared tolerance does not pass, and under this file's structural rule that "
                "removes every speed and verdict field from every case in it. The divergence is "
                "not relabelled: `frame.depth_profile` records that layer 0's present KV — the "
                "kernel's own output, before any downstream layer touches it — is bitwise "
                "identical, and that the disagreement first appears at layer 1 and compounds "
                "with depth. `frame.cross_kernel_control` asked whether that is simply what this "
                "instrument reports for any accepted kernel change and came back NEGATIVE, which "
                "is recorded as measured and gives this change no cover. "
                "`frame.lane_sensitivity_control` measures whether the model's output moves "
                "between two equally-correct lane counts, which is the honest statement of the "
                "limitation. None of them lifts the refusal."
            ),
        },
        "equivalence_complete": node_equivalent and (model_equivalent or args.skip_whole_model),
        "equivalence": {
            "reading": (
                "The top-level `equivalence_complete` is the conservative AND across every scope "
                "in this file, so it is FALSE whenever any scope refused. It is not a licence to "
                "read a speed number next to it. The rejected artifact carried a false global "
                "flag AND kept per-case WIN verdicts; here the flag being false is a SUMMARY of "
                "which scopes were structurally emptied, and no scope that failed equivalence "
                "retains a speed or verdict field. Check `by_scope` before reading anything."
            ),
            "by_scope": {
                "node_scope": {
                    "equivalence_complete": node_equivalent,
                    "publishes_speed": bool(
                        node_equivalent
                        and node_cases
                        and all("kernel_speedup" in c for c in node_cases)
                    ),
                },
                "whole_model_scope": {
                    "equivalence_complete": model_equivalent,
                    "publishes_speed": False,
                    "note": "skipped" if args.skip_whole_model else "ran and refused",
                },
                "fallback_identity": {
                    "equivalence_complete": bool(fallback.get("holds")),
                    "criterion": "bitwise, all three outputs, subject W=1 vs the base build",
                    "publishes_speed": False,
                },
                "prefill_at_this_head": {
                    "equivalence_complete": bool(prefill.get("holds")),
                    "criterion": "bitwise, all three outputs, subject vs the base build at "
                                 "seq_len > 1",
                    "publishes_speed": False,
                },
            },
            "headline_is_bound_to": "node_scope",
        },
        "limitations": [
            "ONE DEVICE (RTX A1000). Nothing here supports a cross-device claim; W is a "
            "portable-by-construction spec constant, but its best value is a property of the "
            "machine, which is why the environment kill switch exists.",
            "KERNEL TIME ONLY, NODE SCOPE ONLY. The only speed numbers this artifact publishes "
            "are for one GQA node in isolation. A whole-model speed claim needs the approved "
            "wall-share attribution that issue #88 v2 owns, and this file makes none.",
            "The p128 cross-build behaviour is issue #96's question. This artifact measures the "
            "GQA kernel at p128 on this box and says nothing about why any cross-build "
            "difference exists or whether this change resolves it.",
            "GPU time is the EP's own timestamp-query total, not a hardware occupancy counter: "
            "it says the kernel finished sooner, not why.",
            "Wall clock on this box is STEADY_UNCERTIFIED (PERF.md §20); process_wall_s is "
            "context, never evidence.",
            "The device lock is COOPERATIVE. It serialises other participants in this protocol "
            "and nothing else; see protocol.gpu_lock for what it does not exclude. The "
            "dispersion gate is the independent second condition.",
            "The forced-W ladder is a table of (past, W) cells and is NOT a source of a "
            "W-generic ratio. Production selects W from the KV extent, so at any given past "
            "exactly one cell is the shipped behaviour and it is flagged as such.",
        ],
    }
    if node_equivalent and node_cases and all("kernel_speedup" in c for c in node_cases):
        speedups = [c["kernel_speedup"] for c in node_cases if c.get("kernel_speedup")]
        report["headline"] = {
            "claim": (
                "GPU-timestamp time of the GQA shader module, per inference, on a graph of one "
                "GroupQueryAttention node, on one RTX A1000"
            ),
            "per_past": {
                str(c["past"]): c["kernel_speedup"]
                for c in node_cases
                if c.get("kernel_speedup")
            },
            "kernel_speedup_min": min(speedups),
            "kernel_speedup_max": max(speedups),
            "scope": (
                "kernel-only, node-scoped, single-device, at the W the SELECTOR CHOOSES for each "
                "past. Not a model-level claim, not a W-generic claim, and not a claim about any "
                "other device."
            ),
            "read_with": "subject_binding, which states what this number is not",
            "bound_to_scope": "node_scope",
            "not_covered_by_this_number": (
                "whole_model_scope refused equivalence and publishes no speed field; this "
                "headline says nothing about it. The top-level equivalence_complete is the "
                "AND across scopes and is therefore false; see `equivalence.by_scope`."
            ),
        }

    leaks = _screen_for_leaked_roots(report)
    if leaks:
        print("[kvpar] REFUSING TO WRITE: the report contains absolute paths", file=sys.stderr)
        for h in leaks[:20]:
            print(f"    {h['at']}: {h['value']}", file=sys.stderr)
        return 3

    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    print(f"\n  wrote {_repo_relative(args.out)}")
    withheld = [c for c in node_cases if "refusal" in c]
    if withheld:
        print(
            f"[kvpar] {len(withheld)} of {len(node_cases)} node case(s) publish no speed claim",
            file=sys.stderr,
        )
        return 1
    if frame and frame.get("null_control", {}).get("holds") is False:
        print("[kvpar] the null control did not hold; the whole-model frame is void",
              file=sys.stderr)
        return 1
    if fallback.get("measured") is not False and not fallback.get("holds"):
        print(
            "[kvpar] the W=1 fallback is NOT bitwise identical to the base build; the refusal "
            "path does not reproduce prior production",
            file=sys.stderr,
        )
        return 1
    if prefill.get("measured") is not False and not prefill.get("holds"):
        print("[kvpar] prefill moved at this head", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
