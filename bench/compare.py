"""Diff two benchmark result JSONs (base vs PR) into a Markdown perf report.

Differs from the MLX EP's ``compare.py`` in one way that matters: **a delta is only flagged
when it is larger than the noise of the two samples it came from**. Both sides carry a robust
relative spread (``rsd``) from ``stats.py``, and ``significant()`` requires the delta to exceed
both the threshold and twice the worse spread. A harness that flags noise gets ignored, and an
ignored harness is worse than none.

Also flagged, and ranked above any timing change:

* a case that **newly falls back to the CPU EP** — the loudest possible perf regression, and
  the one a wall-time table hides, because CPU fallback is always numerically correct;
* an environment mismatch between the two runs (different device, driver, OS or build profile),
  which makes the whole comparison meaningless and says so at the top instead of quietly
  producing a table.

Usage::

    python bench/compare.py --base base.json --pr pr.json [--threshold 0.10] > comment.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from stats import Sample, relative_delta, significant  # noqa: E402

import portability  # noqa: E402

MARKER = "<!-- vulkan-ep-bench -->"


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text("utf-8"))


def _sample(row: "dict | None", key: str) -> "Sample | None":
    if not row or not row.get(key):
        return None
    d = row[key]
    s = Sample(name=d.get("name", ""), samples=list(d.get("samples_ms") or []))
    if not s.samples:
        # No raw samples in the JSON: reconstruct just enough for median/spread comparisons.
        # Two points at median ± mad reproduce the median exactly and the MAD approximately,
        # which is all `significant()` reads. Marked so nothing else relies on it.
        med, mad = d.get("median_ms"), d.get("mad_ms") or 0.0
        if med is None:
            return None
        s.samples = [med - mad, med, med + mad]
    return s


def _env_fingerprint(data: dict) -> dict:
    env = data.get("environment", {})
    host, build = env.get("host", {}), env.get("build", {})
    return {
        "os": f"{host.get('os')} {host.get('release')}",
        "cpu": host.get("cpu"),
        "onnxruntime": host.get("onnxruntime"),
        "profile": build.get("profile"),
        "devices": [d.get("line") for d in env.get("devices", [])],
    }


def device_identity(data: dict) -> "tuple[str | None, str]":
    """``(fingerprint, human)`` for the device a result file was taken on."""
    fp = data.get("device_fingerprint")
    dev = data.get("device") or {}
    human = (
        f"{dev.get('name', '?')} [{dev.get('transfer_class', '?')}, "
        f"period {dev.get('timestamp_period_ns', '?')} ns/tick, "
        f"{(dev.get('max_compute_shared_memory') or 0) // 1024} KiB shared, "
        f"driver {dev.get('driver_version', '?')}]"
        if dev
        else "unidentified device"
    )
    return fp, human


def cross_device_refusal(base: dict, pr: dict) -> "str | None":
    """Return a refusal message when the two runs are not from the same device.

    Hard refusal, not a warning banner. Two devices on this very machine differ in
    `timestampPeriod` by 52x, in shared memory by 1.5x and in transfer class entirely; a table
    of deltas across them is not a degraded comparison, it is a category error. Making it
    *structurally impossible* is the point — a convention would be followed until the first time
    somebody was in a hurry.
    """
    bfp, bh = device_identity(base)
    pfp, ph = device_identity(pr)
    if bfp is None or pfp is None:
        return (
            "one or both result files do not identify the device they ran on "
            f"(base: {bh}; pr: {ph}). Re-run with `--device N`; a benchmark that cannot name "
            "its device cannot be compared to anything."
        )
    if bfp != pfp:
        return (
            "these two runs are from different devices and are not comparable:\n"
            f"    base: {bh}\n            {bfp}\n"
            f"    pr:   {ph}\n            {pfp}\n"
            "Benchmark the same device on both sides. If you meant to compare two devices, "
            "that is a device study, not a regression check — pass --cross-device-study, "
            "which labels every row with its device and prints no verdict."
        )
    return None


def producer_identity(record: dict) -> "tuple[tuple[str, ...], str]":
    """Return ``(fingerprints, human)`` describing who built the graphs in ``record``.

    Falls back to scanning the case rows, so a result file written before ``producers`` existed
    still reports honestly (as unrecorded) rather than silently comparing as if it matched.
    """
    fps = [p.get("fingerprint") for p in record.get("producers", []) if p.get("fingerprint")]
    if not fps:
        fps = [
            c["producer_fingerprint"]
            for c in record.get("cases", [])
            if c.get("producer_fingerprint")
        ]
    uniq = tuple(sorted(set(fps)))
    return uniq, (", ".join(uniq) if uniq else "unrecorded producer")


def producer_refusal(base: dict, pr: dict) -> "str | None":
    """Return a refusal message when the two runs did not benchmark the same graphs.

    Mouse's ``OP_COVERAGE.md`` §4.18 rule — op coverage is relative to a producer, not to a model
    architecture — applies to timings with more force, because a timing has no shape to disagree
    about and so nothing fails loudly. Justin's ``mobius`` emits ``ai.onnx::Attention`` @ 23,
    ``RMSNormalization`` and ``RotaryEmbedding``; the ORT GenAI builder emits the
    ``com.microsoft`` contrib equivalents. Those graphs partition differently, claim differently
    and run differently. A delta between them is a fact about two exporters, and reading it as a
    regression in this repository would be wrong in a way that looks entirely reasonable.

    Unrecorded producers refuse too. "We do not know what built these" is not evidence that the
    same thing built both.
    """
    b_fps, b_human = producer_identity(base)
    p_fps, p_human = producer_identity(pr)
    if not b_fps or not p_fps:
        return (
            f"producer not recorded (base: {b_human}; pr: {p_human}). A benchmark artefact is "
            f"relative to its producer (OP_COVERAGE.md §4.18); two runs whose graph origin is "
            f"unknown cannot be assumed to have benchmarked the same graph. Re-run with a "
            f"harness that records producers."
        )
    if b_fps != p_fps:
        return (
            f"different producers: base built by [{b_human}], pr built by [{p_human}]. Different "
            f"exporters emit different op sets for the same architecture, so this delta is a "
            f"property of the exporters, not of the change under review — pass "
            f"--cross-producer-study, which labels the columns and prints no verdict."
        )
    return None


def producer_mismatch(b: dict, p: dict) -> bool:
    """True when two rows with the same case name were built by different producers.

    As with ``tile_config``, two unrecorded producers mean *unknown*, never *equal*.
    """
    bp, pp = b.get("producer_fingerprint"), p.get("producer_fingerprint")
    return bp is not None and pp is not None and bp != pp


def tile_mismatch(b: dict, p: dict) -> bool:
    """True when two rows were produced by demonstrably different kernels.

    A tile configuration is part of the identity of the kernel. Two ``None``s mean *unknown*,
    which is not the same as *equal* — so an unknown tile config never certifies a comparison,
    it just cannot disprove one.
    """
    bt, pt = b.get("tile_config"), p.get("tile_config")
    return bt is not None and pt is not None and bt != pt


def portability_banner(pr: dict) -> "list[str]":
    """Warn when the PR's numbers came from configurations no floor device could select.

    Not a refusal: measuring a 48 KiB tile on the 4060 is legitimate and useful. But a reader
    scanning a table has no way to tell a number that describes the EP from one that describes
    this desk, and Justin's standing directive (cross-platform generality at all times) is exactly
    that this must be structural rather than remembered. So the table says which it is.
    """
    rows = pr.get("cases", [])
    verdicts = [c.get("portability", {}).get("verdict") for c in rows]
    needs = sum(1 for v in verdicts if v == portability.NEEDS_FALLBACK)
    unknown = sum(1 for v in verdicts if v == portability.UNKNOWN or v is None)
    if not needs and not unknown:
        return []
    out = []
    if needs:
        out.append(
            f"> 🌍 **{needs} of {len(rows)} rows used a configuration above the §7.2 admission "
            f"floor** ({portability.FLOOR_SHARED_MEMORY_BYTES} B shared, "
            f"{portability.FLOOR_WORKGROUP_INVOCATIONS} invocations). Those numbers describe a "
            f"path a device sitting on the floor cannot take; the floor-compliant fallback must "
            f"be measured before any of this is quoted as the EP's behaviour."
        )
    if unknown:
        out.append(
            f"> 🌍 **{unknown} of {len(rows)} rows do not record the configuration that produced "
            f"them.** The engine does not report tile shape or workgroup size yet. Unknown is not "
            f"portable and is not quotable — it is just unknown."
        )
    return out + [""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--pr", required=True)
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="relative delta to flag, e.g. 0.10 for 10%%")
    ap.add_argument("--fail-on-regression", action="store_true")
    ap.add_argument("--cross-device-study", action="store_true",
                    help="the two files are deliberately from different devices; print a "
                         "labelled side-by-side with no regression verdict")
    ap.add_argument("--cross-producer-study", action="store_true",
                    help="the two files were deliberately built by different model producers "
                         "(e.g. mobius vs the ORT GenAI builder); print a labelled side-by-side "
                         "with no regression verdict")
    args = ap.parse_args()

    base, pr = _load(args.base), _load(args.pr)

    refusal = cross_device_refusal(base, pr)
    if refusal and not args.cross_device_study:
        print(f"{MARKER}\n## 🏎️ Vulkan EP benchmark\n\n⛔ **Comparison refused.**\n\n"
              f"```\n{refusal}\n```")
        print(f"⛔ comparison refused: {refusal}", file=sys.stderr)
        return 2

    prod_refusal = producer_refusal(base, pr)
    if prod_refusal and not args.cross_producer_study:
        print(f"{MARKER}\n## 🏎️ Vulkan EP benchmark\n\n⛔ **Comparison refused.**\n\n"
              f"```\n{prod_refusal}\n```")
        print(f"⛔ comparison refused: {prod_refusal}", file=sys.stderr)
        return 2

    bi = {c["name"]: c for c in base.get("cases", [])}
    pi = {c["name"]: c for c in pr.get("cases", [])}

    out = [MARKER, "## 🏎️ Vulkan EP benchmark", ""]

    if args.cross_device_study:
        _, bh = device_identity(base)
        _, ph = device_identity(pr)
        out += [
            "> 📐 **Device study, not a regression check.** The two columns are different "
            "devices, so a delta between them is a property of the hardware, not of the code. "
            "No verdict is issued.",
            "",
            f"* left  — {bh}",
            f"* right — {ph}",
            "",
        ]
    else:
        _, dh = device_identity(base)
        out += [f"Device: **{dh}**", ""]

    if args.cross_producer_study:
        _, bph = producer_identity(base)
        _, pph = producer_identity(pr)
        out += [
            "> 🏭 **Producer study, not a regression check.** The two columns were built by "
            "different exporters, which emit different op sets for the same architecture "
            "(`OP_COVERAGE.md` §4.18). A delta here is a property of the exporters. No verdict "
            "is issued.",
            "",
            f"* left  — built by {bph}",
            f"* right — built by {pph}",
            "",
        ]
    else:
        _, ph = producer_identity(base)
        out += [f"Producer: **{ph}**", ""]

    out += portability_banner(pr)

    if base.get("barrier_backend") != pr.get("barrier_backend"):
        out += [
            f"> ⚠️ **Different barrier backends** (base `{base.get('barrier_backend')}` vs PR "
            f"`{pr.get('barrier_backend')}`). These are two different programs; the delta is "
            "not attributable to the change under review.",
            "",
        ]

    fb, fp = _env_fingerprint(base), _env_fingerprint(pr)
    if fb != fp:
        out += [
            "> ⚠️ **The two runs were not taken on the same machine/build.** Any delta below "
            "mixes a code change with an environment change and should not be read as a "
            "regression or an improvement.",
            "",
            "```diff",
            f"- base: {json.dumps(fb)}",
            f"+ pr:   {json.dumps(fp)}",
            "```",
            "",
        ]

    rows, regressions, improvements, fallbacks, noisy = [], 0, 0, 0, 0
    for name in sorted(set(bi) | set(pi)):
        b, p = bi.get(name), pi.get(name)
        if b is None:
            rows.append((name, "—", "new", float("nan"), "🆕 new case", 1e9))
            continue
        if p is None:
            rows.append((name, "removed", "—", float("nan"), "🗑 removed", -1e9))
            continue

        bs, ps = _sample(b, "vulkan"), _sample(p, "vulkan")
        b_claimed = b.get("claim", {}).get("claimed")
        p_claimed = p.get("claim", {}).get("claimed")

        note, delta = "", float("nan")
        if producer_mismatch(b, p):
            note = "🏭 different producer — different graph, not comparable"
            delta = float("nan")
        elif tile_mismatch(b, p):
            note = "🧩 different tile config — different kernel, not comparable"
            delta = float("nan")
        elif b_claimed and not p_claimed:
            note = "⛔ now falls back to CPU"
            fallbacks += 1
            delta = 1e6  # sort to the top
        elif not b_claimed and p_claimed:
            note = "🎉 now claimed by the EP (was CPU fallback)"
        elif bs and ps:
            delta = relative_delta(bs, ps)
            if significant(bs, ps, args.threshold):
                if delta > 0:
                    note, _ = "🔴 regression", regressions
                    regressions += 1
                else:
                    note = "🟢 improvement"
                    improvements += 1
            elif abs(delta) > args.threshold:
                note = "◻️ within noise (delta < 2x sample spread)"
                noisy += 1

        rows.append((
            name,
            f"{bs.median:.3f}" if bs else "—",
            f"{ps.median:.3f}" if ps else "—",
            delta,
            note,
            delta if delta == delta else 0.0,
        ))

    rows.sort(key=lambda r: r[5], reverse=True)

    parts = []
    if fallbacks:
        parts.append(f"**{fallbacks} new CPU fallback(s)** ⛔")
    if regressions:
        parts.append(f"{regressions} regression(s) 🔴")
    if improvements:
        parts.append(f"{improvements} improvement(s) 🟢")
    if noisy:
        parts.append(f"{noisy} change(s) inside the noise ◻️")
    out.append(f"Median `session.run` (Vulkan EP, end-to-end host latency) — "
               f"{'side-by-side; no verdict' if (args.cross_device_study or args.cross_producer_study) else (' · '.join(parts) if parts else 'no significant change')} "
               f"(threshold ±{args.threshold * 100:.0f}%, noise-gated).")
    out += ["", "| Case | base ms | PR ms | Δ% | note |", "|------|--------:|------:|---:|------|"]
    for name, bms, pms, delta, note, _ in rows:
        d = f"{delta * 100:+.1f}%" if delta == delta and abs(delta) < 1e5 else "—"
        out.append(f"| `{name}` | {bms} | {pms} | {d} | {note} |")

    out += [
        "",
        "<sub>End-to-end host latency (upload + record/replay + submit + fence wait + "
        "readback) — **not** GPU kernel time. Cases marked ⛔ ran on the CPU EP and are not "
        "Vulkan numbers. Deltas are flagged only when they exceed twice the samples' own "
        f"spread. base `{base.get('label')}` · PR `{pr.get('label')}` · "
        f"{pr.get('iters')} iters · ORT "
        f"{pr.get('environment', {}).get('host', {}).get('onnxruntime')}.</sub>",
    ]
    print("\n".join(out))

    if args.fail_on_regression and not (args.cross_device_study or args.cross_producer_study) and (regressions or fallbacks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
