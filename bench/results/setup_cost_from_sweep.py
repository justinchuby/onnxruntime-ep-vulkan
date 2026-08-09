"""Host-side setup cost, read off the sweep's own 54 processes. No new run.

Every process in `crossbuild_decode_window.json` recorded `session_build_ms` (session
construction: translation, `gqa_local_size`, pipeline creation) and `first_run_ms` (the
verification inference) before any timed iteration. Those two fields are the only place the
`c96e7d9 -> 85fbda2` host delta could hide if it is not in device time:

  * `gqa_local_size` + the `std::env::var` lookup run at translate time -> `session_build_ms`
  * `vkCreateComputePipelines` with a `VkSpecializationInfo` vs without -> `session_build_ms`

Both arms of every (workload, repeat) are paired here exactly as the latency ratio is. Refused
records are included: a refusal is about output equivalence, not about how long the session took
to build, and excluding them would silently change the population.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ART = Path(sys.argv[1] if len(sys.argv) > 1
           else "bench/results/crossbuild_decode_window.json")


def main() -> int:
    art = json.loads(ART.read_text(encoding="utf-8"))
    rows = {}
    for r in art["records"]:
        rows.setdefault((r["workload"], r["repeat"]), {})[r["arm"]] = r

    print(f"{'workload':46s} {'n':>2s} {'base_build':>11s} {'cand_build':>11s} "
          f"{'delta_ms':>9s} {'base_first':>11s} {'cand_first':>11s} {'delta_ms':>9s}")
    all_build, all_first = [], []
    for wl in dict.fromkeys(r["workload"] for r in art["records"]):
        b, c, fb, fc = [], [], [], []
        for (w, _rep), arms in rows.items():
            if w != wl or len(arms) != 2:
                continue
            b.append(arms["baseline"].get("session_build_ms"))
            c.append(arms["candidate"].get("session_build_ms"))
            fb.append(arms["baseline"].get("first_run_ms"))
            fc.append(arms["candidate"].get("first_run_ms"))
        b = [x for x in b if x is not None]
        c = [x for x in c if x is not None]
        fb = [x for x in fb if x is not None]
        fc = [x for x in fc if x is not None]
        if not (b and c):
            continue
        db = statistics.median(c) - statistics.median(b)
        df = (statistics.median(fc) - statistics.median(fb)) if (fb and fc) else float("nan")
        all_build.append(db)
        all_first.append(df)
        print(f"{wl[:46]:46s} {len(b):2d} {statistics.median(b):11.2f} "
              f"{statistics.median(c):11.2f} {db:+9.2f} "
              f"{statistics.median(fb) if fb else float('nan'):11.2f} "
              f"{statistics.median(fc) if fc else float('nan'):11.2f} {df:+9.2f}")
    print(f"\n  median session_build delta over workloads: {statistics.median(all_build):+.2f} ms")
    print(f"  median first_run      delta over workloads: {statistics.median(all_first):+.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
