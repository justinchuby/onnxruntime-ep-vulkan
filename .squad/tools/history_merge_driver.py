#!/usr/bin/env python3
"""Custom git merge driver for append-only squad state files.

Why this exists
----------------
`.squad/agents/*/history.md` used to be `merge=union`: git's built-in union
driver produces the set of *unique lines* present on either side of a merge.
That is safe for the case it was chosen for -- two agent branches appending
new sessions to the same file concurrently -- because appended lines from
both sides survive with no conflict.

It is NOT safe for Scribe's history condensation. Condensation *deletes*
lines from history.md on main (folding old sessions into a shorter summary).
Plain union merge has no concept of a deletion: when a branch that forked
*before* the condensation later merges, its copy of history.md still
contains every deleted line, and union merge faithfully reports the union of
"lines in the condensed version" and "lines in the stale branch version" --
which resurrects everything the condensation removed. This happened for real
on 2026-08-03: `switch/history.md` was condensed 92,884 -> 13,914 bytes, and
the next `Merge branch 'squad/switch'` put it back to 106,797+ bytes.

What this driver does instead
------------------------------
Same concurrency guarantee as union merge (no conflicts, safe for parallel
agent appends), but respects a condensation:

1. Compute the 3-way merge base's line set.
2. Whichever side (ours/theirs) is *shorter* than the base is treated as a
   condensation and becomes the authoritative skeleton (its line order/
   removals are kept as-is).
3. Lines from the *other* side that are genuinely new (i.e. not present in
   the merge base) are appended -- these are real concurrent agent writes,
   never lines the condensation intentionally removed, because removed
   lines are by definition already in the base set and therefore excluded.
4. If neither side is shorter than the base (the common case: no
   condensation happened, both sides just appended), the behaviour degrades
   to ordinary union-of-new-lines, matching the previous `merge=union`
   result for that case.

Usage (invoked by git via .gitattributes `merge=squad-history` +
.git/config `merge.squad-history.driver`):
    history_merge_driver.py %O %A %B %P
      %O = ancestor (merge-base) temp file
      %A = ours temp file -- the driver MUST write the merged result here
      %B = theirs temp file
      %P = original path, for logging only

Exits 0 (success, no conflict) unless the temp files can't be read.
"""
import sys


def read_lines(path):
    with open(path, "r", encoding="utf-8", errors="surrogateescape", newline="") as f:
        return f.readlines()


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
        f.writelines(lines)


def main(argv):
    if len(argv) < 4:
        sys.stderr.write("history_merge_driver: expected %O %A %B %P\n")
        return 1

    base_path, ours_path, theirs_path, orig_path = argv[1], argv[2], argv[3], argv[4] if len(argv) > 4 else "?"

    base = read_lines(base_path)
    ours = read_lines(ours_path)
    theirs = read_lines(theirs_path)
    base_set = set(base)

    ours_condensed = len(ours) < len(base)
    theirs_condensed = len(theirs) < len(base)

    if ours_condensed and not theirs_condensed:
        result = list(ours)
        new_side = theirs
    elif theirs_condensed and not ours_condensed:
        result = list(theirs)
        new_side = ours
    else:
        # No condensation on either side (or both -- ambiguous, prefer ours
        # as the skeleton): fall back to ordinary union-of-new-lines, same
        # shape as the old merge=union behaviour agents rely on today.
        result = list(ours)
        new_side = theirs

    result_set = set(result)
    for side in (new_side, ours if new_side is not ours else theirs):
        for line in side:
            if line not in base_set and line not in result_set:
                result.append(line)
                result_set.add(line)

    write_lines(ours_path, result)
    sys.stderr.write(
        "history_merge_driver: %s -> %d lines (base=%d ours=%d theirs=%d, "
        "condensed_side=%s)\n"
        % (
            orig_path,
            len(result),
            len(base),
            len(ours),
            len(theirs),
            "ours" if ours_condensed else ("theirs" if theirs_condensed else "none"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
