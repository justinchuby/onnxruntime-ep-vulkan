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

1. A condensing commit DECLARES itself by writing a marker line,
   `<!-- CONDENSED-AT: <base-blob-sha1> -->`, where `<base-blob-sha1>` is
   the git blob hash of the file's content immediately before condensation
   (i.e. the pre-condensation content, hashed the same way `git hash-object`
   hashes a blob: `sha1("blob " + len(content) + "\0" + content)`). A side
   carrying a marker whose hash matches this merge's actual base content is
   an *authoritative* condensation.
2. Per Morpheus's §8.9.21 part 5 ruling: a condensing commit that both
   condenses AND appends new lines in the same commit can end up *longer*
   than the merge base while still having deleted lines -- the old
   `len(ours) < len(base)` heuristic would then miss it and silently fall
   to the plain-append branch, resurrecting exactly what it was meant to
   protect, with no error anywhere. The marker fixes this: authoritativeness
   is keyed on the declaration, never on length alone.
3. If neither side carries an authoritative marker, length comparison is
   retained as a fallback/disagreement assertion (a second witness, free) --
   whichever side is shorter than the base is treated as a condensation,
   same as the original heuristic-only behaviour. If a marker's hash does
   NOT match the base (stale marker: the file changed underneath it, e.g.
   another commit landed between condensation and merge), the driver does
   not trust it and falls back to the length heuristic instead, logging the
   disagreement.
4. Whichever side is deemed authoritative (by marker or by fallback length)
   becomes the skeleton (its line order/removals kept as-is). Lines from the
   *other* side that are genuinely new (i.e. not present in the merge base)
   are appended -- these are real concurrent agent writes, never lines the
   condensation intentionally removed, because removed lines are by
   definition already in the base set and therefore excluded.
5. If neither side is condensed by marker or by length (the common case: no
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
import hashlib
import re
import sys

MARKER_RE = re.compile(r"<!--\s*CONDENSED-AT:\s*([0-9a-fA-F]{40})\s*-->")


def read_lines(path):
    with open(path, "r", encoding="utf-8", errors="surrogateescape", newline="") as f:
        return f.readlines()


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
        f.writelines(lines)


def git_blob_sha1(lines):
    """Hash file content exactly as `git hash-object` hashes a blob."""
    content = "".join(lines).encode("utf-8", errors="surrogateescape")
    header = ("blob %d\0" % len(content)).encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def declared_marker_hash(lines):
    """Return the base-blob-sha1 named by a CONDENSED-AT marker, if any."""
    for line in lines:
        m = MARKER_RE.search(line)
        if m:
            return m.group(1)
    return None


def main(argv):
    if len(argv) < 4:
        sys.stderr.write("history_merge_driver: expected %O %A %B %P\n")
        return 1

    base_path, ours_path, theirs_path, orig_path = argv[1], argv[2], argv[3], argv[4] if len(argv) > 4 else "?"

    base = read_lines(base_path)
    ours = read_lines(ours_path)
    theirs = read_lines(theirs_path)
    base_set = set(base)
    base_hash = git_blob_sha1(base)

    # Authoritative signal: a declared CONDENSED-AT marker whose named
    # base-blob-sha1 matches this merge's actual base content. Per
    # Morpheus's §8.9.21 part 5 ruling, this is keyed on declaration, not
    # length, so a condense-and-append-in-one-commit still registers.
    ours_marker = declared_marker_hash(ours)
    theirs_marker = declared_marker_hash(theirs)
    ours_declared = ours_marker == base_hash
    theirs_declared = theirs_marker == base_hash

    # Fallback signal: length heuristic, retained only as a disagreement
    # assertion / second witness when no side has an authoritative marker,
    # or when a marker is present but stale (doesn't match this base).
    ours_shorter = len(ours) < len(base)
    theirs_shorter = len(theirs) < len(base)

    if ours_declared or theirs_declared:
        ours_condensed, theirs_condensed = ours_declared, theirs_declared
        basis = "marker"
        if (ours_marker and not ours_declared) or (theirs_marker and not theirs_declared):
            sys.stderr.write(
                "history_merge_driver: %s -> stale CONDENSED-AT marker found "
                "(does not match this merge's base blob %s); ignored, marker "
                "side not trusted on this basis alone\n" % (orig_path, base_hash)
            )
        if bool(ours_shorter) != bool(ours_declared) or bool(theirs_shorter) != bool(theirs_declared):
            sys.stderr.write(
                "history_merge_driver: %s -> length heuristic disagrees with "
                "declared marker (ours_shorter=%s theirs_shorter=%s vs "
                "ours_declared=%s theirs_declared=%s); marker wins\n"
                % (orig_path, ours_shorter, theirs_shorter, ours_declared, theirs_declared)
            )
    else:
        ours_condensed, theirs_condensed = ours_shorter, theirs_shorter
        basis = "length-fallback"

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
        "condensed_side=%s, basis=%s)\n"
        % (
            orig_path,
            len(result),
            len(base),
            len(ours),
            len(theirs),
            "ours" if ours_condensed else ("theirs" if theirs_condensed else "none"),
            basis,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
