#!/bin/bash

# Checks the line coverage of an lcov file, counting each source line once.
#
# Usage: zscripts/check-coverage.bash [lcov-file] [min-percentage]
#
# Why this exists instead of `cargo llvm-cov --fail-under-lines`:
#
# llvm-cov's own summary counts a line once per monomorphization. A function that
# dispatches on a const generic keeps every branch in every instantiation, so the
# branches belonging to another dimension are unreachable in this one -- they are a
# different function -- and the percentage ends up describing the instantiations
# rather than the source. The `DA:` records of the lcov export are already merged
# per source line (a line is hit if any instantiation hit it), which is what a
# line-coverage gate is meant to measure. `LF:`/`LH:` are NOT usable here:
# cargo-llvm-cov writes them from the instantiation-based summary, so they carry
# the artifact.
#
# Each `DA:<line>,<hits>` may appear once per monomorphization; the maximum wins.
#
# What remains uncovered is not a target to grind down: it is recorded, file by
# file, in AGENTS.md, with the reason each gap exists.

set -euo pipefail

FILE="${1:-lcov.info}"
MIN="${2:-92}"

if [ ! -f "$FILE" ]; then
    echo "error: coverage file not found: $FILE" >&2
    exit 2
fi

awk -v min="$MIN" '
    /^SF:/ {
        file = substr($0, 4)
        seen[file] = 1
        next
    }
    /^DA:/ {
        split(substr($0, 4), part, ",")
        line = part[1] + 0
        hits = part[2] + 0
        key = file SUBSEP line
        if (!(key in max_hits) || hits > max_hits[key]) {
            max_hits[key] = hits
        }
        next
    }
    END {
        total = 0
        covered = 0
        for (key in max_hits) {
            total++
            if (max_hits[key] > 0) {
                covered++
            }
        }
        if (total == 0) {
            print "error: no DA records found -- is this an lcov file?" > "/dev/stderr"
            exit 2
        }

        # per-file totals, to name what is low when the check fails
        for (key in max_hits) {
            split(key, part, SUBSEP)
            f = part[1]
            file_total[f]++
            if (max_hits[key] > 0) {
                file_covered[f]++
            }
        }

        pct = 100.0 * covered / total
        printf "line coverage (each source line counted once): %d/%d = %.2f%% (minimum %.2f%%)\n", covered, total, pct, min

        if (pct + 1e-9 < min) {
            print "files below the minimum:" > "/dev/stderr"
            n = 0
            for (f in file_total) {
                fpct = 100.0 * file_covered[f] / file_total[f]
                if (fpct < min) {
                    printf "  %6.2f%%  %5d/%5d  %s\n", fpct, file_covered[f], file_total[f], f > "/dev/stderr"
                    n++
                }
            }
            printf "(%d file(s) below %.2f%%; the check is on the crate total)\n", n, min > "/dev/stderr"
            exit 1
        }
    }
' "$FILE"
