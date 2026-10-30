# Chapter 21 — TRACK_MATCH_STATS=1: which rewrite rules fired, how often, and what they cost
# Run: DEV=CPU python3 examples/21_match_stats.py
#
# With TRACK_MATCH_STATS>=1, PatternMatcher *is* TrackedPatternMatcher (uop/ops.py). For every UPat it counts
# [matches, attempts, time spent in failed attempts, time spent in successful matches]; an atexit hook prints the table
# (PRINT_MATCH_STATS defaults to on unless VIZ is set). Columns:  matches / attempts -- match_ms / total_ms -- file:line  rule
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
env = {**os.environ, "DEV": "CPU", "TRACK_MATCH_STATS": "1", "NO_COLOR": "1"}
p = subprocess.run([sys.executable, os.path.join(HERE, "21_debug_target.py")], env=env, capture_output=True, text=True)
lines = p.stdout.splitlines()
rows = [l for l in lines if re.match(r"^\s*\d+ /\s+\d+ --", l)]
print(lines[0])                                        # the program's own output
print(f"{len(rows) - 1} rules were tried at least once. The 12 most expensive (the table is sorted by total time):")
for l in rows[-13:]: print(l[:150])
print(lines[-1])
by_count = sorted(rows[:-1], key=lambda l: -int(l.split("/")[0]))
print("\nmost frequently *matched* rules:")
for l in by_count[:8]: print(l[:150])
never = [l for l in rows[:-1] if l.strip().startswith("0 /")]
print(f"\n{len(never)} rules were attempted but never matched, e.g.:")
for l in never[:3]: print(l[:150])
