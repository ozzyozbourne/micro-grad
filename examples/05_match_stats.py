# Chapter 5 — which rules fire, and how often?  Run with:
#   DEV=CPU python3 examples/05_match_stats.py   (it sets TRACK_MATCH_STATS=1 itself if you didn't)
# With TRACK_MATCH_STATS>=1, PatternMatcher is swapped for TrackedPatternMatcher (ops.py), which counts every match attempt per UPat.
# At exit it prints "matches / attempts -- match time / total time -- file:line  <rule source>", sorted by time (the slowest rules last).
import os
os.environ.setdefault("TRACK_MATCH_STATS", "1")   # must be set before `import tinygrad`: ContextVars read the env at import
from tinygrad import Tensor
from tinygrad.uop.ops import match_stats

a = Tensor([[1., 2, 3], [4, 5, 6]]).realize()
b = Tensor([10., 20, 30]).realize()
print("result:", (a * 2 + b).sum(axis=1).tolist())
fired = sorted(((v[0], v[1], f"{k.location[0].split('/')[-1]}:{k.location[1]}") for k, v in match_stats.items() if v[0]), reverse=True)
print(f"{len(match_stats)} distinct UPats were tried; {len(fired)} of them matched at least once. Top 12 by number of matches:")
for m, tries, loc in fired[:12]: print(f"  {m:4d} matches / {tries:5d} attempts   {loc}")
