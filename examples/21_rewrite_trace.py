# Chapter 21 — capture the tracked rewrites (what VIZ shows) for the debug target, and turn them into JSON
# Run: DEV=CPU python3 examples/21_rewrite_trace.py
#
# TRACK_MATCH_STATS=2 (implied by VIZ) swaps PatternMatcher for TrackedPatternMatcher (uop/ops.py) and makes every
# @rewrite_group / graph_rewrite call append a TrackedGraphRewrite(loc, sink, matches, name, depth, ...) to
# uop.ops.tracked_ctxs. A match is (uop_before, uop_after, upat_location, seconds), stored as integer trace numbers;
# uop_fields maps trace numbers back to (op, src, arg). At exit the RewriteTrace is pickled to rewrites.pkl.<user>.
# Like the real viz server/CLI we decode that pickle in a *fresh* process with tinygrad/viz/serve.py's own
# load_rewrites + get_full_rewrite (rebuilding the UOps in the process that made them is not supported).
import os, sys, json, subprocess, tempfile, getpass, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp(prefix="ch21trace_")      # private TMPDIR: rewrites.pkl.<user> lands here, not in the shared /tmp
env = {**os.environ, "DEV": "CPU", "TRACK_MATCH_STATS": "2", "PRINT_MATCH_STATS": "0", "TMPDIR": tmp}
p = subprocess.run([sys.executable, os.path.join(HERE, "21_debug_target.py")], env=env, capture_output=True, text=True)
print(p.stdout.replace(tmp, "$TMPDIR").strip())

os.environ["VIZ"] = "0"
from tinygrad.uop.ops import RewriteTrace
from tinygrad.viz import serve as viz
trace = viz.load_pickle(os.path.join(tmp, f"rewrites.pkl.{getpass.getuser()}"), default=RewriteTrace([], [], {}))
shutil.rmtree(tmp)
data = viz.VizData(trace)
viz.load_rewrites(data)
nodes: dict[str, dict] = {}          # one global node table: id -> {label, src}; UOps are hash-consed so ids are stable
short: dict[str, str] = {}           # node signature (label + srcs) -> short id. python id()s can be reused once a
def add_graph(g:dict) -> dict[int, str]:  # frame's UOps are freed, so identity is the content, exactly like hash-consing
  local: dict[int, str] = {}
  for k, v in g.items():             # g is in toposort order (viz.uop_to_json walks x.toposort())
    if v["exclude"]: continue
    label, srcs = viz.ansistrip(v["label"]), [local[s] for _, s in v["src"] if s in local]
    local[k] = short.setdefault(label + "|" + ",".join(srcs), f"u{len(short)}")
    nodes.setdefault(local[k], {"label": label, "src": srcs})
  return local

groups, total, failed = [], 0, []
for ci, ctx in enumerate(data.ctxs):
  steps = []
  for si, rw in enumerate(data.trace.rewrites[ci]):
    frames = []
    if rw.matches:
     try:
      for f in viz.get_full_rewrite(data, rw):
        local = add_graph(f["graph"])
        frames.append({"nodes": list(local.values()), "change": [local[c] for c in (f["change"] or []) if c in local],
                       "upat": None if f["upat"] is None else {"loc": f"{f['upat'][0][0].split('tinygrad/')[-1]}:{f['upat'][0][1]}", "code": f["upat"][1]},
                       "diff": f["diff"]})
        total += 1
     except Exception as e:  # a few intermediate graphs can't be re-labelled by viz's uop_to_json; keep the frames we got
      failed.append(f"{ctx['name']} / {rw.name}: {type(e).__name__} {e}")
    steps.append({"name": rw.name, "depth": rw.depth, "matches": len(rw.matches), "loc": f"{rw.loc[0].split('tinygrad/')[-1]}:{rw.loc[1]}",
                  "bottom_up": rw.bottom_up, "frames": frames})
  groups.append({"name": viz.ansistrip(ctx["name"]), "steps": steps})
  print(f"group {ci}: {groups[-1]['name']!r:28s} {len(steps):3d} rewrites, {sum(s['matches'] for s in steps):4d} matches")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ch21_trace.json")
json.dump({"groups": groups, "nodes": nodes}, open(out, "w"))
for f in failed: print("could not replay:", f)
print(f"{total} frames, {len(nodes)} distinct UOps -> data/ch21_trace.json ({os.path.getsize(out)//1024} KB)")
