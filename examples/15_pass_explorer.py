# Chapter 15 — watch full_rewrite_to_sink stage by stage.
#
# full_rewrite_to_sink (tinygrad/codegen/__init__.py) is a straight-line sequence of ~30 graph_rewrite calls, each with a
# name= string. We record every one of them by wrapping the module-level `graph_rewrite` that codegen/__init__.py imported
# (plus apply_opts and memory_coalescing, which are plain functions in that same sequence).
#
# TRACK_MATCH_STATS=2 makes PatternMatcher a TrackedPatternMatcher, which appends every successful rule application
# (before, after, UPat location) to tinygrad.uop.ops.tracked_ctxs. We read those to count which rules fired in each stage.
#
# run:   DEV=CPU python3 examples/15_pass_explorer.py            (prints the stage table for a small UPCAST reduction)
#        DEV=CPU python3 examples/15_pass_explorer.py --json     (also writes data/ch15_passes.json for the chapter demo)
import os, sys, json, collections
os.environ["TRACK_MATCH_STATS"] = "2"      # must be set before tinygrad is imported
os.environ["PRINT_MATCH_STATS"] = "0"      # ...but don't print the giant global table at exit
from dataclasses import replace
from tinygrad import Tensor
from tinygrad.helpers import Target, printable
from tinygrad.uop import ops as uops_mod
import tinygrad.codegen as cg
from tinygrad.renderer.cstyle import ClangRenderer, OpenCLRenderer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from uop2json import uop_to_graph

TG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(cg.__file__)))  # the repo root

# ---- recording wrappers -------------------------------------------------------------------------------------------
LOG: list[dict] = []
def _matches_since(start_counts):
  # every TrackedGraphRewrite appended (to any group) since `start_counts` — includes nested sub-rewrites
  out = []
  for gi, grp in enumerate(uops_mod.tracked_ctxs):
    for tgr in grp[start_counts.get(gi, 0):]: out.extend(tgr.matches)
  return out
def _snapshot(): return {gi: len(g) for gi, g in enumerate(uops_mod.tracked_ctxs)}

def record(stage, fn, sink, *args, **kwargs):
  snap = _snapshot()
  ret = fn(sink, *args, **kwargs)
  rules = collections.Counter((m[2][0], m[2][1]) for m in _matches_since(snap))
  LOG.append({"name": stage, "in": sink, "out": ret, "rules": rules,
              "bottom_up": kwargs.get("bottom_up", False), "walk": kwargs.get("walk", False)})
  return ret

_graph_rewrite, _memory_coalescing, _apply_opts = cg.graph_rewrite, cg.memory_coalescing, cg.apply_opts
cg.graph_rewrite = lambda sink, pm, *a, name=None, **kw: record(name, _graph_rewrite, sink, pm, *a, name=name, **kw)
cg.memory_coalescing = lambda sink, ren: record("memory_coalescing()", _memory_coalescing, sink, ren)
cg.apply_opts = lambda sink, ren, beam=0: record("apply_opts()", _apply_opts, sink, ren, beam=beam)

def rule_text(loc):
  path, line = loc
  rel = os.path.relpath(path, TG_ROOT) if path.startswith(TG_ROOT) else os.path.basename(path)
  return rel, line, printable(loc)[:150]

# ---- the kernels ---------------------------------------------------------------------------------------------------
CLANG = ClangRenderer(Target("CPU", "CLANG", "x86_64,native"))
CL = OpenCLRenderer(Target("CL"))     # renders OpenCL C; we never compile it, so no GPU is needed
def ast_of(t: Tensor): return t.schedule_linear().src[-1].src[0]
def no_opts(ast): return ast.replace(arg=replace(ast.arg, opts_to_apply=()))   # skip the heuristic: plain loop

KERNELS = {
  "sum_cpu":  ("Tensor.empty(4,8).sum(1) on CPU/Clang (heuristic UPCASTs the reduce)", lambda: ast_of(Tensor.empty(4, 8).sum(1)), CLANG),
  "exp_cpu":  ("Tensor.empty(8).exp() on CPU/Clang, no opts (Clang has no EXP2 → decomposed)", lambda: no_opts(ast_of(Tensor.empty(8).exp())), CLANG),
  "add_gpu":  ("Tensor.empty(16,16)+Tensor.empty(16,16) on OpenCL (GLOBAL/LOCAL/UPCAST)", lambda: ast_of(Tensor.empty(16, 16) + Tensor.empty(16, 16)), CL),
  "sum_gpu":  ("Tensor.empty(16,16).sum(1) on OpenCL (grouped reduce: LOCAL buffer + barrier)", lambda: ast_of(Tensor.empty(16, 16).sum(1)), CL),
}

def run_kernel(key):
  desc, mk, ren = KERNELS[key]
  LOG.clear()
  ast = mk()
  sink = cg.full_rewrite_to_sink(ast, ren)
  return desc, ren, ast, sink, list(LOG)

def to_json(key):
  desc, ren, ast, sink, log = run_kernel(key)
  ids: dict = {}                 # shared across all dumps of this kernel => hash-consed UOps keep their id => graphs morph
  graphs, gidx = [], {}
  def gid(u):
    if u not in gidx: gidx[u] = len(graphs); graphs.append(uop_to_graph(u, ids, max_nodes=700))
    return gidx[u]
  stages = []
  for e in log:
    stages.append({"name": e["name"], "in": gid(e["in"]), "out": gid(e["out"]), "same": e["in"] is e["out"],
                   "nin": len(e["in"].toposort()), "nout": len(e["out"].toposort()), "bottom_up": e["bottom_up"], "walk": e["walk"],
                   "rules": [[*rule_text(loc), n] for loc, n in e["rules"].most_common()]})
  for g in graphs:   # shrink the JSON a bit
    for n in g["nodes"]: n["dtype"] = n["dtype"].replace("dtypes.", "")
  return {"desc": desc, "renderer": type(ren).__name__, "name": sink.arg.name, "stages": stages, "graphs": graphs}

if __name__ == "__main__":
  desc, ren, ast, sink, log = run_kernel("sum_cpu")
  print(f"kernel: {desc}\nrenderer: {type(ren).__name__}   final name: {sink.arg.name}   opts: {sink.arg.applied_opts}\n")
  print(f"{'#':>2} {'stage (name=)':38s} {'nodes':>11s}  top rules fired (count × file:line)")
  for i, e in enumerate(log):
    nin, nout = len(e["in"].toposort()), len(e["out"].toposort())
    top = ", ".join(f"{n}×{rule_text(loc)[0].split('/')[-1]}:{loc[1]}" for loc, n in e["rules"].most_common(3))
    print(f"{i:2d} {e['name']:38s} {nin:4d} → {nout:4d}  {'(unchanged)' if e['in'] is e['out'] else top}")
  if "--json" in sys.argv:
    out = {k: to_json(k) for k in KERNELS}
    path = os.path.join(os.path.dirname(__file__), "..", "data", "ch15_passes.json")
    with open(path, "w") as f: json.dump(out, f, separators=(",", ":"), default=str)
    print(f"\nwrote {os.path.normpath(path)} ({os.path.getsize(path)//1024} KB)")
