"""examples/10_sched_capture.py — watch EVERY named graph_rewrite pass of the scheduler.

Run:  DEV=CPU python3 examples/10_sched_capture.py

The scheduler (tinygrad/schedule/*) is a sequence of named graph_rewrite calls. This file wraps `graph_rewrite`
inside the four scheduler modules so that every call is recorded as (pass name, input graph, output graph).
It also records the IndexingContext that run_rangeify builds (realize_map + range_map), which is the core data
structure of chapter 10.

Chapters 10-12 import this file (via importlib) to generate the JSON data used by the interactive demos, so it also
contains a small UOp -> {nodes, edges} converter that hides "shape argument" sources (the STACK of CONSTs that
movement ops carry, the size CONST of a PARAM/RANGE) to keep graphs readable.
"""
import contextlib
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, UOp, GroupOp
import tinygrad.schedule as S
import tinygrad.schedule.prepare as P
import tinygrad.schedule.indexing as I
import tinygrad.schedule.rangeify as R
import tinygrad.tensor as T
from tinygrad.engine.realize import compile_linear, run_linear

# ---------------------------------------------------------------- hooking
@contextlib.contextmanager
def hooked():
  """records every graph_rewrite done by the scheduler + the IndexingContext of run_rangeify"""
  log = {"passes": [], "rctx": [], "realize_map": {}}
  saved = {}
  for mod in (T, S, P, I, R):
    orig = mod.graph_rewrite
    saved[mod] = orig
    def gr(sink, *a, _orig=orig, **k):
      # snapshot the realize_map before "apply rangeify" consumes (deletes) the STORE entries
      if k.get("name") == "apply rangeify": log["realize_map"] = dict(k["ctx"].realize_map)
      out = _orig(sink, *a, **k)
      log["passes"].append((k.get("name") or "graph_rewrite (unnamed)", sink, out))
      return out
    mod.graph_rewrite = gr
  orig_ctx = I.IndexingContext
  class RecordingIndexingContext(orig_ctx):
    def __init__(self, *a, **k):
      super().__init__(*a, **k)
      log["rctx"].append(self)
  I.IndexingContext = RecordingIndexingContext
  try:
    yield log
  finally:
    for mod, orig in saved.items(): mod.graph_rewrite = orig
    I.IndexingContext = orig_ctx

def schedule(*tensors:Tensor, run=True):
  """schedule (and optionally run) tensors, returning (log, linear, compiled_linear)"""
  # SCACHE=0: a schedule-cache hit would skip all passes (see chapter 12)
  with Context(SCACHE=0), hooked() as log:
    linear, var_vals = Tensor.linear_with_vars(*tensors)
  compiled = compile_linear(linear)
  if run: run_linear(linear, var_vals)
  return log, linear, compiled

def kernel_sources(compiled:UOp) -> list[tuple[str, str]]:
  """(name, source code) for every CALL in a compiled LINEAR"""
  out = []
  for call in compiled.src:
    body = call.body
    if body.op is Ops.PROGRAM:
      srcs = [u.arg for u in body.toposort() if u.op is Ops.SOURCE]
      out.append((body.src[0].arg.name, srcs[0] if srcs else ""))
    elif body.op is Ops.STORE: out.append(("copy", "// bulk copy: CALL(STORE(dst, src)) -> exec_copy, no kernel is compiled"))
    else: out.append((str(body.op), ""))
  return out

# ---------------------------------------------------------------- UOp -> JSON graph
def _arg_label(u:UOp) -> str:
  if u.op in GroupOp.Movement:
    try: return str(u.marg).replace(" ", "")
    except Exception: return ""
  if u.op is Ops.RANGE: return f"r{u.arg[1]} <{u.src[0].render() if u.src[0].op is not Ops.CONST else u.src[0].arg}"
  if u.op in {Ops.PARAM, Ops.BUFFER, Ops.ALLOC}:
    a = u.arg
    kind = {Ops.PARAM: "slot", Ops.BUFFER: "buf", Ops.ALLOC: "slot"}[u.op]
    try: return f"{kind} {a.slot} · {u.max_numel()}×{str(a.dtype).replace('dtypes.', '')}"
    except Exception: return str(a)[:40]
  if u.op is Ops.CONST: return str(u.arg).replace("ConstFloat(", "").rstrip(")") if "ConstFloat" in str(u.arg) else str(u.arg)
  if u.op is Ops.REDUCE: return f"{u.arg[0].name}" + (f", {u.arg[1]} axes" if u.arg[1] else "")
  if u.op is Ops.STAGE: return "" if u.arg is None else ("BUFFERIZE" + ("" if u.arg.removable else " (keep)"))
  if u.op is Ops.SINK and u.arg is not None: return "kernel"
  if u.op is Ops.CALL: return ""
  if u.arg is None: return ""
  return str(u.arg)[:40]

def hidden_srcs(u:UOp) -> set[int]:
  """indices of srcs that are 'shape arguments', not data"""
  if u.op in GroupOp.Movement: return set(range(1, len(u.src)))
  if u.op in {Ops.PARAM, Ops.BUFFER, Ops.ALLOC, Ops.RANGE}: return set(range(len(u.src)))
  return set()

def graph_json(root:UOp, ids:dict|None=None, notes:dict|None=None, max_nodes=260) -> dict:
  ids = {} if ids is None else ids
  keep: list[UOp] = []
  seen = set()
  def visit(u:UOp):
    if u in seen: return
    seen.add(u)
    for i, s in enumerate(u.src):
      if i not in hidden_srcs(u): visit(s)
    keep.append(u)
  visit(root)
  keep = keep[-max_nodes:]
  present = set(keep)
  nodes, edges = [], []
  for u in keep:
    nid = ids.setdefault(u, f"n{len(ids)}")
    try: shape = str(u._shape) if u._shape is not None else ""
    except Exception: shape = ""
    try: dt = str(u.dtype).replace("dtypes.", "")
    except Exception: dt = ""
    n = {"id": nid, "op": u.op.name, "arg": _arg_label(u), "dtype": dt if dt not in ("void",) else "", "shape": shape.replace(" ", "")}
    if notes and u in notes: n["note"] = notes[u]
    nodes.append(n)
    for i, s in enumerate(u.src):
      if i in hidden_srcs(u) or s not in present: continue
      edges.append([ids.setdefault(s, f"n{len(ids)}"), nid, i])
  return {"nodes": nodes, "edges": edges}

def pass_by_name(log, name, nth=0):
  hits = [p for p in log["passes"] if p[0] == name]
  return hits[nth] if nth < len(hits) else None

# ---------------------------------------------------------------- demo
if __name__ == "__main__":
  a = Tensor.empty(3, 4)
  out = (a.permute(1, 0) + 1).sum(0)        # a (3,4) -> (4,3) -> +1 -> sum over the 4 -> (3,)
  log, linear, compiled = schedule(out)
  print("every named graph_rewrite the scheduler ran, in order:")
  for i, (name, src, dst) in enumerate(log["passes"]):
    print(f"  {i:2d}. {name:38s} {len(src.toposort()):3d} uops -> {len(dst.toposort()):3d} uops")
  print("\nrealize_map (UOps that get their own fresh output ranges):")
  for u, axes in log["realize_map"].items(): print(f"  {u.op.name:8s} shape={u.shape}  realized axes={axes}")
  rctx = log["rctx"][0]
  print("\nrange_map: (input ranges, output ranges) of every UOp, in the order run_rangeify visits them:")
  for u, (rin, rout) in rctx.range_map.items():
    print(f"  {u.op.name:8s} {str(u.shape):8s} in={[r.render() for r in rin]}  out={[r.render() for r in rout]}")
  print(f"\nfinal schedule: {linear.op.name} with {len(linear.src)} CALL(s)")
  for name, src in kernel_sources(compiled): print(f"--- {name}\n{src}")
