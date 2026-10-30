"""examples/10_make_data.py — generates the REAL captured data behind chapter 10's demos.

Run:  DEV=CPU python3 examples/10_make_data.py
Writes data/ch10_flow.json, data/ch10_explorer.json, data/ch10_mops.json
"""
import importlib.util, json, os, pathlib
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, Ops, GroupOp
from tinygrad.schedule.indexing import apply_movement_op

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("cap", HERE / "10_sched_capture.py")
cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)
DATA = HERE.parent / "data"

def dump(obj, name):
  with open(DATA / name, "w") as f: json.dump(obj, f, separators=(",", ":"))
  print(f"wrote data/{name}: {os.path.getsize(DATA / name)/1024:.1f} KB")

def rr(rs): return [r.render() for r in rs]

# ------------------------------------------------------------------ 1. the range-flow stepper
def flow(name, code, fn):
  log, linear, compiled = cap.schedule(fn())
  G = cap.pass_by_name(log, "get realize")[1]
  initial = set(cap.pass_by_name(log, "get realize")[1].toposort()) & set(log["realize_map"])
  rctx = log["rctx"][0]
  ids: dict = {}
  # consumers, as run_rangeify computes them
  from tinygrad.schedule.indexing import data_srcs
  topo = G.toposort()
  consumers = {x: [] for x in topo}
  for c in topo:
    for s in data_srcs(c.op, c.src):
      if s in consumers: consumers[s].append(c)
  steps, notes = [], {}
  for u, (rin, rout) in rctx.range_map.items():
    realized = log["realize_map"].get(u)
    if realized is None: why = None
    elif u.op is Ops.STORE: why = "STORE: pm_generate_realize_map always realizes stores"
    elif len(consumers[u]) > 1: why = f"{len(consumers[u])} consumers index it with different ranges -> new ranges"
    else: why = "a consumer broadcasts it (ending ranges): REDUCE/elementwise output is materialized"
    notes[u] = f"in ranges : {rr(rin)}\nout ranges: {rr(rout)}" + (f"\nREALIZED axes {realized}: {why}" if realized is not None else "")
    steps.append({"op": u.op.name, "shape": str(u.shape).replace(" ", ""), "consumers": len(consumers[u]), "in": rr(rin), "out": rr(rout),
                  "realized": realized, "why": why})
  before = cap.graph_json(G, ids, notes)
  # the stepper highlights nodes by id; map range_map keys to ids
  for st, u in zip(steps, rctx.range_map): st["id"] = ids.get(u)
  after = cap.graph_json(cap.pass_by_name(log, "apply rangeify")[2], ids)
  cleaned = cap.graph_json(cap.pass_by_name(log, "symbolic+reduce_collapse+debuf")[2], ids)
  return {"name": name, "code": code, "before": before, "after": after, "cleaned": cleaned, "steps": steps,
          "kernels": [n for n, _ in cap.kernel_sources(compiled)]}

flows = [
  flow("permute + sum", "a = Tensor.empty(3, 4)\nout = (a.permute(1, 0) + 1).sum(0)", lambda: (Tensor.empty(3, 4).permute(1, 0) + 1).sum(0)),
  flow("softmax", "x = Tensor.empty(4, 8)\nout = x.softmax(-1)", lambda: Tensor.empty(4, 8).softmax(-1)),
  flow("pad + shrink + sum", "x = Tensor.empty(4, 4)\nout = x.pad(((1, 1), (1, 1)))[:4, 2:].sum(1)",
       lambda: Tensor.empty(4, 4).pad(((1, 1), (1, 1)))[:4, 2:].sum(1)),
]
dump({"flows": flows}, "ch10_flow.json")

# ------------------------------------------------------------------ 2. before/after explorer over the named passes
STAGES = [
  ("get realize", 1, "tensor graph (input to run_rangeify)", "After transform_to_call + prepare_rangeify: PARAMs instead of buffers, STORE/AFTER for outputs, movement ops still present."),
  ("apply rangeify", 2, "apply rangeify", "pm_apply_rangeify: movement ops are gone, every buffer access is INDEX(buf, idx-expr), realized values are wrapped in STAGE (=BUFFERIZE) closing their ranges."),
  ("symbolic+reduce_collapse+debuf", 2, "symbolic + debuf", "symbolic + pm_reduce_simplify + pm_const_buffer_folding + pm_remove_bufferize: cheap STAGEs are removed (recomputed), index math simplified."),
  ("stage to store", 2, "stage to store", "pm_add_buffers / bufferize_to_store: each surviving STAGE becomes ALLOC + INDEX + STORE + END(ranges) wrapped in AFTER."),
  ("split kernels", 2, "split kernels", "split_kernels: every STORE/END with no open ranges becomes SINK(KernelInfo).call(*buffers) — one CALL per kernel."),
]
def explorer(name, code, fn):
  log, linear, compiled = cap.schedule(fn())
  ids: dict = {}
  stages = []
  for pname, which, title, desc in STAGES:
    p = cap.pass_by_name(log, pname)
    g = cap.graph_json(p[which], ids)
    stages.append({"pass": pname, "title": title, "desc": desc, "graph": g, "n": len(p[which].toposort())})
  passes = [{"name": n, "nin": len(i.toposort()), "nout": len(o.toposort())} for n, i, o in log["passes"]]
  return {"name": name, "code": code, "stages": stages, "passes": passes,
          "kernels": [{"name": n, "src": s} for n, s in cap.kernel_sources(compiled)]}

exps = [
  explorer("permute + sum", "a = Tensor.empty(3, 4)\nout = (a.permute(1, 0) + 1).sum(0)", lambda: (Tensor.empty(3, 4).permute(1, 0) + 1).sum(0)),
  explorer("softmax", "x = Tensor.empty(4, 8)\nout = x.softmax(-1)", lambda: Tensor.empty(4, 8).softmax(-1)),
  explorer("pad + shrink + sum", "x = Tensor.empty(4, 4)\nout = x.pad(((1, 1), (1, 1)))[:4, 2:].sum(1)",
           lambda: Tensor.empty(4, 4).pad(((1, 1), (1, 1)))[:4, 2:].sum(1)),
  explorer("x - x.mean(1)", "x = Tensor.empty(4, 4)\nout = x - x.mean(1, keepdim=True)", lambda: (lambda x: x - x.mean(1, keepdim=True))(Tensor.empty(4, 4))),
]
dump({"programs": exps}, "ch10_explorer.json")

# ------------------------------------------------------------------ 3. movement-op chains, rendered by the real apply_movement_op
def mop_chain(name, code, base_shape, fn):
  y = fn(Tensor.empty(*base_shape))
  u, mops = y.uop, []
  while u.op in GroupOp.Movement: mops.append(u); u = u.src[0]
  rngs = tuple(UOp.range(s, i) for i, s in enumerate(y.shape))
  idx, steps = rngs, []
  for m in mops:   # from the output down to the buffer, exactly like run_rangeify walks the graph
    idx = apply_movement_op(m.op, m.src[0].shape, m.marg, idx)
    steps.append({"op": m.op.name, "arg": str(m.marg).replace(" ", ""), "src_shape": list(m.src[0].shape), "idx": rr(idx)})
  ops_fwd = [{"op": m.op.name, "arg": json.loads(json.dumps(m.marg)), "out_shape": list(m.shape)} for m in reversed(mops)]
  return {"name": name, "code": code, "base": list(base_shape), "out_shape": list(y.shape), "ops": ops_fwd, "steps": steps}

mops = [
  mop_chain("reshape → permute (transpose)", "x.reshape(3,4).permute(1,0)", (12,), lambda x: x.reshape(3, 4).permute(1, 0)),
  mop_chain("pad", "x.pad(((1,1),(0,2)))", (3, 4), lambda x: x.pad(((1, 1), (0, 2)))),
  mop_chain("shrink (slice)", "x[1:3, 1:]", (3, 4), lambda x: x[1:3, 1:]),
  mop_chain("broadcast expand", "x.reshape(3,1).expand(3,4)", (3,), lambda x: x.reshape(3, 1).expand(3, 4)),
  mop_chain("reshape ∘ reshape", "x.reshape(2,6).reshape(4,3)", (12,), lambda x: x.reshape(2, 6).reshape(4, 3)),
  mop_chain("flip (no FLIP op!)", "x.flip(0)", (4,), lambda x: x.flip(0)),
  mop_chain("split an axis", "x.reshape(2,3,2)", (3, 4), lambda x: x.reshape(2, 3, 2)),
]
dump({"chains": mops}, "ch10_mops.json")
