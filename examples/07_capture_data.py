# Chapter 7 — capture REAL data for the chapter's demos (writes data/ch07_*.json).
# Run:  DEV=CPU python3 examples/07_capture_data.py
#
# For each preset (a high-level Tensor expression) we walk down its chain of movement UOps and record, for every
# intermediate view:  the op, its decoded arg (marg), its shape, the real symbolic index expression into the base
# buffer (apply_movement_op + symbolic, the way rangeify composes them), and the realized values.
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from uop2json import uop_to_graph, dump
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, Ops, GroupOp, graph_rewrite
from tinygrad.schedule.indexing import apply_movement_op
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

def index_uop(u: UOp) -> tuple[list[UOp], UOp, UOp]:
  # Mirror rangeify (tinygrad/schedule/indexing.py): start with one RANGE per output axis, push the ranges through every
  # movement op with apply_movement_op, and at each PAD collect the validity of the new ranges — rangeify turns each PAD
  # into WHERE(valid, src, 0) right there (convert_pad_to_where_to_keep_behavior_local), so the valid never has to
  # survive the reshapes further down.
  rngs = [UOp.range(s, i) for i, s in enumerate(u.shape)]
  cur, valids = tuple(rngs), []
  while u.op in GroupOp.Movement:
    cur = apply_movement_op(u.op, u.src[0].shape, u.marg, cur)
    if u.op is Ops.PAD: valids += [r.get_valid() for r in cur]
    cur = tuple(r.get_idx() for r in cur)
    u = u.src[0]
  assert len(cur) == 1, "presets use a flat 1-D base buffer"
  # simplify the index *under* its valid (pm_simplify_valid), as the scheduler does for gated loads
  out = graph_rewrite(cur[0].valid(UOp.const(True).uprod(valids)), symbolic + pm_simplify_valid, name="simplify")
  return rngs, out.get_idx(), out.get_valid()

def index_expr(u: UOp) -> str:
  _, idx, valid = index_uop(u)
  return idx.render() if valid.op is Ops.CONST and valid.arg is True else f"({idx.render()} if {valid.render()} else Invalid)"

def source_cells(u: UOp) -> list[int]:
  # evaluate the symbolic index at every output cell: which flat buffer element does it read? (-1 = Invalid / padding)
  import itertools
  rngs, e, valid = index_uop(u)
  out = []
  for idx in itertools.product(*[range(s) for s in u.shape]):
    sub = {r: r.const_like(i) for r, i in zip(rngs, idx)}
    ok, v = valid.substitute(sub).simplify(), e.substitute(sub).simplify()
    out.append(int(v.arg) if ok.arg is True else -1)
  return out

def clean_graph(root: UOp) -> dict:
  # uop_to_graph, but fold the shape-argument srcs (STACK/CONST under movement ops, sizes under buffers) into the node label
  g = uop_to_graph(root)
  ids = {}
  for u in root.toposort(): ids.setdefault(u, None)
  keep, stack = set(), [root]
  while stack:
    u = stack.pop()
    if u in keep: continue
    keep.add(u)
    data = u.src[:1] if u.op in GroupOp.Movement else (() if u.op in {Ops.BUFFER, Ops.ALLOC, Ops.PARAM} else u.src)
    stack.extend(data)
  topo = [u for u in root.toposort() if u in keep]
  idmap = {u: f"n{i}" for i, u in enumerate(topo)}
  nodes = []
  for u in topo:
    arg = str(u.marg) if u.op in GroupOp.Movement else ("" if u.op in {Ops.BUFFER, Ops.ALLOC} else str(u.arg))
    nodes.append({"id": idmap[u], "op": u.op.name, "arg": arg[:40], "shape": str(u.shape), "dtype": str(u.dtype)})
  edges = [[idmap[s], idmap[u], i] for u in topo for i, s in enumerate(u.src) if s in idmap]
  return {"nodes": nodes, "edges": edges}

def chain(t: Tensor):
  steps, u = [], t.uop
  while u.op in GroupOp.Movement:
    steps.append(u); u = u.src[0]
  steps = steps[::-1]
  out = []
  for v in steps:
    # sanity check: base values are 0,1,2,... so every realized value must equal the buffer cell its index reads (0 if padded)
    vals, src = Tensor(v).flatten().tolist(), source_cells(v)
    assert vals == [float(max(i, 0)) for i in src], (v.op, vals, src)
    out.append({"op": v.op.name, "marg": str(v.marg), "shape": list(v.shape), "index": index_expr(v),
                "values": Tensor(v).tolist(), "src": source_cells(v)})
  return out

presets = {}
base6 = lambda: Tensor([0., 1, 2, 3, 4, 5]).realize()
base12 = lambda: Tensor([float(i) for i in range(12)]).realize()
P = [
  ("flip", "x = arange(6);  x.flip(0)", lambda: base6().flip(0)),
  ("flip2d", "x = arange(12).reshape(3,4);  x.flip(1)", lambda: base12().reshape(3, 4).flip(1)),
  ("transpose_flatten", "arange(6).reshape(2,3).T.flatten()", lambda: base6().reshape(2, 3).T.flatten()),
  ("stride_slice", "arange(12).reshape(3,4)[1:3, ::2]", lambda: base12().reshape(3, 4)[1:3, ::2]),
  ("neg_stride", "arange(12).reshape(3,4)[:, ::-2]", lambda: base12().reshape(3, 4)[:, ::-2]),
  ("repeat", "arange(6).reshape(2,3).repeat(2, 2)", lambda: base6().reshape(2, 3).repeat(2, 2)),
  ("roll", "arange(12).reshape(3,4).roll(1, 1)", lambda: base12().reshape(3, 4).roll(1, 1)),
  ("expand", "arange(6).reshape(2,1,3).expand(2,4,3)", lambda: base6().reshape(2, 1, 3).expand(2, 4, 3)),
  ("pad_shrink", "arange(6).reshape(2,3).pad(((1,1),(2,0)))[1:, :4]", lambda: base6().reshape(2, 3).pad(((1, 1), (2, 0)))[1:, :4]),
  ("diag", "arange(3).diag()  (vector -> diagonal matrix)", lambda: Tensor([0., 1, 2]).realize().diag()),
  ("diagonal", "arange(12).reshape(3,4).diagonal()", lambda: base12().reshape(3, 4).diagonal()),
]
for key, title, fn in P:
  t = fn()
  presets[key] = {"title": title, "final_index": index_expr(t.uop), "steps": chain(t), "graph": clean_graph(t.uop),
                  "result": fn().tolist()}
  print(f"{key:18s} {len(presets[key]['steps']):2d} movement ops  final index: buf[{presets[key]['final_index']}]")

dump(presets, os.path.join(DATA, "ch07_chains.json"))
print("wrote data/ch07_chains.json")

# pad modes: which primitive ops does each mode of Tensor.pad build? (tinygrad/mixin/op.py pad)
from collections import Counter
modes = {}
for mode, kw in [("constant", {}), ("constant value=-1", {"value": -1.0}), ("reflect", {"mode": "reflect"}),
                 ("replicate", {"mode": "replicate"}), ("circular", {"mode": "circular"})]:
  t = base12().reshape(3, 4).pad(((1, 1), (2, 1)), **kw)
  ops = Counter(u.op.name for u in t.uop.toposort() if u.op not in {Ops.CONST, Ops.STACK, Ops.BUFFER})
  modes[mode] = {"ops": dict(ops), "values": base12().reshape(3, 4).pad(((1, 1), (2, 1)), **kw).tolist()}
  print(f"pad mode {mode:18s} {dict(ops)}")
dump(modes, os.path.join(DATA, "ch07_padmodes.json"))
print("wrote data/ch07_padmodes.json")
