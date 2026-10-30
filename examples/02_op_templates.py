#!/usr/bin/env python3
# Chapter 2 — "what UOps does each Tensor method add on top of its input?"
#
#   run:  DEV=CPU python3 examples/02_op_templates.py           # prints the new nodes per op, for an input of shape (2, 3)
#         DEV=CPU python3 examples/02_op_templates.py --dump    # also writes data/ch02_templates.json (used by the
#                                                               # "graph grows as you type" demo in chapter 2)
#
# Method: X is a placeholder f32 tensor (a SIN of an empty buffer, so it is an ordinary lazy value).
# We apply ONE method to X and walk the result graph, stopping at X. Every node we meet is something the method created.
import sys, json, pathlib
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, Ops

B = Tensor([10., 20., 30.])                      # a fixed second operand, shape (3,)

OPS = {  # label -> (function, is it applicable to this shape?)
  "x + 1":          (lambda x: x + 1,          lambda s: True),
  "x * 2":          (lambda x: x * 2,          lambda s: True),
  "x * x":          (lambda x: x * x,          lambda s: True),
  "-x":             (lambda x: -x,             lambda s: True),
  "x.exp()":        (lambda x: x.exp(),        lambda s: True),
  "x.relu()":       (lambda x: x.relu(),       lambda s: True),
  "x.sqrt()":       (lambda x: x.sqrt(),       lambda s: True),
  "x + b":          (lambda x: x + B,          lambda s: len(s) >= 1 and s[-1] == 3),
  "x.sum(axis=-1)": (lambda x: x.sum(axis=-1), lambda s: len(s) >= 1),
  "x.T":            (lambda x: x.T,            lambda s: len(s) == 2),
  "x.reshape(-1)":  (lambda x: x.reshape(-1),  lambda s: len(s) >= 2),
  "x.contiguous()": (lambda x: x.contiguous(), lambda s: True),
}
SHAPES = [(2, 3), (3, 2), (2,), (3,), (6,), ()]

def describe(u: UOp) -> dict:
  arg = "" if u.arg is None else str(u.arg)
  if u.op in (Ops.BUFFER, Ops.ALLOC, Ops.PARAM): arg = f"slot={u.arg.slot}, device={u.arg.device!r}"
  return {"op": u.op.name, "dtype": str(u.dtype), "shape": str(u._shape) if u._shape is not None else "", "arg": arg[:50]}

def template(x: UOp, y: UOp) -> list[dict]:
  """the nodes of y that are not x, in topological order; srcs are 'X' or an index into the list"""
  order: list[UOp] = []
  seen: set = set()
  def walk(u: UOp):
    if u is x or u in seen: return
    for s in u.src: walk(s)
    seen.add(u); order.append(u)
  walk(y)
  idx = {u: i for i, u in enumerate(order)}
  return [{**describe(u), "src": ["X" if s is x else idx[s] for s in u.src]} for u in order]

out: dict = {"start": None, "templates": {}, "realized": {}}
for shape in SHAPES:
  X = Tensor.empty(*shape).sin() if shape else Tensor.empty(1).reshape(()).sin()
  for label, (fn, ok) in OPS.items():
    if not ok(shape): continue
    Y = fn(X)
    out["templates"][f"{label}|{shape}"] = {"shape": str(Y.shape), "nodes": template(X.uop, Y.uop),
                                             "identity": Y.uop is X.uop}
  # what a realized tensor of this shape looks like
  R = (Tensor.empty(*shape).sin() if shape else Tensor.empty(1).reshape(()).sin()).realize()
  out["realized"][str(shape)] = template(UOp.const(0), R.uop)   # (no X inside: the whole graph)

# the starting tensor of the demo, captured for real
A = Tensor([[1., 2., 3.], [4., 5., 6.]])
out["start"] = template(UOp.const(0), A.uop)

# print a readable summary for the (2, 3) input
for key, t in out["templates"].items():
  label, shape = key.split("|")
  if shape != "(2, 3)": continue
  print(f"{label:16s} -> shape {t['shape']:7s} adds {len(t['nodes'])} UOps:", " ".join(n["op"] for n in t["nodes"]) or "(none: returns x itself)")
print("\nstart tensor A = Tensor([[1.,2,3],[4,5,6]]):", " <- ".join(n["op"] for n in reversed(out["start"])))
print("a realized (2, 3) tensor is:", " <- ".join(n["op"] for n in reversed(out["realized"]["(2, 3)"])))

if "--dump" in sys.argv:
  p = pathlib.Path(__file__).resolve().parent.parent / "data" / "ch02_templates.json"
  p.write_text(json.dumps(out))
  print("wrote", p, f"({len(out['templates'])} templates)")
