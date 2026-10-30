# Chapter 8 — capture REAL UOp graphs of Tensor methods for the "decompose this" explorer (writes data/ch08_decomp.json).
# Run:  DEV=CPU python3 examples/08_capture_data.py
import os, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from uop2json import dump
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, Ops, GroupOp

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
x = Tensor([1., 2, 3]).realize()
y = Tensor([4., 5, 6]).realize()
m = Tensor([[1., 2, 3], [4, 5, 6]]).realize()
w = Tensor([[1., 0], [0, 1], [1, 1]]).realize()
idx = Tensor([2, 0]).realize()
img = Tensor([[[[float(4 * i + j) for j in range(4)] for i in range(4)]]]).realize()
ker = Tensor([[[[1., 0], [0, 1]]]]).realize()
NAMES = {x.uop.base: "x", y.uop.base: "y", m.uop.base: "m", w.uop.base: "w", idx.uop.base: "idx", img.uop.base: "img", ker.uop.base: "k"}

def clean_graph(root: UOp) -> dict:
  # the real UOp DAG with shape-argument srcs (STACK/CONST under movement ops, sizes under buffers) folded into labels
  keep, stack = set(), [root]
  while stack:
    u = stack.pop()
    if u in keep: continue
    keep.add(u)
    stack.extend(u.src[:1] if u.op in GroupOp.Movement else (() if u.op in GroupOp.Defines else u.src))
  topo = [u for u in root.toposort() if u in keep]
  ids = {u: f"n{i}" for i, u in enumerate(topo)}
  nodes = []
  for u in topo:
    if u.op in GroupOp.Movement: arg = str(u.marg)
    elif u.op in GroupOp.Defines: arg = ""
    elif u.op is Ops.CONST: arg = str(u.arg).replace("ConstFloat(", "").rstrip(")") if "ConstFloat" in repr(u.arg) else repr(u.arg)
    elif u.op is Ops.REDUCE: arg = f"{u.arg[0].name}, {u.arg[1]} axes"
    else: arg = "" if u.arg is None else str(u.arg)
    n = {"id": ids[u], "op": u.op.name, "arg": arg[:40], "shape": str(u._shape) if u._shape is not None else "", "dtype": str(u.dtype)}
    if u.op in GroupOp.Defines and u in NAMES: n["label"] = f"{u.op.name} {NAMES[u]}"
    nodes.append(n)
  edges = [[ids[s], ids[u], i] for u in topo for i, s in enumerate(u.src) if s in ids]
  return {"nodes": nodes, "edges": edges}

METHODS = [
  ("x - y", "elementwise.py sub", lambda: x - y),
  ("x / y", "elementwise.py div", lambda: x / y),
  ("x.exp()", "elementwise.py exp", lambda: x.exp()),
  ("x.log()", "elementwise.py log", lambda: x.log()),
  ("x.sigmoid()", "elementwise.py sigmoid", lambda: x.sigmoid()),
  ("x.relu()", "elementwise.py relu", lambda: x.relu()),
  ("x.minimum(y)", "elementwise.py minimum", lambda: x.minimum(y)),
  ("x.tanh()", "elementwise.py tanh", lambda: x.tanh()),
  ("x.mean()", "op.py mean", lambda: x.mean()),
  ("m.softmax()", "op.py softmax", lambda: m.softmax()),
  ("m.log_softmax()", "op.py log_softmax", lambda: m.log_softmax()),
  ("m.layernorm()", "op.py layernorm", lambda: m.layernorm()),
  ("m @ w", "op.py dot", lambda: m @ w),
  ("m.argmax(1)", "op.py argmax", lambda: m.argmax(1)),
  ("m.cumsum(1)", "op.py cumsum", lambda: m.cumsum(1)),
  ("Tensor.arange(4)", "op.py arange", lambda: Tensor.arange(4)),
  ("idx.one_hot(3)", "op.py one_hot", lambda: idx.one_hot(3)),
  ("m.gather(1, idx[:,None])", "op.py gather", lambda: m.gather(1, idx.reshape(2, 1))),
  ("img.conv2d(k)", "op.py conv2d", lambda: img.conv2d(ker)),
  ("img.max_pool2d()", "op.py max_pool2d", lambda: img.max_pool2d()),
  ("Tensor.rand(4)", "rand.py rand", lambda: Tensor.rand(4, contiguous=False)),
]
out = []
for expr, where, fn in METHODS:
  t = fn()
  g = clean_graph(t.uop)
  ops = Counter(n["op"] for n in g["nodes"])
  try: val = fn().tolist()
  except Exception as e: val = f"({type(e).__name__})"
  out.append({"expr": expr, "where": where, "shape": str(t.shape), "graph": g, "ops": dict(ops), "value": val})
  print(f"{expr:26s} {len(g['nodes']):4d} nodes  {dict(ops)}")
dump(out, os.path.join(DATA, "ch08_decomp.json"))
print("wrote data/ch08_decomp.json")
