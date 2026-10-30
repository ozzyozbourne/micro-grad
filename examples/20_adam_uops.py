# Chapter 20 — the Adam update, as UOps
# Run: DEV=CPU python3 examples/20_adam_uops.py
#
# nn.optim.Adam is LAMB(..., adam=True). Its _step (nn/optim.py) is ordinary Tensor math plus .assign():
#     b1_t *= b1 ; b2_t *= b2                                  (assign via __imul__)
#     m.assign(b1*m + (1-b1)*g) ; v.assign(b2*v + (1-b2)*g*g)
#     up = (m/(1-b1_t)) / (sqrt(v/(1-b2_t)) + eps)            (m_hat / (sqrt(v_hat)+eps))
#     w.assign(w.detach() - lr*up)                             (Optimizer._apply_update)
# We give one tiny parameter a hand-made gradient, call schedule_step(), and dump the resulting UOp graph before
# anything runs. Then we realize, print the kernels, and check the numbers against a pure-python Adam.
import json, math, os, sys
from tinygrad import Tensor, nn, Context
from tinygrad.uop.ops import Ops
from tinygrad.uop.render import render_uir
from tinygrad.engine.realize import lower_and_compile, run_linear

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from uop2json import uop_to_graph
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def slim(g:dict) -> dict:   # hide STACK/CONST shape plumbing (same helper as examples/20_layers.py)
  ops = {n["id"]: n["op"] for n in g["nodes"]}
  cons = {}
  for s, d, _ in g["edges"]: cons.setdefault(s, []).append(d)
  keep = {n["id"] for n in g["nodes"] if n["op"] not in ("STACK", "CONST") or
          (n["op"] == "CONST" and any(ops[c] not in ("STACK", "BUFFER") for c in cons.get(n["id"], [])))}
  return {"nodes": [n for n in g["nodes"] if n["id"] in keep], "edges": [e for e in g["edges"] if e[0] in keep and e[1] in keep]}

w = Tensor([1.0, -2.0, 0.5])
opt = nn.optim.Adam([w], lr=0.1)
Tensor.realize(w, opt.lr, opt.m[0], opt.v[0], opt.b1_t, opt.b2_t)
names = {w.uop: "w", opt.lr.uop: "lr", opt.m[0].uop: "m", opt.v[0].uop: "v", opt.b1_t.uop: "b1_t", opt.b2_t.uop: "b2_t"}
g_vals = [0.5, -1.0, 2.0]
w.grad = Tensor(g_vals).realize()                 # pretend backward() produced this gradient
names[w.grad.uop] = "g"

with Context(TRAINING=1):
  outs = opt.schedule_step()
print("schedule_step() returned:", [("b1_t", "b2_t", "m", "v", "w")[i] for i in range(len(outs))])
for nm, t in (("b1_t", opt.b1_t), ("m", opt.m[0]), ("w", w)):
  print(f"\n--- {nm}.uop after schedule_step ({len(t.uop.toposort())} UOps) ---")
  print(render_uir(t.uop))

ids: dict = {}
graph = slim(uop_to_graph(w.uop, ids))
for n in graph["nodes"]:          # label the input buffers with their python names
  for u, nm in names.items():
    if ids.get(u) == n["id"]: n["label"] = f"BUFFER {nm}"
def group(t): return sorted({ids[u] for u in t.toposort() if u in ids})
store = w.uop.src[1]
groups = {"b1_t *= b1": group(opt.b1_t.uop), "b2_t *= b2": group(opt.b2_t.uop), "m = b1*m + (1-b1)*g": group(opt.m[0].uop),
          "v = b2*v + (1-b2)*g²": group(opt.v[0].uop), "w - lr*m̂/(√v̂+ε)": group(store.src[1]), "STORE into w": group(w.uop)}

linear = Tensor.schedule_linear(*outs)
compiled = lower_and_compile(linear)
kernels = []
for c in compiled.src:
  prg = c.without_after.body
  kernels.append({"name": prg.src[0].arg.name, "src": next(s.arg for s in prg.src if s.op is Ops.SOURCE)})
print("\nkernels:", [k["name"] for k in kernels])
run_linear(linear)

# pure python Adam (eps=1e-8, b1=0.9, b2=0.999, t=1) to check the numbers
b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.1
ref = []
for wi, gi in zip([1.0, -2.0, 0.5], g_vals):
  m_ = (1 - b1) * gi; v_ = (1 - b2) * gi * gi
  ref.append(wi - lr * (m_ / (1 - b1)) / (math.sqrt(v_ / (1 - b2)) + eps))
print("tinygrad w after 1 step:", [round(x, 6) for x in w.tolist()])
print("python   w after 1 step:", [round(x, 6) for x in ref])
print("m:", [round(x, 6) for x in opt.m[0].tolist()], " v:", [round(x, 8) for x in opt.v[0].tolist()], " b1_t:", opt.b1_t.item())
print("note: at t=1, m_hat = g and v_hat = g*g, so every coordinate moves by ~lr*sign(g) regardless of |g|")

json.dump({"graph": graph, "groups": groups, "kernels": kernels, "uir": {"b1_t": render_uir(opt.b1_t.uop), "w": render_uir(store.src[1])}},
          open(os.path.join(DATA, "ch20_adam.json"), "w"), indent=1)
print("wrote data/ch20_adam.json")
