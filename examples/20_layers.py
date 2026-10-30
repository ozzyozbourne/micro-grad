# Chapter 20 — nn layers are just Tensor ops
# Run: DEV=CPU python3 examples/20_layers.py
#
# tinygrad/nn/__init__.py has no "layer" machinery: no Module base class, no hooks, no registered parameters.
# A layer is a plain Python object that holds Tensors as attributes and has a __call__ that calls Tensor methods.
# This script shows:
#   1. what nn.Linear(3,2)(x) becomes as a UOp graph (and how each Tensor method contributes nodes)
#   2. how many kernels a few layers turn into when realized
#   3. nn.state: get_state_dict / get_parameters walk __dict__, safe_save / safe_load round-trip
#   4. the TRAINING context var (what used to be Tensor.training) changes BatchNorm and gates the optimizer
import json, os, sys, tempfile
from tinygrad import Tensor, nn, Context, GlobalCounters
from tinygrad.uop.ops import Ops
from tinygrad.uop.render import render_uir
from tinygrad.engine.realize import lower_and_compile, run_linear

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from uop2json import uop_to_graph
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def slim(g:dict) -> dict:
  # drop the shape-argument plumbing (STACK of CONSTs feeding movement ops / buffer sizes) so the picture shows the dataflow
  keep_ops = {n["id"] for n in g["nodes"] if n["op"] not in ("STACK", "CONST")}
  consumers = {}
  for s, d, _ in g["edges"]: consumers.setdefault(s, []).append(d)
  ops = {n["id"]: n["op"] for n in g["nodes"]}
  for n in g["nodes"]:  # keep CONSTs that feed real math (e.g. the 0 of relu's max), drop shape consts
    if n["op"] == "CONST" and any(ops[c] not in ("STACK", "BUFFER") for c in consumers.get(n["id"], [])): keep_ops.add(n["id"])
  return {"nodes": [n for n in g["nodes"] if n["id"] in keep_ops], "edges": [e for e in g["edges"] if e[0] in keep_ops and e[1] in keep_ops]}

Tensor.manual_seed(0)
print("=== 1. nn.Linear(3, 2) forward as UOps ===")
lin = nn.Linear(3, 2)
x = Tensor([[1., 2, 3], [4, 5, 6], [7, 8, 9], [1, 0, 1]])
Tensor.realize(lin.weight, lin.bias, x)          # make weights real buffers so the graph starts at BUFFERs
print("weight", lin.weight.shape, "bias", lin.bias.shape, "x", x.shape)
ids: dict = {}
# rebuild Linear step by step, exactly as Linear.__call__ -> Tensor.linear -> Tensor.dot do it (mixin/op.py)
wT = lin.weight.transpose()                       # Linear.__call__: x.linear(self.weight.transpose(), self.bias)
xr = x.reshape(4, 1, 3)                            # dot(): x.reshape(*x.shape[0:-1], 1, x.shape[-1])
wr = wT.reshape(1, 3, 2).transpose(-1, -2)         # dot(): w.reshape(..., 1, *w.shape[-2:]).transpose(-1, -2)
prod = xr * wr                                     # broadcasting MUL: (4,1,3) * (1,2,3) -> (4,2,3)
summed = prod.sum(-1)                              # REDUCE over the shared axis -> (4,2)
y = lin(x)                                         # the real call
stages = [("weight.transpose()", wT), ("x * w (broadcast MUL)", prod), ("(x*w).sum(-1)  == x.dot(w.T)", summed), ("+ bias  == nn.Linear(3,2)(x)", y)]
print(render_uir(y.uop))
census = {}
for u in y.uop.toposort(): census[u.op.name] = census.get(u.op.name, 0) + 1
print("op census:", dict(sorted(census.items(), key=lambda kv: -kv[1])))
graphs = [{"title": t, "graph": slim(uop_to_graph(s.uop, ids))} for t, s in stages]
print("the same graph is what you'd get by hand:", (y.uop is (summed + lin.bias).uop), "(hash-consing: identical UOps are the same object)")

print("\n=== 2. how many kernels does each layer become? ===")
def kernels_of(name, fn):
  out = fn()
  lin_ = Tensor.schedule_linear(out)               # LINEAR of CALLs; the tensors now point at (not yet written) buffers
  lin_ = lower_and_compile(lin_)                    # codegen: each CALL's body becomes a PROGRAM (that's where kernels get names)
  names = [c.src[0].src[0].arg.name if c.src[0].op is Ops.PROGRAM else "copy" for c in lin_.src]
  run_linear(lin_)                                  # actually run it, so later code sees real values
  print(f"  {name:38s} -> {len(names)} kernel(s): {names}")
  return {"layer": name, "kernels": names}
rows = []
xb = Tensor.rand(8, 3, 4, 4); xb.realize()
ln, bn, emb, conv = nn.LayerNorm(16), nn.BatchNorm(3), nn.Embedding(10, 4), nn.Conv2d(3, 4, 3)
Tensor.realize(*nn.state.get_parameters([ln, bn, emb, conv]))
xl = Tensor.rand(5, 16).realize(); idx = Tensor([1, 2, 3, 1]).realize()
rows.append(kernels_of("Linear(3,2)(x)", lambda: lin(x)))
rows.append(kernels_of("LayerNorm(16)(x[5,16])", lambda: ln(xl)))
rows.append(kernels_of("BatchNorm(3)(x[8,3,4,4])  (eval)", lambda: bn(xb)))
with Context(TRAINING=1): rows.append(kernels_of("BatchNorm(3)(x[8,3,4,4])  (TRAINING=1)", lambda: bn(xb)))
rows.append(kernels_of("Embedding(10,4)(idx[4])", lambda: emb(idx)))
rows.append(kernels_of("Conv2d(3,4,3)(x[8,3,4,4])", lambda: conv(xb)))
la, lb = nn.Linear(3, 8), nn.Linear(8, 2); Tensor.realize(*nn.state.get_parameters([la, lb]))
rows.append(kernels_of("Linear(3,8) -> relu -> Linear(8,2)", lambda: lb(la(x).relu())))

print("\n=== 3. nn.state ===")
class Net:
  def __init__(self): self.l1, self.l2, self.norm = nn.Linear(4, 5), nn.Linear(5, 6), nn.LayerNorm(6)
  def __call__(self, x): return self.norm(self.l2(self.l1(x).relu()))
net = Net()
sd = nn.state.get_state_dict(net)
Tensor.realize(*sd.values())                      # Linear weights start as lazy Tensor.uniform graphs; make them real first
for k, v in sd.items(): print(f"  {k:14s} {str(v.shape):8s} is_param={v.is_param}")
print("  get_parameters ->", len(nn.state.get_parameters(net)), "tensors (just the values of get_state_dict)")
bn_sd = nn.state.get_state_dict(bn)
print("  BatchNorm state:", {k: (tuple(v.shape), v.is_param) for k, v in bn_sd.items()})
fn = os.path.join(tempfile.gettempdir(), "ch20_net.safetensors")
nn.state.safe_save(sd, fn)
with open(fn, "rb") as f: raw = f.read()
hlen = int.from_bytes(raw[:8], "little")
print(f"  safe_save wrote {len(raw)} bytes: 8-byte header length ({hlen}) + JSON header + raw tensor bytes")
print("  header:", raw[8:8+hlen].decode().strip()[:150], "...")
loaded = nn.state.safe_load(fn)
print("  safe_load ->", {k: (v.device.split(":")[0], tuple(v.shape), v.uop.op.name) for k, v in list(loaded.items())[:2]}, "...")
print("  loaded['l1.weight'] is a lazy view of the file bytes -- nothing is read until realize:")
print("\n".join("    "+l for l in render_uir(loaded["l1.weight"].uop).splitlines()))
net2 = Net()
nn.state.load_state_dict(net2, loaded, verbose=False)
xt = Tensor.rand(2, 4).realize()
print("  same outputs after load_state_dict:", (net(xt) - net2(xt)).abs().max().item() == 0.0)

print("\n=== 4. TRAINING context var ===")
from tinygrad.helpers import TRAINING
print("  TRAINING default:", TRAINING.value)
opt = nn.optim.SGD(nn.state.get_parameters(net), lr=0.1)
try: opt.step()
except RuntimeError as e: print("  opt.step() without TRAINING ->", str(e).strip().splitlines()[0])
with Context(TRAINING=1):
  print("  inside Context(TRAINING=1):", TRAINING.value)
  _ = bn(xb).realize()
print("  BatchNorm running_mean after one TRAINING forward:", [round(v, 4) for v in bn.running_mean.tolist()])
print("  num_batches_tracked:", bn.num_batches_tracked.item(), "(we ran two TRAINING forwards: one in section 2, one here)")

json.dump({"linear_stages": graphs, "linear_uir": render_uir(y.uop), "kernel_rows": rows}, open(os.path.join(DATA, "ch20_layers.json"), "w"), indent=1)
print("\nwrote data/ch20_layers.json")
