# Chapter 20 — ONE training step of a small MLP, traced through the whole stack
# Run: DEV=CPU python3 examples/20_train_step.py
#
#   forward graph -> loss -> backward (gradient UOps) -> optimizer assign -> schedule (LINEAR of CALLs)
#   -> codegen (one C function per kernel) -> run
#
# Nothing here is special API: it is exactly what `loss.realize(*opt.schedule_step())` does inside
# Tensor.realize (tensor.py), split into its two halves so we can look in between:
#   linear = Tensor.schedule_linear(...)      # tensor graph -> LINEAR(CALL, CALL, ...)
#   run_linear(linear)                        # lower_and_compile + execute every CALL
import json, math, os, random, sys
from collections import Counter
from tinygrad import Tensor, nn, Context
from tinygrad.uop.ops import Ops, UOp
from tinygrad.uop.render import render_uir
from tinygrad.engine.realize import lower_and_compile, run_linear

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from uop2json import uop_to_graph
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# ---------- synthetic data: a two-armed spiral (pure python, no numpy) ----------
def spiral(n_per_class=64, seed=0):
  rnd = random.Random(seed)
  pts, labels = [], []
  for c in range(2):
    for i in range(n_per_class):
      r = 0.15 + 0.85 * i / n_per_class
      t = 1.6 * math.pi * i / n_per_class + c * math.pi + rnd.gauss(0, 0.12)
      pts.append([r * math.cos(t), r * math.sin(t)]); labels.append(c)
  return pts, labels

class MLP:
  def __init__(self, hidden=32):
    self.l1, self.l2 = nn.Linear(2, hidden), nn.Linear(hidden, 2)
  def __call__(self, x:Tensor) -> Tensor: return self.l2(self.l1(x).relu())

if __name__ == "__main__":
  Tensor.manual_seed(0)
  pts, labels = spiral()
  X, Y = Tensor(pts), Tensor(labels)                  # (128, 2) float32, (128,) int32
  model = MLP()
  params = nn.state.get_parameters(model)
  opt = nn.optim.Adam(params, lr=0.02)
  # realize everything that exists *before* the step so the step's graph starts at real BUFFERs
  Tensor.realize(*params, X, Y, *opt.m, *opt.v, opt.b1_t, opt.b2_t, opt.lr)
  names = {X.uop: "X", Y.uop: "Y", opt.lr.uop: "adam.lr", opt.b1_t.uop: "adam.b1_t", opt.b2_t.uop: "adam.b2_t"}
  for k, v in nn.state.get_state_dict(model).items(): names[v.uop] = k
  for i, (k, _) in enumerate(nn.state.get_state_dict(model).items()): names[opt.m[i].uop], names[opt.v[i].uop] = f"adam.m[{k}]", f"adam.v[{k}]"
  bufname = {u.buffer: n for u, n in names.items()}
  print("parameters:", {k: tuple(v.shape) for k, v in nn.state.get_state_dict(model).items()})
  print("X", X.shape, X.dtype, " Y", Y.shape, Y.dtype)

  def census(u:UOp) -> str:
    c = Counter(x.op.name for x in u.toposort())
    return f"{sum(c.values())} UOps: " + ", ".join(f"{k}={v}" for k, v in c.most_common(9))

  with Context(TRAINING=1):
    opt.zero_grad()
    # ---------- 1. forward ----------
    logits = model(X)
    print("\n[1] forward  logits:", logits.shape, "|", census(logits.uop))
    loss = logits.sparse_categorical_crossentropy(Y)
    print("    loss    :", loss.shape, "|", census(loss.uop))
    fwd_uops = set(loss.uop.toposort())
    # ---------- 2. backward ----------
    loss.backward()
    g = model.l2.bias.grad
    print("\n[2] backward: every param now has .grad, a *lazy* Tensor (nothing computed yet)")
    for k, v in nn.state.get_state_dict(model).items():
      new = len(set(v.grad.uop.toposort()) - fwd_uops)
      print(f"    {k:10s}.grad {str(v.grad.shape):8s} graph has {len(v.grad.uop.toposort()):3d} UOps, {new:3d} of them new (not in the forward graph)")
    print("    l2.bias.grad as UIR (shared forward subgraph included):")
    uir = render_uir(g.uop).splitlines()
    print("\n".join("      " + l for l in uir[-12:]))
    print(f"      ... ({len(uir)} lines total)")
    # ---------- 3. optimizer ----------
    before = model.l2.bias.uop
    outs = opt.schedule_step()                         # builds assign()s, does NOT run anything
    after = model.l2.bias.uop
    print("\n[3] opt.schedule_step() returned", len(outs), "tensors to realize: b1_t, b2_t, 4 m's, 4 v's, then the 4 params")
    print("    l2.bias.uop before:", before.op, "  after:", after.op, "with src ops", [s.op.name for s in after.src])
    store = after.src[1]
    print("    the STORE writes into", store.src[0].op.name, "the value", store.src[1].op.name, f"({len(store.src[1].toposort())} UOps deep)")
    # ---------- 4. schedule ----------
    linear = Tensor.schedule_linear(loss, *outs)
  print(f"\n[4] Tensor.schedule_linear -> {linear.op.name} with {len(linear.src)} CALLs")

  # ---------- 5. codegen every kernel ----------
  compiled = lower_and_compile(linear)
  def base(u:UOp) -> UOp:
    while u.op in (Ops.BITCAST, Ops.SHRINK, Ops.RESHAPE): u = u.src[0]
    return u
  bufname[loss.uop.buffer] = "loss"
  def argname(u:UOp) -> str:
    b = base(u)
    n = bufname.get(b.buffer) if b.op is Ops.BUFFER else None
    return n if n is not None else "tmp" + str(tuple(u.shape))
  kernels, last_writer = [], {}
  for i, c in enumerate(compiled.src):
    c = c.without_after
    prg = c.body
    args = list(c.src[1:])
    outs_i, ins_i = [args[j] for j in prg.arg.outs], [args[j] for j in prg.arg.ins]
    src = next(s.arg for s in prg.src if s.op is Ops.SOURCE)
    deps = sorted({last_writer[a] for a in ins_i if a in last_writer})
    for a in outs_i: last_writer[a] = i
    kernels.append({"i": i, "name": prg.src[0].arg.name, "outs": [argname(a) for a in outs_i], "ins": [argname(a) for a in ins_i],
                    "deps": deps, "src": src, "ast": render_uir(linear.src[i].src[0]), "opts": str(prg.src[0].arg.applied_opts)})
  # label phases from dataflow: forward = everything the loss kernel transitively depends on
  loss_k = next(k["i"] for k in kernels if "loss" in k["outs"])
  fwd, stack = set(), [loss_k]
  while stack:
    j = stack.pop()
    if j in fwd: continue
    fwd.add(j); stack += kernels[j]["deps"]
  for k in kernels:
    touches_opt = any(n.startswith("adam.") for n in k["outs"] + k["ins"]) or any(n in names.values() and not n.startswith(("X", "Y", "adam")) for n in k["outs"])
    k["phase"] = "forward" if k["i"] in fwd else "update" if touches_opt else "backward"
  print("\n[5] the kernels, in schedule order (phase is derived from dataflow, see the script):")
  for k in kernels:
    print(f"    {k['i']:2d} {k['phase']:8s} {k['name']:14s} writes {', '.join(k['outs']):16s} reads {', '.join(k['ins'])}")
  print("    per phase:", dict(Counter(k["phase"] for k in kernels)))
  for k in (kernels[0], kernels[loss_k]):
    print(f"\n    --- generated C for kernel {k['i']} ({k['name']}, {k['phase']}) ---")
    print("\n".join("    " + l for l in k["src"].strip().splitlines()))

  # ---------- 6. run ----------
  run_linear(linear)
  print(f"\n[6] run_linear done. loss = {loss.item():.4f}")
  print("    Adam bias-correction state b1_t, b2_t =", opt.b1_t.item(), opt.b2_t.item(), "(= 0.9**1, 0.999**1)")

  # ---------- data for the chapter's timeline demo ----------
  json.dump({"kernels": kernels, "loss": loss.item()}, open(os.path.join(DATA, "ch20_step.json"), "w"), indent=1)
  print("\nwrote data/ch20_step.json")
