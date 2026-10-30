# Chapter 8 — how many DISTINCT primitive ops does a real model's tensor graph use?
# Run:  DEV=CPU python3 examples/08_model_opcount.py
#
# We build two small but real models from tinygrad.nn — a conv net (Conv2d, BatchNorm, relu, max_pool2d, Linear,
# cross-entropy) and a transformer block (LayerNorm, Linear, scaled_dot_product_attention, gelu) — and count the
# Ops in the lazy graph of the loss, and of loss + all gradients.
from collections import Counter
from tinygrad import Tensor, nn, Context
from tinygrad.uop.ops import Ops, UOp
from tinygrad.nn.state import get_parameters

Tensor.manual_seed(0)
# "plumbing" ops that carry shapes / storage rather than compute
PLUMBING = {Ops.CONST, Ops.STACK, Ops.BUFFER, Ops.ALLOC, Ops.SINK, Ops.DETACH, Ops.CONTIGUOUS_BACKWARD, Ops.STAGE, Ops.COPY, Ops.AFTER, Ops.STORE}
MOVEMENT = {Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND, Ops.PAD, Ops.SHRINK}

class ConvNet:
  def __init__(self):
    self.c1, self.bn = nn.Conv2d(1, 4, 3), nn.BatchNorm(4)
    self.l1 = nn.Linear(4 * 3 * 3, 10)
  def __call__(self, x): return self.l1(self.bn(self.c1(x)).relu().max_pool2d((2, 2)).flatten(1))

class Block:
  def __init__(self, d=8, heads=2):
    self.ln1, self.ln2, self.h = nn.LayerNorm(d), nn.LayerNorm(d), heads
    self.qkv, self.o, self.f1, self.f2 = nn.Linear(d, 3 * d), nn.Linear(d, d), nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
  def __call__(self, x):
    b, t, d = x.shape
    q, k, v = [z.reshape(b, t, self.h, d // self.h).transpose(1, 2) for z in self.qkv(self.ln1(x)).chunk(3, dim=-1)]
    x = x + self.o(q.scaled_dot_product_attention(k, v, is_causal=True).transpose(1, 2).reshape(b, t, d))
    return x + self.f2(self.f1(self.ln2(x)).gelu())

CAPTURE: dict = {}
def report(name, model, x, y):
  x = x.realize()   # realize the random input so its THREEFRY/SIN (Box-Muller) graph is not counted as "model"
  for p in get_parameters(model): p.realize()
  with Context(TRAINING=1):
    loss = model(x).sparse_categorical_crossentropy(y) if y.ndim == 1 else model(x).flatten(0, 1).sparse_categorical_crossentropy(y.flatten())
  params = [p for p in get_parameters(model) if p.is_floating_point()]   # BatchNorm also holds an int counter
  fwd = Counter(u.op for u in loss.uop.toposort())
  grads = loss.gradient(*params)
  both = Counter(u.op for u in UOp.sink(loss.uop, *[g.uop for g in grads]).toposort())
  CAPTURE[name] = {"forward": {o.name: n for o, n in fwd.items()}, "forward+backward": {o.name: n for o, n in both.items()}}
  for label, c in [("forward", fwd), ("forward+backward", both)]:
    compute = {o: n for o, n in c.items() if o not in PLUMBING}
    alu = sorted((o.name for o in compute if o not in MOVEMENT and o is not Ops.REDUCE))
    mv = sorted((o.name for o in compute if o in MOVEMENT))
    print(f"{name:16s} {label:17s} {sum(c.values()):5d} UOps | {len(compute):2d} distinct compute ops: "
          f"{len(mv)} movement {mv} + REDUCE + {len(alu)} ALU {alu}")
  return both

c1 = report("ConvNet", ConvNet(), Tensor.randn(2, 1, 8, 8), Tensor([3, 7]))
c2 = report("TransformerBlock", Block(), Tensor.randn(2, 4, 8), Tensor([[1, 2, 3, 4], [5, 6, 7, 0]]))
print("\nall ops used by either model (fwd+bwd), most common first:")
tot = c1 + c2
print(", ".join(f"{o.name}:{n}" for o, n in tot.most_common()))
print(f"\nOps enum has {len(Ops)} members in total; the two models use {len(set(tot))} of them.")

import json, os
with open(os.path.join(os.path.dirname(__file__), "..", "data", "ch08_opcount.json"), "w") as f: json.dump(CAPTURE, f, indent=1)
print("wrote data/ch08_opcount.json")
