#!/usr/bin/env python3
# Chapter 2 — Tensors are lazy. What a Tensor holds, what every op builds, and what forces computation.
#
#   run:  DEV=CPU python3 examples/02_lazy_tensor.py
import weakref
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import GlobalCounters
import tinygrad.tensor as tensor_mod

def tree(u: UOp, indent=0, seen=None) -> None:
  """compact one-line-per-node printer: OP dtype shape [arg]. (print(u) gives the full official repr)"""
  seen = set() if seen is None else seen
  shape = u._shape if u._shape is not None else ""
  arg = ""
  if u.op in (Ops.BUFFER, Ops.ALLOC, Ops.PARAM): arg = f"slot={u.arg.slot} device={u.arg.device}"
  elif u.arg is not None: arg = str(u.arg)
  ref = " (shared, see above)" if u in seen and u.src else ""
  print(f"{'  '*indent}{u.op.name:8s} {str(u.dtype):17s} {str(shape):8s} {arg}{ref}")
  if ref: return
  seen.add(u)
  for s in u.src: tree(s, indent+1, seen)

def section(s): print(f"\n### {s}")

section("1. what a Tensor object is")
a = Tensor([1., 2., 3.])
print("Tensor.__slots__ =", Tensor.__slots__)
print("MRO:", " -> ".join(c.__name__ for c in Tensor.__mro__))
print("a.uop is a", type(a.uop).__name__, "| a.shape", a.shape, "| a.dtype", a.dtype, "| a.device", a.device)
print("a.grad =", a.grad, "| a.is_param =", a.is_param)
print("shape/dtype/device are not stored on the Tensor, they are read from the UOp:",
      a.shape == a.uop.shape and a.dtype == a.uop.dtype and a.device == a.uop.device)

section("2. every op builds a UOp, nothing is computed")
b = Tensor([10., 20., 30.])
GlobalCounters.reset()
c = a * b + 1
d = c.exp().sum()
print("kernels run so far:", GlobalCounters.kernel_count)
print("d.uop:"); tree(d.uop)

section("3. Tensor and UOp share the same mixins: the Tensor graph IS the UOp graph")
print("(a*b+1).uop is (a.uop*b.uop+1) ->", (a*b+1).uop is (a.uop*b.uop + 1))
print("UOp MRO also contains:", [c.__name__ for c in UOp.__mro__ if "Mixin" in c.__name__])

section("4. creation: which UOps appear")
for name, fn in [("Tensor([1,2,3])", lambda: Tensor([1, 2, 3])),
                 ("Tensor([1,2], device='PYTHON')", lambda: Tensor([1, 2], device="PYTHON")),
                 ("Tensor(3.5)", lambda: Tensor(3.5)),
                 ("Tensor.empty(2,2)", lambda: Tensor.empty(2, 2)),
                 ("Tensor.ones(2)", lambda: Tensor.ones(2)),
                 ("Tensor.full((2,3), 7, buffer=False)", lambda: Tensor.full((2, 3), 7, buffer=False))]:
  t = fn()
  print(f"-- {name}:  dtype={t.dtype} device={t.device}")
  tree(t.uop, 1)
r = Tensor.rand(3)
print(f"-- Tensor.rand(3): {len(r.uop.toposort())} UOps (a threefry PRNG graph), top op = {r.uop.op.name}")

section("5. broadcasting lives in the shape rule of the ALU op")
x, y = Tensor.empty(3, 1), Tensor.empty(4)
z = x + y
print("x.shape", x.shape, "+ y.shape", y.shape, "->", z.shape)
tree(z.uop)
print("explicit x.expand(3,5):"); tree(x.expand(3, 5).uop)

section("6. realize() swaps graphs for buffers in EVERY live tensor (all_tensors + _apply_map_to_tensors)")
e = a * b
f = e + 1
print("live tensors tracked in tensor.all_tensors:", len(tensor_mod.all_tensors))
print("f before e.realize():"); tree(f.uop)
e.realize()
print("e after realize:"); tree(e.uop)
print("f after e.realize() (f was rewritten too, it now reads e's BUFFER):"); tree(f.uop)

section("7. tolist()/item()/numpy() force realization -- but of a COPY, not in place")
g = a + b
for i in range(2):
  GlobalCounters.reset(); g.tolist(); print(f"g.tolist() #{i+1}: ran {GlobalCounters.kernel_count} kernels, g.uop.op is still {g.uop.op.name}")
GlobalCounters.reset(); g.realize(); print(f"g.realize(): ran {GlobalCounters.kernel_count} kernels, g.uop.op is now {g.uop.op.name}")
GlobalCounters.reset(); print("g.tolist() =", g.tolist(), f"ran {GlobalCounters.kernel_count} kernels")
print("Tensor(5).item() =", Tensor(5).item(), "(a CONST with no device is cloned to CPU first)")
try: bool(a)
except TypeError as err: print("bool(a) ->", type(err).__name__, err)

section("8. contiguous() adds a STAGE: 'materialize me here'")
tree((a + b).contiguous().uop)

section("9. assign(): a STORE wrapped in AFTER")
w = Tensor([1, 2, 3]).realize()
w.assign(w + 1)
tree(w.uop)
print("w after realize:", w.realize().tolist(), "| w.uop.op =", w.uop.op.name)

section("10. weak references: dead tensors leave all_tensors")
n0 = len(tensor_mod.all_tensors)
tmp = Tensor([1.0]) * 2
print("after creating tmp:", len(tensor_mod.all_tensors) - n0, "new live tensors")
ref = weakref.ref(tmp); del tmp
print("after del tmp: weakref is dead ->", ref() is None, "| net new live tensors:", len(tensor_mod.all_tensors) - n0)
