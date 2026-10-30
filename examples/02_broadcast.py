#!/usr/bin/env python3
# Chapter 2 — broadcasting. Two separate mechanisms:
#   1. ALU ops (ADD, MUL, ...) broadcast IMPLICITLY: UOp._shape computes _broadcast_shape of the inputs, no op is added.
#   2. explicit .expand() / ._broadcast_to() emits RESHAPE (squeeze the 1s) -> EXPAND (adds dims on the LEFT) -> PERMUTE (back).
#
#   run:  DEV=CPU python3 examples/02_broadcast.py
from tinygrad import Tensor
from tinygrad.uop.ops import _broadcast_shape, _align_left, broadcast_axes

def X(shape):  # an ordinary lazy f32 value of this shape
  return Tensor.empty(*shape).sin() if shape else Tensor.empty(1).reshape(()).sin()

def chain(u, x):  # the ops between u and x, outermost first
  out = []
  while u is not x: out.append(f"{u.op.name}{u.shape}"); u = u.src[0]
  return " <- ".join(out) or "(nothing: x itself)"

print("### 1. implicit broadcasting in ALU ops: only the shape rule changes")
for a, b in [((3, 1), (4,)), ((2, 1, 4), (3, 1)), ((5,), ()), ((2, 3), (3, 2))]:
  try:
    z = X(a) + X(b)
    print(f"{str(a):10s} + {str(b):8s} -> {z.shape}   z.uop.op={z.uop.op.name} srcs={[s.shape for s in z.uop.src]}")
  except Exception as e:
    print(f"{str(a):10s} + {str(b):8s} -> {type(e).__name__}: {e}")

print("\n### 2. the rule, step by step: _align_left then per-dim 'the non-1 sizes must agree'")
a, b = (2, 1, 4), (3, 1)
print("_align_left", a, b, "=", _align_left(a, b), "-> _broadcast_shape =", _broadcast_shape(a, b))

print("\n### 3. explicit _broadcast_to: RESHAPE -> EXPAND(left) -> PERMUTE")
for a, b in [((3, 1), (3, 5)), ((3,), (4, 3)), ((1, 3), (2, 4, 3)), ((2, 1, 4), (2, 3, 4)), ((1,), (5,)), ((3, 1), (2, 3, 4)), ((), (2, 3))]:
  x = X(a)
  print(f"{str(a):10s} -> {str(b):10s} broadcast_axes={str(broadcast_axes(a, b)):8s} : {chain(x._broadcast_to(b).uop, x.uop)}")
