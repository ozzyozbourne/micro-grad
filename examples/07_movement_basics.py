# Chapter 7 — movement ops, part 1: what a movement op *is* in the tensor graph.
# Run:  DEV=CPU python3 examples/07_movement_basics.py
#
# What this prints:
#   1. the raw UOp for each of the five primitive movement ops (note: the "arg" lives in src[1:], not in .arg)
#   2. the decoded arguments (UOp.marg) and the inferred shape (UOp._shape)
#   3. which primitive ops the friendly Tensor methods turn into
from tinygrad import Tensor
from tinygrad.uop.ops import Ops, GroupOp

print("GroupOp.Movement =", sorted(o.name for o in GroupOp.Movement))
print("is FLIP an Op?   =", hasattr(Ops, "FLIP"))

# A Tensor made from a Python list lives in a flat 1-D buffer; its 2-D shape is itself a RESHAPE view.
a = Tensor([[0., 1, 2], [3, 4, 5]]).realize()
print("\n--- the base tensor a (shape", a.shape, ")")
print(a.uop)

def short(u):
  # compact one-line description of a movement UOp: op, decoded arg, and the ops of its shape-arg srcs
  return f"{u.op.name:8s} marg={u.marg!s:28s} srcs={[s.op.name for s in u.src]}  ->  shape {u.shape}"

print("\n--- the five primitives, applied to a")
print(short(a.reshape(3, 2).uop))
print(short(a.permute(1, 0).uop))
print(short(a.reshape(1, 2, 3)._mop(Ops.EXPAND, (4,)).uop))   # raw EXPAND: arg = the NEW LEADING dims
print(short(a.pad(((1, 1), (0, 2))).uop))                       # PAD marg = (offset, new_size) per axis
print(short(a.shrink(((0, 1), (1, 3))).uop))                    # SHRINK marg = (offset, size) per axis

print("\n--- a full movement UOp: the args are UOps (STACKs of CONSTs), shared by hash-consing")
print(a.pad(((1, 1), (0, 2))).uop)

def prims(t):
  # list the primitive movement ops (bottom-up) that sit between t and its buffer
  out, u = [], t.uop
  while u.op in GroupOp.Movement:
    out.append(f"{u.op.name}{u.marg}")
    u = u.src[0]
  return out[::-1][1:]   # drop the RESHAPE that gives the buffer its shape

print("\n--- high-level methods -> primitives")
x = Tensor([[0., 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]).realize()   # shape (3, 4)
for name, t in [
    ("x.T", x.T), ("x.flatten()", x.flatten()), ("x.unsqueeze(1)", x.unsqueeze(1)),
    ("x.reshape(1,3,4).squeeze(0)", x.reshape(1, 3, 4).squeeze(0)),
    ("x[1]", x[1]), ("x[1:3, ::2]", x[1:3, ::2]), ("x[:, None]", x[:, None]),
    ("x[:, 1:2].expand(3, 5)", x[:, 1:2].expand(3, 5)),
    ("x.repeat(2, 1)", x.repeat(2, 1)), ("x.roll(1, 1)", x.roll(1, 1)),
    ("x.pad(((0,0),(1,1)))", x.pad(((0, 0), (1, 1)))),
    ("x.flip(1)", x.flip(1)),
  ]:
  p = prims(t)
  print(f"{name:28s} {str(t.shape):10s} {len(p):2d} ops: " + " -> ".join(p))

print("\n--- the values (realized), to convince ourselves the tricks are right")
print("x.flip(1)    =", x.flip(1).tolist())
print("x.roll(1, 1) =", x.roll(1, 1).tolist())
print("x[1:3, ::2]  =", x[1:3, ::2].tolist())
