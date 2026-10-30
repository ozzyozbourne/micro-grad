"""examples/10_movement_index.py — movement ops become index arithmetic (tinygrad/schedule/indexing.py: apply_movement_op)

Run:  DEV=CPU python3 examples/10_movement_index.py

For each chain of movement ops we
  1. create one RANGE per output axis (the loop variables of the consumer),
  2. walk from the output down to the buffer, calling the REAL apply_movement_op for each op
     (this is exactly what run_rangeify does for every movement op it visits),
  3. verify the final index expression by evaluating it for every output element and comparing with the real
     values tinygrad computes for the same view of arange(N).
"""
import itertools
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, GroupOp
from tinygrad.schedule.indexing import apply_movement_op

def show(desc, base_shape, fn):
  print(f"=== {desc}")
  base = Tensor.arange(int(__import__('math').prod(base_shape))).reshape(base_shape).contiguous().realize()
  y = fn(base)
  # collect the movement ops between the output and the realized buffer
  u, mops = y.uop, []
  while u.op in GroupOp.Movement: mops.append(u); u = u.src[0]
  rngs = tuple(UOp.range(s, i) for i, s in enumerate(y.shape))     # RANGE(size, axis id) per output axis
  idx = rngs
  print(f"  output shape {y.shape}, loop vars {[r.render() for r in rngs]}")
  for m in mops:
    idx = apply_movement_op(m.op, m.src[0].shape, m.marg, idx)
    print(f"  {m.op.name:8s} arg={str(m.marg):22s} into shape {str(m.src[0].shape):9s} -> {[i.render() for i in idx]}")
  # the base after .contiguous().realize() is a reshape of a flat buffer, so the last index is a flat offset
  expr = idx[0] if len(idx) == 1 else None
  if expr is not None:
    real = y.flatten().tolist()
    ok = True
    for flat, pos in enumerate(itertools.product(*[range(s) for s in y.shape])):
      v = expr.substitute({r: r.const_like(p) for r, p in zip(rngs, pos)}).simplify()
      val = None if v.op is Ops.CONST and str(v.arg) == "Invalid" else (v.arg if v.op is Ops.CONST else v.render())
      # a padded (Invalid) position reads nothing: the real tensor shows the pad value 0 there
      want = real[flat]
      got = 0 if val is None else val
      if got != want: ok = False
    print(f"  check: evaluating the index for all {len(real)} outputs reproduces y.tolist() -> {'OK' if ok else 'MISMATCH'}")

show("transpose: reshape(3,4) then permute(1,0)", (12,), lambda x: x.reshape(3, 4).permute(1, 0))
show("slice: x[1:3, 1:]  (SHRINK adds the offset)", (3, 4), lambda x: x[1:3, 1:])
show("pad: x.pad(((1,1),(0,2)))  (PAD adds a validity condition)", (3, 4), lambda x: x.pad(((1, 1), (0, 2))))
show("broadcast: x.reshape(3,1).expand(3,4)  (EXPAND drops the range)", (3,), lambda x: x.reshape(3, 1).expand(3, 4))
show("two reshapes collapse: reshape(2,6).reshape(4,3)", (12,), lambda x: x.reshape(2, 6).reshape(4, 3))
show("flip(0): there is no FLIP op, but the index still comes out as 3-r0", (4,), lambda x: x.flip(0))
