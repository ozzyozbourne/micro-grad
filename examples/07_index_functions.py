# Chapter 7 — movement ops, part 2: a movement op is an INDEX FUNCTION (output index -> input index [+ valid]).
# Run:  DEV=CPU NOOPT=1 python3 examples/07_index_functions.py
#
# What this prints:
#   1. apply_movement_op (tinygrad/schedule/indexing.py) for each primitive, on symbolic RANGE variables r0, r1, ...
#   2. whole chains of movement ops composed the way rangeify composes them, then simplified with symbolic
#   3. the C code tinygrad generates for a view: no copies, just an index expression
#   4. how many kernels views and contiguous() cost
from tinygrad import Tensor, Device
from tinygrad.uop.ops import UOp, Ops, GroupOp, graph_rewrite
from tinygrad.schedule.indexing import apply_movement_op
from tinygrad.uop.symbolic import symbolic, pm_simplify_valid
from tinygrad.codegen import to_program

print("=== 1. each primitive as a function of the OUTPUT ranges r0, r1, ...")
def one(op, in_shape, arg, out_shape):
  rngs = tuple(UOp.range(s, i) for i, s in enumerate(out_shape))         # one loop variable per output axis
  ins = apply_movement_op(op, in_shape, arg, rngs)                       # -> one index expression per INPUT axis
  print(f"{op.name:8s} in{in_shape} arg={arg} out{out_shape}:  input index = [{', '.join(i.render() for i in ins)}]")
one(Ops.RESHAPE, (2, 3), (3, 2), (3, 2))
one(Ops.RESHAPE, (2, 3), (6,), (6,))
one(Ops.PERMUTE, (2, 3, 4), (2, 0, 1), (4, 2, 3))
one(Ops.EXPAND, (3,), (4,), (4, 3))
one(Ops.PAD, (2, 3), ((1, 4), (0, 5)), (4, 5))
one(Ops.SHRINK, (4, 5), ((1, 2), (2, 3)), (2, 3))

print("\n=== 2. whole chains: push the output ranges through every movement op, top to bottom (what rangeify does)")
def flat_index(t: Tensor) -> str:
  u = t.uop
  cur, valids = tuple(UOp.range(s, i) for i, s in enumerate(u.shape)), []
  while u.op in GroupOp.Movement:                 # walk from the view down to the buffer
    cur = apply_movement_op(u.op, u.src[0].shape, u.marg, cur)
    if u.op is Ops.PAD: valids += [r.get_valid() for r in cur]   # rangeify turns each PAD into a WHERE right here
    cur = tuple(r.get_idx() for r in cur)
    u = u.src[0]
  out = graph_rewrite(cur[0].valid(UOp.const(True).uprod(valids)), symbolic + pm_simplify_valid, name="simplify")
  return out.render()
base = Tensor([0., 1, 2, 3, 4, 5]).realize()          # a flat buffer of 6 floats
m = base.reshape(2, 3)
for name, t in [("reshape(2,3)", m), ("reshape(2,3).T", m.T), ("flip(0)  [12 mops!]", base.flip(0)),
                ("reshape(2,3).flip(1)", m.flip(1)), ("reshape(2,3).T.flip(0)", m.T.flip(0)),
                ("[::2]", base[::2]), ("[::-1]", base[::-1]), ("[1:5]", base[1:5]),
                ("reshape(1,6).expand(3,6)", base.reshape(1, 6).expand(3, 6)), ("repeat(2)", base.repeat(2)),
                ("roll(2)", base.roll(2)), ("reshape(2,3).pad(((1,0),(0,1)))", m.pad(((1, 0), (0, 1)))),
                ("base[:3].diag()", base[:3].diag())]:
  print(f"{name:34s} shape {str(t.shape):8s} buf[{flat_index(t)}]")

print("\n=== 3. the generated C for (m.T.flip(0) + 1): the view is only an index expression")
print("value:", (m.T.flip(0) + 1).tolist())   # realize a fresh copy first: schedule_linear() below rebinds `out` without running it
out = m.T.flip(0) + 1
for call in out.schedule_linear().src:
  if call.src[0].op is Ops.SINK:
    prg = to_program(call.src[0], Device["CPU"].renderer)
    print(next(x for x in prg.src if x.op is Ops.SOURCE).arg)

print("=== 4. kernel counts (CALLs of compiled kernels in the schedule)")
def kernels(*ts):
  return sum(1 for c in Tensor.schedule_linear(*ts).src if c.op is Ops.CALL and c.src[0].op is Ops.SINK)
print("m.T                     :", kernels(m.T), "kernels (a view of a realized buffer needs no work)")
print("m.reshape(3,2).contiguous():", kernels(m.reshape(3, 2).contiguous()), "kernels (contiguous already — same memory)")
print("base[1:5].contiguous()  :", kernels(base[1:5].contiguous()), "kernels (a contiguous slice is a sub-buffer view)")
print("m.T.contiguous()        :", kernels(m.T.contiguous()), "kernel  (a real copy into a fresh buffer)")
print("(m.T + 1)               :", kernels(m.T + 1), "kernel  (the permute is fused into the add's load)")
print("contiguous() is a STAGE UOp:", m.T.contiguous().uop.op)
