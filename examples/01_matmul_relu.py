#!/usr/bin/env python3
# Chapter 1 — a second program through the pipeline: a small matmul followed by relu.
#
#   run:  DEV=CPU python3 examples/01_matmul_relu.py
#         DEV=CPU DEBUG=2 python3 examples/01_matmul_relu.py     # + one line per executed kernel/copy
#         DEV=CPU DEBUG=4 python3 examples/01_matmul_relu.py     # + applied opts and the generated C source
#         DEV=CPU DEBUG=4 NOOPT=1 python3 examples/01_matmul_relu.py   # the same kernel without optimizations
#
# Point of this example: matmul is NOT a primitive. `x @ w` becomes movement ops (RESHAPE / EXPAND / PERMUTE),
# an elementwise MUL and a REDUCE(ADD). relu() becomes a MAX-like WHERE/CMPLT. The scheduler fuses all of it
# into ONE kernel with three loops.
from tinygrad import Tensor, Device
from tinygrad.uop.ops import Ops
from tinygrad.helpers import DEBUG
from tinygrad.codegen import to_program

x = Tensor([[1., -2, 3, -4],
            [5., -6, 7, -8]])               # shape (2, 4)
w = Tensor([[1., 0, -1],
            [0., 1,  0],
            [1., 1,  1],
            [-1, 0,  1]])                   # shape (4, 3)
y = (x @ w).relu()                          # shape (2, 3), still nothing computed

def op_histogram(u):
  h: dict[str, int] = {}
  for n in u.toposort(): h[n.op.name] = h.get(n.op.name, 0) + 1
  return h

print("y.shape =", y.shape, " y.dtype =", y.dtype, " y.device =", y.device)
print("ops in the lazy tensor graph:", op_histogram(y.uop))
print("the top of the graph (relu is a WHERE of a comparison):")
top = y.uop
print(f"  {top.op.name}( {', '.join(s.op.name for s in top.src)} )")

# schedule: Tensor -> LINEAR of CALLs. (realize() would do this and then run it)
linear = y.schedule_linear()
print(f"\nschedule: {len(linear.src)} CALLs ->", [si.src[0].op.name for si in linear.src])
ast = next(si.src[0] for si in linear.src if si.src[0].op is Ops.SINK)
print("ops in the kernel AST:", op_histogram(ast))
print("RANGEs in the kernel AST:", sorted([(u.arg, int(u.vmax)+1) for u in ast.toposort() if u.op is Ops.RANGE], key=str))

if DEBUG < 4:   # with DEBUG>=4 tinygrad prints the source itself
  prg = to_program(ast, Device["CPU"].renderer)
  print("\ngenerated C for", prg.src[0].arg.name, "with opts", prg.src[0].arg.applied_opts)
  print(prg.src[2].arg)

# schedule_linear() already re-pointed y.uop at its (still empty!) output BUFFER, so we must run the schedule.
# (calling y.tolist() without this would read uninitialized memory: y "has a buffer" and nothing is left to compute)
from tinygrad.engine.realize import run_linear
run_linear(linear)
print("result:", y.tolist())

