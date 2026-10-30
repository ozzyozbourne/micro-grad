# Chapter 4 — the SAME IR (UOp) at three altitudes.  Run with:  DEV=CPU python3 examples/04_three_altitudes.py
#   altitude 1: the tensor graph   (what Tensor ops build: BUFFER, RESHAPE, PERMUTE, REDUCE ... shapes, no loops)
#   altitude 2: the kernel AST     (after scheduling: PARAM, RANGE, INDEX, REDUCE, STORE, END, SINK ... loops, no shapes)
#   altitude 3: the linear program (after codegen: a flat list of UOps, LOAD/STORE, accumulators, ready to print as C)
from tinygrad import Tensor, Device, Context
from tinygrad.uop.ops import Ops
from tinygrad.codegen import to_program
from tinygrad.engine.realize import run_linear

a = Tensor([[1., 2, 3], [4, 5, 6]]).realize()   # realize so the inputs are plain BUFFERs on CPU
b = Tensor([10., 20, 30]).realize()
c = (a * 2 + b).sum(axis=1)                       # expected: [2+4+6+60, 8+10+12+60] = [72, 90]

print("=== altitude 1: tensor graph (c.uop) ===")
print(c.uop.render_uir())

print("\n=== altitude 2: kernel AST (src[0] of the CALL in the schedule) ===")
linear = c.schedule_linear()                       # LINEAR( CALL(SINK ast, buffers...), ... )
call = linear.src[-1]
ast = call.src[0]
print("schedule:", linear.op, "with", len(linear.src), "CALL(s); the CALL's srcs:", [s.op for s in call.src])
print(ast.render_uir())

print("\n=== altitude 3: linearized program (PROGRAM.src[1] is a LINEAR of UOps) ===")
with Context(NOOPT=1):                             # no optimizations so the program stays readable
  prg = to_program(ast, Device["CPU"].renderer)
print("PROGRAM srcs:", [s.op for s in prg.src])
lin = prg.src[1].src
for i, u in enumerate(lin):
  srcs = ", ".join(f"%{lin.index(s)}" for s in u.src)
  arg = "" if u.arg is None else f"  : {u.arg if u.op is not Ops.SINK else u.arg.name}"
  print(f"%{i:<3}{u.op.name.lower():7s} {str(u.dtype.name):9s} {srcs}{arg}")

print("\n=== ...which renders to C (PROGRAM.src[2] is a SOURCE) ===")
print(prg.src[2].arg)

run_linear(linear)                                 # schedule_linear only PLANS the work; this runs it
print("result:", c.tolist())
