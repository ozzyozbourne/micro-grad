"""examples/11_recompute_tradeoff.py — fusion is a trade: one kernel for x - x.mean() means redundant work on a GPU

Run:  DEV=CPU python3 examples/11_recompute_tradeoff.py

x.mean() of a (16, 16) tensor has shape (): it has no axes, so run_rangeify has nothing to realize and the REDUCE
is fused into the output kernel. We render that single kernel twice from the SAME kernel AST:
  * for the CPU (clang): the reduce loop does not depend on the output loop, so it is emitted once, before it
  * for OpenCL (no GPU needed, we only render source): the output loop becomes 256 work-groups (gidx0) and EVERY
    work-group recomputes the full 256-element sum (with a 16-wide local GROUP reduction).
"""
from tinygrad import Tensor
from tinygrad.uop.ops import Ops
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import OpenCLRenderer
from tinygrad.device import Device

x = Tensor.empty(16, 16)
linear, _ = (x - x.mean()).linear_with_vars()
print(f"{len(linear.src)} kernel(s) scheduled")
ast = linear.src[0].body
for title, renderer in [("CPU (clang)", Device["CPU"].renderer), ("OpenCL", OpenCLRenderer(Target("CL")))]:
  prg = to_program(ast, renderer)
  src = [u.arg for u in prg.toposort() if u.op is Ops.SOURCE][0]
  print(f"===== {title}\n{src}")
