"""examples/10_realize_map.py — which UOps get realized, and the ranges every UOp gets (run_rangeify's debug table)

Run:  DEV=CPU python3 examples/10_realize_map.py

DEBUG_RANGEIFY=1 makes run_rangeify (tinygrad/schedule/indexing.py) print one line per UOp as it walks the graph
from the outputs down to the buffers (reversed toposort):

    ***  2 Ops.EXP2   (4, 8)   0 [r4][r5]
    ^^^  ^ ^^^^^^^^   ^^^^^^   ^ ^^^^^^^^
    |    | op         shape    | ranges ("in -> out" when they differ; yellow in a terminal = realized axes)
    |    | number of consumers  number of "ending ranges" (ranges a consumer broadcasts over)
    "***" = this UOp is in the realize_map: it gets FRESH output ranges and will be wrapped in a STAGE (bufferize)

The final line of each program tells you how many kernels survived (the realize map is only the list of CANDIDATES;
chapter 11 shows how pm_remove_bufferize throws the cheap ones away again).
"""
from tinygrad import Tensor
from tinygrad.helpers import Context

def run(title, make):
  print(f"\n######## {title}")
  out = make()
  with Context(DEBUG_RANGEIFY=1, SCACHE=0):
    linear, var_vals = Tensor.linear_with_vars(out)
  print(f"-> {len(linear.src)} kernel(s)")

run("elementwise + reduce: (a.permute(1,0)+1).sum(0)", lambda: (Tensor.empty(3, 4).permute(1, 0) + 1).sum(0))
run("softmax over the last axis", lambda: Tensor.empty(4, 8).softmax(-1))
run("one value, two different readers: y = x*3; y.sum(0) + y.max(0)", lambda: (lambda y: y.sum(0) + y.max(0))(Tensor.empty(8, 4) * 3))
run("same value read transposed: y = x+1; y + y.T", lambda: (lambda y: y + y.T)(Tensor.empty(4, 4) + 1))
run("pad then slice then reduce", lambda: Tensor.empty(4, 4).pad(((1, 1), (1, 1)))[:4, 2:].sum(1))
