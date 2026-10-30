"""examples/10_apply_rangeify.py — the graph BEFORE and AFTER run_rangeify, printed as real UOps

Run:  DEV=CPU python3 examples/10_apply_rangeify.py

Uses the pass recorder from 10_sched_capture.py to grab the input of the "get realize" pass (the tensor graph that
run_rangeify receives) and the output of "apply rangeify" (the same computation written with RANGEs and INDEX).
"""
import importlib.util, pathlib
from tinygrad import Tensor
from tinygrad.uop.ops import Ops

spec = importlib.util.spec_from_file_location("cap", pathlib.Path(__file__).parent / "10_sched_capture.py")
cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)

def before_after(title, out):
  log, linear, compiled = cap.schedule(out)
  before = cap.pass_by_name(log, "get realize")[1]
  after = cap.pass_by_name(log, "apply rangeify")[2]
  cleaned = cap.pass_by_name(log, "symbolic+reduce_collapse+debuf")[2]
  print(f"################ {title}")
  print("---- BEFORE run_rangeify (shapes + movement ops):")
  print(before)
  print("---- AFTER 'apply rangeify' (RANGE / INDEX / STAGE, no movement ops):")
  print(after)
  count = lambda g, op: sum(1 for u in g.toposort() if u.op is op)
  print(f"---- STAGE count: after 'apply rangeify' = {count(after, Ops.STAGE)}, "
        f"after 'symbolic+reduce_collapse+debuf' = {count(cleaned, Ops.STAGE)}; kernels = {len(linear.src)}\n")

# 1. PAD becomes a WHERE on a validity condition, SHRINK becomes +2, the sum gets its own RANGE
before_after("x.pad(((1,1),(1,1)))[:4, 2:].sum(1)", Tensor.empty(4, 4).pad(((1, 1), (1, 1)))[:4, 2:].sum(1))
# 2. a value read by two consumers that disagree on ranges gets a STAGE (bufferize) ... which is later removed
before_after("y = x+1; y + y.T", (lambda y: y + y.T)(Tensor.empty(4, 4) + 1))
