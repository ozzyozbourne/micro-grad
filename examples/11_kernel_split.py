"""examples/11_kernel_split.py — from STAGE to STORE to CALL: the last three passes of get_kernel_graph

Run:  DEV=CPU python3 examples/11_kernel_split.py

Program: out = x - x.mean(1, keepdim=True) on a (4, 4) tensor -> 2 kernels (row means, then subtract).
We print the graph after
  1. "symbolic+reduce_collapse+debuf": the surviving STAGE (the row mean) is still a value node
  2. "stage to store": bufferize_to_store turned it into AFTER(ALLOC, END(STORE(INDEX(ALLOC, r), value), r))
  3. "split kernels": split_store wrapped every closed STORE/END into SINK(KernelInfo).call(*buffers)
"""
import importlib.util, pathlib
from tinygrad import Tensor
from tinygrad.uop.ops import Ops

spec = importlib.util.spec_from_file_location("cap", pathlib.Path(__file__).parent / "10_sched_capture.py")
cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)

x = Tensor.empty(4, 4)
log, linear, compiled = cap.schedule(x - x.mean(1, keepdim=True), run=False)
for name in ["symbolic+reduce_collapse+debuf", "stage to store", "split kernels"]:
  g = cap.pass_by_name(log, name)[2]
  ops = {}
  for u in g.toposort(): ops[u.op.name] = ops.get(u.op.name, 0) + 1
  print(f"======== after '{name}'   ({len(g.toposort())} UOps; STAGE={ops.get('STAGE',0)} ALLOC={ops.get('ALLOC',0)} "
        f"STORE={ops.get('STORE',0)} END={ops.get('END',0)} CALL={ops.get('CALL',0)})")
  print(g)
print(f"======== create_schedule: {linear.op.name} of {len(linear.src)} CALLs")
for i, call in enumerate(linear.src):
  print(f"  CALL {i}: body {call.body.op.name}, args: " + ", ".join(f"{s.op.name}({s.max_numel()} x {s.dtype})" for s in call.src[1:]))
