"""examples/11_remove_bufferize.py — realize CANDIDATES vs. kernels that survive pm_remove_bufferize

Run:  DEV=CPU python3 examples/11_remove_bufferize.py

run_rangeify proposes STAGE (= BUFFERIZE) nodes; the pass "symbolic+reduce_collapse+debuf" (get_kernel_graph in
tinygrad/schedule/rangeify.py) then runs pm_remove_bufferize, whose cost function remove_bufferize() decides for each
INDEX(STAGE(src, ranges...), idx...) whether to keep the buffer or substitute idx for ranges inside src (= recompute).
The rules (rangeify.py remove_bufferize):
  * STAGE created from user .contiguous()/STORE is never removable (BufferizeOpts.removable=False / it is a STORE)
  * if src reads more than 3 buffers                          -> keep (re-reading 4+ buffers per use is too costly)
  * if any REDUCE inside src reads a buffer                    -> keep (never recompute a reduction)
  * otherwise                                                 -> remove: recompute src at every use
"""
import importlib.util, pathlib
from tinygrad import Tensor
from tinygrad.uop.ops import Ops

spec = importlib.util.spec_from_file_location("cap", pathlib.Path(__file__).parent / "10_sched_capture.py")
cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)

def report(title, out):
  log, linear, compiled = cap.schedule(out, run=False)
  cands = [u for u in log["realize_map"] if u.op is not Ops.STORE]
  n_before = sum(1 for u in cap.pass_by_name(log, "apply rangeify")[2].toposort() if u.op is Ops.STAGE)
  n_after = sum(1 for u in cap.pass_by_name(log, "symbolic+reduce_collapse+debuf")[2].toposort() if u.op is Ops.STAGE)
  print(f"{title:46s} candidates={[f'{u.op.name}{u.shape}' for u in cands]}")
  print(f"{'':46s} STAGEs: {n_before} after 'apply rangeify' -> {n_after} after 'symbolic+reduce_collapse+debuf'  => {len(linear.src)} kernel(s)")

report("y=x+1; y + y.T", (lambda y: y + y.T)(Tensor.empty(4, 4) + 1))
report("y=x*3; y.sum(0) + y.max(0)", (lambda y: y.sum(0) + y.max(0))(Tensor.empty(8, 4) * 3))
report("softmax", Tensor.empty(4, 8).softmax(-1))
print("\n-- the '> 3 buffers' rule: y = (sum of k inputs)*3, then y.sum(0) + y.max(0)")
for k in range(1, 6):
  ins = [Tensor.empty(8, 4) for _ in range(k)]
  y = ins[0]
  for t in ins[1:]: y = y + t
  y = y * 3
  report(f"k={k} input buffers", y.sum(0) + y.max(0))
print("\n-- the 'reduce reads a buffer' rule: the candidate contains a REDUCE")
x = Tensor.empty(8, 4)
y = x.sum(1, keepdim=True) * 2            # (8,1): broadcast back below -> candidate, contains a reduce over x
report("y = x.sum(1)*2;  x - y", x - y)
