# Chapter 18 — what TinyJit does on call 1, 2 and 3+, and what the captured LINEAR contains.
# run: DEV=CPU python3 examples/18_jit_inside.py
from tinygrad import Tensor, TinyJit, nn
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.helpers import GlobalCounters

Tensor.manual_seed(0)
l1, l2 = nn.Linear(8, 16), nn.Linear(16, 4)
Tensor.realize(*nn.state.get_parameters([l1, l2]))   # realize the weights first, otherwise their (lazy, random) init is part of the graph
@TinyJit
def step(x): return l2(l1(x).relu()).softmax().realize()

for i in range(4):
  x = Tensor.randn(2, 8).realize()
  GlobalCounters.reset()
  phase = {0: "ignore (run normally)", 1: "capture", 2: "replay"}.get(step.cnt, "replay")
  out = step(x)
  print(f"call {i+1}: cnt was {i} -> {phase:22s} kernels run: {GlobalCounters.kernel_count}  out[0]={[round(v, 4) for v in out.tolist()[0]]}")

cap = step.captured
lin = cap._linear
print("\ncaptured:", type(cap).__name__, "| expected_names", cap.expected_names,
      "| input info (view, vars, dtype, device):", [(v.op.name, vs, str(dt), d) for v, vs, dt, d in cap.expected_input_info])
print("the captured LINEAR has", len(lin.src), "compiled CALLs:")
for c in lin.src:
  body = c.without_after.body
  def kind(a):
    a0 = a
    while a.op in (Ops.SHRINK, Ops.BITCAST, Ops.RESHAPE): a = a.src[0]
    if a.op is Ops.PARAM: return f"PARAM#{a.arg.slot}(jit input)"
    if a.op is Ops.ALLOC: return "ALLOC(planned scratch)"
    if a.op is Ops.BUFFER: return f"BUFFER({a.arg.buffer.nbytes}B, baked in)"
    return a.op.name
  name = body.src[0].arg.name if body.op is Ops.PROGRAM else body.op.name
  print(f"  {name:20s} args: {[kind(a) for a in get_call_arg_uops(c.without_after)]}")

ret_buf = out.uop.buffer
print("\nreturned tensor is the SAME buffer every replay:", ret_buf is step(Tensor.randn(2, 8).realize()).uop.buffer)
print("the weights are captured by reference (BUFFER), not copied: l1.weight buffer in linear?",
      any(u.op is Ops.BUFFER and u.arg.buffer is l1.weight.uop.buffer for u in lin.toposort()))
