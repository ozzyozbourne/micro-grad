"""examples/12_schedule.py — create_linear_with_vars step by step, and what the final schedule looks like

Run:  DEV=CPU NOOPT=1 python3 examples/12_schedule.py 2>/dev/null

Tensor.linear_with_vars builds a SINK of the outputs (each an AFTER(BUFFER, STORE(...))) and hands it to
create_linear_with_vars (tinygrad/schedule/__init__.py). We capture that SINK and replay the steps one by one.
"""
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
import tinygrad.tensor as T
import tinygrad.schedule as S
from tinygrad.engine.realize import run_linear

def short(u:UOp) -> str:
  if u.op is Ops.BUFFER: return f"BUFFER#{u.arg.slot}({u.max_numel()})"
  if u.op is Ops.PARAM: return f"PARAM{u.arg.slot}"
  if u.op is Ops.ALLOC: return f"ALLOC{u.arg.slot}({u.max_numel()})"
  if u.op is Ops.BITCAST and u.src[0].op is Ops.SHRINK:
    sh = u.src[0]; return f"arena#{sh.src[0].arg.slot}[{sh.src[1].arg}:+{sh.src[2].arg}]"
  return u.op.name

def show_calls(title, linear:UOp):
  print(f"--- {title}: {linear.op.name} of {len(linear.src)}")
  for i, c in enumerate(linear.src):
    print(f"   CALL {i}: body={c.body.op.name:7s} args=({', '.join(short(s) for s in c.src[1:])})")

captured = {}
orig = T.create_linear_with_vars
def hook(big_sink):
  captured["big_sink"] = big_sink
  return orig(big_sink)
T.create_linear_with_vars = hook

x = Tensor.empty(4, 8).realize()
out = x / x.sum(1, keepdim=True)
with Context(SCACHE=0): linear, var_vals = Tensor.linear_with_vars(out)
big = captured["big_sink"]
print("0. the big SINK Tensor.linear_with_vars hands over:", [f"{s.op.name}({short(s.src[0])}, {s.src[1].op.name})" for s in big.src])

with Context(SCACHE=0):
  call = S.transform_to_call(big)
  print(f"1. transform_to_call -> CALL(body={call.body.op.name}, args=({', '.join(short(s) for s in call.src[1:])}))  "
        f"key={call.body.key.hex()[:8]}")
  lin_call = graph_rewrite(call, S.pm_schedule, name="schedule to linear", enter_calls=True)
  show_calls("2. 'schedule to linear': CALL(LINEAR...) body", lin_call.body)
  resolved = graph_rewrite(lin_call, S.pm_resolve_linear_call, name="resolve linear call")
  show_calls("3. 'resolve linear call': PARAM -> real BUFFER, ALLOC -> new BUFFER", resolved)
  planned = S.memory_plan_rewrite(graph_rewrite(resolved, S.pm_copy_from_store), {b for b in call.src[1:] if b.op is Ops.BUFFER})
  show_calls("4. memory_plan_rewrite: intermediates become views into one arena", planned)
show_calls("what Tensor.linear_with_vars actually returned", linear)

print("\n=== ordering: a kernel that reads the OLD value of a must run before the assign that overwrites it")
a = Tensor([1., 2., 3., 4.]).realize()
b = a * 10            # reads the OLD a  -> must run before the assign (WAR dependency)
a.assign(a + 1)       # overwrites a in place
c = a * 100           # reads the NEW a  -> must run after the assign (RAW dependency)
lin, vv = Tensor.linear_with_vars(b, a, c)
show_calls("schedule for (b, a, c)", lin)
run_linear(lin, vv)
print("b =", b.tolist(), " a =", a.tolist(), " c =", c.tolist())
