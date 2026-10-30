"""examples/12_run_linear.py — how engine/realize.py consumes the schedule: compile_linear -> link_linear -> pm_exec

Run:  DEV=CPU NOOPT=1 python3 examples/12_run_linear.py 2>/dev/null

Tensor.realize() is just:
    linear, var_vals = Tensor.linear_with_vars(*to_realize)
    run_linear(linear, var_vals)
and run_linear does   link_linear(compile_linear(linear)) and then one pm_exec.rewrite per CALL.
We do the same steps by hand and print what each CALL body looks like after each step.
"""
from tinygrad import Tensor
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import compile_linear, link_linear, pm_exec, ExecContext

def bodies(title, linear):
  print(f"--- {title}")
  for i, c in enumerate(linear.src):
    c = c.without_after
    b = c.body
    extra = f" name={b.src[0].arg.name}" if b.op is Ops.PROGRAM else (f" (kernel AST, {len(b.toposort())} uops)" if b.op is Ops.SINK else "")
    print(f"   CALL {i}: body={b.op.name}{extra}  args={len(c.src)-1}")

x = Tensor([[float(8*r + c + 1) for c in range(8)] for r in range(4)]).realize()
out = x / x.sum(1, keepdim=True)
linear, var_vals = Tensor.linear_with_vars(out)
bodies("1. the schedule from linear_with_vars", linear)

compiled = compile_linear(linear)          # lower_and_compile: to_program() on every SINK (cached, maybe in parallel)
bodies("2. compile_linear: every kernel AST became a PROGRAM", compiled)
prg = compiled.src[0].without_after.body
print("   PROGRAM srcs:", [s.op.name for s in prg.src])

linked = link_linear(compiled)              # hcq_link: resolve addresses (a no-op rewrite for plain CPU calls)
bodies("3. link_linear", linked)

ctx = ExecContext(var_vals)
for call in linked.src:                      # what run_linear's loop does (minus stats)
  pm_exec.rewrite(call.without_after, ctx)  # CALL(PROGRAM) -> exec_kernel, CALL(STORE) -> exec_copy
print("--- 4. executed. each row of out sums to", [round(sum(r), 4) for r in out.tolist()])
