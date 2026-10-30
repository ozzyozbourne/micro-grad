# Chapter 18 — @function: capture a Python function as ONE reusable CALL in the tensor graph (graph-level capture).
# run: DEV=CPU python3 examples/18_function.py
from tinygrad import Tensor, function
from tinygrad.uop.ops import Ops

w = Tensor.randn(4, 4).realize()

# 1) w is used inside but not passed in: it is an "implicit" input. The default refuses that.
@function
def strict(x: Tensor) -> Tensor: return (x @ w).relu()
try: strict(Tensor.randn(2, 4).realize())
except RuntimeError as e: print("1 strict     :", str(e).replace("\n", " | "))

# 2) allow_implicit=True turns w into an extra parameter of the CALL
@function(allow_implicit=True)
def layer(x: Tensor) -> Tensor: return (x @ w).relu()

x = Tensor.randn(2, 4).realize()
y = layer(x)                                 # traces the Python function ONCE per call, but builds no kernels
call = next(u for u in y.uop.toposort() if u.op is Ops.CALL)
print("2 y.uop      :", y.uop.op.name, "of a", call.op.name, "| CallInfo:", call.arg)
print("  CALL srcs  :", [s.op.name for s in call.src], " <- (body, x, w, output ALLOC/BUFFER)")
params = sorted({u for u in call.body.toposort() if u.op is Ops.PARAM}, key=lambda p: p.arg.slot)
print("  body PARAMs:", [(p.arg.slot, p.max_numel()) for p in params], " slot 0 = x (8 elems), 1 = w (16), 2 = the output (8)")
print("  is_inline_call:", call.is_inline_call, "(inlined into the caller's graph at schedule time)")

# 3) a second call with a different x reuses the SAME body UOp (hash-consing), so the schedule cache hits
y2 = layer(Tensor.randn(2, 4).realize())
call2 = next(u for u in y2.uop.toposort() if u.op is Ops.CALL)
print("3 same body object for a new input:", call2.body is call.body)
print("  values:", [round(v, 3) for v in y.tolist()[0]], "| reference:", [round(v, 3) for v in (x @ w).relu().tolist()[0]])

# 4) precompile=True: the body is scheduled once on its own (a CALL of a LINEAR) instead of being inlined
@function(precompile=True, allow_implicit=True)
def layer_pc(x: Tensor) -> Tensor: return (x @ w).relu()
z = layer_pc(x)
callp = next(u for u in z.uop.toposort() if u.op is Ops.CALL)
print("4 precompile :", callp.arg, "| is_inline_call:", callp.is_inline_call, "| result matches:", z.tolist() == y.tolist())
