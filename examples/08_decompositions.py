# Chapter 8 — how high-level Tensor methods decompose into primitive UOps.
# Run:  DEV=CPU python3 examples/08_decompositions.py
#
# For each method we print the lazy UOp graph as a compact expression. Movement ops are printed in lower case
# (reshape/permute/expand/pad/shrink) with their decoded args; ALU/reduce ops in UPPER CASE. Nothing is computed:
# these are the graphs the Tensor methods in tinygrad/mixin/*.py build.
from tinygrad import Tensor
from tinygrad.uop.ops import Ops, GroupOp, UOp

x = Tensor([1., 2, 3]).realize()
y = Tensor([4., 5, 6]).realize()
NAMES = {x.uop.base: "x", y.uop.base: "y"}

def expr(u: UOp) -> str:
  if u.base in NAMES and u.op is not Ops.REDUCE and all(v.op in GroupOp.Movement | {Ops.BUFFER} for v in u.toposort() if v.op not in {Ops.CONST, Ops.STACK}):
    return NAMES[u.base]                       # x / y (hide the RESHAPE that gives the buffer its shape)
  if u.op is Ops.CONST: return str(u.arg).replace("ConstFloat(", "").replace(")", "") if "ConstFloat" in repr(u.arg) else repr(u.arg)
  if u.op in GroupOp.Movement: return f"{u.op.name.lower()}{u.marg}({expr(u.src[0])})"
  if u.op is Ops.CAST: return f"cast<{u.dtype.name}>({expr(u.src[0])})"
  if u.op is Ops.REDUCE: return f"REDUCE<{u.arg[0].name},axes={u.arg[1]}>({expr(u.src[0])})"
  return f"{u.op.name}(" + ", ".join(expr(s) for s in u.src) + ")"

def ops_used(t: Tensor) -> str:
  used = sorted({u.op.name for u in t.uop.toposort()} - {"CONST", "STACK", "BUFFER", "RESHAPE"})
  return ", ".join(used)

print("=== elementwise (tinygrad/mixin/elementwise.py)")
for name, t in [("x - y", x - y), ("-x", -x), ("x / y", x / y), ("x.exp()", x.exp()), ("x.log()", x.log()),
                ("x.sigmoid()", x.sigmoid()), ("x.relu()", x.relu()), ("x.maximum(y)", x.maximum(y)),
                ("x.minimum(y)", x.minimum(y)), ("x.cos()", x.cos()), ("x.rsqrt()", x.rsqrt()), ("x ** 3", x ** 3),
                ("x >= y", x >= y), ("x != y", x != y), ("x.clamp(0, 2)", x.clamp(0, 2))]:
  print(f"{name:14s} = {expr(t.uop)}")

print("\n=== reductions (tinygrad/mixin/reduce.py, op.py)")
for name, t in [("x.sum()", x.sum()), ("x.max()", x.max()), ("x.min()", x.min()), ("x.mean()", x.mean()),
                ("x.softmax()", x.softmax())]:
  print(f"{name:14s} = {expr(t.uop)}")

print("\n=== bigger methods: which primitive ops appear (movement + ALU + REDUCE)")
m = Tensor([[1., 2, 3], [4, 5, 6]]).realize()
idx = Tensor([2, 0]).realize()
for name, t in [("m.log_softmax()", m.log_softmax()), ("m.layernorm()", m.layernorm()), ("m.argmax(1)", m.argmax(1)),
                ("m.cumsum(1)", m.cumsum(1)), ("Tensor.arange(4)", Tensor.arange(4)), ("idx.one_hot(3)", idx.one_hot(3)),
                ("m.gather(1, idx[:, None])", m.gather(1, idx.reshape(2, 1))), ("m[:, idx]", m[:, idx]),
                ("m @ m.T", m @ m.T), ("m.sparse_cat_xent(idx)", m.sparse_categorical_crossentropy(idx))]:
  print(f"{name:26s} {len(t.uop.toposort()):4d} UOps | {ops_used(t)}")

print("\n=== check: the decompositions compute the right numbers")
print("x.sigmoid() =", [round(v, 4) for v in x.sigmoid().tolist()])
print("m.softmax() =", [[round(v, 4) for v in r] for r in m.softmax().tolist()])
print("m.argmax(1) =", m.argmax(1).tolist(), "  m.cumsum(1) =", m.cumsum(1).tolist())
print("arange(4)   =", Tensor.arange(4).tolist(), "  one_hot =", idx.one_hot(3).tolist())
print("gather      =", m.gather(1, idx.reshape(2, 1)).tolist(), "  m[:, idx] =", m[:, idx].tolist())
