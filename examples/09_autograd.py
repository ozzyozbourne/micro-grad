# Chapter 9 — autograd is graph rewriting: gradients are just more lazy UOps.
# Run:  DEV=CPU python3 examples/09_autograd.py
from tinygrad import Tensor
from tinygrad.uop.ops import Ops, GroupOp, UOp

NAMES: dict = {}
def expr(u: UOp) -> str:
  # compact printer: buffers by name, movement ops lower-case with args, everything else UPPER(srcs)
  if u.base in NAMES and all(v.op in GroupOp.Movement | {Ops.BUFFER, Ops.CONST, Ops.STACK} for v in u.toposort()): return NAMES[u.base]
  if u.op is Ops.CONST: return str(u.arg).replace("ConstFloat(", "").rstrip(")") if "ConstFloat" in repr(u.arg) else repr(u.arg)
  if u.op in GroupOp.Movement: return f"{u.op.name.lower()}{u.marg}({expr(u.src[0])})"
  if u.op is Ops.CAST: return f"cast<{u.dtype.name}>({expr(u.src[0])})"
  if u.op is Ops.REDUCE: return f"REDUCE<{u.arg[0].name},{u.arg[1]}>({expr(u.src[0])})"
  return f"{u.op.name}(" + ", ".join(expr(s) for s in u.src) + ")"

print("=== 1. d/dx sum(x*x): one rule per op, summed where x fans out")
x = Tensor([1., 2., 3.]).realize(); NAMES[x.uop.base] = "x"
y = (x * x).sum()
g, = y.gradient(x)                       # Tensor.gradient -> compute_gradient (tinygrad/mixin/gradient.py)
print("forward :", expr(y.uop))
print("gradient:", expr(g.uop))
print("hash-consing: both MUL-rule outputs are the SAME UOp ->", g.uop.src[0] is g.uop.src[1])
print("value   :", g.tolist())

print("\n=== 2. .backward() fills .grad on every float tensor in the graph (there is no requires_grad flag)")
x2 = Tensor([1., 2., 3.]).realize()
(x2 * x2).sum().backward()
print("x2.grad is a lazy Tensor whose uop is", x2.grad.uop.op.name, "-> realized value", x2.grad.tolist())

print("\n=== 3. matmul: dL/dA = G @ B.T and dL/dB = A.T @ G fall out of the MUL + REDUCE + movement rules")
A = Tensor([[1., 2, 3], [4, 5, 6]]).realize(); NAMES[A.uop.base] = "A"
B = Tensor([[1., 0], [0, 1], [1, 1]]).realize(); NAMES[B.uop.base] = "B"
L = (A @ B).sum()
gA, gB = L.gradient(A, B)
print("forward :", expr(L.uop))
print("dL/dA   :", expr(gA.uop))
print("dL/dA   =", gA.tolist(), "   expected ones(2,2) @ B.T =", (Tensor.ones(2, 2) @ B.T).tolist())
print("dL/dB   =", gB.tolist(), "   expected A.T @ ones(2,2) =", (A.T @ Tensor.ones(2, 2)).tolist())

print("\n=== 4. movement-op gradients are the inverse movement op")
v = Tensor([1., 2, 3, 4, 5, 6]).realize(); NAMES[v.uop.base] = "v"
w8 = Tensor([1., 2, 3, 4, 5, 6, 7, 8]).realize()
for name, f in [("reshape(2,3)", lambda t: t.reshape(2, 3)), ("reshape(2,3).T", lambda t: t.reshape(2, 3).T),
                ("pad((1,1))", lambda t: t.pad(((1, 1),))), ("shrink((1,4))", lambda t: t[1:4]),
                ("reshape(1,6).expand(3,6)", lambda t: t.reshape(1, 6).expand(3, 6)), ("flip(0)", lambda t: t.flip(0))]:
  out = f(v)
  upstream = Tensor([float(i) for i in range(1, out.numel() + 1)]).reshape(out.shape)   # a recognisable upstream gradient
  gv, = out.gradient(v, gradient=upstream)
  print(f"{name:26s} upstream {str(upstream.tolist()):58s} -> dv = {gv.tolist()}")

print("\n=== 5. MLP: gradients of a real loss, checked against finite differences")
Tensor.manual_seed(1)
X = Tensor.randn(4, 3).realize(); Y = Tensor([0, 2, 1, 2]).realize()
W1, b1 = Tensor.randn(3, 5).realize(), Tensor.zeros(5).contiguous().realize()
W2, b2 = Tensor.randn(5, 3).realize(), Tensor.zeros(3).contiguous().realize()
def loss_fn(W1, b1, W2, b2): return ((X @ W1 + b1).relu() @ W2 + b2).sparse_categorical_crossentropy(Y)
loss = loss_fn(W1, b1, W2, b2)
grads = loss.gradient(W1, b1, W2, b2)
print("loss =", round(loss.item(), 5), " gradient shapes:", [g.shape for g in grads])
eps = 1e-3
W1l, gW1 = W1.tolist(), grads[0].tolist()
for (i, j) in [(0, 0), (1, 3), (2, 4)]:
  plus = [row[:] for row in W1l]; plus[i][j] += eps
  minus = [row[:] for row in W1l]; minus[i][j] -= eps
  fd = (loss_fn(Tensor(plus), b1, W2, b2).item() - loss_fn(Tensor(minus), b1, W2, b2).item()) / (2 * eps)
  print(f"dL/dW1[{i},{j}]  autograd {gW1[i][j]: .5f}   finite-diff {fd: .5f}")

print("\n=== 6. DETACH stops the gradient; the graph walk skips it")
z = Tensor([2., 3.]).realize()
print("d/dz (z * z.detach()).sum() =", (z * z.detach()).sum().gradient(z)[0].tolist(), " (only one branch contributes)")
print("d/dz (z * z).sum()          =", (z * z).sum().gradient(z)[0].tolist())
