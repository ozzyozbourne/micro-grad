# Chapter 9 — because gradients are ordinary lazy UOps, forward and backward are scheduled (and fused) together.
# Run:  DEV=CPU python3 examples/09_fusion.py
import re, os, json
from tinygrad import Tensor, Device
from tinygrad.uop.ops import Ops
from tinygrad.codegen import to_program

def kernels(*ts: Tensor) -> list[str]:
  # schedule (without running) and return the names of the compiled kernels, in order
  names = []
  for call in Tensor.schedule_linear(*ts).src:
    if call.op is Ops.CALL and call.src[0].op is Ops.SINK:
      src = next(x for x in to_program(call.src[0], Device["CPU"].renderer).src if x.op is Ops.SOURCE).arg
      names.append(re.search(r"void (\w+)\(", src).group(1))
  return names

def setup():
  Tensor.manual_seed(0)
  X = Tensor.randn(16, 8).realize(); Y = Tensor([i % 4 for i in range(16)]).realize()
  W1, W2 = Tensor.randn(8, 32).realize(), Tensor.randn(32, 4).realize()
  return X, Y, W1, W2
def mlp_loss(X, Y, W1, W2): return (X @ W1).relu().__matmul__(W2).sparse_categorical_crossentropy(Y)

print("=== MLP  loss = xent(relu(X @ W1) @ W2)")
X, Y, W1, W2 = setup()
fwd = kernels(mlp_loss(X, Y, W1, W2))
print(f"forward only            : {len(fwd):2d} kernels  {fwd}")
CAPTURE = {"fwd": fwd}

X, Y, W1, W2 = setup()
loss = mlp_loss(X, Y, W1, W2)
g1, g2 = loss.gradient(W1, W2)                          # build the gradient graph first (lazy) ...
loss.realize()                                          # ... run the forward on its own ...
bwd = kernels(g1, g2)                                   # ... then schedule only the gradients
print(f"backward after forward  : {len(bwd):2d} kernels  {bwd}")
CAPTURE["bwd_after_fwd"] = bwd
print(f"  -> separately: {len(fwd)} + {len(bwd)} = {len(fwd) + len(bwd)} kernels (forward intermediates get recomputed)")

X, Y, W1, W2 = setup()
loss = mlp_loss(X, Y, W1, W2); loss.realize()          # PITFALL: realize first, then ask for the gradient
g1, = loss.gradient(W1)
print(f"gradient AFTER realize  : loss.uop is now {loss.uop.base.op.name}; dL/dW1 = {g1.uop.op.name} of a constant, all zero: {g1.abs().sum().item() == 0}")

X, Y, W1, W2 = setup()
loss = mlp_loss(X, Y, W1, W2)
g1, g2 = loss.gradient(W1, W2)
both = kernels(loss, g1, g2)                            # one schedule for loss AND gradients
print(f"forward+backward at once: {len(both):2d} kernels  {both}")
CAPTURE["fwd_bwd"] = both

X, Y, W1, W2 = setup()
loss = mlp_loss(X, Y, W1, W2)
g1, g2 = loss.gradient(W1, W2)
step = kernels(loss, W1.assign(W1 - 0.1 * g1), W2.assign(W2 - 0.1 * g2))
print(f"fwd+bwd+SGD update      : {len(step):2d} kernels  {step}")
CAPTURE["fwd_bwd_sgd"] = step

print("\n=== tiny examples")
x = Tensor([1., 2, 3]).realize()
y = (x * x).sum(); gx, = y.gradient(x)
print("sum(x*x) and its grad   :", kernels(y, gx))
A, B = Tensor.randn(4, 4).realize(), Tensor.randn(4, 4).realize()
L = (A @ B).relu().sum(); gA, gB = L.gradient(A, B)
print("relu(A@B).sum() + grads :", kernels(L, gA, gB))

# save for the chapter's kernel-count visualization
with open(os.path.join(os.path.dirname(__file__), "..", "data", "ch09_fusion.json"), "w") as f: json.dump(CAPTURE, f, indent=1)
print("wrote data/ch09_fusion.json")
