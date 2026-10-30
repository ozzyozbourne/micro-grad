# Chapter 8 — matmul and conv2d are "movement ops + MUL + REDUCE". Print the real shape traces.
# Run:  DEV=CPU python3 examples/08_matmul_conv.py
from tinygrad import Tensor
from tinygrad.uop.ops import Ops, GroupOp, UOp

def mop_chain(u: UOp) -> list[str]:
  # walk down from u through movement ops to the buffer, return them bottom-up with shapes
  out = []
  while u.op in GroupOp.Movement:
    out.append(f"{u.op.name:8s} {str(u.marg):42s} -> {u.shape}")
    u = u.src[0]
  out.append(f"{u.op.name:8s} {'':42s}    {u.shape}")
  return out[::-1]

print("=== matmul: A(2,3) @ B(3,4)   (tinygrad/mixin/op.py dot)")
A = Tensor([[1., 2, 3], [4, 5, 6]]).realize()
B = Tensor([[1., 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]).realize()
C = A @ B
red = C.uop
print("result:", red.op.name, red.arg, "shape", red.shape)
perm = red.src[0]
print("under it:", perm.op.name, perm.marg, "shape", perm.shape)
mul = perm.src[0]
print("under it:", mul.op.name, "shape", mul.shape, "  <- broadcast happens HERE, inside MUL's _shape")
print("A side:"); print("  " + "\n  ".join(mop_chain(mul.src[0])))
print("B side:"); print("  " + "\n  ".join(mop_chain(mul.src[1])))
print("A @ B =", C.tolist())

print("\n=== conv2d: x(1,1,5,5) * w(2,1,3,3), stride 1, padding 1   (op.py conv2d -> movement.py _pool)")
x = Tensor([[[[float(5*i+j) for j in range(5)] for i in range(5)]]]).realize()
w = Tensor([[[[0., 0, 0], [0, 1, 0], [0, 0, 0]]], [[[1., 1, 1], [1, 1, 1], [1, 1, 1]]]]).realize()
out = x.conv2d(w, padding=1)
u = out.uop
while u.op is not Ops.MUL: u = u.src[0]
print("input side (pad, then _pool's repeat/shrink/reshape/permute, then conv2d's reshape/expand/permute):")
print("  " + "\n  ".join(mop_chain(u.src[0])))
print("weight side:"); print("  " + "\n  ".join(mop_chain(u.src[1])))
print("MUL shape:", u.shape)
r = out.uop
chain = []
while r is not u:
  chain.append(f"{r.op.name}{'<'+r.arg[0].name+','+str(r.arg[1])+'>' if r.op is Ops.REDUCE else (r.marg if r.op in GroupOp.Movement else '')} -> {r.shape}")
  r = r.src[0]
print("after the MUL:", " | ".join(chain[::-1]))
res = out.tolist()
print("out[0,0] (identity kernel) =", res[0][0])
print("out[0,1] (3x3 box sum)     =", res[0][1])

print("\n=== _pool alone: 1-D sliding windows of size 3 over arange(6)")
p = Tensor([0., 1, 2, 3, 4, 5]).realize()._pool((3,))
print("shape", p.shape, "=", p.tolist())
print("\n".join(mop_chain(p.uop)))
