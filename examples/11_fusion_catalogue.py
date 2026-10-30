"""examples/11_fusion_catalogue.py — how many kernels does each program become, and why?

Run:  DEV=CPU NOOPT=1 DEBUG=2 python3 examples/11_fusion_catalogue.py 2>/dev/null

NOOPT=1 disables the kernel optimizer (no upcast/unroll), so kernel names read as the plain loop nest:
E_16 = elementwise over 16, r_4_8 = 4 outputs each reducing 8 elements, etc. DEBUG=2 prints one '***' line per
executed CALL: device, running kernel count, name, number of buffer args.
"""
from tinygrad import Tensor
from tinygrad.helpers import Context

def case(title, make, **ctx):
  print(f"\n### {title}")
  with Context(**ctx):
    make().realize()

case("elementwise chain: (a*2 + b).relu()", lambda: (Tensor.empty(16) * 2 + Tensor.empty(16)).relu())
case("elementwise + reduce: (a*b).sum(1)", lambda: (Tensor.empty(4, 8) * Tensor.empty(4, 8)).sum(1))
case("softmax(-1) on (4, 8)", lambda: Tensor.empty(4, 8).softmax(-1))
case("matmul + bias + relu", lambda: (Tensor.empty(4, 8) @ Tensor.empty(8, 3) + Tensor.empty(3)).relu())
case("x - x.mean(1, keepdim=True)", lambda: (lambda x: x - x.mean(1, keepdim=True))(Tensor.empty(4, 4)))
case("x - x.mean()   (full reduce)", lambda: (lambda x: x - x.mean())(Tensor.empty(4, 4)))
case("(a + 1).contiguous() * 2", lambda: (Tensor.empty(16) + 1).contiguous() * 2)
case("y = a*3; y.sum(0) + y.max(0)", lambda: (lambda y: y.sum(0) + y.max(0))(Tensor.empty(8, 4) * 3))
c = Tensor.empty(8).realize()
case("c.assign(c*2 + 1)", lambda: c.assign(c * 2 + 1))
case("c.assign(c.flip(0) + 1)   (reads other positions of c)", lambda: c.assign(c.flip(0) + 1))
case("COPY: ((a on CPU:1) + 1).to('CPU') * 2", lambda: (Tensor.empty(8, device="CPU:1") + 1).to("CPU") * 2)
case("Tensor.empty(65536).sum()   (split_reduceop)", lambda: Tensor.empty(65536).sum())
ts = [Tensor.empty(16) for _ in range(6)]
case("sum of 6 inputs with MAX_KERNEL_BUFFERS=4", lambda: ts[0] + ts[1] + ts[2] + ts[3] + ts[4] + ts[5], MAX_KERNEL_BUFFERS=4)
