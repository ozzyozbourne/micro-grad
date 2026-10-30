# Chapter 21 — the tiny program every introspection tool in this chapter is pointed at
# Run: DEV=CPU DEBUG=2 python3 examples/21_debug_target.py     (try DEBUG=1..7, VIZ=-1, NOOPT=1, BEAM=2 ...)
from tinygrad import Tensor
x = Tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])   # python list -> a PYTHON-device buffer, copied to CPU
y = (x * 2 + 1).sum(axis=1)                          # one fused elementwise+reduce kernel
print(y.tolist())                                    # .tolist() realizes: schedule -> codegen -> run
