# Chapter 18 — TinyJit's rules: what is baked in, what raises JitError, and what silently goes wrong.
# run: DEV=CPU python3 examples/18_jit_rules.py
from tinygrad import Tensor, TinyJit, Variable
from tinygrad.engine.jit import JitError
def attempt(name, fn):
  try: print(f"{name}: OK ->", fn())
  except Exception as e: print(f"{name}: {type(e).__name__}: {str(e)[:150]}")

# 1. item() inside the jit
@TinyJit
def f1(x): return (x*2).sum().item()
attempt("item inside", lambda: [f1(Tensor([1.,2,3]).realize()) for _ in range(3)])

# 2. python value baked in
@TinyJit
def f2(x, k): return (x*k).realize()
attempt("python scalar baked", lambda: [f2(Tensor([1.,2,3]).realize(), k).tolist() for k in (1,2,3,4)])

# 3a. shape change of a reshaped input: the input's view (RESHAPE) is part of expected_input_info -> JitError
@TinyJit
def f3(x): return (x+1).realize()
attempt("shape change (2-D view)", lambda: [f3(Tensor.ones(n, 2).contiguous().realize()).shape for n in (3,3,3,4)])

# 3b. size change of a plain 1-D buffer: its view is a bare NOOP, nothing to compare -> stale replay, no error!
@TinyJit
def f3b(x): return (x+1).realize()
attempt("size change (plain 1-D buffer)", lambda: [f3b(Tensor([1.]*n).realize()).tolist() for n in (3,3,3,4)])

# 4. duplicate inputs
@TinyJit
def f4(x, y): return (x+y).realize()
t = Tensor([1.,2]).realize()
attempt("duplicate", lambda: f4(t, t))

# 5. variable
@TinyJit
def f5(x, n): return x[:n].sum().realize()
big = Tensor([1.,2,3,4,5,6,7,8]).realize()
v = Variable("n", 1, 8)
attempt("variable", lambda: [f5(big, v.bind(n)).item() for n in (2,3,5,8)])

# 6. nothing to jit
@TinyJit
def f6(x): return x
attempt("nothing", lambda: [f6(Tensor([1.]).realize()) for _ in range(3)])

# 7. random inside jit
@TinyJit
def f7(x): return (x + Tensor.rand(3)).realize()
attempt("rand", lambda: [[round(v,3) for v in f7(Tensor.zeros(3).contiguous().realize()).tolist()] for _ in range(4)])
