# Chapter 15 — integer division and modulo decomposition (tinygrad/codegen/decomp/op.py).
#
# "early decompositions": DIV (floor) -> SHR by a power of two, else TRUNCDIV (+ sign fix if operands may be negative)
#                         MOD (floor) -> AND by 2^k-1, else TRUNCMOD (+ sign fix)
# "late decompositions":  TRUNCDIV by a constant d on a non-negative x -> (x*m) >> s       (fast_idiv / magicgu)
#                         TRUNCMOD by d                                -> x - d*((x*m) >> s)
# The rules are only added when the renderer can render the replacement: they check renderer.code_for_op (SHR, AND, ...).
#
# run:   DEV=CPU python3 examples/15_fast_idiv.py
from tinygrad import Tensor, Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.decomp.op import magicgu, fast_idiv
from tinygrad.uop.ops import UOp
from tinygrad.helpers import Context

print("== magicgu(vmax, d) -> (m, s) such that x//d == (x*m) >> s for every 0 <= x <= vmax ==")
for vmax, d in [(255, 3), (1000, 7), (2**16, 10), (2**31 - 1, 7), (2**31 - 1, 1000)]:
  m, s = magicgu(vmax, d)
  ok = all((x * m) >> s == x // d for x in range(0, min(vmax, 200000) + 1))
  print(f"  vmax={vmax:<11} d={d:<5} m={m:<12} s={s:<3} m*vmax={m*vmax:<22} fits int32: {m*vmax <= 2**31-1!s:5}  check(x<=2e5): {ok}")

print("\n== fast_idiv on UOps: the bounds of x (vmin/vmax) decide what we get ==")
ren = Device["CPU"].renderer
for vmax, d in [(999, 7), (2**31 - 1, 7), (2**31 - 1, 12), (2**31 - 1, 1000)]:
  x = UOp.variable("x", 0, vmax, dtypes.int32)
  print(f"  x in [0,{vmax}], x//{d}  ->  {fast_idiv(ren, x, d).render()}")
print(f"  x in [-5,100], x//7  ->  {fast_idiv(ren, UOp.variable('x', -5, 100, dtypes.int32), 7)}   (negative x: not handled here)")

print("\n== the same thing inside a real kernel on the CPU (Clang) ==")
# NOTE: DISABLE_FAST_IDIV defaults to 1 in helpers.py, so by default the TRUNCDIV->(x*m)>>s rule is NOT added.
a = Tensor.empty(1000, dtype=dtypes.int32)   # an int32 input buffer: a load can be anything in the int32 range
r = Tensor.arange(1000, dtype=dtypes.int32)  # not a buffer: becomes RANGE index math, known to be in [0, 999]
def store_line(ast) -> str:
  src = to_program(ast, ren).src[2].arg
  return [l.strip() for l in src.splitlines() if l.strip().startswith("*")][-1]
for expr, out in [("a // 8", a // 8), ("a % 8", a % 8), ("a // 7", a // 7), ("a + arange//7", a + r // 7), ("a + arange%7", a + r % 7)]:
  ast = out.schedule_linear().src[-1].src[0]   # (scheduling a tensor consumes it, so grab the AST once)
  print(f"  {expr:14s} default            : {store_line(ast)}")
  with Context(DISABLE_FAST_IDIV=0):
    print(f"  {'':14s} DISABLE_FAST_IDIV=0: {store_line(ast)}")
x = Tensor(list(range(-3, 1000)), dtype=dtypes.int32)
print("  check //7:", (x // 7).tolist()[:5], (x // 7).tolist()[-3:], "  %7:", (x % 7).tolist()[:5], (x % 7).tolist()[-3:])
