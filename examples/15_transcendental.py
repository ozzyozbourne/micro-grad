# Chapter 15 — transcendental decomposition (tinygrad/codegen/decomp/transcendental.py).
#
# 1. which renderers natively support EXP2/LOG2/SIN (their code_for_op) and hence which get the polynomial rewrite
# 2. how big xexp2 / xsin make the graph
# 3. a pure-python float32 port of xexp2 and xsin (the same port the chapter's plotter uses in JS), checked bit-for-bit
#    against what tinygrad actually computes on the CPU (Clang has no EXP2/SIN, so the CPU kernel runs the polynomial)
#
# run:   DEV=CPU python3 examples/15_transcendental.py
import math, struct, ctypes
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.uop.ops import UOp, Ops, graph_rewrite
from tinygrad.renderer.cstyle import ClangRenderer, OpenCLRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.codegen.decomp.transcendental import get_transcendental_patterns

print("== 1. native support (Ops in renderer.code_for_op) ==")
for ren in [ClangRenderer(Target("CPU", "CLANG", "x86_64,native")), CPULLVMRenderer(Target("CPU", "LLVM", "x86_64,native")),
            OpenCLRenderer(Target("CL")), PTXRenderer(Target("CUDA", "PTX", "sm_89")), WGSLRenderer(Target("WEBGPU"))]:
  sup = {op.name: op in ren.code_for_op for op in (Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT)}
  print(f"  {type(ren).__name__:16s} {sup}")

print("\n== 2. graph size after the rewrite (float32 input variable d) ==")
x = UOp.variable("d", -1e30, 1e30, dtypes.float32)    # an opaque float32 input
pm = get_transcendental_patterns(tuple(ClangRenderer.code_for_op.keys()), False)
for op in (Ops.EXP2, Ops.LOG2, Ops.SIN):
  u = x.alu(op)
  out = graph_rewrite(u, pm)
  print(f"  {op.name:5s}: 1 UOp  ->  {len(out.toposort()):4d} UOps   (ops used: {sorted({y.op.name for y in out.toposort()})})")

# ---- 3. float32 port --------------------------------------------------------------------------------------------
# Two evaluation modes:
#   CPU=False  "as written": every UOp rounds to float32 on its own, adds happen in graph order.
#   CPU=True   "as this CPU computes it": the Clang kernel is compiled with -O2 -march=native, so clang contracts
#              acc*x+c into a single FMA (one rounding), and the C renderer prints the nested ADD chain of the Cody-Waite
#              reduction as a flat "(a+b+c+d+x)" (cstyle.py strips parens of a same-op operand), which C evaluates
#              left-to-right: the small correction terms are summed first, the big term q*-3.14147949 and x last.
CPU = False
def f32(v: float) -> float: return ctypes.c_float(v).value                     # round to float32 (overflow -> inf)
def f2i(v: float) -> int: return struct.unpack("<i", struct.pack("<f", v))[0]    # bitcast float32 -> int32
def f2u(v: float) -> int: return struct.unpack("<I", struct.pack("<f", v))[0]    # bitcast float32 -> uint32
def u2f(v: int) -> float: return struct.unpack("<f", struct.pack("<I", v & 0xFFFFFFFF))[0]
def fma(a, b, c): return f32(a * b + c)  # a*b of two float32s is exact in a python double -> a single rounding
def polyN(x, coeffs):                    # helpers.polyN: reduce(lambda acc,c: acc*x+c, p, 0.0); consts are float32 UOps
  acc = 0.0
  for c in map(f32, coeffs): acc = fma(acc, x, c) if CPU else f32(f32(acc * x) + c)
  return acc
def rintk(v):                            # round away from 0, then cast to int32 (C truncation)
  return int(f32(v + (-0.5 if v < 0 else 0.5)))
def pow2if(q): return u2f((q + 127) * (1 << 23))   # shl(q+bias, 23) in int32 (wraps), then bitcast to float32
def ldexp2k(u, q):                       # u * 2^(q>>1) * 2^(q - (q>>1)); >> is floor division
  return f32(f32(u * pow2if(q // 2)) * pow2if(q - q // 2))

EXP2_C = [0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0]
def xexp2(d):
  d = f32(d)
  if math.isnan(d): return math.nan
  x = 0.0 if math.isinf(d) else d        # _lazy_map_numbers: inf/nan -> 0 for the core computation
  q = rintk(x)                           # q = round(x)
  s = f32(x - q)                         # s in [-0.5, 0.5]
  u = ldexp2k(polyN(s, EXP2_C), q)       # 2^x = 2^s * 2^q
  if d >= 128: return math.inf
  if d < -150: return 0.0
  return u

SIN_C = [2.6083159809786593541503e-06, -0.0001981069071916863322258, 0.00833307858556509017944336, -0.166666597127914428710938, 1.0]
def sin_poly(d): return f32(d * polyN(f32(d * d), SIN_C))
CW = [f32(c) for c in (-3.1414794921875, -0.00011315941810607910156, -1.9841872589410058936e-09, -1.2154201256553420762e-10)]
def cody_waite(d):                      # float32 branch of cody_waite_reduction: r = d - q*pi with pi split in 4 parts
  q = rintk(f32(d * f32(0.318309886183790671537767526745028724)))
  qf = float(q)
  if CPU:                               # rendered as ((q*c3)+(q*c2)+(q*c1)+(q*c0)+d), evaluated left to right
    r = f32(qf * CW[3])
    for c in CW[2::-1]: r = fma(qf, c, r)
    return f32(r + d), q
  r = d
  for c in CW: r = f32(f32(qf * c) + r)
  return r, q
TWO_OVER_PI = [0x00000000, 0x28be60db, 0x9391054a, 0x7f09d5f4, 0x7d4d3770, 0x36d8a566, 0x4f10e410]
M64 = (1 << 64) - 1
def payne_hanek(d):                     # float32 branch of payne_hanek_reduction, uint64 arithmetic wraps
  bits = f2u(d)
  exponent = (bits >> 23) & 0xFF
  f = u2f((bits & 0x807FFFFF) | 0x3F000000)       # frexp mantissa in [0.5, 1)
  e = (exponent - 127 + 1) & 0xFFFFFFFF           # frexp exponent (uint32 math)
  ia = int(f32(f * 4.294967296e9)) & M64          # 32 bits of mantissa as an integer
  i, e = e >> 5, e & 31                           # which 32-bit words of 2/pi we need, and the bit shift inside them
  offset = 32 - e
  a = [TWO_OVER_PI[i + k] if i + k < len(TWO_OVER_PI) - 1 else 0 for k in range(4)]   # the _take() where-chain
  shl = lambda x, y: ((x << y) & M64) & 0xFFFFFFFF
  shr = lambda x, y: (x >> y) & 0xFFFFFFFF
  hi, mi, lo = shl(a[0], e) | shr(a[1], offset), shl(a[1], e) | shr(a[2], offset), shl(a[2], e) | shr(a[3], offset)
  p = ((((ia * hi) & M64) << 32) + ((ia * mi) & M64) + (((ia * lo) & M64) >> 32)) & M64   # x * 2/pi in fixed point
  q = p >> 62                                     # top two bits: the quadrant
  p &= 0x3FFFFFFFFFFFFFFF                         # the fraction
  r = f32(f32(float(p)) * f32(3.4061215800865545e-19))  # fraction * pi/2 / 2^62
  return (r, q) if f < 0.5 else (f32(r - f32(math.pi / 2)), q + 1)
def xsin(d):
  d = f32(d)
  if math.isnan(d) or math.isinf(d): return math.nan
  sign = 0.0 if d == 0 else (-1.0 if d < 0 else 1.0)
  xa = f32(d * sign)
  if xa < 30.0:                         # switch_over=30: small angles use Cody-Waite
    r, q = cody_waite(xa); res = sin_poly(r); res = f32(res * (-1.0 if q & 1 else 1.0))
  else:                                 # large angles use Payne-Hanek
    r, q = payne_hanek(xa); res = sin_poly(f32(r + (f32(math.pi / 2) if q & 1 else 0.0))); res = f32(res * (-1.0 if q & 2 else 1.0))
  return f32(res * sign)

def ulps(a, b):                         # distance in float32 units in the last place
  if math.isnan(a) and math.isnan(b): return 0
  ia, ib = f2i(a), f2i(b)
  ia, ib = (ia if ia >= 0 else -(ia & 0x7FFFFFFF)), (ib if ib >= 0 else -(ib & 0x7FFFFFFF))
  return abs(ia - ib)

print("\n== 3. tinygrad on CPU vs the python float32 port vs libm ==")
xs = [i * 0.37 - 60.0 for i in range(325)] + [-149.3, 127.5, 0.001, 0.0]
sx = [i * 0.091 - 29.0 for i in range(640)] + [31.0, 123.456, 1000.25, 39800.5, 1e5, -77777.7, 1e6]
for name, fn, port, ref, inputs in [("exp2", Tensor.exp2, xexp2, lambda v: 2.0 ** v, xs), ("sin", Tensor.sin, xsin, math.sin, sx)]:
  got = fn(Tensor(inputs, dtype=dtypes.float32)).tolist()
  for CPU in (False, True):
    same = sum(f2i(g) == f2i(port(v)) for g, v in zip(got, inputs))
    print(f"  {name}: port with CPU={CPU!s:5}: bit-exact with tinygrad for {same}/{len(inputs)} inputs, "
          f"max {max(ulps(g, port(v)) for g, v in zip(got, inputs))} ulp apart")
  err, at = max((ulps(g, f32(ref(f32(v)))), v) for g, v in zip(got, inputs))
  print(f"  {name}: tinygrad's max error vs libm (on the float32 input): {err} ulp at x={at:.6g}")
CPU = True
for v in (0.5, 10.25, -3.7, 100.0):
  print(f"  exp2({v:6}) tinygrad={Tensor([v]).exp2().item():<14.9g} port={xexp2(v):<14.9g} libm={2.0 ** f32(v):.9g}")
for v in (0.5, 3.14159, 31.0, 1000.25):
  print(f"  sin({v:7}) tinygrad={Tensor([v]).sin().item():<14.9g} port={xsin(v):<14.9g} libm={math.sin(f32(v)):.9g}")
