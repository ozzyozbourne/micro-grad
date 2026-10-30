#!/usr/bin/env python3
# Chapter 3 — dtype emulation: what codegen does when a dtype must not (or can not) be used natively.
#
#   run:  DEV=CPU python3 examples/03_emulation.py
#
# The decomposition lives in tinygrad/codegen/decomp/dtype.py (pm_dtype_decomps). It runs inside full_rewrite_to_sink
# ("decomp dtypes" step) for every dtype in EMULATED_DTYPES or missing from renderer.supported_dtypes().
#   floats (bf16, fp8, half) -> stored as uintN bits, computed in float32 (f2f bit-twiddling conversions)
#   long / ulong             -> two 32-bit ints (l2i: add-with-carry, schoolbook multiply, ...)
# We render C for a 1-element kernel so the output stays readable. We stop at render: no compile, no run.
import os, subprocess
from tinygrad import Tensor, dtypes, Device, Context
from tinygrad.uop.ops import Ops
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.late.linearizer import linearize

ren = Device["CPU"].renderer
print("CPU renderer:", type(ren).__name__)

def kernel_ast(t: Tensor):
  return next(si.src[0] for si in t.schedule_linear().src if si.src[0].op is Ops.SINK)

def render(ast, emulate: str) -> str:
  with Context(EMULATED_DTYPES=emulate):
    return ren.render(linearize(full_rewrite_to_sink(ast, ren)))

# --- bfloat16 * 2 -------------------------------------------------------------------------
ast = kernel_ast(Tensor.empty(1, dtype=dtypes.bfloat16) * 2)
native = render(ast, "")
print("\n=== bf16 * 2, native (__bf16 in C) ===")
print(native)
# try to compile the native version with this machine's clang (stderr silenced; we only report the outcome)
saved = os.dup(2); devnull = os.open(os.devnull, os.O_WRONLY); os.dup2(devnull, 2)
try:
  ren.compiler.compile(native); outcome = "compiled fine"
except (subprocess.CalledProcessError, RuntimeError) as e: outcome = f"FAILED ({type(e).__name__}): this clang target has no __bf16"
finally: os.dup2(saved, 2); os.close(devnull)
print("native compile:", outcome)

print("\n=== bf16 * 2, EMULATED_DTYPES=bfloat16 (stored as unsigned short, math in float) ===")
print(render(ast, "bfloat16"))

# --- int64 add/mul -------------------------------------------------------------------------
x = Tensor.empty(1, dtype=dtypes.int64)
ast = kernel_ast(x * 3 + x)
print("=== int64: x*3 + x, native ===")
print(render(ast, ""))
print("=== int64: x*3 + x, EMULATED_DTYPES=long (each long is two 32-bit ints) ===")
print(render(ast, "long"))

# --- and it really runs: emulated bf16 end to end -------------------------------------------
with Context(EMULATED_DTYPES="bfloat16"):
  y = Tensor([1.5, -2.25, 3.0], dtype=dtypes.bfloat16) * 2
  print("emulated bf16 result:", y.tolist(), y.dtype)
