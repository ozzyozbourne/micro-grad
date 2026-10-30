# Chapter 15 — "decomp dtypes": emulating dtypes the renderer can't handle (tinygrad/codegen/decomp/dtype.py).
#
# pm_dtype_decomps collects every fp8/bf16/half/long/ulong dtype it sees, then do_dtype_decomps emulates the ones
# that are in EMULATED_DTYPES or missing from renderer.supported_dtypes():  long -> pairs of int32 words (pm_long_decomp),
# small floats -> float32 math on their bit patterns (pm_float_decomp, f2f).
#
# run:   DEV=CPU python3 examples/15_dtype_decomp.py
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target, Context
from tinygrad.uop.ops import Ops
from tinygrad.uop.render import render_uir
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.renderer.cstyle import ClangRenderer

wgsl, clang = WGSLRenderer(Target("WEBGPU")), ClangRenderer(Target("CPU", "CLANG", "x86_64,native"))
print("WGSL supports int64?", dtypes.long in wgsl.supported_dtypes(), "| Clang supports int64?", dtypes.long in clang.supported_dtypes())

def ast_of(t): return t.schedule_linear().src[-1].src[0]
for title, ren, mk, ctx in [("int64 * 3 on Clang", clang, lambda: Tensor.empty(4, dtype=dtypes.long) * 3, {}),
                            ("int64 * 3 on WGSL", wgsl, lambda: Tensor.empty(4, dtype=dtypes.long) * 3, {}),
                            ("bfloat16 + 1 on Clang, EMULATED_DTYPES=bfloat16", clang, lambda: Tensor.empty(4, dtype=dtypes.bfloat16) + 1,
                             {"EMULATED_DTYPES": "bfloat16"})]:
  with Context(**ctx):
    sink = full_rewrite_to_sink(ast_of(mk()), ren)
  params = [f"slot{u.arg.slot}:{u.dtype}x{u.max_numel()}" for u in sink.toposort() if u.op is Ops.PARAM]
  print(f"\n== {title}: {len(sink.toposort())} UOps, params {params}")
  print("\n".join(render_uir(sink).splitlines()[:14]) + "\n...")
