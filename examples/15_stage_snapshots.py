# Chapter 15 — print the kernel at the interesting points of full_rewrite_to_sink, in tinygrad's own compact text form
# (render_uir, the same printer DEBUG>=5 uses). Kernel: Tensor.empty(4,8).sum(1) on the CPU (Clang) renderer.
#
# run:   DEV=CPU python3 examples/15_stage_snapshots.py
from tinygrad import Tensor
from tinygrad.helpers import Target
from tinygrad.uop.render import render_uir
import tinygrad.codegen as cg
from tinygrad.renderer.cstyle import ClangRenderer

# record (stage name -> output sink) by wrapping the graph_rewrite that codegen/__init__.py calls
snapshots: dict[str, object] = {}
_gr, _mc = cg.graph_rewrite, cg.memory_coalescing
def gr(sink, pm, *a, name=None, **kw):
  snapshots[name] = ret = _gr(sink, pm, *a, name=name, **kw)
  return ret
def mc(sink, ren):
  snapshots["memory_coalescing()"] = ret = _mc(sink, ren)
  return ret
cg.graph_rewrite, cg.memory_coalescing = gr, mc

ast = Tensor.empty(4, 8).sum(1).schedule_linear().src[-1].src[0]
final = cg.full_rewrite_to_sink(ast, ClangRenderer(Target("CPU", "CLANG", "x86_64,native")))

SHOW = ["postopt symbolic",                  # optimized AST: the UPCAST range is still a RANGE
        "expander",                          # UPCAST RANGE -> CONST tuple (0..7) with a shape; REDUCE gets "horizontal" axes
        "remove reduces",                    # REDUCE -> register accumulator (ALLOC REG + STORE/AFTER/END)
        "*** expand broadcast / add loads",  # broadcasting made explicit, INDEX of GLOBAL buffers wrapped in LOAD
        "devectorize2",                      # every shaped (vector) op split into scalar lanes + STACK
        "memory_coalescing()",               # 8 scalar loads at idx+0..7 -> 2 SHRINK(len 4) float4 loads
        "lower all index dtypes",            # weakint index math commits to int32
        "late decompositions"]               # MUL by 8 -> SHL by 3 (Clang has SHL in code_for_op)
for name in SHOW:
  s = snapshots[name]
  print(f"===== after {name!r}  ({len(s.toposort())} UOps)")
  print(render_uir(s))
# the very end: 'cast consts' commits every remaining bare (weak) const, e.g. the 4 in "add %5, 4" becomes an i32 CAST
print(f"===== final sink returned by full_rewrite_to_sink ({len(final.toposort())} UOps)")
print(render_uir(final))
