# Chapter 15 — the GPU-only stages of full_rewrite_to_sink, shown on the OpenCL renderer (rendering needs no GPU).
#
#  kernel A: a 16x16 elementwise add  -> simplify ranges (16x16 merges into 256), group gpudims, range to special
#  kernel B: a 16x16 sum over axis 1  -> remove reduces (a grouped reduce through a LOCAL buffer), add implicit barriers
#
# run:   DEV=CPU python3 examples/15_gpu_lowering.py
from tinygrad import Tensor
from tinygrad.helpers import Target
from tinygrad.uop.render import render_uir
import tinygrad.codegen as cg
from tinygrad.renderer.cstyle import OpenCLRenderer

snap: dict[str, tuple] = {}
_gr = cg.graph_rewrite
def gr(sink, pm, *a, name=None, **kw):
  ret = _gr(sink, pm, *a, name=name, **kw)
  snap[name] = (sink, ret)
  return ret
cg.graph_rewrite = gr
ren = OpenCLRenderer(Target("CL"))
print(f"OpenCLRenderer: has_local={ren.has_local} global_max={ren.global_max} local_max={ren.local_max} supports_float4={ren.supports_float4}")

def show(title, ast, stages, before_too=()):
  snap.clear()
  out = cg.full_rewrite_to_sink(ast, ren)
  print(f"\n##### {title}: final kernel name {out.arg.name}, opts {out.arg.applied_opts}")
  for name in stages:
    before, after = snap[name]
    if name in before_too: print(f"=== before {name!r}\n{render_uir(before)}")
    print(f"=== after {name!r}\n{render_uir(after)}")

show("A: Tensor.empty(16,16) + Tensor.empty(16,16)", (Tensor.empty(16, 16) + Tensor.empty(16, 16)).schedule_linear().src[-1].src[0],
     ["simplify ranges", "group gpudims", "range to special"], before_too=("simplify ranges", "range to special"))
show("B: Tensor.empty(16,16).sum(1)", Tensor.empty(16, 16).sum(1).schedule_linear().src[-1].src[0],
     ["postopt symbolic", "remove reduces", "add implicit barriers"])
