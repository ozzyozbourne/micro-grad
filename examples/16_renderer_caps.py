# Chapter 16 — the Renderer "inventory": the class attributes every lowering stage (ch.15) and the linearizer consult.
#
# Renderers are plain classes (renderer/__init__.py Renderer, renderer/cstyle.py CStyleLanguage, ...). Building one needs a
# Target and imports its Compiler, so as in 16_renderers.py we stub the compilers that need GPU toolchains.
#
# run:   DEV=CPU python3 examples/16_renderer_caps.py
import sys, types
from tinygrad.helpers import Target
from tinygrad.device import Compiler
from tinygrad.uop.ops import Ops
import tinygrad.runtime.support.compiler_cuda as ccuda, tinygrad.runtime.support.compiler_amd as camd
import tinygrad.runtime.support.compiler_llvm as cllvm
class StubCompiler(Compiler):
  def __init__(self, *args, **kwargs): super().__init__(None)
ccuda.NVRTCCompiler = ccuda.NVCCCompiler = ccuda.PTXCompiler = ccuda.NVPTXCompiler = StubCompiler
camd.HIPCompiler = camd.HIPCCCompiler = StubCompiler
cllvm.AMDLLVMCompiler = StubCompiler
sys.modules["tinygrad.runtime.ops_metal"] = types.SimpleNamespace(MetalCompiler=StubCompiler)
from tinygrad.renderer.cstyle import ClangRenderer, OpenCLRenderer, MetalRenderer, CUDARenderer, HIPRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer, AMDLLVMRenderer
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.renderer.x86 import X86Renderer

rens = [ClangRenderer(Target("CPU", "CLANG", "x86_64,native")), CPULLVMRenderer(Target("CPU", "LLVM", "x86_64,native")),
        X86Renderer(Target("CPU", "X86", "x86_64,native")), OpenCLRenderer(Target("CL")), MetalRenderer(Target("METAL", "", "Apple9")),
        CUDARenderer(Target("CUDA", "", "sm_89")), PTXRenderer(Target("CUDA", "PTX", "sm_89")), HIPRenderer(Target("AMD", "HIP", "gfx1100")),
        AMDLLVMRenderer(Target("AMD", "LLVM", "gfx1100")), WGSLRenderer(Target("WEBGPU"))]
def short(v):
  if isinstance(v, tuple) and len(v) == 3 and len(set(v)) == 1 and v[0] > 10**6: return "huge"
  return str(v)
print(f"{'renderer':16s} {'float4':6s} {'local':5s} {'global_max':20s} {'local_max':16s} {'smem':6s} {'TCs':3s} {'#ops':4s} missing from code_for_op")
interesting = [Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIP, Ops.FDIV, Ops.MAX, Ops.MULACC, Ops.TRUNCDIV, Ops.TRUNCMOD, Ops.CMPEQ, Ops.SHL]
for r in rens:
  missing = [o.name for o in interesting if o not in r.code_for_op]
  print(f"{type(r).__name__:16s} {str(r.supports_float4):6s} {str(r.has_local):5s} {short(r.global_max):20s} {short(r.local_max):16s} "
        f"{r.shared_max:<6d} {len(r.tensor_cores):<3d} {len(r.code_for_op):<4d} {' '.join(missing)}")
print("\nC-family string options (CStyleLanguage subclasses):")
for r in rens:
  if not hasattr(r, "kernel_typedef"): continue
  wi = {k: v(0) for k, v in r.code_for_workitem.items()}
  print(f"  {type(r).__name__:15s} kernel={r.kernel_typedef[:34]!r:38s} buffer_prefix={r.buffer_prefix!r:12s} smem={r.smem_prefix!r:22s} "
        f"workitem={wi}")
