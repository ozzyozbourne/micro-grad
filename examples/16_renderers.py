# Chapter 16 — one kernel AST, many renderers.
#
# to_program(ast, renderer) runs the whole codegen pipeline for a given Renderer: full_rewrite_to_sink (lowering, ch.15,
# which already depends on the renderer: code_for_op, extra_matcher, has_local, global_max...), linearize, render, compile.
# Rendering needs no GPU at all. Compiling does, so for every renderer whose compiler can't run here we swap in the
# base Compiler(), whose compile() just returns the source bytes (device.py: "empty compiler is the default").
# Clang, LLVM (CPU) and X86 compile for real on this machine.
#
# run:   DEV=CPU python3 examples/16_renderers.py           (prints a summary + the first lines of every source)
#        DEV=CPU python3 examples/16_renderers.py --json    (also writes data/ch16_renderers.json)
import sys, os, json, types
from dataclasses import replace
from tinygrad import Tensor
from tinygrad.helpers import Target
from tinygrad.device import Compiler
from tinygrad.codegen import to_program
from tinygrad.uop.ops import Ops

# --- compilers we cannot load here (no nvrtc, no comgr, no Metal framework): stub them so the renderer can be built
import tinygrad.runtime.support.compiler_cuda as ccuda, tinygrad.runtime.support.compiler_amd as camd
class StubCompiler(Compiler):
  def __init__(self, *args, **kwargs): super().__init__(None)
ccuda.NVRTCCompiler = ccuda.NVCCCompiler = ccuda.PTXCompiler = ccuda.NVPTXCompiler = StubCompiler
camd.HIPCompiler = camd.HIPCCCompiler = StubCompiler
sys.modules["tinygrad.runtime.ops_metal"] = types.SimpleNamespace(MetalCompiler=StubCompiler)
import tinygrad.runtime.support.compiler_llvm as cllvm
cllvm.AMDLLVMCompiler = StubCompiler

from tinygrad.renderer.cstyle import ClangRenderer, OpenCLRenderer, MetalRenderer, CUDARenderer, HIPRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer, AMDLLVMRenderer
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.renderer.x86 import X86Renderer

RENDERERS = [  # (tab name, constructor, compiles for real here?)
  ("Clang (CPU)",     lambda: ClangRenderer(Target("CPU", "CLANG", "x86_64,native")), True),
  ("LLVM IR (CPU)",   lambda: CPULLVMRenderer(Target("CPU", "LLVM", "x86_64,native")), True),
  ("x86-64 (ISA)",    lambda: X86Renderer(Target("CPU", "X86", "x86_64,native")), True),
  ("OpenCL",          lambda: OpenCLRenderer(Target("CL")), False),
  ("Metal",           lambda: MetalRenderer(Target("METAL", "", "Apple9")), False),
  ("CUDA",            lambda: CUDARenderer(Target("CUDA", "", "sm_89")), False),
  ("PTX",             lambda: PTXRenderer(Target("CUDA", "PTX", "sm_89")), False),
  ("HIP (AMD)",       lambda: HIPRenderer(Target("AMD", "HIP", "gfx1100")), False),
  ("LLVM IR (AMDGPU)", lambda: AMDLLVMRenderer(Target("AMD", "LLVM", "gfx1100")), False),
  ("WGSL (WebGPU)",   lambda: WGSLRenderer(Target("WEBGPU")), False),
]

def kernels():
  x = Tensor.empty(4, 8)
  a = (x.sum(1) + x.max(1)).schedule_linear().src[-1].src[0]
  a = a.replace(arg=replace(a.arg, opts_to_apply=()))         # no optimizations: plain loops are easier to compare
  b = (Tensor.empty(64).exp() * 2).schedule_linear().src[-1].src[0]   # default (heuristic) optimizations
  return {"reduce": ("(x.sum(1) + x.max(1)) on a 4x8 tensor, no opts", a), "exp": ("(x.exp()*2) on 64 floats, default opts", b)}

def render_all(ast):
  out = []
  for name, mk, real in RENDERERS:
    ren = mk()
    if not real: ren.compiler = StubCompiler()
    error = None
    try: prg = to_program(ast, ren)   # PROGRAM(SINK, LINEAR, SOURCE, BINARY)
    except Exception as e:            # a real compile error: record it, and still show what the renderer produced
      error, ren.compiler = f"{type(e).__name__}: {str(e).strip().splitlines()[0]}", StubCompiler()
      prg = to_program(ast, ren)
    sink, lin, source, binary = prg.src
    if isinstance(ren, X86Renderer): text = ren.asm_str(list(lin.src), sink.arg.function_name)   # SOURCE is hex for ISA
    else: text = source.arg
    info = prg.arg
    out.append({"name": name, "cls": type(ren).__name__, "real_compile": real, "fn": sink.arg.function_name,
                "source": text, "lines": len(text.splitlines()), "nuops": len(lin.src), "binary_bytes": len(binary.arg),
                "global_size": list(info.global_size), "local_size": list(info.local_size),
                "opts": [str(o) for o in sink.arg.applied_opts], "estimates": str(sink.arg.estimates), "error": error})
  return out

if __name__ == "__main__":
  data = {}
  for key, (desc, ast) in kernels().items():
    print(f"===== kernel '{key}': {desc}")
    data[key] = {"desc": desc, "renderers": render_all(ast)}
    for r in data[key]["renderers"]:
      print(f"--- {r['name']:17s} {r['cls']:16s} fn={r['fn']:<12s} global={r['global_size']} local={r['local_size']} "
            f"linear uops={r['nuops']:3d} source lines={r['lines']:3d} binary={r['binary_bytes']}B{'' if r['real_compile'] else ' (stub: = source)'}")
      if r["error"]: print(f"      !! compile failed: {r['error']}")
    for r in data[key]["renderers"]:
      if key == "reduce" and r["name"] in ("Clang (CPU)", "CUDA", "x86-64 (ISA)"):
        print(f"\n[{r['name']}]\n" + "\n".join(r["source"].splitlines()[:28]))
    print()
  if "--json" in sys.argv:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "ch16_renderers.json")
    with open(path, "w") as f: json.dump(data, f, separators=(",", ":"))
    print(f"wrote {os.path.normpath(path)} ({os.path.getsize(path)//1024} KB)")
