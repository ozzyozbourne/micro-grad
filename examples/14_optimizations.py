# Chapter 14 — Optimizations, tensor cores & BEAM
# Run: DEV=CPU python3 examples/14_optimizations.py
#
# Everything here is real: we take kernel ASTs from the scheduler, apply OptOps with the real Scheduler
# (codegen/opt/postrange.py), lower them with the real full_rewrite_to_sink, and render with real renderers.
# GPU renderers (OpenCL, Metal, CUDA) are instantiated WITHOUT their compilers (no GPU / no driver here):
# we only need .render(), never .compile(). BEAM search at the end really compiles and times CPU kernels.
import json, os, sys, time
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import Ops, UOp, KernelInfo, AxisType
from tinygrad.helpers import Context, Target
from tinygrad.renderer import Renderer, Estimates, tc
from tinygrad.renderer.cstyle import OpenCLRenderer, MetalRenderer, CUDARenderer
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups, pm_alloc_to_buf
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.codegen.opt import postrange, heuristic
from tinygrad.codegen.opt.postrange import Scheduler, split_targets
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

def asts(*ts):
  return [c.src[0] for c in Tensor.schedule_linear(*ts).src if c.op is Ops.CALL and c.src[0].op is Ops.SINK]

def bare(cls, target, tcs=()):
  # a renderer object without its compiler: Renderer.__init__ only stores the target
  r = cls.__new__(cls); Renderer.__init__(r, target); r.tensor_cores = list(tcs); return r

CPU = Device["CPU"].renderer                                   # ClangRenderer: has_local=False, no tensor cores
CL = OpenCLRenderer(Target("CL", "CL", ""))                    # has_local=True, has_shared=True, no tensor cores
METAL = bare(MetalRenderer, Target("METAL", "METAL", "Apple9"), tc.metal)
CUDA = bare(CUDARenderer, Target("CUDA", "CUDA", "sm_80"), tc.get_cuda("sm_80"))

def lower(ast: UOp, ren):
  """the same steps do_to_program performs, minus compile: optimize+lower, linearize, render. returns (KernelInfo, src, Estimates, n_uops)"""
  sink = full_rewrite_to_sink(ast, ren, optimize=not ast.arg.optimized)
  lst = line_rewrite(linearize(sink), pm_linearize_cleanups + pm_alloc_to_buf)
  return sink.arg, ren.render(lst), Estimates.from_uops(tuple(lst), ignore_indexing=True), len(lst)

def with_opts(ast: UOp, opts) -> UOp:
  # the official way to force an explicit opt list: KernelInfo.opts_to_apply (apply_opts in postrange.py applies them in order)
  return ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts)))

def est(e): return {"ops": int(e.ops), "lds": int(e.lds), "mem": int(e.mem)}
def shape_of(k: Scheduler): return [[int(s), t.name] for s, t in zip(k.full_shape, k.axis_types)]
def fmt(k: Scheduler): return " ".join(f"{s}:{t.name}" for s, t in zip(k.full_shape, k.axis_types))

# ---------------------------------------------------------------- 1. what exists
print("OptOps members :", [o.name for o in OptOps])
print("AxisType order :", [a.name for a in AxisType])
print("split_targets  :", {k.name: [v.name for v in vs] for k, vs in split_targets.items()})
OUT["optops"] = [o.name for o in OptOps]

# ---------------------------------------------------------------- 2. shift_to on one RANGE of 16
ast16 = asts(Tensor.empty(16) + 1)[0]
k = Scheduler(ast16, CPU)
print("\n== shift_to: E(16)+1 on CPU.  before:", fmt(k))
print(k.ast.render_uir())
k.apply_opt(Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)))
print("after Opt(SPLIT, axis=0, arg=(4, UPCAST)):", fmt(k))
print(k.ast.render_uir())
k2 = Scheduler(ast16, CPU); k2.apply_opt(Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST, True)))
OUT["shift"] = {"before": Scheduler(ast16, CPU).ast.render_uir(), "after": k.ast.render_uir(), "after_top": k2.ast.render_uir()}
for nm, opts in [("noopt", None), ("upcast4", [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST))])]:
  if opts is None:
    with Context(NOOPT=1): ki, src, e, n = lower(ast16, CPU)
  else: ki, src, e, n = lower(with_opts(ast16, opts), CPU)
  OUT["shift"][nm] = src
print("C with UPCAST 4:\n" + OUT["shift"]["upcast4"])

# ---------------------------------------------------------------- 3. the same matmul: NOOPT vs heuristic vs manual opts
A, B = Tensor.empty(16, 16), Tensor.empty(16, 16)
mm = asts(A @ B)[0]
print("\n== matmul 16x16 @ 16x16. the AST has ranges:", fmt(Scheduler(mm, CPU)))
# NOTE: Opt.axis indexes the CURRENT Scheduler.rngs list, which is re-sorted after every opt (UPCAST ranges sort after WEAK/GLOBAL ones,
# reduce ranges last), so after splitting axis 0 the other output axis is still axis 1 and the reduce axis moves to index 4.
manual = [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 4, (4, AxisType.UPCAST))]
OUT["mm_variants"] = {}
for rname, ren in [("CPU (Clang)", CPU), ("OpenCL", CL)]:
  OUT["mm_variants"][rname] = {}
  for vname in ["NOOPT=1", "default (hand_coded_optimizations)", "manual opts_to_apply"]:
    if vname.startswith("NOOPT"):
      with Context(NOOPT=1): ki, src, e, n = lower(mm, ren)
    elif vname.startswith("default"): ki, src, e, n = lower(mm, ren)
    else: ki, src, e, n = lower(with_opts(mm, manual if ren is CPU else manual[:2] + [Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL))]), ren)
    OUT["mm_variants"][rname][vname] = {"name": ki.function_name, "opts": [str(o) for o in ki.applied_opts], "src": src, "est": est(e), "uops": n}
    print(f"[{rname:11s}] {vname:36s} -> {ki.function_name:18s} uops={n:4d} {est(e)}  opts={list(ki.applied_opts)}")
print("\n-- CPU, manual opts (UPCAST 4 on both output axes, UNROLL the reduce by 4):")
print(OUT["mm_variants"]["CPU (Clang)"]["manual opts_to_apply"]["src"])

# ---------------------------------------------------------------- 4. trace hand_coded_optimizations
log = []
_orig_apply = Scheduler.apply_opt
def traced_apply(self, opt, append_opt=True):
  try:
    ret = _orig_apply(self, opt, append_opt)
    log.append({"opt": str(opt), "ok": True, "shape": shape_of(self), "inner": not append_opt})
    return ret
  except KernelOptError as ex:
    log.append({"opt": str(opt), "ok": False, "err": str(ex)[:120], "shape": shape_of(self), "inner": not append_opt})
    raise
Scheduler.apply_opt = traced_apply
traces = {}
for tname, t, ren in [("matmul 64x64 · CPU", Tensor.empty(64, 64) @ Tensor.empty(64, 64), CPU),
                      ("matmul 64x64 · OpenCL", Tensor.empty(64, 64) @ Tensor.empty(64, 64), CL),
                      ("sum (64,256)→(64,) · OpenCL", Tensor.empty(64, 256).sum(1), CL),
                      ("sum 4096→() · OpenCL", Tensor.empty(4096).sum(), CL),
                      ("matvec (256,)@(256,64) · OpenCL", Tensor.empty(256) @ Tensor.empty(256, 64), CL),
                      ("matmul 64x64 fp32 · Metal (TC)", Tensor.empty(64, 64) @ Tensor.empty(64, 64), METAL)]:
  ast = [a for a in asts(t) if any(u.op is Ops.REDUCE for u in a.toposort())][-1]
  k = Scheduler(ast, ren); k.convert_loop_to_global()
  start = shape_of(k)
  log.clear()
  k2 = hand_coded_optimizations(k)
  ki, src, e, n = lower(k2.get_optimized_ast(), ren)
  traces[tname] = {"start": start, "log": list(log), "final": shape_of(k2), "opts": [str(o) for o in k2.applied_opts],
                   "name": ki.function_name, "src": src, "est": est(e)}
  print(f"\n== heuristic trace: {tname}   start {' '.join(f'{s}:{t}' for s,t in start)}")
  for l in log: print(("   ok  " if l["ok"] else "   FAIL") + (" (inner)" if l["inner"] else "        "), l["opt"], "->", " ".join(f"{s}:{t}" for s, t in l["shape"]), "" if l["ok"] else "  # " + l["err"])
  print("   final:", ki.function_name, list(k2.applied_opts))
Scheduler.apply_opt = _orig_apply
OUT["traces"] = traces
print("\n-- OpenCL source for the GROUPed sum (local memory + barrier):")
print(traces["sum 4096→() · OpenCL"]["src"])

# ---------------------------------------------------------------- 5. the opt playground (matmul 16x16x16, OpenCL)
def menu(k: Scheduler):
  red = k.axes_of(reduce=True)
  items = []
  for a in range(k.shape_len):
    if k.full_shape[a] < 4: continue
    if a in red:
      items += [Opt(OptOps.SPLIT, a, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, a, (4, AxisType.LOCAL)),            # UNROLL, GROUP
                Opt(OptOps.SPLIT, a, (4, AxisType.LOCAL, True))]                                                 # GROUPTOP
    else:
      items += [Opt(OptOps.SPLIT, a, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, a, (4, AxisType.LOCAL))]         # UPCAST, LOCAL
  return items
def label(opt: Opt, k: Scheduler):
  amt, t = opt.arg[0], opt.arg[1]
  top = len(opt.arg) > 2 and opt.arg[2]
  red = opt.axis in k.axes_of(reduce=True)
  kind = ("GROUPTOP" if top else "GROUP") if (t is AxisType.LOCAL and red) else ("UNROLL" if red else t.name)
  return f"{kind} axis {opt.axis} by {amt}"
pg_ast = asts(Tensor.empty(16, 16) @ Tensor.empty(16, 16))[0]
nodes = {}
def explore(k: Scheduler, key: str, depth: int):
  ki, src, e, n = lower(k.get_optimized_ast(), CL)
  node = {"opts": [str(o) for o in k.applied_opts], "shape": shape_of(k), "name": ki.function_name, "src": src, "est": est(e), "uops": n, "children": []}
  nodes[key] = node
  if depth == 0: return
  for i, opt in enumerate(menu(k)):
    k2 = k.copy()
    try: k2.apply_opt(opt)
    except KernelOptError as ex:
      node["children"].append({"label": label(opt, k), "opt": str(opt), "err": str(ex)}); continue
    ck = key + "." + str(i)
    node["children"].append({"label": label(opt, k), "opt": str(opt), "key": ck})
    explore(k2, ck, depth - 1)
k0 = Scheduler(pg_ast, CL); k0.convert_loop_to_global()
explore(k0, "r", 2)
with open(os.path.join(HERE, "..", "data", "ch14_playground.json"), "w") as f: json.dump(nodes, f, indent=0)
print(f"\n== playground: {len(nodes)} real variants of matmul 16x16x16 rendered for OpenCL")
for key in ["r", "r.0", "r.0.2", "r.5"]:
  if key in nodes: print(f"  {key:8s} {nodes[key]['name']:22s} {nodes[key]['est']} uops={nodes[key]['uops']}")

# ---------------------------------------------------------------- 6. tensor cores
print("\n== tensor cores (renderer/tc.py)")
tcs = []
for rname, lst in [("Metal (Apple7+)", tc.metal), ("CUDA sm_80", tc.get_cuda("sm_80")), ("CUDA sm_89", tc.get_cuda("sm_89")),
                   ("AMD RDNA3", tc.amd_rdna3), ("AMD RDNA4", tc.amd_rdna4), ("AMD CDNA3", tc.amd_cdna3)]:
  for t in lst:
    tcs.append({"renderer": rname, "dims_NMK": t.dims, "threads": t.threads, "in": t.dtype_in.name, "out": t.dtype_out.name})
  print(f"  {rname:16s}", sorted({(t.dims, t.threads) for t in lst}), sorted({f'{t.dtype_in.name}->{t.dtype_out.name}' for t in lst}))
OUT["tensor_cores"] = tcs
print("  CPU (Clang) tensor_cores:", CPU.tensor_cores, "  OpenCL:", CL.tensor_cores)
tcv = {}
for rname, ren, dt in [("Metal fp32 8x8x8", METAL, dtypes.float), ("CUDA fp16 m16n8k16", CUDA, dtypes.half)]:
  a = asts(Tensor.empty(16, 16, dtype=dt) @ Tensor.empty(16, 16, dtype=dt))[0]
  k = Scheduler(a, ren); k.convert_loop_to_global()
  k.apply_opt(Opt(OptOps.TC, 0, (-1, 0, 1)))
  tc_ast = k.get_optimized_ast()
  wmma = [u for u in tc_ast.toposort() if u.op is Ops.WMMA][0]
  ki, src, e, n = lower(tc_ast, ren)
  tcv[rname] = {"shape": shape_of(k), "wmma_arg": str(wmma.arg), "name": ki.function_name, "src": src, "est": est(e)}
  print(f"  TC on {rname}: shape after TC opt = {fmt(k)}\n     WMMA arg = {wmma.arg}")
OUT["tc_variants"] = tcv
print(tcv["Metal fp32 8x8x8"]["src"])

# ---------------------------------------------------------------- 7. BEAM=2 on the CPU, recording every timed candidate
from tinygrad.codegen.opt import search
records = []
_orig_time = search._time_program
def recording_time(prg, *args, **kwargs):
  tms = _orig_time(prg, *args, **kwargs)
  records.append({"opts": [str(o) for o in prg.src[0].arg.applied_opts], "t": min(tms), "uops": len(prg.src[1].src)})
  return tms
search._time_program = recording_time
_orig_bs = search.beam_search
result = {}
def wrapped_bs(s, *a, **kw):
  t0 = time.perf_counter(); ret = _orig_bs(s, *a, **kw)
  result.update({"start": shape_of(s), "final_opts": [str(o) for o in ret.applied_opts], "final": shape_of(ret), "secs": time.perf_counter() - t0})
  return ret
search.beam_search = wrapped_bs
X, W = Tensor.empty(32, 32).realize(), Tensor.empty(32, 32).realize()
with Context(BEAM=2, IGNORE_BEAM_CACHE=1, DEBUG=0):
  (X @ W).realize()
search._time_program, search.beam_search = _orig_time, _orig_bs
print(f"\n== BEAM=2 on CPU, matmul 32x32: timed {len(records)} candidates in {result['secs']:.1f}s")
print("   start shape:", result["start"], "\n   winner opts:", result["final_opts"], "\n   winner shape:", result["final"])
for r in sorted(records, key=lambda r: r["t"])[:5]: print(f"   {r['t']*1e6:8.2f} us  {r['opts']}")
OUT["beam"] = {"records": records, **result}

with open(os.path.join(HERE, "..", "data", "ch14_opts.json"), "w") as f: json.dump(OUT, f, indent=1, default=str)
print("\nwrote data/ch14_opts.json")
