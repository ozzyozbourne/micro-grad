#!/usr/bin/env python3
# Chapter 1 — trace ONE tinygrad program through every stage of the pipeline.
#
#   run:  DEV=CPU python3 examples/01_pipeline_trace.py
#         DEV=CPU python3 examples/01_pipeline_trace.py --dump   # also writes data/ch01_pipeline.json for the guide
#
# We do NOT re-implement anything here. We "spy" on the real functions tinygrad calls during realize():
#   Tensor.linear_with_vars           (tinygrad/tensor.py)          tensor graph -> bufferized graph -> LINEAR
#   transform_to_call                 (tinygrad/schedule/__init__.py) bufferized graph -> CALL(SINK) with PARAMs
#   prepare_rangeify / run_rangeify   (tinygrad/schedule/prepare.py, indexing.py)  movement ops -> RANGE indexing
#   get_kernel_graph                  (tinygrad/schedule/rangeify.py) graph cut into per-kernel CALLs
#   create_schedule                   (tinygrad/schedule/__init__.py) kernels toposorted into a LINEAR
#   to_program                        (tinygrad/codegen/__init__.py) per-kernel AST -> PROGRAM(SINK, LINEAR, SOURCE, BINARY)
# and print what goes in and what comes out of each.
import sys, pathlib
from tinygrad import Tensor, Device
from tinygrad.uop.ops import UOp, Ops
import tinygrad.schedule as sched
import tinygrad.schedule.rangeify as rangeify_mod
from tinygrad.codegen import to_program
from tinygrad.engine.realize import run_linear

captured: dict[str, list] = {}
def spy(module, name):
  """replace module.name with a wrapper that records (args, result) and then returns the real result"""
  real = getattr(module, name)
  def wrapper(*args, **kwargs):
    ret = real(*args, **kwargs)
    captured.setdefault(name, []).append((args, ret))
    return ret
  setattr(module, name, wrapper)

# these names are looked up at call time in the module namespace, so patching the module attribute is enough
spy(sched, "transform_to_call")
spy(sched, "prepare_rangeify")
spy(sched, "get_kernel_graph")
spy(sched, "create_schedule")
spy(rangeify_mod, "run_rangeify")

def banner(s): print(f"\n{'='*12} {s} {'='*(60-len(s))}")
def count_ops(u:UOp) -> str:
  ops: dict[str, int] = {}
  for x in u.toposort(): ops[x.op.name] = ops.get(x.op.name, 0) + 1
  return ", ".join(f"{k}:{v}" for k,v in sorted(ops.items(), key=lambda kv: -kv[1]))

# ---------------------------------------------------------------- stage 0: the program
a = Tensor([1., 2., 3., 4.])
b = Tensor([5., 6., 7., 8.])
c = (a * b + 1).sum()          # nothing has been computed yet!

banner("STAGE 1: the lazy tensor graph (c.uop)")
print(c)                       # the Tensor repr: it holds a UOp
print(c.uop)
tensor_graph = c.uop

banner("STAGE 2: scheduling  (Tensor.linear_with_vars)")
linear, var_vals = c.linear_with_vars()
bufferized_sink = captured["transform_to_call"][0][0][0]
call = captured["transform_to_call"][0][1]
print("-- input to transform_to_call: outputs got storage (BUFFER + STORE + AFTER):")
print(bufferized_sink)
print("\n-- output: one CALL; its body is a SINK where every BUFFER became a PARAM slot:")
print("CALL body ops:", count_ops(call.src[0]))
print("CALL args    :", [f"{s.op.name}({s.arg.device}, slot={s.arg.slot})" for s in call.src[1:]])
prepared = captured["prepare_rangeify"][0][1]
rangeified = captured["run_rangeify"][0][1]
kernel_graph = captured["get_kernel_graph"][0][1]
print("\n-- after prepare_rangeify: COPY became ALLOC + STORE + AFTER")
print("ops:", count_ops(prepared))
print("\n-- after run_rangeify: movement ops are gone, every access is an INDEX by RANGEs")
print("ops:", count_ops(rangeified))
print("\n-- after get_kernel_graph: the graph is cut into CALLs, one per kernel")
kcalls = [u for u in kernel_graph.toposort() if u.op is Ops.CALL]
print(f"{len(kcalls)} CALLs:", [count_ops(k.src[0]) for k in kcalls])
sched_linear = captured["create_schedule"][0][1]

banner("STAGE 3: the schedule (LINEAR of CALLs, after memory planning)")
for i, si in enumerate(linear.src):
  kind = "kernel" if si.src[0].op is Ops.SINK else "copy (STORE call)" if si.src[0].op is Ops.STORE else si.src[0].op.name
  args = [f"{s.op.name}" + (f"[{s.arg.device}]" if s.op is Ops.BUFFER else "") for s in si.src[1:]]
  print(f"  {i}: CALL  body={si.src[0].op.name:5s} -> {kind:18s} args={args}")

banner("STAGE 4: the kernel AST (body of the compute CALL)")
kernel_call = next(si for si in linear.src if si.src[0].op is Ops.SINK)
ast = kernel_call.src[0]
print(ast)

banner("STAGE 5: codegen  (to_program = full_rewrite_to_sink -> linearize -> render -> compile)")
prg = to_program(ast, Device["CPU"].renderer)
full_sink, lin, source, binary = prg.src
print("PROGRAM srcs:", [s.op.name for s in prg.src])
print("kernel name :", full_sink.arg.name, "| applied opts:", full_sink.arg.applied_opts)
print(f"-- LINEAR: {len(lin.src)} UOps in program order")
for i, u in enumerate(lin.src):
  print(f"  {i:2d} {u.op.name:8s} {str(u.dtype):14s} src={[lin.src.index(s) for s in u.src if s in lin.src]} arg={u.arg if u.op is not Ops.PARAM else u.arg.slot}")
print("-- SOURCE (rendered C):")
print(source.arg)
print(f"-- BINARY: {len(binary.arg)} bytes of compiled machine code (clang, via Compiler.compile_cached)")

banner("STAGE 6: run  (run_linear)")
run_linear(linear, var_vals)
print("c is now backed by a realized buffer:", c.uop)
print("c.item() =", c.item(), "  (expected", sum(x*y+1 for x,y in zip([1,2,3,4],[5,6,7,8])), ")")

if "--dump" in sys.argv:
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
  from uop2json import uop_to_graph, dump
  def artifacts(tensor_graph, linear) -> dict:
    """everything the guide's pipeline demo shows, for the program whose spies are in `captured`"""
    ids: dict = {}
    bufferized_sink, call = captured["transform_to_call"][0][0][0], captured["transform_to_call"][0][1]
    ast = next(si.src[0] for si in linear.src if si.src[0].op is Ops.SINK)
    prg = to_program(ast, Device["CPU"].renderer)
    full_sink, lin, source, binary = prg.src
    def g(u, enter_calls=False):
      if not enter_calls: return uop_to_graph(u, ids)
      # same format as tools/uop2json.py, but walking INTO the bodies of CALLs (src[0]) as well
      nodes, edges = [], []
      for x in u.toposort(enter_calls=True):
        nid = ids.setdefault(x, f"n{len(ids)}")
        nodes.append({"id": nid, "op": x.op.name, "dtype": str(x.dtype), "arg": str(x.arg)[:59], "shape": str(x._shape) if x._shape is not None else ""})
        edges += [[ids.setdefault(s, f"n{len(ids)}"), nid, i] for i, s in enumerate(x.src)]
      return {"nodes": nodes, "edges": edges}
    return {
      "tensor_graph": g(tensor_graph), "bufferized": g(bufferized_sink), "call": g(call, True),
      "prepared": g(captured["prepare_rangeify"][0][1]), "rangeified": g(captured["run_rangeify"][0][1]),
      "kernel_graph": g(captured["get_kernel_graph"][0][1], True), "sched_linear": g(captured["create_schedule"][0][1], True),
      "linear": g(linear, True), "ast": g(ast), "full_sink": g(full_sink),
      "lin_list": [{"i": i, "op": u.op.name, "dtype": str(u.dtype), "src": [lin.src.index(s) for s in u.src if s in lin.src],
                    "arg": str(u.arg.slot if u.op is Ops.PARAM else u.arg)[:60]} for i,u in enumerate(lin.src)],
      "kernel_name": full_sink.arg.name, "opts": str(full_sink.arg.applied_opts),
      "source": source.arg, "binary_len": len(binary.arg),
      "text": {"tensor_graph": str(tensor_graph), "bufferized": str(bufferized_sink), "call": str(call),
               "prepared": str(captured["prepare_rangeify"][0][1]), "rangeified": str(captured["run_rangeify"][0][1]),
               "kernel_graph": str(captured["get_kernel_graph"][0][1]), "sched_linear": str(captured["create_schedule"][0][1]),
               "ast": str(ast), "linear": str(linear), "full_sink": str(full_sink)},
    }
  out = {"sum": artifacts(tensor_graph, linear)}
  # the second program: a small matmul + relu, traced through the very same spies
  captured.clear()
  x = Tensor([[1., -2, 3, -4], [5., -6, 7, -8]]); w = Tensor([[1., 0, -1], [0., 1, 0], [1., 1, 1], [-1, 0, 1]])
  y = (x @ w).relu()
  y_graph = y.uop
  y_linear, _ = y.linear_with_vars()
  out["matmul"] = artifacts(y_graph, y_linear)
  run_linear(y_linear)
  out["matmul"]["result"], out["sum"]["result"] = y.tolist(), c.item()
  p = pathlib.Path(__file__).resolve().parent.parent / "data" / "ch01_pipeline.json"
  dump(out, str(p))
  print("wrote", p)
