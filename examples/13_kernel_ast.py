# Chapter 13 — Anatomy of a kernel AST
# Run: DEV=CPU python3 examples/13_kernel_ast.py
#
# We build five tiny tensor programs, ask the scheduler for the LINEAR of kernels that would realize them
# (Tensor.schedule_linear), and pull out each kernel's AST: the SINK(KernelInfo) that sits in CALL.src[0].
# That SINK is exactly what engine/realize.py hands to codegen.to_program(ast, renderer).
# We then print the AST three ways (UOp repr, render_uir, a census of ops), and lower it to real C on the
# CPU backend with NOOPT=1 (no optimizations) and with the default hand-coded optimizations.
import json, os, re, sys
from collections import Counter
from tinygrad import Tensor, Device
from tinygrad.uop.ops import Ops, UOp, KernelInfo
from tinygrad.helpers import Context
from tinygrad.codegen import to_program

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from uop2json import uop_to_graph

def kernel_calls(*ts: Tensor) -> list[UOp]:
  # schedule_linear returns LINEAR(CALL, CALL, ...); a compute kernel is a CALL whose body (src[0]) is a SINK with KernelInfo.
  # CALL.src[1+slot] is the BUFFER UOp that PARAM(slot) in the body will be bound to at launch.
  lin = Tensor.schedule_linear(*ts)
  return [c for c in lin.src if c.op is Ops.CALL and c.src[0].op is Ops.SINK and isinstance(c.src[0].arg, KernelInfo)]
def kernel_asts(*ts: Tensor) -> list[UOp]: return [c.src[0] for c in kernel_calls(*ts)]

# small concrete inputs (python lists -> real buffers, so the kernel ASTs really LOAD from PARAMs)
A  = Tensor([[1., 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).realize()   # (3,4)
B  = Tensor([[1., 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]).realize()      # (3,4)
M  = Tensor([[1., 0, 2], [0, 1, 1]]).realize()                          # (2,3)
x  = Tensor([1., 2, 3]).realize()                                       # (3,)
col = Tensor([[1.], [2], [3]]).realize()                                # (3,1)
row = Tensor([[10., 20, 30, 40]]).realize()                             # (1,4)

cases = {
  "add":    ("A + B                 # (3,4) + (3,4)", lambda: A + B),
  "sum":    ("A.sum(axis=1)         # (3,4) -> (3,)", lambda: A.sum(axis=1)),
  "matmul": ("M @ A                 # (2,3) @ (3,4) -> (2,4)", lambda: M @ A),
  "pad":    ("x.pad((1,1)) * 2      # (3,) -> (5,)", lambda: x.pad((1, 1)) * 2),
  "bcast":  ("col + row             # (3,1) + (1,4) -> (3,4)", lambda: col + row),
}
inputs = {"A": A, "B": B, "M": M, "x": x, "col": col, "row": row}

cpu = Device["CPU"].renderer

# record the range list the kernel name is built from: wrap Scheduler.get_optimized_ast (codegen/opt/postrange.py)
from tinygrad.codegen.opt import postrange
_goa, last_rngs = postrange.Scheduler.get_optimized_ast, []
def _recording_goa(self, name_override=None):
  last_rngs[:] = [[int(sz), t.name] for sz, t in zip(self.full_shape, self.axis_types)]
  return _goa(self, name_override)
postrange.Scheduler.get_optimized_ast = _recording_goa
out = {}
for name, (code, fn) in cases.items():
  calls = kernel_calls(fn())
  assert len(calls) == 1, f"{name}: expected one kernel, got {len(calls)}"
  call = calls[0]; ast = call.src[0]
  print(f"\n===================== {name}: {code}")
  if name == "add":
    print("--- repr(ast): the full UOp tree (x4:= names a shared node)")
    print(ast)
  print("--- ast.render_uir()")
  print(ast.render_uir())
  census = Counter(u.op.name for u in ast.toposort())
  print("--- op census:", dict(census))
  rngs = sorted([u for u in ast.toposort() if u.op is Ops.RANGE], key=lambda u: u.arg)
  print("--- RANGEs (arg = (AxisType, axis_id)):", [(str(r.arg[0]), r.arg[1], r.src[0].arg) for r in rngs])

  # lower to real code. to_program == the same call engine/realize.py makes (it is cached on ast.key + NOOPT + ...)
  with Context(NOOPT=1): p0 = to_program(ast, cpu)
  rngs0 = list(last_rngs)
  p1 = to_program(ast, cpu)
  rngs1 = list(last_rngs)
  ki0, ki1 = p0.src[0].arg, p1.src[0].arg   # PROGRAM.src = (SINK, LINEAR, SOURCE, BINARY); the SINK's KernelInfo is now filled in
  print(f"--- NOOPT=1 : name={ki0.function_name}  applied_opts={ki0.applied_opts}  estimates={ki0.estimates}")
  print(p0.src[2].arg.strip())
  print(f"--- default : name={ki1.function_name}  applied_opts={ki1.applied_opts}  estimates={ki1.estimates}")
  print(f"--- ranges the names were built from: NOOPT {rngs0}  default {rngs1}")

  # run it for real so the page can show the result
  result = fn().realize().tolist()
  print("--- result:", result)
  # which input tensor sits in which PARAM slot? compare the CALL's buffer UOps with the inputs' realized buffers
  slots = {}
  for slot, b in enumerate(call.src[1:]):
    nm = next((k for k, t in inputs.items() if t.uop.base is b), "out" if slot == 0 else "?")
    slots[slot] = nm
  print("--- PARAM slots:", slots)

  ids: dict = {}
  graph = uop_to_graph(ast, ids)
  # structured IR for the in-browser interpreter / pseudo-C (op, src ids, decoded args)
  ir = []
  for u in ast.toposort():
    d = {"id": ids[u], "op": u.op.name, "src": [ids[s] for s in u.src], "dtype": str(u.dtype)}
    if u.op is Ops.CONST: d["val"] = "Invalid" if u.is_invalid else u.arg
    if u.op is Ops.PARAM: d["slot"], d["size"] = u.arg.slot, u.src[0].arg
    if u.op is Ops.RANGE: d["axis_type"], d["axis_id"], d["size"] = u.arg[0].name, u.arg[1], u.src[0].arg
    if u.op is Ops.REDUCE: d["rop"] = u.arg[0].name
    if u.op is Ops.CAST: d["to"] = str(u.arg)
    ir.append(d)
  out[name] = {
    "code": code, "name_ansi": [ki0.name, ki1.name], "uir": ast.render_uir(), "repr": str(ast) if name == "add" else None, "census": dict(census),
    "graph": graph, "ir": ir, "result": result, "slots": slots,
    "noopt": {"rngs": rngs0, "name": ki0.name, "fname": ki0.function_name, "opts": [str(o) for o in ki0.applied_opts], "src": p0.src[2].arg.strip(),
              "est": {"ops": ki0.estimates.ops, "lds": ki0.estimates.lds, "mem": ki0.estimates.mem}},
    "opt": {"rngs": rngs1, "name": ki1.name, "fname": ki1.function_name, "opts": [str(o) for o in ki1.applied_opts], "src": p1.src[2].arg.strip(),
            "est": {"ops": ki1.estimates.ops, "lds": ki1.estimates.lds, "mem": ki1.estimates.mem}},
  }

out["_inputs"] = {k: v.tolist() for k, v in inputs.items()}

# ---- the AST zoo: more kernels, NOOPT C only ----
print("\n===================== zoo")
E = Tensor.empty
zoo = {
  "softmax (3 kernels)": lambda: E(2, 8).softmax(),
  "cumsum": lambda: E(5).cumsum(),
  "x / x.sum()": lambda: E(4) / E(4).sum(),
  "argmax": lambda: E(3, 5).argmax(1),
  "gather a[idx]": lambda: E(6, 4)[Tensor([1, 3])],
  "conv2d": lambda: E(1, 1, 4, 4).conv2d(E(2, 1, 3, 3)),
  "cat": lambda: E(2, 3).cat(E(2, 3)),
}
out["_zoo"] = {}
for zname, fn in zoo.items():
  ks = []
  for ast in kernel_asts(fn()):
    with Context(NOOPT=1): p0 = to_program(ast, cpu)
    ki = p0.src[0].arg
    ks.append({"uir": ast.render_uir(), "graph": uop_to_graph(ast), "name": ki.name, "fname": ki.function_name,
               "src": p0.src[2].arg.strip(), "census": dict(Counter(u.op.name for u in ast.toposort()))})
    print(f"{zname:22s} -> {ki.function_name:14s} {dict(Counter(u.op.name for u in ast.toposort() if u.op is not Ops.CONST))}")
  out["_zoo"][zname] = ks
# which PARAM slot holds which input? slot 0 is always the output; the rest follow the order the scheduler found them
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"), exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ch13_asts.json"), "w") as f: json.dump(out, f, indent=1, default=str)
print("\nwrote data/ch13_asts.json")
