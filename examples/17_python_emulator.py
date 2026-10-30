# Chapter 17 — the PYTHON device: a pure-Python interpreter that executes the linearized UOps one by one.
# run: DEV=PYTHON NOOPT=1 python3 examples/17_python_emulator.py
# (also works with DEV=CPU: the script forces the PYTHON device itself)
# It captures the real execution trace (TRACE=1 hook of runtime/ops_python.py) into data/ch17_python_trace.json
import os, sys, json, pickle, pathlib
os.environ["TRACE"] = "1"   # PythonProgram.__call__ checks getenv("TRACE") and prints every executed UOp
from tinygrad import Tensor, Context
from tinygrad.engine.realize import compile_linear, run_linear
import tinygrad.runtime.ops_python as ops_python

with Context(DEV="PYTHON", NOOPT=1):
  a = Tensor([1., 2, 3], device="PYTHON").realize()
  b = Tensor([10., 20, 30], device="PYTHON").realize()
  c = (a * b).sum()
  lin = c.schedule_linear()
  prg = compile_linear(lin).src[-1].body          # the CALL's body is now a PROGRAM
  uops = pickle.loads(prg.src[3].arg)             # PYTHON "binary" = the pickled UOp list (PythonRenderer.render)
  idx = {u: i for i, u in enumerate(uops)}

  print(f"kernel {prg.src[0].arg.name}: {len(uops)} uops, global_size={prg.arg.global_size}")
  for i, u in enumerate(uops):
    print(f"  {i:2d} {u.op.name:8s} {str(u.dtype):16s} src={[idx[s] for s in u.src if s in idx]} arg={u.arg}")

  # hook the emulator's print to record the trace instead of printing it
  names, trace = {}, []
  def name_of(m):
    return names.setdefault(id(m), f"mem{len(names)}")
  def fmt(v):
    if isinstance(v, memoryview): return name_of(v)
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], memoryview): return f"&{name_of(v[0])}[{v[1]}]"
    if isinstance(v, list): return [fmt(x) for x in v]
    if isinstance(v, float): return round(v, 4)
    return v if isinstance(v, (int, str)) else str(v)
  mems = {}
  def hook(i, op, dtype, arg, src_values, src_dtypes):
    for sv in src_values:
      for lane in sv if isinstance(sv, list) else []:
        m = lane[0] if isinstance(lane, tuple) else lane
        if isinstance(m, memoryview): mems[name_of(m)] = m
    trace.append({"pc": i, "op": op.name, "src": [fmt(sv[0]) if isinstance(sv, list) and len(sv) == 1 else fmt(sv) for sv in src_values],
                  "mem": {k: [round(x, 4) for x in m.tolist()] for k, m in mems.items()}})
  ops_python.print = hook
  run_linear(lin)
  del ops_python.print
  result = c.item()

print(f"\nexecuted {len(trace)} UOp steps (the RANGE loop body ran 3 times); result = {result}")
for t in trace[:16]: print(f"  pc={t['pc']:2d} {t['op']:8s} src={t['src']}")
print("  ...")
for t in trace[-6:]: print(f"  pc={t['pc']:2d} {t['op']:8s} src={t['src']}")
print("final memory:", trace[-1]["mem"])

out = {"kernel": prg.src[0].arg.name, "result": result,
       "uops": [{"i": i, "op": u.op.name, "dtype": str(u.dtype), "arg": str(u.arg)[:60], "src": [idx[s] for s in u.src if s in idx]} for i, u in enumerate(uops)],
       "trace": trace}
path = pathlib.Path(__file__).resolve().parent.parent / "data" / "ch17_python_trace.json"
path.write_text(json.dumps(out))
print("wrote", path.name, f"({path.stat().st_size} bytes)")
