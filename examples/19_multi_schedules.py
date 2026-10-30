# Chapter 19 — capture the real compiled schedules (kernels + copies, per device) of common multi-device programs.
# run: DEV=CPU python3 examples/19_multi_schedules.py        (writes data/ch19_schedules.json)
import json, pathlib
from tinygrad import Tensor, Context
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import compile_linear, run_linear, get_call_arg_uops

def dev_str(d): return list(d) if isinstance(d, tuple) else d
def describe(a):
  """walk a CALL argument down to its storage: MSELECT(i) picks lane i of a multi buffer, MSTACK stacks per-device buffers"""
  chain = []
  while a.op in (Ops.MSELECT, Ops.SHRINK, Ops.BITCAST, Ops.RESHAPE):
    if a.op is Ops.MSELECT: chain.append(f"MSELECT({a.arg})")
    a = a.src[0]
  if a.op is Ops.MSTACK: chain.append(f"MSTACK[{len(a.src)}]")
  return {"via": chain, "device": dev_str(a.device)}

def capture(label, code, tensor):
  lin = tensor.schedule_linear()
  cl = compile_linear(lin)
  calls = []
  for c in cl.src:
    c = c.without_after
    args = get_call_arg_uops(c)
    if c.body.op is Ops.STORE:
      calls.append({"kind": "copy", "src": describe(args[1]), "dst": describe(args[0]),
                    "nbytes": args[0].max_numel() * args[0].dtype.itemsize})
    else:
      src = c.body.src[2].arg
      calls.append({"kind": "kernel", "name": c.body.src[0].arg.name, "device": dev_str(args[0].device),
                    "args": [describe(a) for a in args], "device_num": "_device_num" in src, "source": src})
  run_linear(lin)
  value = tensor.tolist()
  print(f"\n== {label}: {code}\n   -> {sum(c['kind']=='kernel' for c in calls)} kernels, {sum(c['kind']=='copy' for c in calls)} copies")
  for i, c in enumerate(calls):
    if c["kind"] == "copy": print(f"   {i:2d} COPY   {c['nbytes']:3d}B  {c['src']['device']}{''.join('.'+v for v in c['src']['via'])} -> {c['dst']['device']}")
    else: print(f"   {i:2d} KERNEL {c['name']:10s} on {c['device']}" + ("   (uses _device_num)" if c["device_num"] else ""))
  print("   value:", value)
  return {"label": label, "code": code, "calls": calls, "value": value}

D2, D3 = ("CPU:0", "CPU:1"), ("CPU:0", "CPU:1", "CPU:2")
t = Tensor([[float(4*i + j) for j in range(4)] for i in range(4)]).realize()
out = []
s = t.shard(D2, axis=0)
out.append(capture("shard", "s = t.shard(('CPU:0','CPU:1'), axis=0)", s))
out.append(capture("elementwise", "s * 2 + 1", s * 2 + 1))
out.append(capture("needs _device_num", "s + Tensor.arange(4).reshape(4,1)", s + Tensor.arange(4).float().reshape(4, 1)))
out.append(capture("reduce other axis", "s.sum(axis=1)", s.sum(axis=1)))
out.append(capture("reduce sharded axis (naive allreduce)", "s.sum(axis=0)", s.sum(axis=0)))
w = Tensor([[1., 0], [0, 1], [1, 1], [2, 0]]).realize()
out.append(capture("data parallel matmul", "s @ w.shard(devs)   # x split on rows, w replicated", s @ w.shard(D2)))
xc = t.shard(D2, axis=1)
out.append(capture("model parallel matmul", "t.shard(devs,axis=1) @ w.shard(devs,axis=0)   # split the contraction dim", xc @ w.shard(D2, axis=0)))
out.append(capture("gather", "s.to('CPU')", s.to("CPU")))
t3 = Tensor([[float(i*10 + j) for j in range(6)] for i in range(3)]).realize()
s3 = t3.shard(D3, axis=0).realize()
with Context(RING=2):   # force the ring algorithm (by default ring needs >2 devices and >256k elements)
  out.append(capture("ring allreduce, 3 devices", "with Context(RING=2): t3.shard(3 devs, axis=0).sum(axis=0)", s3.sum(axis=0)))

path = pathlib.Path(__file__).resolve().parent.parent / "data" / "ch19_schedules.json"
path.write_text(json.dumps(out))
print("\nwrote", path.name, f"({path.stat().st_size} bytes)")
