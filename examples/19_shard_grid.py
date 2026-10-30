# Chapter 19 — capture the real compiled schedule for a grid of (num devices, shard axis, operation) on a 12x12 tensor.
# run: DEV=CPU python3 examples/19_shard_grid.py        (writes data/ch19_grid.json, used by the sharding visualizer)
import json, pathlib
from tinygrad import Tensor
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import compile_linear, run_linear, get_call_arg_uops

def devstr(d): return list(d) if isinstance(d, tuple) else d
def summarize(t):
  lin = t.schedule_linear()
  calls = []
  for c in compile_linear(lin).src:
    c = c.without_after
    a = get_call_arg_uops(c)
    if c.body.op is Ops.STORE:
      src = a[1]
      while src.op in (Ops.SHRINK, Ops.BITCAST, Ops.RESHAPE) : src = src.src[0]
      lane = src.arg if src.op is Ops.MSELECT else None
      while src.op in (Ops.MSELECT, Ops.SHRINK, Ops.BITCAST, Ops.RESHAPE): src = src.src[0]
      sdev = src.device[lane] if lane is not None and isinstance(src.device, tuple) else src.device
      calls.append({"k": "copy", "src": devstr(sdev), "dst": devstr(a[0].device), "n": a[0].max_numel() * a[0].dtype.itemsize})
    else: calls.append({"k": "kernel", "name": c.body.src[0].arg.name, "dev": devstr(a[0].device)})
  run_linear(lin)
  return calls

X = Tensor([[float(12*i + j) for j in range(12)] for i in range(12)]).realize()
W = Tensor([[float((i + 2*j) % 5) for j in range(12)] for i in range(12)]).realize()
OPS = {
  "x * 2 + 1":            lambda x, d: x * 2 + 1,
  "x.sum(axis=0)":        lambda x, d: x.sum(axis=0),
  "x.sum(axis=1)":        lambda x, d: x.sum(axis=1),
  "x.sum()":              lambda x, d: x.sum(),
  "x @ W  (W replicated)":  lambda x, d: x @ W.shard(d),
  "x @ W  (W axis=0)":    lambda x, d: x @ W.shard(d, axis=0),
  "x @ W  (W axis=1)":    lambda x, d: x @ W.shard(d, axis=1),
}
out = {}
for nd in (2, 3, 4):
  devs = tuple(f"CPU:{i}" for i in range(nd))
  for axis in (0, 1, None):
    xs = X.shard(devs, axis=axis).realize()
    for name, fn in OPS.items():
      try:
        y = fn(xs, devs)
        calls = summarize(y)
        ref = fn(X, None).tolist() if "@" not in name else (X @ W).tolist()
        ok = y.tolist() == ref
        out[f"{nd}|{axis}|{name}"] = {"calls": calls, "out_device": devstr(y.device), "out_axis": y.uop.axis if isinstance(y.device, tuple) else None,
                                     "ok": ok}
      except Exception as e:
        out[f"{nd}|{axis}|{name}"] = {"error": f"{type(e).__name__}: {e}"[:200]}
      r = out[f"{nd}|{axis}|{name}"]
      if "error" in r: print(f"{nd} devs, axis={axis!s:4} {name:22s} ERROR {r['error'][:80]}")
      else:
        k = sum(c["k"] == "kernel" for c in r["calls"]); cp = sum(c["k"] == "copy" for c in r["calls"])
        print(f"{nd} devs, axis={axis!s:4} {name:22s} {k:2d} kernels {cp:2d} copies  out axis={r['out_axis']!s:4}  correct={r['ok']}")
path = pathlib.Path(__file__).resolve().parent.parent / "data" / "ch19_grid.json"
path.write_text(json.dumps(out))
print("wrote", path.name, f"({path.stat().st_size} bytes)")
