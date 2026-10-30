#!/usr/bin/env python3
"""Chapter 22: (re)build the precomputed traces used by the end-to-end explorer when the live API is unavailable.

Run:  python3 examples/22_build_presets.py          (DEV=CPU is forced by tools/trace.py)

Writes data/ch22_presets.json (index: id, title, blurb, code, file) and one data/ch22_preset_<id>.json trace per preset.
Every trace is produced by tools/trace.py exactly like a live trace from the page's "Trace" button.
"""
import os, sys, json, importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("tg_trace", os.path.join(ROOT, "tools", "trace.py"))
tg_trace = importlib.util.module_from_spec(spec); spec.loader.exec_module(tg_trace)

PRESETS = [
  ("elementwise", "Elementwise", "a*b+1 then relu: one kernel, no loop survives (the 8 elements are upcast).", """\
# elementwise: one kernel. inputs are realized first, so the trace starts from real BUFFERs
a = Tensor([[1., 2, 3, 4], [5, 6, 7, 8]]).realize()
b = Tensor([[0.5, -1, 2, 0], [1, 1, -2, 3]]).realize()
out = (a * b + 1).relu()
"""),
  ("sum", "Sum (reduce)", "sum over an axis: a REDUCE range becomes an accumulator loop in C.", """\
# reduce along an axis: a RANGE of type REDUCE becomes an accumulator loop
x = Tensor([[1., 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
            [1, 1, 1, 1, 2, 2, 2, 2]]).realize()
out = x.sum(axis=1)
"""),
  ("matmul", "Matmul", "a @ b = reshape + expand + mul + sum, fused into a single kernel.", """\
# matmul = reshape + expand + mul + sum, fused into ONE kernel
a = Tensor([[1., 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]).realize()
b = Tensor([[1., 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0]]).realize()
out = a @ b
"""),
  ("softmax", "Softmax", "max-reduce, exp-sum-reduce, divide: watch the graph get cut into 3 kernels.", """\
# softmax: a max-reduce, an exp-sum-reduce and a divide. how many kernels?
x = Tensor([[1., 2, 3, 4, 5, 6, 7, 8],
            [0, 0, 0, 0, 1, 1, 1, 1]]).realize()
out = x.softmax(axis=-1)
"""),
  ("conv2d", "Conv2d", "conv2d = pad + pool (movement) + mul + sum; the windowing becomes index math.", """\
# conv2d: pool (movement ops) + mul + sum. 1 image, 1 in-channel, 2 filters of 3x3
x = Tensor([[[[float(i * 6 + j) for j in range(6)] for i in range(6)]]]).realize()  # (1,1,6,6)
w = Tensor([[[[0., 1, 0], [1, -4, 1], [0, 1, 0]]],
            [[[1., 1, 1], [1, 1, 1], [1, 1, 1]]]]).realize()                         # (2,1,3,3)
out = x.conv2d(w)                                                                    # (1,2,4,4)
"""),
  ("mlp", "MLP fwd+bwd", "2-layer MLP, loss.backward(): autograd adds UOps, then forward+backward are scheduled together.", """\
# a 2-layer MLP forward + backward: autograd builds MORE UOps, all scheduled together
x  = Tensor([[1., 2, 3], [0, -1, 1]]).realize()                                     # (2,3)
w1 = Tensor([[.1, -.2, .3, .4], [.5, .6, -.7, .8], [-.9, .1, .2, .3]]).realize()     # (3,4)
w2 = Tensor([[.2], [-.3], [.4], [.1]]).realize()                                     # (4,1)
loss = (x @ w1).relu().matmul(w2).square().mean()
loss.backward()                       # fills .grad of every in-scope float tensor
out = {"loss": loss, "w1.grad": w1.grad, "w2.grad": w2.grad}
"""),
  ("padslice", "Pad / slice", "pad, shrink, flip, permute: movement ops vanish into index arithmetic and a gate.", """\
# movement ops only: pad, shrink (slice), flip, permute -> index arithmetic + a WHERE for the padding
x = Tensor([[1., 2, 3], [4, 5, 6]]).realize()
out = x.pad(((1, 1), (2, 0)))[1:4, ::-1].permute(1, 0) * 10
"""),
  ("fused", "Fused chain", "mean -> subtract -> exp2 -> sum: a reduce feeding a broadcast feeding another reduce.", """\
# a fused chain: reduce -> broadcast back -> elementwise -> reduce
x = Tensor([[1., 2, 3, 4], [2, 4, 6, 8]]).realize()
m = x.mean(axis=1, keepdim=True)
out = ((x - m) * 2).exp2().sum(axis=1)
"""),
]

if __name__ == "__main__":
  index = []
  for pid, title, blurb, code in PRESETS:
    res = tg_trace.run_trace(code, {"noopt": False, "beam": 0})
    if not res.get("ok"):
      print(f"{pid}: FAILED {res.get('error')}\n{res.get('traceback', '')}"); sys.exit(1)
    fn = f"ch22_preset_{pid}.json"
    s = json.dumps(res, separators=(",", ":"))
    with open(os.path.join(ROOT, "data", fn), "w") as f: f.write(s)
    kernels = [c.get("fn", c["name"]) for c in res["calls"]]
    print(f"{pid:12s} {len(s)/1024:6.1f} KB  {len(res['calls'])} call(s): {', '.join(kernels):40s} nodes={len(res['nodes'])}")
    index.append({"id": pid, "title": title, "blurb": blurb, "code": code, "file": fn, "kernels": [c["fn"] for c in res["calls"] if c["kind"] == "kernel"]})
  with open(os.path.join(ROOT, "data", "ch22_presets.json"), "w") as f: json.dump({"presets": index}, f, indent=1)
  print(f"wrote data/ch22_presets.json ({len(index)} presets)")
