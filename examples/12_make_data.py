"""examples/12_make_data.py — real schedules, buffer lifetimes, memory plans and schedule-cache keys for chapter 12

Run:  DEV=CPU NOOPT=1 python3 examples/12_make_data.py
Writes data/ch12_mlp.json (schedule viewer + Gantt chart) and data/ch12_cache.json (schedule cache walkthrough)
"""
import importlib.util, json, os, pathlib
from tinygrad import Tensor, Variable
from tinygrad.helpers import Context
from tinygrad.uop.ops import UOp, Ops
import tinygrad.schedule as S

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("cap", HERE / "10_sched_capture.py")
cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)
DATA = HERE.parent / "data"

def dump(obj, name):
  with open(DATA / name, "w") as f: json.dump(obj, f, separators=(",", ":"))
  print(f"wrote data/{name}: {os.path.getsize(DATA / name)/1024:.1f} KB")

# ---- capture the LINEAR right before and right after memory_plan_rewrite
mp = {}
_orig_mp = S.memory_plan_rewrite
def _hook(linear, held):
  out = _orig_mp(linear, held)
  mp.update(pre=linear, post=out, held=held)
  return out
S.memory_plan_rewrite = _hook

def stored_slots(sink:UOp) -> set[int]:
  """kernel-local PARAM slots that the kernel STOREs into = its outputs"""
  outs = set()
  for u in sink.toposort():
    if u.op is Ops.STORE:
      tgt = u.src[0]
      while tgt.op is not Ops.PARAM and len(tgt.src): tgt = tgt.src[0]
      if tgt.op is Ops.PARAM: outs.add(tgt.arg.slot)
  return outs

# ================================================================= 1. a small MLP
B, D_IN, H, D_OUT = 64, 128, 256, 10
x = Tensor.empty(B, D_IN).realize()
dims = [D_IN, H, H, H, D_OUT]       # 3 hidden layers + output layer
params = [(Tensor.empty(i, o).realize(), Tensor.empty(o).realize()) for i, o in zip(dims, dims[1:])]
names = {x.uop.base: "x (input)"}
for i, (w, b) in enumerate(params): names[w.uop.base] = f"W{i+1}"; names[b.uop.base] = f"b{i+1}"
h = x
for i, (w, b) in enumerate(params):
  h = h @ w + b
  if i < len(params) - 1: h = h.relu()
out = h.softmax(-1)
CODE = """x = Tensor.empty(64, 128)                       # batch 64
W1, b1 = Tensor.empty(128, 256), Tensor.empty(256)
W2, b2 = Tensor.empty(256, 256), Tensor.empty(256)
W3, b3 = Tensor.empty(256, 256), Tensor.empty(256)
W4, b4 = Tensor.empty(256, 10),  Tensor.empty(10)
h1 = (x @ W1 + b1).relu()
h2 = (h1 @ W2 + b2).relu()
h3 = (h2 @ W3 + b3).relu()
out = (h3 @ W4 + b4).softmax(-1)
out.realize()"""

with Context(SCACHE=0), cap.hooked() as log:
  linear, var_vals = Tensor.linear_with_vars(out)
names[out.uop.base] = "out (softmax)"
compiled = cap.compile_linear(linear)
srcs = cap.kernel_sources(compiled)
pre, post, held = mp["pre"], mp["post"], mp["held"]

def buf_name(b:UOp) -> str:
  for k, v in names.items():
    if k is b or (k.op is Ops.BUFFER and k.arg.slot == b.arg.slot): return v
  return TMP_NAMES.get(b.arg.slot, f"tmp buf {b.arg.slot}")

# intermediate buffers are named after the kernel that writes them (this order is checked below)
TMP_LABELS = ["h1", "h2", "h3", "logits", "row max", "row sum(exp)"]
TMP_NAMES: dict = {}
for i, c in enumerate(pre.src):
  for slot in stored_slots(c.body):
    b = c.src[1 + slot]
    if b not in held and i < len(TMP_LABELS): TMP_NAMES[b.arg.slot] = TMP_LABELS[i]
bufs: dict = {}
calls = []
writer: dict = {}
for i, (c, cpost) in enumerate(zip(pre.src, post.src)):
  outs = stored_slots(c.body)
  args = []
  for slot, (b, bp) in enumerate(zip(c.src[1:], cpost.src[1:])):
    key = f"b{b.arg.slot}"
    planned = bp.op is Ops.BITCAST
    off = int(bp.src[0].src[1].arg) if planned else None
    if key not in bufs:
      bufs[key] = {"key": key, "name": buf_name(b), "nbytes": int(b.max_numel() * b.dtype.itemsize), "numel": int(b.max_numel()),
                   "dtype": str(b.dtype).replace("dtypes.", ""), "held": b in held, "first": i, "last": i, "offset": off}
    bufs[key]["last"] = i
    args.append({"slot": slot, "buf": key, "out": slot in outs, "offset": off})
  deps = sorted({writer[a["buf"]] for a in args if not a["out"] and a["buf"] in writer})
  for a in args:
    if a["out"]: writer[a["buf"]] = i
  ast = c.body
  calls.append({"i": i, "name": srcs[i][0], "src": srcs[i][1], "args": args, "deps": deps,
                "ast": cap.graph_json(ast, {}), "ast_text": str(ast)[:6000]})
arena = [u for u in post.toposort() if u.op is Ops.BUFFER and u not in set(pre.toposort())]
plan = {"arena_bytes": int(arena[0].max_numel()) if arena else 0,
        "unplanned_bytes": sum(v["nbytes"] for v in bufs.values() if not v["held"])}
linear_text = str(post)
dump({"code": CODE, "calls": calls, "bufs": list(bufs.values()), "plan": plan,
      "passes": [p[0] for p in log["passes"]]}, "ch12_mlp.json")
print(f"MLP: {len(calls)} kernels, intermediates {plan['unplanned_bytes']} B -> arena {plan['arena_bytes']} B")

# ================================================================= 2. the schedule cache
def f(x): return x / x.sum(1, keepdim=True)
FCODE = "def f(x): return x / x.sum(1, keepdim=True)"
variants = []
def capture_variant(label, desc, make):
  before = set(S.schedule_cache)
  with cap.hooked() as lg:
    t = make()
    lin, vv = Tensor.linear_with_vars(t)
  rb = cap.pass_by_name(lg, "replace bufs")
  key = [k for k in S.schedule_cache if k not in before]
  # the key is the hash of the CALL body built from the output of "replace bufs" (see lower_sink_to_linear)
  body = rb[2]
  k = body.key.hex()
  hit = not key
  variants.append({"label": label, "desc": desc, "key": k[:8], "hit_in_tinygrad": hit, "var_vals": vv,
                   "before": cap.graph_json(rb[1], {}), "after": cap.graph_json(rb[2], {}), "n_kernels": len(lin.src),
                   "linear_text": str(S.schedule_cache[body.key])[:5000] if body.key in S.schedule_cache else ""})
  print(f"cache variant {label:28s} key {k[:8]}  {'hit' if hit else 'MISS'}  var_vals={vv}")

S.schedule_cache.clear()
capture_variant("call 1: x = empty(4, 8)", "first call: nothing cached yet", lambda: f(Tensor.empty(4, 8)))
capture_variant("call 2: new data, same shape", "different BUFFER, same shape/dtype/ops", lambda: f(Tensor.empty(4, 8)))
capture_variant("call 3: shape (5, 8)", "a different shape is a different graph", lambda: f(Tensor.empty(5, 8)))
capture_variant("call 4: dtype half", "a different dtype is a different graph", lambda: f(Tensor.empty(4, 8, dtype="half")))
xs = Tensor.empty(16, 8).realize()
v = Variable("bs", 1, 16)
capture_variant("call 5: x[:bs], bs=3", "symbolic shape: the bound value is stripped from the key", lambda: f(xs[:v.bind(3)]))
capture_variant("call 6: x[:bs], bs=11", "same Variable, different value", lambda: f(xs[:v.bind(11)]))
dump({"code": FCODE, "variants": variants}, "ch12_cache.json")
