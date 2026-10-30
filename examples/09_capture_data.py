# Chapter 9 — capture REAL data for the autograd demos (writes data/ch09_*.json).
# Run:  DEV=CPU python3 examples/09_capture_data.py
#
#  ch09_walk.json  : for a few small graphs, every step of compute_gradient's reverse walk. We re-run the exact loop of
#                    tinygrad/mixin/gradient.py:compute_gradient (same _deepwalk, same pm_gradient.rewrite, same
#                    broadcast-sum and accumulation), recording the forward value and the gradient at every node.
#  ch09_rules.json : every pm_gradient rule (its source line) + a worked numeric example checked by finite differences.
import os, sys, math, inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from uop2json import dump
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, GroupOp, broadcast_axes
from tinygrad.dtype import sum_acc_dtype
import tinygrad.mixin.gradient as gradmod
from tinygrad.mixin.gradient import pm_gradient, _deepwalk

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
def val(u: UOp):
  try: return Tensor(u).tolist()
  except Exception as e: return f"({type(e).__name__})"
def rnd(v, n=4):
  if isinstance(v, list): return [rnd(x, n) for x in v]
  return round(v, n) if isinstance(v, float) else v

def label(u: UOp, names) -> tuple[str, str]:
  if u.op in GroupOp.Movement: return u.op.name, str(u.marg)
  if u.op is Ops.REDUCE: return u.op.name, f"{u.arg[0].name}, {u.arg[1]} axes"
  if u.op is Ops.CONST: return "CONST", str(u.arg).replace("ConstFloat(", "").rstrip(")")
  if u.op in GroupOp.Defines: return f"BUFFER {names.get(u, '')}", ""
  return u.op.name, ""

def data_srcs(u: UOp):
  return u.src[:1] if u.op in GroupOp.Movement else (() if u.op in GroupOp.Defines else u.src)

def capture_walk(title, loss: Tensor, targets: dict[str, Tensor]):
  names = {t.uop.base: n for n, t in targets.items()}
  root, root_grad = loss.uop, loss.const_like(1.0).uop
  tset = {t.uop for t in targets.values()}
  # forward graph with shape args folded
  keep, stack = [], [root]
  seen = set()
  while stack:
    u = stack.pop()
    if u in seen: continue
    seen.add(u); stack.extend(data_srcs(u))
  topo = [u for u in root.toposort() if u in seen]
  ids = {u: f"n{i}" for i, u in enumerate(topo)}
  nodes = []
  for u in topo:
    op, arg = label(u, names)
    nodes.append({"id": ids[u], "op": op.split()[0], "label": op, "arg": arg, "shape": str(u._shape), "value": rnd(val(u))})
  edges = [[ids[s], ids[u], i] for u in topo for i, s in enumerate(data_srcs(u))]
  # ---- the reverse walk: a faithful copy of compute_gradient, instrumented ----
  walk, in_path = _deepwalk(root, tset)
  grads = {root: root_grad}
  steps = [{"node": ids[root], "kind": "seed", "grad": rnd(val(root_grad)),
            "text": "seed: dL/dL = 1 (Tensor.gradient uses const_like(1.0) when no gradient is passed)"}]
  for t0 in reversed(walk):
    if t0 not in grads:
      if t0 in ids: steps.append({"node": ids[t0], "kind": "skip", "op": t0.op.name})
      continue
    lgrads = pm_gradient.rewrite(t0, ctx=grads[t0])
    contrib = []
    for i, (k, v) in enumerate(zip(t0.src, lgrads)):
      if v is None: continue
      summed = False
      if k._shape is not None and v._shape is not None and k._shape != v._shape:
        v = v.cast(sum_acc_dtype(v.dtype))._rop(Ops.ADD, broadcast_axes(k.shape, v.shape)).reshape(k.shape).cast(v.dtype); summed = True
      acc = k in grads
      grads[k] = grads[k] + v if acc else v
      if k in ids:
        contrib.append({"src": ids[k], "index": i, "value": rnd(val(v)), "accumulated": acc, "broadcast_summed": summed,
                        "total": rnd(val(grads[k]))})
    steps.append({"node": ids[t0], "kind": "rule", "op": t0.op.name, "grad": rnd(val(grads[t0])), "contrib": contrib})
  result = {n: rnd(val(grads[t.uop])) for n, t in targets.items()}
  # cross-check with the public API
  api = loss.gradient(*targets.values())
  for (n, _), g in zip(targets.items(), api): assert rnd(g.tolist()) == result[n], (n, g.tolist(), result[n])
  return {"title": title, "graph": {"nodes": nodes, "edges": edges}, "steps": steps, "result": result,
          "targets": {n: ids[t.uop] for n, t in targets.items() if t.uop in ids}}

walks = {}
x = Tensor([1., 2., 3.]).realize()
walks["square"] = capture_walk("L = (x * x).sum()", (x * x).sum(), {"x": x})
x = Tensor([1., -2., 3.]).realize(); w = Tensor([0.5, 0.5, -1.]).realize(); b = Tensor([0.25]).realize()
walks["neuron"] = capture_walk("L = (x * w + b).relu().sum()", (x * w + b).relu().sum(), {"x": x, "w": w, "b": b})
a = Tensor([[1., 2], [3, 4]]).realize(); c = Tensor([[1., -1], [0, 2]]).realize()
walks["matmul"] = capture_walk("L = (a @ c).sum()", (a @ c).sum(), {"a": a, "c": c})
z = Tensor([0.5, -1.0]).realize()
walks["sigmoid"] = capture_walk("L = z.sigmoid().sum()", z.sigmoid().sum(), {"z": z})
dump(walks, os.path.join(DATA, "ch09_walk.json"))
for k, v in walks.items():
  print(f"walk {k:8s} {len(v['graph']['nodes']):3d} nodes  steps: " + " ".join(s.get("op", s["kind"]) + ("(skip)" if s["kind"] == "skip" else "") for s in v["steps"]))
  print(f"              grads={v['result']}")

# ---- rule table: source line of each pm_gradient rule + a worked example ----
src_lines, start = inspect.getsourcelines(gradmod)
lines = {i + 1: l.rstrip() for i, l in enumerate(src_lines)}
def rule_src(op_name):
  out = []
  for n, l in lines.items():
    if f"UPat(Ops.{op_name}," in l or f"UPat(Ops.{op_name})" in l or (op_name in ("CMPLT", "CMPEQ") and "UPat((Ops.CMPLT, Ops.CMPEQ))" in l):
      blk = [l]; m = n + 1
      while m in lines and lines[m].startswith("    "): blk.append(lines[m]); m += 1
      out.append({"line": n, "code": "\n".join(blk)})
  return out

def worked(fn, inputs: list[list[float]], upstream=None):
  # apply fn to inputs, gradient wrt every input with upstream (default ones); finite-difference check
  ts = [Tensor(v).realize() for v in inputs]
  out = fn(*ts)
  up = Tensor.ones(*out.shape).contiguous().realize() if upstream is None else Tensor(upstream).realize()
  gs = out.gradient(*ts, gradient=up)
  res = {"inputs": inputs, "output": rnd(out.tolist()), "upstream": rnd(up.tolist()), "grads": [rnd(g.tolist()) for g in gs]}
  fd = []
  eps = 1e-3
  for i, v in enumerate(inputs):
    row = []
    for j in range(len(v)):
      pl = [list(q) for q in inputs]; mi = [list(q) for q in inputs]
      pl[i][j] += eps; mi[i][j] -= eps
      fp = (fn(*[Tensor(q) for q in pl]) * up).sum().item(); fm = (fn(*[Tensor(q) for q in mi]) * up).sum().item()
      row.append(round((fp - fm) / (2 * eps), 3))
    fd.append(row)
  res["finite_diff"] = fd
  return res

R = [
  ("ADD", "both inputs get the upstream gradient unchanged", "\\frac{\\partial (a+b)}{\\partial a} = 1", lambda a, b: a.alu(Ops.ADD, b), [[1., 2, 3], [4., 5, 6]]),
  ("MUL", "each input gets upstream × the OTHER input", "\\frac{\\partial (ab)}{\\partial a} = b", lambda a, b: a.alu(Ops.MUL, b), [[1., 2, 3], [4., 5, 6]]),
  ("MAX", "the larger input gets the gradient; ties split it 50/50", "\\partial \\max(a,b) = [a>b] + \\tfrac12[a=b]", lambda a, b: a.alu(Ops.MAX, b), [[1., 5, 3], [4., 2, 3]]),
  ("WHERE", "the condition gets nothing; each branch gets the gradient where it was selected", "\\text{where}(c,a,b)", lambda a, b: (a > 2).where(a, b), [[1., 3, 4], [10., 20, 30]]),
  ("RECIP", "-upstream · ret²  (ret = 1/x, so -1/x²)", "\\frac{d}{dx}\\frac1x = -\\frac{1}{x^2}", lambda a: a.alu(Ops.RECIP), [[0.5, 1., 2.]]),
  ("EXP2", "ret · ln2 · upstream", "\\frac{d}{dx}2^x = 2^x\\ln 2", lambda a: a.alu(Ops.EXP2), [[0., 1., 2.]]),
  ("LOG2", "upstream / (x · ln2)", "\\frac{d}{dx}\\log_2 x = \\frac{1}{x\\ln 2}", lambda a: a.alu(Ops.LOG2), [[0.5, 1., 4.]]),
  ("SQRT", "upstream / (2 · ret)", "\\frac{d}{dx}\\sqrt x = \\frac{1}{2\\sqrt x}", lambda a: a.alu(Ops.SQRT), [[1., 4., 9.]]),
  ("SIN", "sin(π/2 − x) · upstream  (= cos x, written with SIN so no COS op is needed)", "\\frac{d}{dx}\\sin x = \\cos x = \\sin(\\tfrac\\pi2 - x)", lambda a: a.alu(Ops.SIN), [[0., 1., 3.]]),
  ("POW", "e·b^(e−1) for the base, ret·ln b for the exponent, with special cases at 0", "\\partial_b b^e = e b^{e-1},\\ \\partial_e b^e = b^e \\ln b", lambda a, b: a.alu(Ops.POW, b), [[1., 2, 3], [2., 2, 0.5]]),
  ("CAST", "cast the gradient back to the input dtype", "", lambda a: a.cast(dtypes.float64).cast(dtypes.float), [[1.5, 2.5, 3.5]]),
  ("TRUNC", "zero: trunc is piecewise constant", "", lambda a: a.alu(Ops.TRUNC), [[1.5, 2.5, -3.5]]),
  ("CMPLT", "no gradient (None) to either side: comparisons are boolean", "", lambda a, b: (a < b).where(a, 0.0), [[1., 5, 3], [4., 2, 6]]),
  ("REDUCE", "ADD: broadcast upstream back to the input shape (reduce ↔ expand). MAX: mask of the argmax positions / tie count. MUL: product of the others", "\\frac{\\partial}{\\partial x_i}\\sum_j x_j = 1", lambda a: a._rop(Ops.ADD, (0,)).reshape(1), [[1., 2, 3]]),
  ("REDUCE", "MAX case: the gradient goes to the argmax positions, split evenly between ties", "", lambda a: a._rop(Ops.MAX, (0,)).reshape(1), [[1., 3, 3]]),
  ("RESHAPE", "reshape the gradient back to the input shape", "", lambda a: a.reshape(2, 2), [[1., 2, 3, 4]]),
  ("PERMUTE", "permute back with the inverse permutation argsort(arg)", "", lambda a: a.reshape(2, 2).permute(1, 0).reshape(4), [[1., 2, 3, 4]]),
  ("EXPAND", "pass the gradient through; compute_gradient then SUMS over the broadcast axes", "", lambda a: a._mop(Ops.EXPAND, (2,)).reshape(6), [[1., 2, 3]]),
  ("PAD", "shrink the gradient back to the un-padded window", "", lambda a: a.pad(((1, 2),)), [[1., 2, 3]]),
  ("SHRINK", "pad the gradient with zeros back to the input shape", "", lambda a: a[1:3], [[1., 2, 3, 4]]),
]
rules = []
for op, desc, tex, fn, inputs in R:
  n_in = len(inputs)
  up = None
  if op in ("PAD", "SHRINK", "RESHAPE", "PERMUTE", "EXPAND"):
    probe = fn(*[Tensor(v) for v in inputs])
    up = [[float(i + 1) for i in range(probe.shape[1])] for _ in range(1)][0] if probe.ndim == 2 else [float(i + 1) for i in range(probe.numel())]
    if probe.ndim == 2: up = [[float(probe.shape[1] * r + c + 1) for c in range(probe.shape[1])] for r in range(probe.shape[0])]
  ex = worked(fn, inputs, up)
  title = op if not any(r["op"] == op for r in rules) else op + " (MAX)"
  rules.append({"op": op, "title": title, "desc": desc, "tex": tex, "src": rule_src(op), **ex})
  print(f"rule {op:8s} lines {[s['line'] for s in rules[-1]['src']]}  grads {ex['grads']}  fd {ex['finite_diff']}")
# rules without a numeric example, but shown in the table
for op, desc in [("STACK", "each stacked source gets its slice ctx[i] of the gradient"),
                 ("CONTIGUOUS_BACKWARD", "identity in forward; makes the GRADIENT contiguous (a STAGE) in backward"),
                 ("STAGE", "contiguous() is a no-op for gradients: pass through"),
                 ("COPY", "copy the gradient back to the source device"),
                 ("UNSHARD", "re-shard the gradient on the same axis (multi-device, Chapter 19)"),
                 ("SINK", "a SINK of several outputs hands each its own gradient"),
                 ("STORE", "assign: the stored value gets the gradient"), ("AFTER", "ordering / assign plumbing: route the gradient to the value or the call"),
                 ("BITCAST", "no gradient through a bitcast")]:
  rules.append({"op": op, "title": op, "desc": desc, "tex": "", "src": rule_src(op)})
rules.append({"op": "reduce_gradient", "title": "reduce_gradient()", "desc": "the REDUCE rule's helper: ADD / MAX / MUL cases", "tex": "",
              "src": [{"line": 8, "code": "\n".join(lines[i] for i in range(8, 19))}]})
dump({"rules": rules, "file": "tinygrad/mixin/gradient.py"}, os.path.join(DATA, "ch09_rules.json"))
print("wrote data/ch09_walk.json, data/ch09_rules.json")
