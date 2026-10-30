# Chapter 6 — symbolic algebra & range analysis.  Run with:  DEV=CPU python3 examples/06_symbolic.py
# Builds index-style integer expressions out of UOps and runs them through tinygrad's `symbolic` PatternMatcher.
import json, pathlib, itertools
from tinygrad import Tensor, Device, Context
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, PatternMatcher, graph_rewrite, sym_infer
from tinygrad.uop.symbolic import symbolic, sym, pm_simplify_valid, pm_drop_and_clauses
from tinygrad.schedule.indexing import apply_movement_op
from tinygrad.codegen import to_program

def section(t): print(f"\n=== {t} ===")
R = lambda u: u.render(simplify=False)
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
x, y = UOp.variable("x", 0, 100), UOp.variable("y", 0, 3)

section("1. range analysis: every integer UOp knows an interval [vmin, vmax]")
for e in [x, y, x + y, x * 4 + y, (x * 4 + y) // 4, (x * 4 + y) % 4, x - y, y * -3, x.maximum(50), (y < 4), (x < 50)]:
  print(f"  {R(e):16s} in [{e.vmin}, {e.vmax}]")

section("2. symbolic: before -> after (graph_rewrite(expr, symbolic))")
cases = {
  "constant folding":        UOp.const(3) * 4 + 2,
  "x*1, x+0":                x * 1 + 0,
  "combine terms":           x * 2 + x * 3,
  "move consts to the end":  (x + 3) + y + 5,
  "x//c*c + x%c -> x":       (x // 4) * 4 + x % 4,
  "exact division":          (x * 4 + y) // 4,
  "mod of a congruent sum":  (x * 4 + y) % 4,
  "nested div":              (x // 3) // 4,
  "div of a mod":            (x % 12) // 4,
  "lt folding":              (x * 4 + y) < 8,
  "range says it's const":   y // 4,
  "comparison always true":  y < 4,
  "max with known order":    y.maximum(5),
  "range mod its bound":     UOp.range(16, 0) % 16,
  "x - x (interval is loose)": x - x,
  "divide by gcd (w in [1,10])": (x * 4) // (UOp.variable("w", 1, 10) * 2),
}
captured = []
for name, e in cases.items():
  out = graph_rewrite(e, symbolic)
  captured.append({"name": name, "before": R(e), "after": R(out), "vmin": str(e.vmin), "vmax": str(e.vmax)})
  print(f"  {name:28s} {R(e):22s} -> {R(out):12s} interval before: [{e.vmin}, {e.vmax}]")

section("3. where div/mod come from: a reshape turns one index into div/mod expressions")
# this mirrors _apply_reshape in tinygrad/schedule/indexing.py: flatten the OUTPUT indices with the output strides,
# then peel INPUT indices off with % and // ... and then simplify
import sys, linecache, re
from tinygrad.uop.ops import UPat
from tinygrad.uop import divandmod
def which_branch(d: UOp) -> int:
  # re-run fold_divmod_general (uncached) under a tracer and report the line of its OUTERMOST return
  code, depth, line = divandmod.fold_divmod_general.__wrapped__.__code__, [0], [0]
  def tracer(frame, event, arg):
    if frame.f_code is code:
      if event == "call": depth[0] += 1
      elif event == "return":
        if depth[0] == 1: line[0] = frame.f_lineno
        depth[0] -= 1
    return tracer
  sys.settrace(tracer)
  try: divandmod.fold_divmod_general.__wrapped__(d)
  finally: sys.settrace(None)
  # name = the nearest "# rule_name ..." comment above that return
  for ln in range(line[0], 0, -1):
    if m := re.match(r"# (x//y is constant|exact division|[a-z]+_[a-z_]+)", linecache.getline(divandmod.__file__, ln).strip()): return line[0], m[1]
  return line[0], "?"
def rule_name(p: UPat, fn) -> str:
  # a named function, else the rule's trailing comment, else the nearest comment above it
  if fn.__name__ != "<lambda>": return fn.__name__
  if "#" in (t := linecache.getline(p.location[0], p.location[1])): return t.split("#", 1)[1].strip()
  above, ln = [], p.location[1] - 1   # the block of comment lines directly above the rule
  while (t := linecache.getline(p.location[0], ln).strip()).startswith("#"): above.insert(0, t); ln -= 1
  # prefer a plain comment (not a NOTE/TODO, commented-out code or a "** section **" header), else a section header
  plain = [t.strip("# ") for t in above if not t.startswith("# **") and not any(w in t for w in ("NOTE", "TODO", "UPat"))]
  if above := plain or [t.strip("# *") for t in above if t.startswith("# **")]: return above[0][:80]
  return linecache.getline(p.location[0], p.location[1]).strip()[:80]
def gdump(u: UOp) -> dict:
  ids = {x: f"n{i}" for i, x in enumerate(u.toposort())}
  lab = lambda x: R(x) if x.op in (Ops.PARAM, Ops.RANGE, Ops.CONST) else None   # leaves show their rendered name/value
  return {"nodes": [[ids[x], x.op.name, lab(x)] for x in ids if x.op is not Ops.STACK and not (x.op is Ops.CONST and x.src)],
          "edges": [[ids[s], ids[x], i] for x in ids for i, s in enumerate(x.src) if s.op is not Ops.STACK and x.op is not Ops.RANGE]}
class Logged(PatternMatcher):
  def __init__(self, patterns, log): super().__init__(patterns); self.log, self.fns = log, {id(p): fn for p, fn in patterns}
  def rewrite(self, uop, ctx=None):
    for p, match, early_reject in self.pdict.get(uop.op, []):
      if (ret := match(uop, ctx)) is not None and ret is not uop:
        f, ln = p.location[0].split("/")[-1], p.location[1]
        e = {"before": R(uop), "after": R(ret), "rule": f"{f}:{ln}", "g0": gdump(uop), "g1": gdump(ret)}
        if f == "divandmod.py" and ln == 103: e["branch_line"], e["branch"] = which_branch(uop)
        else: e["branch"] = rule_name(p, self.fns[id(p)])
        self.log.append(e)
        return ret
    return None
def reshape_index(in_shape, out_shape):
  rngs = [UOp.range(s, i) for i, s in enumerate(out_shape)]
  acc, axes_in = 1, []
  for s, r in list(zip(out_shape, rngs))[::-1]: axes_in.append(r * acc); acc *= s
  combined = UOp.const(0).usum(axes_in)
  axes_out = []
  for s in in_shape[::-1]: axes_out.append(combined % s); combined = combined // s
  raw = UOp.sink(*axes_out[::-1])
  log: list = []
  out = graph_rewrite(raw, Logged((symbolic + pm_simplify_valid + pm_drop_and_clauses).patterns, log))
  return raw, out, log
steppers = []
for in_shape, out_shape in [((12,), (3, 4)), ((3, 4), (12,)), ((3, 4), (2, 6)), ((2, 3, 4), (6, 4)), ((4, 6), (2, 2, 6))]:
  raw, out, log = reshape_index(in_shape, out_shape)
  print(f"  reshape {in_shape} -> {out_shape}: index into the input")
  print(f"     raw:        {', '.join(R(s) for s in raw.src)}")
  print(f"     simplified: {', '.join(R(s) for s in out.src)}   ({len(log)} rule applications)")
  steppers.append({"in": in_shape, "out": out_shape, "raw": [R(s) for s in raw.src], "simplified": [R(s) for s in out.src], "log": log,
                   "nodes_raw": len(raw.toposort()), "nodes_simplified": len(out.toposort())})
print(f"  the {len(steppers[2]['log'])} rule applications for (3,4) -> (2,6), in order:")
for st in steppers[2]["log"]:
  print(f"     {st['rule']:16s} {st['before']:30s} -> {st['after']:36s} [{st['branch']}]")

section("4. Variables, bind, and symbolic shapes")
n = UOp.variable("n", 1, 10)
print("n:", R(n), " range:", (n.vmin, n.vmax), " is_variable:", n.is_variable)
bn = n.bind(3)
print("bound:", bn.arg.val, " unbind ->", (R(bn.unbind()[0]), bn.unbind()[1]))
t = Tensor.ones(10).contiguous().realize()[:bn]
print("Tensor shape with a bound Variable:", tuple(R(d) if isinstance(d, UOp) else d for d in t.shape), " (the dim is the PARAM itself, val=3)")
print("(t*2).sum() with n=3:", (t * 2).sum().item())
expr = (n * 4 + 2) // 2
print("sym_infer(", R(expr), ", n=3 ) =", sym_infer(expr, {"n": 3}), "   simplified:", R(expr.simplify()))

section("5. interval check against brute force (vmin/vmax are sound bounds)")
for e in [(x * 4 + y) // 4, (x + y) % 7, (y * 3 - x) // 5, x.maximum(y * 40)]:
  vals = [e.substitute({x: UOp.const(a), y: UOp.const(b)}).simplify().arg for a in range(0, 101, 5) for b in range(4)]
  print(f"  {R(e):20s} vmin/vmax=[{e.vmin}, {e.vmax}]  sampled=[{min(vals)}, {max(vals)}]")

section("6. Invalid: gated indices from PAD, simplified under their gate")
r = UOp.range(6, 0)
idx = apply_movement_op(Ops.PAD, (4,), ((1, 1),), (r,))[0]   # pad a length-4 axis by 1 on each side
print("PAD (4,)->(6,) index:", R(idx))
print("   get_valid():", R(idx.get_valid()), "   get_idx():", R(idx.get_idx()))
r8 = UOp.range(8, 0)
for e in [(r8 < 4).where(r8 // 4 + r8 % 4 * 2, UOp.invalid()), ((r8 < 4) & (r8 >= 1)).where((r8 + 3) // 4, UOp.invalid()),
          (r8 < 4).where(r8, UOp.invalid()) + 1, (r8 < 4).where(r8, UOp.invalid()) * 0]:
  print(f"  {R(e):48s} -> {R(graph_rewrite(e, sym))}")

section("7. why it matters: the index math that reaches the kernel")
t = Tensor([float(i) for i in range(24)]).reshape(4, 6).realize()
mk = lambda: t.reshape(2, 2, 6).permute(1, 0, 2).reshape(4, 6).contiguous()   # fresh graph each time
ast = mk().schedule_linear().src[-1].src[0]
with Context(NOOPT=1): prg = to_program(ast, Device["CPU"].renderer)
print("\n".join(l for l in prg.src[2].arg.split("\n") if "for" in l or "val0" in l))
print("result row 1:", mk().tolist()[1])
v = Tensor([1., 2, 3, 4]).realize()
with Context(NOOPT=1): prg = to_program(v.pad(((1, 1),)).contiguous().schedule_linear().src[-1].src[0], Device["CPU"].renderer)
print("\n".join(l for l in prg.src[2].arg.split("\n") if "val0" in l))
print("pad result:", v.pad(((1, 1),)).contiguous().tolist())
for st in steppers: print(f"  reshape {str(st['in']):10s}->{str(st['out']):10s} index graph: {st['nodes_raw']:3d} UOps raw -> {st['nodes_simplified']:3d} simplified")

# data for the interactive demos
interval_checks = []
a, b = UOp.variable("a", 0, 10), UOp.variable("b", 0, 10)
templates = {"a+b": a + b, "a-b": a - b, "a*b": a * b, "a//b": a // b, "a%b": a % b, "max(a,b)": a.maximum(b), "a<b": a < b,
             "(a*4+b)//4": (a * 4 + b) // 4, "(a*4+b)%4": (a * 4 + b) % 4, "a*-3+b": a * -3 + b, "(a<b).where(a,b)": (a < b).where(a, b)}
for (alo, ahi), (blo, bhi) in itertools.product([(0, 10), (-5, 5), (3, 7), (-8, -2)], [(0, 10), (1, 4), (-3, 3), (2, 2), (-6, -1)]):
  aa, bb = UOp.variable("a", alo, ahi), UOp.variable("b", blo, bhi)
  for name, e in templates.items():
    ee = e.substitute({a: aa, b: bb})
    lim = lambda v: v if isinstance(v, bool) or abs(v) < 2**62 else ("-inf" if v < 0 else "inf")   # weakint dtype bounds are huge
    interval_checks.append({"expr": name, "a": [alo, ahi], "b": [blo, bhi], "vmin": lim(ee.vmin), "vmax": lim(ee.vmax)})
# extra real before/after pairs for the "simplify this" game (not printed)
z = UOp.variable("z", 0, 7)
game_exprs = [(x * 4 + z) // 8, (x * 8 + z) % 8, (z + 8) // 8, (z * 2 + 1) % 2, (x // 2) // 2 // 2, x * 3 + x, (x + 1) * 2 - 2,
              (z < 8), (z < 3), ((x * 2) < 7), (x // 4) * 4 + x % 4 + 1, (x % 16) // 4, (x % 16) % 4, z.maximum(-1), (x * 6 + z) // 3,
              (z * 4 + 3) // 4, UOp.range(8, 0) // 8, (x + 5) % 5, ((x * 4) + 2) // 2]
game = []
for e in game_exprs:
  log = []
  out = graph_rewrite(e, Logged(symbolic.patterns, log))
  assert out is graph_rewrite(e, symbolic)   # the logging matcher gives exactly the same answer
  game.append({"before": R(e), "after": R(out), "vmin": str(e.vmin), "vmax": str(e.vmax),
               "rules": [{k: v for k, v in st.items() if k not in ("g0", "g1")} for st in log]})
json.dump({"cases": captured, "game": game, "steppers": steppers, "intervals": interval_checks}, open(DATA / "ch06_symbolic.json", "w"), indent=1, default=str)
