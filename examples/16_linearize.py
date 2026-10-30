# Chapter 16 — from a DAG with RANGE/END to a list, then PROGRAM: SINK -> LINEAR -> SOURCE -> BINARY.
#
# Kernel: (x.sum(1) + x.max(1)) for x of shape (4,8), no optimizations (opts_to_apply=()), rendered by Clang.
# It has one outer loop and two *sibling* inner loops, which is exactly the case the linearizer has to get right.
#
# run:   DEV=CPU python3 examples/16_linearize.py           (prints everything)
#        DEV=CPU python3 examples/16_linearize.py --json    (also writes data/ch16_linear.json for the chapter's demos)
import sys, os, re, json
from dataclasses import replace
from tinygrad import Tensor, Device
from tinygrad.helpers import prod
from tinygrad.uop.ops import UOp, Ops, ProgramInfo
from tinygrad.uop.render import render_uir
import tinygrad.codegen as cg
from tinygrad.codegen.late.linearizer import linearize, CFGContext
from tinygrad.codegen import line_rewrite, pm_linearize_cleanups, pm_alloc_to_buf, pm_to_program

# capture the sink right before the "add control flow" stage so we can look at CFGContext ourselves
seen: dict = {}
_gr = cg.graph_rewrite
def gr(sink, pm, *a, name=None, **kw):
  if name == "add control flow": seen["before_cf"] = sink
  return _gr(sink, pm, *a, name=name, **kw)
cg.graph_rewrite = gr

x = Tensor.empty(4, 8)
ast = (x.sum(1) + x.max(1)).schedule_linear().src[-1].src[0]
ast = ast.replace(arg=replace(ast.arg, opts_to_apply=()))
ren = Device["CPU"].renderer
sink = cg.full_rewrite_to_sink(ast, ren)

def short(u: UOp) -> str:
  if u.op is Ops.RANGE: return f"RANGE(axis {u.arg[1]})"
  if u.op is Ops.END: return "END(" + ",".join(f"axis {r.arg[1]}" for r in u.src[1:]) + ")"
  return u.op.name

print("== 1. CFGContext on the sink entering 'add control flow' ==")
cfg = CFGContext(seen["before_cf"])
for rng, after in cfg.edges.items():
  print(f"  {short(rng):16s} must come after {short(after)}   -> the RANGE's bound gets wrapped: AFTER(bound, {short(after)})")
print("  in the final sink these show up as AFTER uops feeding the RANGE bounds:")
for u in sink.toposort():
  if u.op is Ops.RANGE: print(f"    {short(u):16s} src[0] = {u.src[0].op.name}({', '.join(short(s) for s in u.src[0].src)})")

print("\n== 2. linearize(sink): a toposort with priorities ==")
lst = linearize(sink)
def prio(u: UOp):
  run_count = prod([int(r.vmax) + 1 for r in u.ranges])
  p = {Ops.PARAM: -20, Ops.LOAD: -1, Ops.STORE: 1, Ops.RANGE: 5, Ops.END: -5}.get(u.op, 0)
  if u.op in (Ops.BUFFER, Ops.ALLOC): p = -18
  return (run_count, len(u.ranges), p, u.arg.slot if u.op is Ops.PARAM else None)
depth, rows = 0, []
for i, u in enumerate(lst):
  if u.op is Ops.END: depth -= 1
  rows.append((i, u, depth))
  print(f"  {i:3d} {'  '*depth}{short(u):20s} dtype={str(u.dtype).replace('dtypes.',''):6s} prio(run_count,#ranges,op_prio,extra)={prio(u)}")
  if u.op is Ops.RANGE: depth += 1

print("\n== 3. pm_to_program, one rule application at a time ==")
prg = UOp(Ops.PROGRAM, src=(sink,), arg=ProgramInfo.from_sink(sink, ren.target))
step = 0
while (nxt := pm_to_program.rewrite(prg, ren)) is not None:
  prg, step = nxt, step + 1
  print(f"  step {step}: PROGRAM.src = ({', '.join(s.op.name for s in prg.src)})")
sink2, lin, source, binary = prg.src
print(f"  ProgramInfo: global_size={prg.arg.global_size} local_size={prg.arg.local_size} globals={prg.arg.globals} outs={prg.arg.outs} ins={prg.arg.ins}")
print(f"  Estimates (filled in by do_estimates): {sink2.arg.estimates}")
print(f"  LINEAR has {len(lin.src)} uops, SOURCE has {len(source.arg.splitlines())} lines, BINARY is {len(binary.arg)} bytes of ELF object")
print(source.arg)

# ---- map every C line back to the UOps that produced it (used by the chapter's "UOp -> source line" demo) ----------
lines_lst = line_rewrite(lst, pm_linearize_cleanups + pm_alloc_to_buf)   # exactly what do_linearize hands to render()
assert len(lines_lst) == len(lst)
# Spy on the renderer: CStyleLanguage._render() calls string_rewrite.rewrite(u, ctx) for every uop it doesn't skip, and
# afterwards ctx.r[u] is either the variable name it declared on its own line (val0, alu0, Lidx0...) or, if the uop was
# inlined into its consumer, the expression string itself. Emitted uops map 1:1, in order, onto the kernel lines.
class Spy:
  def __init__(self, pm): self.pm, self.seen, self.r = pm, [], None
  def rewrite(self, u, ctx):
    self.seen.append(u); self.r = ctx.r
    return self.pm.rewrite(u, ctx)
spy = ren.string_rewrite = Spy(type(ren).string_rewrite)
_, kernel_lines, bufs = ren._render(lines_lst)
del ren.string_rewrite                     # back to the class attribute
emitted = [u for u in spy.seen if re.fullmatch(r"[A-Za-z_]\w*", spy.r[u])]
assert len(emitted) == len(kernel_lines), (len(emitted), len(kernel_lines))
idx = {u: i for i, u in enumerate(lines_lst)}
emitted_at = {idx[u]: k for k, u in enumerate(emitted)}
src_lines = source.arg.splitlines()
pos, kline_to_src = 0, {}
for k, kl in enumerate(kernel_lines):      # locate each kernel line inside the full source text
  while src_lines[pos] != kl: pos += 1
  kline_to_src[k] = pos
line_uops: dict[int, set[int]] = {}
for i, k in emitted_at.items():           # a line owns its uop plus every src that was inlined into it (not emitted itself)
  todo, owned = [i], set()
  while todo:
    j = todo.pop()
    if j in owned: continue
    owned.add(j)
    todo.extend(idx[s] for s in lines_lst[j].src if s in idx and idx[s] not in emitted_at and lines_lst[idx[s]].op is not Ops.PARAM)
  line_uops[kline_to_src[k]] = owned
for name, (p, _) in bufs:                  # the PARAM declarations "float* data0_4 = args->arg0;"
  for li, l in enumerate(src_lines):
    if f"{name} = args->" in l: line_uops[li] = {idx[p]}
print("\n== 4. which UOps produced which C line ==")
for li, l in enumerate(src_lines):
  if li in line_uops: print(f"  {l.strip()[:60]:60s} <- " + ", ".join(f"%{j} {lines_lst[j].op.name}" for j in sorted(line_uops[li])))

if "--json" in sys.argv:
  nodes = [{"i": i, "op": u.op.name, "label": short(lines_lst[i]), "dtype": str(u.dtype).replace("dtypes.", ""),
            "arg": str(u.arg)[:40] if u.op not in (Ops.SINK,) else u.arg.function_name, "src": [idx[s] for s in u.src if s in idx],
            "depth": d, "prio": list(prio(lst[i])), "ranges": sorted(r.arg[1] for r in lst[i].ranges),
            "r": spy.r.get(u)}      # what _render stored in ctx.r for this uop: its variable name, or its inlined expression
           for (i, _, d), u in zip(rows, lines_lst)]
  pos = {u: i for i, u in enumerate(lst)}
  data = {"kernel": "(x.sum(1) + x.max(1)), x: 4x8 float32, no opts, Clang", "fn": sink.arg.function_name, "uir": render_uir(sink),
          "nodes": nodes, "source": src_lines, "line_uops": {str(k): sorted(v) for k, v in line_uops.items()},
          "cfg_edges": [[short(r), short(a)] for r, a in cfg.edges.items()],
          # sink.toposort() order, as positions in the final list: linearize() sorts this list by priority (+tuplize) to
          # get its "ideal" numbering, so the chapter's JS port of linearize() needs it to reproduce the real result
          "topo": [pos[u] for u in sink.toposort()],
          # linearize()'s "ideal" numbering nkey, recomputed exactly as linearizer.py:36 does: sort by priority, ties broken
          # by u.tuplize (TUPLE_ORDER=1 is the default). Indexed by position in the final list.
          "nkey": [k for _, k in sorted((pos[u], k) for k, u in enumerate(sorted(sink.toposort(), key=lambda u: prio(u) + u.tuplize)))]}
  path = os.path.join(os.path.dirname(__file__), "..", "data", "ch16_linear.json")
  with open(path, "w") as f: json.dump(data, f, separators=(",", ":"))
  print(f"\nwrote {os.path.normpath(path)} ({os.path.getsize(path)//1024} KB)")
