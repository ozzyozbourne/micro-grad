# Chapter 16 — why "add control flow" (CFGContext + pm_add_control_flow) exists: turn it off and watch the linearizer break.
#
# linearize() is "only" a priority toposort. The DAG says what must come before what, but nothing in the plain DAG says
# that two *sibling* loops must not overlap: a RANGE only has to come before its body and END only after it. With two
# siblings the priorities happen to give a correct order anyway; with three siblings of different sizes they do not.
# CFGContext adds the missing edges (sibling i+1's RANGE bound is wrapped in AFTER(bound, END of sibling i)).
#
# We monkeypatch graph_rewrite to skip the "add control flow" stage and check whether every END closes the most recently
# opened RANGE (= the loops are properly nested, which C's "}" requires).
#
# run:   DEV=CPU python3 examples/16_why_control_flow.py
from dataclasses import replace
from tinygrad import Tensor, Device
from tinygrad.uop.ops import Ops
import tinygrad.codegen as cg
from tinygrad.codegen.late.linearizer import linearize

SKIP_CF = [False]
_gr = cg.graph_rewrite
def gr(sink, pm, *a, name=None, **kw):
  if name == "add control flow" and SKIP_CF[0]: return sink      # pretend the stage does not exist
  return _gr(sink, pm, *a, name=name, **kw)
cg.graph_rewrite = gr

def loop_events(lst):
  """the RANGE/END skeleton of a linear program, and whether it is properly nested"""
  stack, ev, ok = [], [], True
  for u in lst:
    if u.op is Ops.RANGE:
      stack.append(u); ev.append(f"RANGE{u.arg[1]}(len {int(u.vmax)+1})")
    elif u.op is Ops.END:
      ev.append(f"END{u.src[1].arg[1]}")
      if not stack or stack[-1] is not u.src[1]: ok = False
      elif stack: stack.pop()
  return ok, ev

x, y, z = Tensor.empty(4, 8), Tensor.empty(4, 16), Tensor.empty(4, 3)
cases = {"two siblings   x.sum(1)+x.max(1)": x.sum(1) + x.max(1),
         "three siblings x.sum(1)+y.max(1)+z.sum(1)": x.sum(1) + y.max(1) + z.sum(1)}
ren = Device["CPU"].renderer
for desc, t in cases.items():
  ast = t.schedule_linear().src[-1].src[0]
  ast = ast.replace(arg=replace(ast.arg, opts_to_apply=()))     # no opts: keep every reduce as a real loop
  print(f"== {desc}")
  for skip in (False, True):
    SKIP_CF[0] = skip
    ok, ev = loop_events(linearize(cg.full_rewrite_to_sink(ast, ren)))
    print(f"  {'WITHOUT' if skip else 'with   '} add control flow: {'properly nested' if ok else 'BROKEN NESTING'}")
    print("     " + "  ".join(ev))
