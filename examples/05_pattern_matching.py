# Chapter 5 — UPat, PatternMatcher & graph_rewrite.  Run with:  DEV=CPU python3 examples/05_pattern_matching.py
# Builds a tiny rewrite system from scratch, then looks at the real machinery underneath it.
import json, pathlib
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, graph_rewrite
from tinygrad.uop.upat import _get_code
from tinygrad.uop.symbolic import symbolic_simple, symbolic, sym

def section(t): print(f"\n=== {t} ===")
x = UOp.variable("x", 0, 10)
c = UOp.const
R = lambda u: u.render(simplify=False)     # render WITHOUT running symbolic, so we see the raw graph

section("1. a UPat is a template; UPat.var('x') matches anything and names it")
p = UPat.var("x") * 1                      # operator overloading builds UPat(Ops.MUL, src=[UPat(name='x'), UPat.cvar(arg=1)])
print("op:", p.op, "| src permutations tried:", len(p.src), "| early_reject:", p.early_reject)
print("match x*1:", [{k: R(v) for k, v in m.items()} for m in p.match(x * 1, {})])
print("match 1*x:", [{k: R(v) for k, v in m.items()} for m in p.match(c(1) * x, {})], " <- commutative: src is a list, so both orders are tried")
print("match x*2:", p.match(x * 2, {}))

section("2. a tiny PatternMatcher: identities + constant folding")
tiny = PatternMatcher([
  (UPat.var("x") * 1, lambda x: x),                                                       # x*1 -> x
  (UPat.var("x") + 0, lambda x: x),                                                       # x+0 -> x
  (UPat(Ops.ADD, src=(UPat.cvar("a"), UPat.cvar("b"))), lambda a, b: c(a.arg + b.arg)),   # const+const
  (UPat(Ops.MUL, src=(UPat.cvar("a"), UPat.cvar("b"))), lambda a, b: c(a.arg * b.arg)),   # const*const
])
e = (x * 1 + (c(2) + 3)) * (c(1) + 0)
print("before:", R(e))
print("after :", R(graph_rewrite(e, tiny)))
print("pdict keys (rules are indexed by op):", {k.name: len(v) for k, v in tiny.pdict.items()})

section("3. returning None means 'no match, try the next rule'")
only_small = PatternMatcher([(UPat.var("x") * UPat.cvar("k"), lambda x, k: x + x if k.arg == 2 else None)])
print("x*2 ->", R(graph_rewrite(x * 2, only_small)), "   x*3 ->", R(graph_rewrite(x * 3, only_small)), " (fxn returned None: unchanged)")

section("4. the order nodes are matched in: default vs bottom_up=True")
class Logged(PatternMatcher):
  log: list = []
  def rewrite(self, uop, ctx=None):
    ret = super().rewrite(uop, ctx)
    Logged.log.append([R(uop), None if ret is None else R(ret)])
    return ret
logged = Logged(tiny.patterns)
traces = {}
for bu in (False, True):
  Logged.log = []
  out = graph_rewrite(e, logged, bottom_up=bu)
  traces["bottom_up" if bu else "default"] = {"result": R(out), "log": Logged.log}
  print(f"bottom_up={bu}: result {R(out)}")
  for i, (before, after) in enumerate(Logged.log): print(f"   {i:2d}. try {before:22s} -> {after if after else '(no match)'}")

section("5. fixed point: a rewritten node is rewritten again (unless walk=True)")
expand = PatternMatcher([(UPat.var("x") * 4, lambda x: (x * 2) * 2), (UPat.var("x") * 2, lambda x: x + x)])
print("graph_rewrite(x*4)            ->", R(graph_rewrite(x * 4, expand)))
print("graph_rewrite(x*4, walk=True) ->", R(graph_rewrite(x * 4, expand, walk=True)), " (single pass, no re-traversal of the replacement)")

section("6. ctx: rules can read/write shared state (here: count the CONSTs we fold)")
counting = PatternMatcher([(UPat(Ops.ADD, src=(UPat.cvar("a"), UPat.cvar("b"))), lambda ctx, a, b: ctx.append(a.arg + b.arg) or c(a.arg + b.arg))])
seen: list = []
print("result:", R(graph_rewrite(c(1) + 2 + 3 + 4, counting, ctx=seen)), " ctx after:", seen)

section("7. UPats are compiled to Python source (upat.py)")
code, consts = _get_code(UPat.var("x") * 1, has_ctx=False)
print(code.split("\n", 1)[1]); print("constants:", consts)
code, consts = _get_code((UPat.var("x") // UPat.cvar("c1")) // UPat.cvar("c2"), has_ctx=False)
print(code.split("\n", 1)[1])

section("8. the real matchers are just bigger lists of the same thing")
for name, pm in [("symbolic_simple", symbolic_simple), ("symbolic", symbolic), ("sym", sym)]:
  print(f"{name:16s} {len(pm.patterns):4d} rules over {len(pm.pdict):2d} ops")
print("symbolic_simple on our expression:", R(graph_rewrite(e, symbolic_simple)))
print("composition is '+':", len((tiny + expand).patterns), "rules")

pathlib.Path(__file__).resolve().parent.parent.joinpath("data", "ch05_traces.json").write_text(json.dumps(traces, indent=1))
