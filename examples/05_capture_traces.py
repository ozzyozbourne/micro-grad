# Chapter 5 data capture: real graph_rewrite traces used to VERIFY the JavaScript mirror of unified_rewrite / walk_rewrite.
# Run with:  DEV=CPU python3 examples/05_capture_traces.py      (writes data/ch05_rewrite_traces.json)
# For every (expression, rule set, mode) we log each call the engine makes into the PatternMatcher: the node tried and what it returned.
import json, pathlib
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, graph_rewrite

R = lambda u: u.render(simplify=False)
c = UOp.const
RULES = {  # name -> (UPat, fxn). The JS demo implements the same rules with the same source-order semantics.
  "mul1":  (UPat.var("x") * 1, lambda x: x),
  "add0":  (UPat.var("x") + 0, lambda x: x),
  "cadd":  (UPat(Ops.ADD, src=(UPat.cvar("a"), UPat.cvar("b"))), lambda a, b: c(a.arg + b.arg)),
  "cmul":  (UPat(Ops.MUL, src=(UPat.cvar("a"), UPat.cvar("b"))), lambda a, b: c(a.arg * b.arg)),
  "mul0":  (UPat.var("x") * 0, lambda x: c(0)),
  "assoc": ((UPat.var("x") + UPat.cvar("c1")) + UPat.cvar("c2"), lambda x, c1, c2: x + (c1.arg + c2.arg)),
  "xx":    (UPat.var("x") + UPat.var("x"), lambda x: x * 2),
  "mul4":  (UPat.var("x") * 4, lambda x: (x * 2) * 2),
  "mul2":  (UPat.var("x") * 2, lambda x: x + x),
}
x, y = UOp.variable("x", 0, 10), UOp.variable("y", 0, 10)
EXPRS = {
  "E1": (x * 1 + (c(2) + 3)) * (c(1) + 0),
  "E2": (x + 0) * (y * 1) + c(0) * (x + y),
  "E3": (x + y) * (x + y) * 1 + ((x + 2) + 3),
  "E4": x * 4,
  "E5": x + x,
}
SETS = {"tiny": ["mul1", "add0", "cadd", "cmul"], "more": ["mul1", "add0", "cadd", "cmul", "mul0", "assoc"],
        "expand": ["mul4", "mul2"], "loop": ["xx", "mul2"]}
COMBOS = [("E1", "tiny"), ("E2", "more"), ("E3", "more"), ("E4", "expand"), ("E5", "loop")]
MODES = {"default": {}, "bottom_up": {"bottom_up": True}, "walk": {"walk": True}, "walk+bottom_up": {"walk": True, "bottom_up": True}}

class Logged(PatternMatcher):
  log: list = []
  def rewrite(self, uop, ctx=None):
    ret = super().rewrite(uop, ctx)
    Logged.log.append([R(uop), None if ret is None else R(ret)])
    return ret

out = []
for ename, sname in COMBOS:
  pm = Logged([RULES[r] for r in SETS[sname]])
  for mname, kw in MODES.items():
    Logged.log = []
    try: res, err = R(graph_rewrite(EXPRS[ename], pm, **kw)), None
    except RuntimeError as e: res, err = None, str(e).split("\n")[0]
    out.append({"expr": ename, "rules": sname, "mode": mname, "input": R(EXPRS[ename]), "result": res, "error": err,
                "n_calls": len(Logged.log), "log": Logged.log[:60]})
    print(f"{ename} {sname:6s} {mname:15s} {R(EXPRS[ename]):34s} -> {res if err is None else 'RuntimeError: ' + err}   ({len(Logged.log)} matcher calls)")
pathlib.Path(__file__).resolve().parent.parent.joinpath("data", "ch05_rewrite_traces.json").write_text(json.dumps(out, indent=1))
