# Chapter 21 — NOOPT and BEAM change the *same* kernel; process replay is "same input, compare generated code"
# Run: DEV=CPU python3 examples/21_noopt_beam.py
#
# The kernel AST is identical in all three runs; only the optimizer stage differs:
#   NOOPT=1  -> no OptOps applied (the plain loops the AST describes)
#   default  -> the hand-coded heuristic (codegen/opt/heuristic.py)
#   BEAM=2   -> beam search over OptOps, timing real candidates (codegen/opt/search.py)
# We print the applied opts and the C for each. Then the idea behind process replay: lower the same captured AST
# twice and diff the source -- a refactor that changes nothing must give an empty diff.
import difflib, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
prog = """
from tinygrad import Tensor
a = Tensor.rand(64, 64).realize()
b = (a @ a.T).realize()
"""
for title, extra in (("NOOPT=1", {"NOOPT": "1"}), ("default (hand-coded heuristic)", {}), ("BEAM=2", {"BEAM": "2", "IGNORE_BEAM_CACHE": "1"})):
  env = {**os.environ, "DEV": "CPU", "DEBUG": "4", "NO_COLOR": "1", **extra}
  p = subprocess.run([sys.executable, "-c", prog], env=env, capture_output=True, text=True)
  out = p.stdout
  # the matmul kernel is the one whose name starts with r_ (a reduce)
  name = re.search(r"void (r_\w+)\(", out).group(1)
  m = re.search(rf"^{name}\s+opts: (.*)$", out, re.M)
  src = out[out.index(f"void {name}("):]
  src = src[:src.index("\n}\n") + 2]
  run = re.search(rf"^\*\*\* CPU\s+\d+ {name}\s.*tm\s+([\d.]+\w+)/", out, re.M)
  print(f"===== {title}: kernel {name}  opts={m.group(1) if m else '()'}  time={run.group(1) if run else '?'}")
  lines = src.splitlines()
  print("\n".join(lines[:14]) + (f"\n  ... ({len(lines)} lines)" if len(lines) > 14 else ""))
  print()

print("===== process replay, in miniature")
from tinygrad import Tensor, Device
from tinygrad.uop.ops import Ops
from tinygrad.codegen import to_program
from tinygrad.helpers import Context
x = Tensor.rand(4, 4).realize()
ast = next(c.src[0] for c in Tensor.schedule_linear((x * 2).sum(1)).src if c.src[0].op is Ops.SINK)
ren = Device["CPU"].renderer
def src_of(prg): return next(s.arg for s in prg.src if s.op is Ops.SOURCE)
good = src_of(to_program(ast, ren))                        # "master"
same = src_of(to_program(ast, ren))                        # "your branch", a pure refactor
with Context(NOOPT=1): changed = src_of(to_program(ast, ren))   # "your branch", which (accidentally) changed codegen
for nm, cmp in (("refactor", same), ("behavior change", changed)):
  diff = [l for l in difflib.unified_diff(good.splitlines(), cmp.splitlines(), lineterm="") if not l.startswith(("---", "+++"))]
  print(f"{nm}: {len(diff)} diff lines" + ("".join("\n  " + l for l in diff[:12]) if diff else "  -> process replay passes"))
