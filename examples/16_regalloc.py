# Chapter 16 — the x86 ISA renderer: instruction selection + linear-scan register allocation (codegen/late/regalloc.py).
#
# For an ISARenderer, do_to_program runs two extra graph_rewrites ("pre instruction selection", "instruction selection")
# that turn LOAD/STORE/ALU uops into Ops.INS uops (arg = (X86Ops.xxx, dtype, *registers)) whose registers are *virtual*
# (Register("v12", cons=(rax, rcx, ...))). do_linearize then builds LinearScanRegallocContext over the linear list,
# which computes live ranges and assigns a real register to every virtual one (spilling to the stack if it runs out).
# We subclass that context to keep a reference to it, then print the live ranges and the assignment.
#
# run:   DEV=CPU python3 examples/16_regalloc.py           (prints the table + final assembly)
#        DEV=CPU python3 examples/16_regalloc.py --json    (also writes data/ch16_regalloc.json for the live-range chart)
import sys, os, json
from dataclasses import replace
from tinygrad import Tensor
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops
import tinygrad.codegen as cg
from tinygrad.renderer.x86 import X86Renderer, X86Ops

caught = []
class SpyRegalloc(cg.LinearScanRegallocContext):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    caught.append(self)
cg.LinearScanRegallocContext = SpyRegalloc   # do_linearize looks the name up in tinygrad.codegen at call time

x = Tensor.empty(4, 8)
ast = (x.sum(1) + x.max(1)).schedule_linear().src[-1].src[0]
ast = ast.replace(arg=replace(ast.arg, opts_to_apply=()))
ren = X86Renderer(Target("CPU", "X86", "x86_64,native"))
prg = cg.to_program(ast, ren)
ra = caught[0]

# virtual registers are named v0, v1, ... except loop counters/accumulators, which are named by a UOp key hash: shorten
names = {}
def nm(r) -> str:
  if r.name not in names: names[r.name] = r.name if len(r.name) < 8 else f"k{sum(1 for n in names.values() if n.startswith('k'))}"
  return names[r.name]
def text(i, u) -> str:
  if u.op is not Ops.INS: return u.op.name.lower()
  if u.arg[0] in (X86Ops.IMM, X86Ops.REG): return f"({u.arg[0].name.lower()})"
  if u.arg[0] is X86Ops.LABEL: return "label:"
  try: s = ren.asm_str([u], "f").split("\n", 1)[1].strip()
  except Exception: s = u.arg[0].name.lower()     # jumps print their target via the full list only
  for long, short in names.items(): s = s.replace(long, short)
  return s
for v in ra.live_range: nm(v)

rows = [{"i": i, "text": text(i, u), "real": {nm(v): str(r) for v, r in ra.reals.get(i, {}).items()}} for i, u in enumerate(ra.uops)]
vregs = []
for v, lr in ra.live_range.items():
  first_real = next((ra.reals[i][v] for i in lr if v in ra.reals.get(i, {})), None)
  vregs.append({"name": nm(v), "cls": "xmm" if str(v.cons[0]).startswith("xmm") else "gpr", "uses": lr,
                "start": min(lr), "end": max(lr), "real": str(first_real), "spilled": v in ra.spills})
print(f"{len(ra.uops)} linear uops, {len(vregs)} virtual registers, {len(ra.spills)} spilled")
print(f"{'vreg':6s} {'class':5s} {'live range':12s} real   use positions")
for v in sorted(vregs, key=lambda v: v["start"]):
  print(f"{v['name']:6s} {v['cls']:5s} [{v['start']:3d},{v['end']:3d}]    {v['real']:6s} {v['uses']}")
print("\n--- instructions with virtual registers -> real registers (only INS that define/use registers) ---")
for r in rows:
  if r["real"]: print(f"{r['i']:3d}  {r['text']:42s} {r['real']}")
asm = ren.asm_str(list(prg.src[1].src), prg.src[0].arg.function_name)
print("\n--- final assembly (after regalloc, callee-saved spills and stack frame were inserted) ---\n" + asm)

if "--json" in sys.argv:
  path = os.path.join(os.path.dirname(__file__), "..", "data", "ch16_regalloc.json")
  with open(path, "w") as f: json.dump({"rows": rows, "vregs": vregs, "asm": asm.splitlines()}, f, separators=(",", ":"))
  print(f"\nwrote {os.path.normpath(path)} ({os.path.getsize(path)//1024} KB)")
