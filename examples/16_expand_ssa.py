# Chapter 16 — how CStyleLanguage._render decides between "inline this expression" and "declare a variable".
#
# _render names every UOp (val0, alu0, bidx0, cast0, Lidx0 ...) and asks string_rewrite for its C fragment. Then:
#   - CAST of a CONST, INDEX, single-use REG LOAD, single-use ALU (except WHERE) / CAST / BITCAST / STACK -> inlined:
#     ctx.r[u] becomes the fragment itself, and the consumer pastes it in
#   - everything else gets its own line:  <type> <name> = <fragment>;
# EXPAND_SSA=1 disables the "single-use ALU/CAST" inlining, so you see one SSA value per line.
#
# run:   DEV=CPU python3 examples/16_expand_ssa.py
#        EXPAND_SSA=1 DEV=CPU python3 examples/16_expand_ssa.py
import os
from tinygrad import Tensor, Device
from tinygrad.codegen import to_program

ren = Device["CPU"].renderer
x = Tensor.empty(4, 4)
ast = ((x * 2 + 1).sqrt() * x.max(1, keepdim=True)).schedule_linear().src[-1].src[0]   # default (heuristic) opts
src = to_program(ast, ren).src[2].arg
print(f"EXPAND_SSA={os.environ.get('EXPAND_SSA', '0')}: {len(src.splitlines())} lines")
print(src)
