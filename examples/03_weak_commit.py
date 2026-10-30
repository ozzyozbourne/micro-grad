#!/usr/bin/env python3
# Chapter 3 — where do weak literals become concrete? Follow the dtypes of one expression from Tensor graph to C.
#
#   run:  DEV=CPU python3 examples/03_weak_commit.py
#
# 1. in the Tensor graph the literal is a bare CONST with a *derived* weak dtype (dtype_from_uop: int arg -> weakint)
# 2. _broadcasted keeps a weak CONST weak (it only re-kinds it, weakint -> weakfloat), a concrete operand is CAST
# 3. codegen (full_rewrite_to_sink: pm_commit_weak / pm_lower_weak / pm_cast_const) gives every CONST a concrete type
# 4. the renderer prints a typed literal
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, GroupOp
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.late.linearizer import linearize
from tinygrad import Device

ren = Device["CPU"].renderer

def kernel_ast(t: Tensor):
  return next(si.src[0] for si in t.schedule_linear().src if si.src[0].op is Ops.SINK)

for label, mk in [("Tensor.empty(4, dtype=int8) + 1000", lambda: Tensor.empty(4, dtype=dtypes.int8) + 1000),
                  ("Tensor.empty(4, dtype=half) * 0.5", lambda: Tensor.empty(4, dtype=dtypes.half) * 0.5),
                  ("Tensor.empty(4, dtype=int32) + 1.5", lambda: Tensor.empty(4, dtype=dtypes.int32) + 1.5)]:
  t = mk()
  print(f"\n=== {label} ===")
  print("tensor dtype:", t.dtype, "| top op:", t.uop.op.name)
  for i, s in enumerate(t.uop.src): print(f"  src[{i}]: {s.op.name:6s} dtype={s.dtype}" + (f"  arg={s.arg!r}" if s.op is Ops.CONST else
                                          f"  <- {s.src[0].op.name} {s.src[0].dtype}" if s.op is Ops.CAST else ""))
  if t.dtype in dtypes.weaks:
    print("weak result: realizing it needs a concrete dtype first -> commit_dtype() =", t.commit_dtype())
    t = t.cast(t.commit_dtype())
  lowered = full_rewrite_to_sink(kernel_ast(t), ren)
  alu = [u for u in lowered.toposort() if u.op in GroupOp.ALU]
  print("after full_rewrite_to_sink, ALU ops:", ", ".join(f"{u.op.name}:{u.dtype.name}" for u in alu))
  # a CONST's dtype is derived from its Python arg (always weak), so a typed constant is CAST(CONST): the CAST carries the width
  topo = lowered.toposort()
  # (small ints 0..4 are index/loop constants of the kernel itself; skip them to keep the output short)
  casts = sorted({f"CAST({u.src[0].arg!r}:{u.src[0].dtype.name}) -> {u.dtype.name}" for u in topo
                  if u.op is Ops.CAST and u.src[0].op is Ops.CONST and u.src[0].arg not in (0, 1, 2, 3, 4)})
  print("  typed constants:", ", ".join(casts))
  print("  weak node that is NOT a bare CONST under a concrete CAST?",
        any(u.dtype in dtypes.weaks and u.op is not Ops.CONST for u in topo))
  src = ren.render(linearize(lowered))
  print("C:", next(l.strip() for l in src.split("\n") if l.strip().startswith("*(") and "data0" in l))
