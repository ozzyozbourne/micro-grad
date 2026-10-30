#!/usr/bin/env python3
# Chapter 1 — the same program twice: which stages are skipped the second time?
#
#   run:  DEV=CPU DEBUG=2 python3 examples/01_caches.py
#
# Four caches sit on the pipeline (all keyed by UOp content hashes, `UOp.key`):
#   schedule_cache    tinygrad/schedule/__init__.py   CALL body  -> LINEAR          (SCACHE=0 disables)
#   to_program_cache  tinygrad/codegen/__init__.py    kernel AST -> PROGRAM         (in-process)
#   compile cache     tinygrad/device.py              C source   -> binary          (on disk, CCACHE=0 disables)
#   runtime_cache     tinygrad/engine/realize.py      PROGRAM    -> loaded function (in-process)
from tinygrad import Tensor
import tinygrad.schedule as sched, tinygrad.codegen as codegen, tinygrad.engine.realize as realize

def program(data: list[float]):
  a = Tensor(data)                        # different VALUES in the input buffer ...
  return (a * a + 1).sum()                # ... but the same graph STRUCTURE (buffers become PARAM slots)

def program_with_const(k: float):
  a = Tensor([1., 2., 3., 4.])
  return (a * a + k).sum()                # a different CONST is a different graph -> new kernel

runs = [("program([1,2,3,4])", lambda: program([1., 2., 3., 4.])),
        ("program([5,6,7,8])", lambda: program([5., 6., 7., 8.])),
        ("program_with_const(2.0)", lambda: program_with_const(2.0))]
for i, (name, fn) in enumerate(runs):
  print(f"--- run {i+1}: {name}")
  out = fn().item()
  print(f"result={out}  | schedule_cache={len(sched.schedule_cache)} to_program_cache={len(codegen.to_program_cache)} runtime_cache={len(realize.runtime_cache)}")
