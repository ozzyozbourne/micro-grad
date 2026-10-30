#!/usr/bin/env python3
# Chapter 3 — "what dtype will this expression get?" Every answer below is produced by running tinygrad.
#
#   run:  DEV=CPU python3 examples/03_promotion_quiz.py   (add --dump to write data/ch03_quiz.json for the quiz game)
import sys, json, pathlib
from tinygrad import Tensor, dtypes

i8, u8, half = dtypes.int8, dtypes.uint8, dtypes.half
CASES = [
  ("Tensor([1, 2, 3])",                    lambda: Tensor([1, 2, 3])),
  ("Tensor([1., 2, 3])",                   lambda: Tensor([1., 2, 3])),
  ("Tensor([True, False])",                lambda: Tensor([True, False])),
  ("Tensor([2**40])",                      lambda: Tensor([2**40])),
  ("Tensor([2**63])",                      lambda: Tensor([2**63])),
  ("Tensor(3)",                            lambda: Tensor(3)),
  ("Tensor(3.0)",                          lambda: Tensor(3.0)),
  ("Tensor([1, 2, 3]) + 1",                lambda: Tensor([1, 2, 3]) + 1),
  ("Tensor([1, 2, 3]) + 1.5",              lambda: Tensor([1, 2, 3]) + 1.5),
  ("Tensor([1, 2, 3]) / 2",                lambda: Tensor([1, 2, 3]) / 2),
  ("Tensor([1, 2, 3]) // 2",               lambda: Tensor([1, 2, 3]) // 2),
  ("Tensor([1, 2, 3], dtype=int8) + 1",    lambda: Tensor([1, 2, 3], dtype=i8) + 1),
  ("Tensor([1, 2, 3], dtype=int8) + 1000", lambda: Tensor([1, 2, 3], dtype=i8) + 1000),
  ("Tensor([1, 2, 3], dtype=uint8) * 2.5", lambda: Tensor([1, 2, 3], dtype=u8) * 2.5),
  ("Tensor([1.], dtype=half) + 1.0",       lambda: Tensor([1.], dtype=half) + 1.0),
  ("Tensor([1.], dtype=half) + Tensor([1.])", lambda: Tensor([1.], dtype=half) + Tensor([1.])),
  ("int8 tensor + uint8 tensor",           lambda: Tensor([1], dtype=i8) + Tensor([1], dtype=u8)),
  ("uint32 tensor + int32 tensor",         lambda: Tensor([1], dtype=dtypes.uint32) + Tensor([1], dtype=dtypes.int32)),
  ("int64 tensor + uint64 tensor",         lambda: Tensor([1], dtype=dtypes.int64) + Tensor([1], dtype=dtypes.uint64)),
  ("int64 tensor + half tensor",           lambda: Tensor([1], dtype=dtypes.int64) + Tensor([1.], dtype=half)),
  ("Tensor([True]) + Tensor([True])",      lambda: Tensor([True]) + Tensor([True])),
  ("Tensor([True]) + 1",                   lambda: Tensor([True]) + 1),
  ("Tensor([1, 2]) < 2",                   lambda: Tensor([1, 2]) < 2),
  ("Tensor([1, 4]).sqrt()",                lambda: Tensor([1, 4]).sqrt()),
  ("Tensor([1, 2], dtype=uint8).sum()",    lambda: Tensor([1, 2], dtype=u8).sum()),
  ("Tensor([1, 2], dtype=int8).sum()",     lambda: Tensor([1, 2], dtype=i8).sum()),
  ("Tensor([1., 2], dtype=half).sum()",    lambda: Tensor([1., 2], dtype=half).sum()),
  ("Tensor([True, False]).sum()",          lambda: Tensor([True, False]).sum()),
  ("Tensor.arange(5)",                     lambda: Tensor.arange(5)),
  ("Tensor.full((2,), 7)",                 lambda: Tensor.full((2,), 7)),
  ("Tensor(3) * 0.5",                      lambda: Tensor(3) * 0.5),
  ("Tensor([1, 2]).mean()",                lambda: Tensor([1, 2]).mean()),
  ("Tensor([1, 2]) ** 2",                  lambda: Tensor([1, 2]) ** 2),
  ("Tensor([1, 2]) ** 0.5",                lambda: Tensor([1, 2]) ** 0.5),
  ("Tensor([1., 2]).bitcast(dtypes.int32)", lambda: Tensor([1., 2]).bitcast(dtypes.int32)),
]

rows = []
for code, fn in CASES:
  t = fn()
  val = t.tolist()
  rows.append({"code": code, "dtype": t.dtype.name, "value": str(val)})
  print(f"{code:42s} -> {t.dtype.name:10s} {val}")

if "--dump" in sys.argv:
  p = pathlib.Path(__file__).resolve().parent.parent / "data" / "ch03_quiz.json"
  p.write_text(json.dumps(rows, indent=1))
  print("wrote", p)
