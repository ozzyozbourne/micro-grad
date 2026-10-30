#!/usr/bin/env python3
# Chapter 2 — which calls actually run kernels? We measure instead of guessing.
#
#   run:  DEV=CPU python3 examples/02_realize_triggers.py   (add --dump to write data/ch02_triggers.json)
#
# For each snippet we build fresh inputs, reset GlobalCounters, run the snippet and report how many
# kernels (compute kernels + copies) were executed. 0 means "only the graph was built".
import sys, json, pathlib
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters

def fresh():
  a = Tensor([1., 2., 3., 4.]).realize()
  b = Tensor([5., 6., 7., 8.]).realize()
  return a, b

SNIPPETS = [
  ("c = a + b",                         lambda a, b: a + b),
  ("c = (a * b).sum()",                 lambda a, b: (a * b).sum()),
  ("c = a + b; c.shape",                lambda a, b: (a + b).shape),
  ("c = a + b; c.dtype",                lambda a, b: (a + b).dtype),
  ("c = a.reshape(2, 2).T",             lambda a, b: a.reshape(2, 2).T),
  ("c = a + b; c.realize()",            lambda a, b: (a + b).realize()),
  ("c = a + b; c.tolist()",             lambda a, b: (a + b).tolist()),
  ("(a * b).sum().item()",              lambda a, b: (a * b).sum().item()),
  ("c = a + b; c.contiguous()",         lambda a, b: (a + b).contiguous()),
  ("c = a.exp(); str(c)  (repr)",       lambda a, b: repr(a.exp())),
  ("a.assign(a + 1)",                   lambda a, b: a.assign(a + 1)),
  ("a.assign(a + 1).realize()",         lambda a, b: a.assign(a + 1).realize()),
  ("a.realize()  (already a buffer)",   lambda a, b: a.realize()),
  ("Tensor.ones(4)",                    lambda a, b: Tensor.ones(4)),
  ("Tensor.ones(4).realize()",          lambda a, b: Tensor.ones(4).realize()),
  ("Tensor.rand(4)",                    lambda a, b: Tensor.rand(4)),
  ("Tensor.rand(4).realize()",          lambda a, b: Tensor.rand(4).realize()),
  ("Tensor([1., 2.]).realize()",        lambda a, b: Tensor([1., 2.]).realize()),
  ("c = (a + b)[1]",                    lambda a, b: (a + b)[1]),
  ("(a + b)[1].item()",                 lambda a, b: (a + b)[1].item()),
  ("Tensor(3) + 4  (no device)",        lambda a, b: Tensor(3) + 4),
  ("(Tensor(3) + 4).item()",            lambda a, b: (Tensor(3) + 4).item()),
]

results = []
for code, fn in SNIPPETS:
  a, b = fresh()
  GlobalCounters.reset()
  fn(a, b)
  results.append({"code": code, "kernels": GlobalCounters.kernel_count})
  print(f"{code:36s} -> {GlobalCounters.kernel_count} kernel(s)")

if "--dump" in sys.argv:
  p = pathlib.Path(__file__).resolve().parent.parent / "data" / "ch02_triggers.json"
  p.write_text(json.dumps(results, indent=1))
  print("wrote", p)
