# Chapter 21 — the UOp spec: what a valid graph looks like, and what an invalid one fails with
# Run: DEV=CPU python3 examples/21_spec.py
#
# uop/spec.py defines the spec as PatternMatchers that return True for valid UOps (spec_tensor, spec_program,
# spec_kernel_graph, spec_full). type_verify(uops, spec) runs the matcher on every UOp in toposort order and raises
#   RuntimeError("UOp verification failed at {i} on {op} {dtype} {len(src)} {[(src.op, src.dtype, src.arg)...]} {arg}")
# on the first one that does not return True. SPEC=1 (the default) runs it at a few checkpoints (schedule + codegen);
# SPEC=2 additionally checks every UOp *as it is constructed* (UOpMetaClass.__call__ in uop/ops.py).
import json, os, subprocess, sys
from tinygrad import dtypes
from tinygrad.uop.ops import UOp, Ops
from tinygrad.uop.spec import type_verify, spec_tensor, spec_program, spec_full
HERE = os.path.dirname(os.path.abspath(__file__))

a = UOp.const(1.0, dtypes.float)   # NOTE: a typed const is CAST(CONST(weak)), see UOp.const
b = UOp.const(2.0, dtypes.float)
i = UOp.const(3, dtypes.int)
cases = [
  ("valid: float ADD float", (a + b), spec_tensor),
  ("ADD of float and int (operands must share the result dtype)", UOp(Ops.ADD, src=(a, i)), spec_tensor),
  ("bitwise AND on floats", UOp(Ops.AND, src=(a, b)), spec_tensor),
  ("CMPLT comparing a float with an int", UOp(Ops.CMPLT, src=(a, i)), spec_tensor),
  ("WHERE with a float condition", UOp(Ops.WHERE, src=(a, a, b)), spec_tensor),
  ("CAST whose arg is not a DType", UOp(Ops.CAST, src=(a,), arg="float"), spec_tensor),
  ("SQRT of an int (surprise: dtype inference makes it float32, so it passes)", UOp(Ops.SQRT, src=(i,)), spec_tensor),
  ("a movement op inside a *program* (spec_program forbids RESHAPE)", a.reshape((1,)), spec_program),
]
def safe_uir(u:UOp) -> str:
  try: return u.render_uir()
  except Exception: return f"UOp(Ops.{u.op.name}, src=<{len(u.src)} srcs>, arg={u.arg!r})   (render_uir itself chokes on it)"
results = []
for title, u, spec in cases:
  try:
    type_verify(u, spec)
    msg = "OK"
  except Exception as e:
    msg = f"{type(e).__name__}: {e}"
  print(f"--- {title}\n    {msg}")
  results.append({"title": title, "uir": safe_uir(u), "spec": {id(spec_tensor): "spec_tensor", id(spec_program): "spec_program"}[id(spec)], "error": msg})

print("\n--- SPEC=2: the check runs at construction time, inside UOpMetaClass.__call__")
code = ("from tinygrad import dtypes\nfrom tinygrad.uop.ops import UOp, Ops\n"
        "UOp(Ops.ADD, src=(UOp.const(1.0, dtypes.float), UOp.const(3, dtypes.int)))\nprint('constructed fine, nobody checked')")
for spec in ("1", "2"):
  p = subprocess.run([sys.executable, "-c", code], env={**os.environ, "DEV": "CPU", "SPEC": spec}, capture_output=True, text=True)
  out = (p.stdout + p.stderr).strip().splitlines()
  last = next((l for l in out if l.startswith(("RuntimeError", "constructed"))), out[-1])
  print(f"    SPEC={spec}: {last[:160]}")
  results.append({"title": f"SPEC={spec}: construct ADD(float, int)", "uir": "UOp(Ops.ADD, src=(UOp.const(1.0, dtypes.float), UOp.const(3, dtypes.int)))", "spec": "spec_full (SPEC>=2)" if spec == "2" else "(none at construction)", "error": last})
json.dump(results, open(os.path.join(HERE, "..", "data", "ch21_spec.json"), "w"), indent=1)
print("wrote data/ch21_spec.json")
