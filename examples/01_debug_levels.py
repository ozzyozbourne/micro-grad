#!/usr/bin/env python3
# Chapter 1 — what each DEBUG level prints, for the same tiny program.
#
#   run:  python3 examples/01_debug_levels.py            # re-runs itself with DEBUG=0..5 and shows each output
#         python3 examples/01_debug_levels.py --dump     # also writes data/ch01_debug.json for the DEBUG explorer
#         DEV=CPU DEBUG=4 python3 examples/01_debug_levels.py --child    # just the program, at one level
import os, sys, subprocess, json, pathlib

if "--child" in sys.argv:
  from tinygrad import Tensor
  x = Tensor([[1., -2, 3, -4], [5., -6, 7, -8]])
  w = Tensor([[1., 0, -1], [0., 1, 0], [1., 1, 1], [-1, 0, 1]])
  print((x @ w).relu().tolist())
  sys.exit(0)

outputs = {}
for level in range(6):
  env = {**os.environ, "DEV": "CPU", "DEBUG": str(level), "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"}
  res = subprocess.run([sys.executable, __file__, "--child"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  # drop the library-loading chatter that DEBUG>=1 prints when devices open, and tqdm carriage returns
  lines = [l.split("\r")[-1] for l in res.stdout.splitlines() if not l.startswith(("loading ", "opened device"))]
  outputs[level] = "\n".join(lines)
  print(f"================ DEBUG={level} ================\n{outputs[level]}\n")

if "--dump" in sys.argv:
  p = pathlib.Path(__file__).resolve().parent.parent / "data" / "ch01_debug.json"
  p.write_text(json.dumps(outputs))
  print("wrote", p)
