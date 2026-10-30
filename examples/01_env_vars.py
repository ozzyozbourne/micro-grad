#!/usr/bin/env python3
# Chapter 1 — every environment knob tinygrad reads through a ContextVar, with its default value.
#
#   run:  python3 examples/01_env_vars.py
#         DEBUG=2 BEAM=1 python3 examples/01_env_vars.py     # values come from the environment
#
# ContextVar (tinygrad/helpers.py) reads os.environ once at import, and can be overridden in code with
# `with Context(DEBUG=4): ...`. All of them register themselves in ContextVar._cache.
from tinygrad import Context
from tinygrad.helpers import ContextVar, DEBUG
import tinygrad.codegen, tinygrad.engine.realize   # importing pulls in the modules that define more ContextVars

for key, cv in sorted(ContextVar._cache.items()):
  print(f"{key:24s} = {cv.value!r}")
print(f"\n{len(ContextVar._cache)} ContextVars")
with Context(DEBUG=4): print("inside `with Context(DEBUG=4)`: DEBUG.value =", DEBUG.value)
print("outside again: DEBUG.value =", DEBUG.value)
