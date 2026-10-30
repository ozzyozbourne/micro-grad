"""Shared helper: convert a tinygrad UOp graph into the JSON format the guide's graph renderer (Guide.uopGraph) consumes.

Format:
  {"nodes": [{"id": "n0", "op": "ADD", "dtype": "dtypes.float", "arg": "None", "shape": "(4,)", "label": "ADD"}...],
   "edges": [["n0", "n3", 0], ...]}   # edge = [src_id, consumer_id, src_index]
Node ids are stable within a single dump. Use `same_ids=True` mapping (pass a shared `ids` dict) when dumping several
graphs that should morph into each other: UOps are hash-consed, so the same UOp object keeps the same id.

Usage:
  import sys; sys.path.insert(0, "/home/user/workspace/tinygrad-guide/tools")
  from uop2json import uop_to_graph, dump
  dump({"graph": uop_to_graph(t.uop)}, "/home/user/workspace/tinygrad-guide/data/ch04_example.json")
"""
import json
from tinygrad.uop.ops import UOp

def _short(x, n=60) -> str:
  s = str(x)
  return s if len(s) <= n else s[:n-1] + "…"

def uop_to_graph(root: UOp, ids: dict | None = None, max_nodes: int = 400) -> dict:
  ids = {} if ids is None else ids
  nodes, edges = [], []
  topo = list(root.toposort())
  if len(topo) > max_nodes: topo = topo[-max_nodes:]
  present = set(topo)
  for u in topo:
    nid = ids.setdefault(u, f"n{len(ids)}")
    try: dtype = str(u.dtype)
    except Exception: dtype = "?"
    try: shape = str(u._shape) if u._shape is not None else ""
    except Exception: shape = ""
    nodes.append({"id": nid, "op": u.op.name, "dtype": dtype, "arg": _short(u.arg), "shape": shape})
    for i, s in enumerate(u.src):
      if s in present: edges.append([ids.setdefault(s, f"n{len(ids)}"), nid, i])
  return {"nodes": nodes, "edges": edges}

def dump(obj, path: str):
  with open(path, "w") as f: json.dump(obj, f, indent=1, default=str)
