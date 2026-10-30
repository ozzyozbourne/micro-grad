"""examples/11_make_data.py — the fusion case-study catalogue behind chapter 11 (real kernels, real C code)

Run:  DEV=CPU NOOPT=1 python3 examples/11_make_data.py
      (NOOPT=1 keeps the generated C close to the loop nest rangeify produced: no upcasting/unrolling)
Writes data/ch11_cases.json and data/ch11_game.json

For every program we record
  * the tensor graph exactly as run_rangeify receives it (input of the "get realize" pass)
  * the realize_map CANDIDATES and which of them SURVIVED pm_remove_bufferize (matched through the RANGE ids
    each candidate's STAGE closes — the STAGE of a realized UOp closes exactly the fresh ranges run_rangeify gave it)
  * which kernel(s) every node of the tensor graph ends up in: a kernel is rooted at a STORE or at a surviving
    STAGE; walking down from the root, we stop at other roots / buffers (those become loads). A node reached from
    two roots is RECOMPUTED in both kernels.
  * the final kernels (CALLs of the LINEAR) with their C source.
"""
import importlib.util, json, os, pathlib
from collections import deque
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.uop.ops import UOp, Ops, GroupOp
from tinygrad.schedule.indexing import data_srcs

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("cap", HERE / "10_sched_capture.py")
cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)
DATA = HERE.parent / "data"

def dump(obj, name):
  with open(DATA / name, "w") as f: json.dump(obj, f, separators=(",", ":"))
  print(f"wrote data/{name}: {os.path.getsize(DATA / name)/1024:.1f} KB")

def range_ids(rs) -> frozenset:
  return frozenset(r.arg[1] for r in UOp.sink(*rs).ranges if r.op is Ops.RANGE) if len(rs) else frozenset()

def analyze(make, ctx=None):
  with Context(**(ctx or {})):
    outs = make()
    outs = outs if isinstance(outs, tuple) else (outs,)
    log, linear, compiled = cap.schedule(*outs)
  G = cap.pass_by_name(log, "get realize")[1]
  rctx, rmap = log["rctx"][0], log["realize_map"]
  post = cap.pass_by_name(log, "limit buffers")[2]
  stages = [u for u in post.toposort() if u.op is Ops.STAGE]
  # match STAGEs that survived to candidates by the ranges they close
  survived, used = set(), set()
  cands = [u for u in rmap if u.op is not Ops.STORE]
  for st in stages:
    sig = range_ids(st.src[1:])
    best = [c for c in cands if c not in used and sig and sig <= range_ids(rctx.range_map[c][1])]
    if not best: best = [c for c in cands if c not in used and c.op is st.src[0].op]
    if best: survived.add(best[0]); used.add(best[0])
  stores = [u for u in G.toposort() if u.op is Ops.STORE]
  roots = stores + [c for c in G.toposort() if c in survived]
  # kernel membership
  member: dict[UOp, list[int]] = {}
  deps: dict[int, set[int]] = {i: set() for i in range(len(roots))}
  root_idx = {r: i for i, r in enumerate(roots)}
  def walk(k: int, u: UOp, top: bool):
    if not top and u in root_idx:            # another kernel's output: we LOAD it
      deps[k].add(root_idx[u]); member.setdefault(u, [])
      return
    if u.op is Ops.AFTER:                     # buffer written by other kernels (their STOREs are roots)
      for s in u.src[1:]:
        if s in root_idx: deps[k].add(root_idx[s])
      walk(k, u.src[0], False); return
    lst = member.setdefault(u, [])
    if k in lst: return
    lst.append(k)
    if u.op in GroupOp.Defines: return
    for i, s in enumerate(u.src):
      if i in cap.hidden_srcs(u): continue
      walk(k, s, False)
  for i, r in enumerate(roots): walk(i, r, True)
  # order kernels like create_schedule (Kahn / BFS over dependencies)
  indeg = {i: len(d) for i, d in deps.items()}
  children = {i: [j for j in deps if i in deps[j]] for i in deps}
  q, order = deque(i for i in deps if indeg[i] == 0), []
  while q:
    i = q.popleft(); order.append(i)
    for j in children[i]:
      indeg[j] -= 1
      if indeg[j] == 0: q.append(j)
  renum = {old: new for new, old in enumerate(order)}
  ids: dict = {}
  notes = {}
  for u, ks in member.items():
    ks2 = sorted(renum[k] for k in ks)
    if u.op in GroupOp.Defines: notes[u] = "buffer: read by kernel(s) " + ", ".join(f"K{k}" for k in ks2)
    elif len(ks2) > 1: notes[u] = "RECOMPUTED in kernels " + ", ".join(f"K{k}" for k in ks2)
    elif ks2: notes[u] = f"computed in kernel K{ks2[0]}"
  for c in cands: notes[c] = notes.get(c, "") + ("\nrealize_map candidate -> SURVIVED: becomes a kernel output" if c in survived
                                                 else "\nrealize_map candidate -> REMOVED by pm_remove_bufferize (recomputed instead)")
  graph = cap.graph_json(G, ids, notes)
  kern_of = {ids[u]: sorted(renum[k] for k in ks) for u, ks in member.items() if u in ids}
  for n in graph["nodes"]: n["k"] = kern_of.get(n["id"], [])
  roots_info = [{"id": ids.get(roots[old]), "op": roots[old].op.name, "shape": str(roots[old].shape).replace(" ", "")} for old in order]
  kerns = cap.kernel_sources(compiled)
  calls = []
  for c in linear.src:
    calls.append({"nbufs": len(c.src) - 1, "out_numel": int(c.src[1].max_numel()) if len(c.src) > 1 else 0})
  return {"graph": graph, "roots": roots_info,
          "candidates": [{"id": ids.get(c), "op": c.op.name, "shape": str(c.shape).replace(" ", ""), "survived": c in survived} for c in cands],
          "kernels": [{"name": n, "src": s, **calls[i]} for i, (n, s) in enumerate(kerns)],
          "n_kernels": len(linear.src), "n_roots": len(roots)}

def realized(*shape):
  return Tensor.empty(*shape).realize()

CASES = [
  ("ew", "Elementwise chain", "a, b = Tensor.empty(16), Tensor.empty(16)\nout = (a*2 + b).relu()",
   lambda: (Tensor.empty(16) * 2 + Tensor.empty(16)).relu(),
   "Every op has the same 16 RANGE-indexed elements; nothing is realized except the output STORE. One loop, one kernel."),
  ("ewred", "Elementwise + reduce", "a, b = Tensor.empty(4, 8), Tensor.empty(4, 8)\nout = (a*b).sum(1)",
   lambda: (Tensor.empty(4, 8) * Tensor.empty(4, 8)).sum(1),
   "The multiply lives *inside* the reduce loop: the REDUCE's own range r1 indexes a and b directly. A reduce never forces its INPUT to be realized."),
  ("softmax", "Softmax", "x = Tensor.empty(4, 8)\nout = x.softmax(-1)",
   lambda: Tensor.empty(4, 8).softmax(-1),
   "Two reduces (max, then sum of exp) whose results are broadcast back over the row: each reduce result must be complete before it can be broadcast, so each is a kernel. exp(x-max) is a candidate too (two consumers), but it is cheap, so it is recomputed (in K1 for the sum and again in K2 for the output) instead of stored."),
  ("mm", "Matmul + bias + relu", "a, w, b = Tensor.empty(4, 8), Tensor.empty(8, 3), Tensor.empty(3)\nout = (a @ w + b).relu()",
   lambda: (Tensor.empty(4, 8) @ Tensor.empty(8, 3) + Tensor.empty(3)).relu(),
   "The matmul is a broadcast-multiply + REDUCE; the bias add and relu are elementwise on the reduce output and are applied at the STORE. One kernel."),
  ("xmean1", "x - x.mean(axis)", "x = Tensor.empty(4, 4)\nout = x - x.mean(1, keepdim=True)",
   lambda: (lambda x: x - x.mean(1, keepdim=True))(Tensor.empty(4, 4)),
   "The row mean is broadcast back over the row (ending ranges) -> realized: kernel 0 computes the 4 means, kernel 1 subtracts."),
  ("xmean", "x - x.mean() (full)", "x = Tensor.empty(4, 4)\nout = x - x.mean()",
   lambda: (lambda x: x - x.mean())(Tensor.empty(4, 4)),
   "Surprise: ONE kernel. The full mean has shape (), so it has no output ranges to realize; the reduce is simply evaluated inside the output kernel (and codegen hoists it out of the loop)."),
  ("contig", "contiguous() forces a split", "a = Tensor.empty(16)\nout = (a + 1).contiguous() * 2",
   lambda: (Tensor.empty(16) + 1).contiguous() * 2,
   "contiguous() is a STAGE; prepare_rangeify turns it into a STORE into a fresh ALLOC (stage_to_anon_store). A STORE is always realized and never removed: two kernels."),
  ("twice", "A tensor used twice", "y = Tensor.empty(8, 4) * 3\nout = y.sum(0) + y.max(0)",
   lambda: (lambda y: y.sum(0) + y.max(0))(Tensor.empty(8, 4) * 3),
   "y has two consumers that iterate it with different reduce ranges, so it is a realize candidate — but storing y*3 costs more than recomputing one multiply, so pm_remove_bufferize removes it: both reduces read the input and multiply again. One kernel."),
  ("assign", "assign (in place)", "c = Tensor.empty(8).realize()\nc.assign(c * 2 + 1)",
   None,
   "assign builds AFTER(c, STORE(c, c*2+1)). Every element only reads its own position, so one kernel reads and writes the same buffer (arg 1)."),
  ("assign_hz", "assign with a hazard", "c = Tensor.empty(8).realize()\nc.assign(c.flip(0) + 1)",
   None,
   "Element i reads element 7-i of the buffer it is overwriting: a write-after-read hazard. realize_store_after_src realizes the source first (dest is in the source's toposort), so the flipped values are written to a temporary first: two kernels."),
  ("copy", "COPY between devices", "a = Tensor.empty(8, device=\"CPU:1\")\nout = (a + 1).to(\"CPU\") * 2",
   lambda: ((Tensor.empty(8, device="CPU:1") + 1).to("CPU") * 2),
   "A COPY is an anonymous STORE into a buffer on the destination device (copy_to_anon_store). The cross-device STORE first needs its source materialized on CPU:1, and later becomes a CALL(STORE) — a bulk copy, not a compiled kernel."),
  ("bigsum", "2-stage split reduce", "x = Tensor.empty(65536)\nout = x.sum()",
   lambda: Tensor.empty(65536).sum(),
   "split_reduceop (prepare.py) sees a reduce of >= 32768 elements into few outputs and rewrites it as reshape(256, 256).sum(1).contiguous().sum(): the first kernel has 256 parallel outputs."),
  ("limit", "MAX_KERNEL_BUFFERS=4", "ts = [Tensor.empty(16) for _ in range(6)]\nout = ts[0]+ts[1]+ts[2]+ts[3]+ts[4]+ts[5]\n# run with MAX_KERNEL_BUFFERS=4",
   lambda: (lambda ts: ts[0] + ts[1] + ts[2] + ts[3] + ts[4] + ts[5])([Tensor.empty(16) for _ in range(6)]),
   "pm_limit_bufs counts the buffers each binary op reaches; past MAX_KERNEL_BUFFERS-1 inputs it inserts a fresh bufferize (stage()) on its elementwise sources. Devices like WEBGPU (8 bindings) need this; on CPU it is off unless you set the variable."),
]

cases = []
for key, title, code, make, why in CASES:
  ctx = {"MAX_KERNEL_BUFFERS": 4} if key == "limit" else None
  if key == "assign":
    c = realized(8)
    make = lambda c=c: c.assign(c * 2 + 1)
  if key == "assign_hz":
    c = realized(8)
    make = lambda c=c: c.assign(c.flip(0) + 1)
  info = analyze(make, ctx)
  info.update({"key": key, "title": title, "code": code, "why": why})
  if info["n_roots"] != info["n_kernels"]: print(f"  NOTE {key}: {info['n_roots']} roots vs {info['n_kernels']} kernels")
  print(f"{key:10s} {info['n_kernels']} kernel(s): {[k['name'] for k in info['kernels']]}")
  cases.append(info)
dump({"cases": cases}, "ch11_cases.json")

# ---- the "predict the kernel count" game: only counts + kernel names
GAME = [
  ("(a + b) * c", lambda: (Tensor.empty(16) + Tensor.empty(16)) * Tensor.empty(16), "elementwise only"),
  ("x.sum(1).sum()", lambda: Tensor.empty(4, 8).sum(1).sum(), "a reduce feeding a reduce: the inner REDUCE is consumed without broadcast, its output ranges are the outer reduce's range"),
  ("(a @ b).max(1)", lambda: (Tensor.empty(4, 8) @ Tensor.empty(8, 3)).max(1), "a reduce of a reduce, again no broadcast in between"),
  ("x / x.sum(1, keepdim=True)", lambda: (lambda x: x / x.sum(1, keepdim=True))(Tensor.empty(4, 8)), "the row sum is broadcast back"),
  ("x.layernorm()", lambda: Tensor.empty(4, 8).layernorm(), "mean (broadcast) -> var (broadcast) -> output"),
  ("x.std()", lambda: Tensor.empty(4, 8).std(), "the mean is a scalar but std is a scalar too... the mean is broadcast over a 32-element reduce"),
  ("x.argmax(1)", lambda: Tensor.empty(4, 8).argmax(1), "max is broadcast back to compare with every element"),
  ("x.T.contiguous()", lambda: Tensor.empty(4, 8).T.contiguous(), "contiguous of a view is a STORE of a permuted read"),
  ("x.cat(x + 1)", lambda: (lambda x: x.cat(x + 1))(Tensor.empty(4, 8)), "PAD + ADD: two padded halves summed"),
  ("(x > 0).where(x, 0)", lambda: (lambda x: (x > 0).where(x, 0))(Tensor.empty(4, 8)), "elementwise"),
  ("conv2d 1x2x6x6 * 3x2x3x3", lambda: Tensor.empty(1, 2, 6, 6).conv2d(Tensor.empty(3, 2, 3, 3)), "the im2col-style pad/reshape/expand view folds into index math"),
  ("Tensor.empty(8).cumsum()", lambda: Tensor.empty(8).cumsum(), "cumsum is a pad+reduce view trick (a reduce over a triangular window)"),
  ("Tensor.empty(1024,1024).sum()", lambda: Tensor.empty(1024, 1024).sum(), "a big reduce into one output -> split_reduceop"),
  ("Tensor.empty(1024,1024).sum(0)", lambda: Tensor.empty(1024, 1024).sum(0), "big reduce, but already 1024 outputs: no split"),
  ("two outputs: (x+1, (x*2).sum())", lambda: (lambda x: ((x + 1), (x * 2).sum()))(Tensor.empty(4, 8)), "two STOREs = two kernels (different loop nests)"),
  ("2-layer MLP (no bias)", lambda: (Tensor.empty(2, 8) @ Tensor.empty(8, 16)).relu() @ Tensor.empty(16, 4), "the hidden activation is consumed by a reduce that broadcasts it over 4 outputs"),
]
game = []
for code, make, hint in GAME:
  with Context(SCACHE=0):
    out = make()
    out = out if isinstance(out, tuple) else (out,)
    linear, _ = Tensor.linear_with_vars(*out)
  compiled = cap.compile_linear(linear)
  names = [n for n, _ in cap.kernel_sources(compiled)]
  print(f"game: {code:36s} -> {len(names)} {names}")
  game.append({"code": code, "n": len(names), "names": names, "hint": hint})
dump({"questions": game}, "ch11_game.json")
