# Chapter 4 data capture (writes data/ch04_*.json for the interactive demos).
# Run with:  DEV=CPU python3 examples/04_capture_data.py   (it sets TRACK_MATCH_STATS=2 itself, before tinygrad is imported)
# TRACK_MATCH_STATS=2 makes every graph_rewrite record its input graph + every match (this is what VIZ uses),
# so we can see exactly which Ops appear at which stage of the compiler for a corpus of small programs.
import os, sys, json, re, pathlib
os.environ["TRACK_MATCH_STATS"] = "2"          # must be set before `import tinygrad`: ContextVars read the env at import
os.environ.setdefault("PRINT_MATCH_STATS", "0")  # don't print the big match-stats table at exit
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
from uop2json import uop_to_graph, dump
from tinygrad import Tensor, Device, Context, UOp, dtypes
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.ops import tracked_keys, tracked_ctxs, uop_fields
from tinygrad.uop.spec import spec_tensor, spec_program, spec_kernel_graph
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.engine.realize import run_linear
DATA = HERE.parent / "data"

# ---------------- 1. the same IR at three altitudes ----------------
a = Tensor([[1., 2, 3], [4, 5, 6]]).realize()
b = Tensor([10., 20, 30]).realize()
c = (a * 2 + b).sum(axis=1)
ids: dict = {}
tensor_graph = uop_to_graph(c.uop, ids)
tensor_uir = c.uop.render_uir()
linear = c.schedule_linear()
ast = linear.src[-1].src[0]
ast_graph = uop_to_graph(ast, ids)
with Context(NOOPT=1): prg = to_program(ast, Device["CPU"].renderer)
lin = list(prg.src[1].src)
prog_graph = uop_to_graph(lin[-1], ids)
order = {u: i for i, u in enumerate(lin)}
prog_list = [{"i": i, "op": u.op.name, "dtype": u.dtype.name, "src": [order[s] for s in u.src if s in order],
              "arg": "" if u.arg is None else (u.arg.name if u.op is Ops.SINK else str(u.arg))} for i, u in enumerate(lin)]
# give each program node its position in the linear list as a label suffix
for n in prog_graph["nodes"]:
  u = next(k for k, v in ids.items() if v == n["id"])
  if u in order: n["sub"] = f"%{order[u]} {u.dtype.name}" + (f"  {n['arg']}" if n["arg"] not in ("None", "") and u.op not in (Ops.PARAM, Ops.SINK, Ops.BUFFER) else "")
run_linear(linear)
dump({"tensor": tensor_graph, "tensor_uir": tensor_uir, "ast": ast_graph, "ast_uir": ast.render_uir(),
      "program": prog_graph, "program_list": prog_list, "c_source": prg.src[2].arg, "result": c.tolist()},
     DATA / "ch04_altitudes.json")

# ---------------- 2. a small graph for the toposort animation ----------------
x, y = UOp.variable("x", 0, 10), UOp.variable("y", 0, 10)
s = (x + y) * (x + y) + x * 2
tids: dict = {}
tg = uop_to_graph(s, tids)
for n in tg["nodes"]:
  u = next(k for k, v in tids.items() if v == n["id"])
  n["label"] = u.op.name
  n["sub"] = u.render(simplify=False) if u.op is not Ops.STACK else "() shape"
dump({"graph": tg, "root": tids[s], "order": [tids[u] for u in s.toposort()],
      "src_order": {tids[u]: [tids[q] for q in u.src] for u in s.toposort()}}, DATA / "ch04_toposort.json")

# ---------------- 3. a corpus of programs: which Ops appear at which stage? ----------------
def corpus():
  t = Tensor([float(i) for i in range(12)]).reshape(3, 4).realize()   # NOTE: Tensor.arange is deviceless (virtual) until used with a device tensor
  m = Tensor.ones(4, 4).contiguous().realize()
  I10 = Tensor(list(range(10))).realize()
  yield "elementwise", lambda: (t * 2 + 1).realize()
  yield "reduce", lambda: t.sum(axis=1).realize()
  yield "matmul", lambda: (t @ m).realize()
  yield "conv2d", lambda: Tensor.ones(1, 1, 5, 5).contiguous().conv2d(Tensor.ones(2, 1, 3, 3).contiguous()).realize()
  yield "pad/shrink/flip", lambda: t.pad(((1, 1), (0, 2)))[1:3, ::-1].contiguous().realize()
  yield "cast/bitcast", lambda: (t.cast(dtypes.int32).bitcast(dtypes.uint32) >> 1).realize()
  yield "where/cmp", lambda: ((t > 3).where(t, -t) != 2).realize()
  yield "transcendentals", lambda: (t.exp2() + t.log2() + t.sin() + t.sqrt() + t.reciprocal()).realize()
  yield "int div/mod", lambda: (I10 // 3 + I10 % 3).realize()
  yield "bitwise", lambda: ((I10 ^ 5) | (I10 & 3)).realize()
  yield "rand", lambda: Tensor.rand(4).realize()
  yield "argmax/cumsum", lambda: (t.argmax(axis=1).float().sum() + t.cumsum(axis=1).sum()).realize()
  yield "gather", lambda: t[Tensor([2, 0, 1])].realize()
  def symbolic():
    v = UOp.variable("n", 1, 10).bind(3)
    return (Tensor.ones(10).contiguous().realize()[:v] * 2).sum().realize()
  yield "symbolic shape", symbolic
  def grad():
    w = Tensor.ones(3, 3).contiguous().realize()
    (w * w).sum().backward()
    return w.grad.realize()
  yield "backward", grad
  yield "assign", lambda: m.assign(m + 1).realize()
  yield "multi-device", lambda: (Tensor([float(i) for i in range(8)]).reshape(4, 2).shard(("CPU:0", "CPU:1"), axis=0) * 2).sum().realize()
  yield "copy to device", lambda: Tensor([1., 2, 3], device="PYTHON").to("CPU").realize()
  yield "detach/contiguous_backward", lambda: (t.detach() * 2 + t.contiguous_backward()).realize()
  yield "pow", lambda: (t ** 2.5 + 2 ** t).realize()

phases = {"tensor": set(), "schedule": set(), "ast": set(), "codegen": set(), "program": set(), "runtime": set()}
def ops_of(num, out):
  stack, seen = [num], set()
  while stack:
    n = stack.pop()
    if n in seen: continue
    seen.add(n); op, src, _ = uop_fields[n]; out.add(op.name); stack.extend(src)
corpus_names = []
for name, fn in corpus():
  corpus_names.append(name)
  fn()
for k, ctxs in zip(tracked_keys, tracked_ctxs):
  kname = k.display_name
  kind = "schedule" if kname.startswith("Schedule") else "runtime" if kname.startswith(("HCQ", "default")) else "codegen"
  for j, r in enumerate(ctxs):
    ops_of(r.sink, phases[kind])
    for before, after, *_ in r.matches: ops_of(after, phases[kind])
    if kind == "codegen" and j == 0: ops_of(r.sink, phases["ast"])
    if kind == "schedule" and r.name == "drop after": ops_of(r.sink, phases["tensor"])
for prg in to_program_cache.values():
  for u in prg.src[1].src: phases["program"].add(u.op.name)

# ---------------- 4. the Ops atlas: every member of the enum ----------------
import inspect, tinygrad.uop as tu
text = inspect.getsource(tu).splitlines()
section, comment, op_info = "", [], {}
for line in text:
  st = line.strip()
  if st.startswith("class GroupOp"): break
  if m := re.match(r"# \*\* (\d+) -- (.*) \*\*", st): section, comment = f"{m.group(1)} · {m.group(2)}", []; continue
  if st.startswith("#"): comment.append(st[1:].strip()); continue
  if not st: comment = []; continue
  for nm in re.findall(r"([A-Z_0-9]+) = auto\(\)", st):
    trailing = st.split("#", 1)[1].strip() if "#" in st else ""
    op_info[nm] = {"section": section, "comment": " ".join(comment + ([trailing] if trailing else []))}
  if re.search(r"= auto\(\)", st): comment = []   # a comment block describes only the op line right after it
groups = [g for g in dir(GroupOp) if not g.startswith("_") and g != "All"]
atlas = []
for op in Ops:
  atlas.append({"name": op.name, "value": op.value, **op_info.get(op.name, {}),
                "groups": [g for g in groups if op in getattr(GroupOp, g)],
                "spec_tensor": op in spec_tensor.pdict, "spec_program": op in spec_program.pdict,
                "spec_kernel_graph": op in spec_kernel_graph.pdict,
                "seen": [p for p, s in phases.items() if op.name in s]})
dump({"ops": atlas, "corpus": corpus_names}, DATA / "ch04_ops_atlas.json")
print(f"captured {len(atlas)} ops; seen per phase:", {k: len(v) for k, v in phases.items()})
print("never seen in the corpus:", [o["name"] for o in atlas if not o["seen"]])
