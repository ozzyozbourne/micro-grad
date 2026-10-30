#!/usr/bin/env python3
"""End-to-end pipeline tracer for the tinygrad study guide (chapter 22).

Give it a small tinygrad program, get back ONE JSON document describing every stage the program goes through:

  lazy tensor graph -> scheduling rewrite passes -> schedule (LINEAR of CALLs) -> per kernel: AST, applied opts,
  every codegen pass of full_rewrite_to_sink, linearized UOp list, rendered C source, estimates -> run: timings + values.

Snippet convention
  * `Tensor`, `dtypes`, `nn`, `UOp`, `Ops` (and `np` if numpy is installed) are pre-imported.
  * The snippet assigns the variable `out` (a Tensor, or a list/tuple/dict of Tensors). If it doesn't, and the last
    statement is an expression, that expression becomes `out`.
  * Anything realized *while the snippet runs* (e.g. `x = Tensor([...]).realize()`) is the "prelude"; the trace focuses
    on the single `realize()` of `out` performed by the tracer afterwards.

How the capture works (all of it is tinygrad's own tracking machinery, nothing is re-implemented):
  * The worker process is started with TRACK_MATCH_STATS=2 (what VIZ=1 sets). At import time tinygrad.uop.ops then swaps
    PatternMatcher for TrackedPatternMatcher, and every `@rewrite_group`-wrapped call records into
    `ops.tracked_keys` / `ops.tracked_ctxs`: one group per top-level step (the schedule, each kernel's do_to_program,
    HCQ compile/link), one TrackedGraphRewrite per graph_rewrite (name, call site, depth, every match as
    (before trace_num, after trace_num, UPat location, time)).
  * TrackedGraphRewrite only stores the *input* sink (as a trace_num); to also get the output graph we wrap
    RewriteContext.unified_rewrite / walk_rewrite and remember (input UOp, output UOp) for the TrackedGraphRewrite that is
    active (ops.active_rewrites[-1]) when the rewrite starts.
  * PARALLEL=0 keeps kernel lowering in-process (the worker pool disables tracking).
  * Per-call timings come from wrapping tinygrad.engine.realize.track_stats (the function that prints DEBUG=2 lines).

Usage
  python3 tools/trace.py snippet.py [--noopt] [--beam N] [--out trace.json]
  echo 'out = Tensor([1.,2,3]).exp()' | python3 tools/trace.py -
  from trace import run_trace; run_trace(code, {"noopt": True})       # what server.py does
"""
import sys, os, json, subprocess, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
TINYGRAD_ROOT = "/home/user/workspace/tinygrad"

# ---------------------------------------------------------------------------------------------------------------------
# parent side: spawn a worker with the right environment and a timeout
# ---------------------------------------------------------------------------------------------------------------------

def run_trace(code: str, opts: dict | None = None, timeout: float = 60.0) -> dict:
  opts = opts or {}
  env = {k: v for k, v in os.environ.items() if k not in ("VIZ", "DEBUG", "PROFILE", "BEAM", "NOOPT", "TRACK_MATCH_STATS")}
  beam = max(0, min(int(opts.get("beam", 0) or 0), 4))
  env.update(DEV="CPU", TRACK_MATCH_STATS="2", PARALLEL="0", NO_COLOR="1", PYTHONUNBUFFERED="1",
             NOOPT="1" if opts.get("noopt") else "0", BEAM=str(beam), PYTHONPATH=TINYGRAD_ROOT)
  t0 = time.time()
  with tempfile.TemporaryDirectory(prefix="tgtrace_") as td:
    inp, outp = os.path.join(td, "req.json"), os.path.join(td, "trace.json")
    with open(inp, "w") as f: json.dump({"code": code, "opts": opts}, f)
    try:
      p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", inp, outp], cwd=td, env=env,
                         capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
      return {"ok": False, "error": f"timeout: the trace took longer than {timeout:.0f} s", "stderr": (e.stderr or "")[-4000:] if isinstance(e.stderr, str) else ""}
    if not os.path.exists(outp):
      return {"ok": False, "error": f"worker crashed (exit code {p.returncode})", "stderr": p.stderr[-6000:], "stdout": p.stdout[-3000:]}
    with open(outp) as f: ret = json.load(f)
  ret["wall_s"] = round(time.time() - t0, 3)
  if p.stderr.strip(): ret["stderr"] = p.stderr[-3000:]
  return ret

# ---------------------------------------------------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------------------------------------------------

MAX_GRAPH_NODES = 1500     # per serialized graph (keeps the nodes closest to the root, like uop2json)
MAX_TABLE_NODES = 25000    # total unique nodes in one trace
MAX_PASSES = 160           # per group
MAX_IDS_PER_RULE = 40
MAX_VALUES = 256

def worker(inp_path: str, out_path: str):
  import ast, io, traceback, contextlib, atexit, re
  req = json.load(open(inp_path))
  code, opts = req["code"], req.get("opts", {})
  result: dict = {"ok": False, "version": 1, "code": code, "opts": opts,
                  "env": {"DEV": "CPU", "NOOPT": os.environ.get("NOOPT"), "BEAM": os.environ.get("BEAM")}}
  stdout_buf = io.StringIO()
  stage = "import"
  try:
    from tinygrad import Tensor, dtypes, nn, Device
    from tinygrad.uop import ops as O
    from tinygrad.uop.ops import UOp, Ops, KernelInfo
    from tinygrad.uop.render import _render_arg, render_uir
    from tinygrad.helpers import printable, GlobalCounters, perf_counter_us
    from tinygrad.renderer import Renderer
    import tinygrad.engine.realize as R
    from tinygrad.codegen import to_program_cache, to_program_key
    atexit.unregister(O.print_match_stats)   # don't write rewrites.pkl / print stats at exit
    assert O.TRACK_MATCH_STATS.value >= 2, "worker must run with TRACK_MATCH_STATS=2"

    # ---- hook: remember (input, output) of every tracked graph_rewrite ----
    # NOTE: untracked rewrites (substitute()/simplify() without a name run under TRACK_MATCH_STATS=0) also go through
    # RewriteContext while an outer tracked rewrite is active, so only the *first* (outermost) call per ctx is recorded.
    io_map: dict[int, tuple] = {}
    started: set[int] = set()
    def wrap(meth):
      orig = getattr(O.RewriteContext, meth)
      def f(self, root):
        ctx = O.active_rewrites[-1] if O.active_rewrites else None
        mine = ctx is not None and id(ctx) not in started
        if mine: started.add(id(ctx))
        ret = orig(self, root)
        if mine: io_map[id(ctx)] = (ctx, root, ret)
        return ret
      setattr(O.RewriteContext, meth, f)
    wrap("unified_rewrite"); wrap("walk_rewrite")

    # ---- hook: per-call runtime stats ----
    runs: list[dict] = []
    orig_track = R.track_stats
    def track_stats(ctx, call, st, ets):
      wall_us = float(perf_counter_us() - st)
      try:
        body = call.body
        kind = "kernel" if body.op is Ops.PROGRAM else ("copy" if body.op is Ops.STORE else body.op.name.lower())
        name = body.src[0].arg.name if body.op is Ops.PROGRAM else (f"copy {call.src[1].device} <- {call.src[2].device}" if kind == "copy" else kind)
        est = R.estimate_uop(call)
        runs.append({"kind": kind, "name": name, "key": body.key.hex() if body.op is Ops.PROGRAM else None,
                     "time_us": round(ets[0]*1e6, 2) if ets and ets[0] is not None else None, "wall_us": round(wall_us, 2),
                     "ops": _int(est.ops), "mem": _int(est.mem), "lds": _int(est.lds)})
      except Exception as e: runs.append({"kind": "?", "name": f"<stats error {e}>"})
      return orig_track(ctx, call, st, ets)
    R.track_stats = track_stats

    # ---- serialization: one global node table, graphs are just root ids ----
    ids: dict = {}
    table: dict[str, dict] = {}
    tn2id: dict[int, str] = {}
    truncated = {"graphs": 0, "table": False}
    def nid(u): return ids.setdefault(u, f"n{len(ids)}")
    def node_info(u) -> dict:
      try: dt = str(u.dtype).replace("dtypes.", "")
      except Exception: dt = "?"
      try: sh = str(u._shape) if u._shape is not None else ""
      except Exception: sh = ""
      try: arg = _render_arg(u)
      except Exception: arg = str(u.arg)
      arg = arg if len(arg) <= 90 else arg[:89] + "…"
      d = {"op": u.op.name, "dt": dt, "src": [nid(s) for s in u.src]}
      if arg: d["arg"] = arg
      if sh: d["sh"] = sh
      return d
    def ser(u) -> dict | None:
      if u is None: return None
      if not isinstance(u, UOp): return None
      topo = list(u.toposort(enter_calls=True))
      n = len(topo)
      if n > MAX_GRAPH_NODES: topo, truncated["graphs"] = topo[-MAX_GRAPH_NODES:], truncated["graphs"] + 1
      for x in topo:
        i = nid(x)
        if i not in table:
          if len(table) >= MAX_TABLE_NODES: truncated["table"] = True; break
          table[i] = node_info(x)
        if (tn:=x.__dict__.get("trace_num")) is not None: tn2id[tn] = i
      return {"root": nid(u), "n": n}

    def ser_pass(ctx) -> dict:
      rec = {"name": ctx.name, "depth": ctx.depth, "loc": f"{_rel(ctx.loc[0])}:{ctx.loc[1]}", "bottom_up": ctx.bottom_up,
             "walk": ctx.walk, "matches": len(ctx.matches)}
      io_ = io_map.get(id(ctx))
      if io_ is not None:
        _, root, ret = io_
        rec["in"] = ser(root)
        rec["out"] = ser(ret) if ret is not root else rec["in"]
        rec["noop"] = ret is root
      rules: dict = {}
      for before, after, loc, et in ctx.matches:
        k = f"{_rel(loc[0])}:{loc[1]}"
        r = rules.setdefault(k, {"loc": k, "code": printable(loc)[:160], "count": 0, "us": 0.0, "before": [], "after": []})
        r["count"] += 1; r["us"] += et*1e6
        if len(r["before"]) < MAX_IDS_PER_RULE and before in tn2id: r["before"].append(tn2id[before])
        if len(r["after"]) < MAX_IDS_PER_RULE and after in tn2id: r["after"].append(tn2id[after])
      for r in rules.values(): r["us"] = round(r["us"], 1)
      rec["rules"] = sorted(rules.values(), key=lambda r: -r["count"])
      return rec

    def ser_group(gi: int) -> dict:
      key, ctxs = O.tracked_keys[gi], O.tracked_ctxs[gi]
      kind = "kernel" if isinstance(key.ret, Renderer) else ("runtime" if "HCQ" in key.display_name else "schedule")
      g = {"name": key.display_name, "kind": kind, "n_passes": len(ctxs), "passes": []}
      if kind == "kernel": g["fn"], g["prg_key"] = key.keys[1], key.keys[0].hex()
      for ctx in ctxs[:MAX_PASSES]: g["passes"].append(ser_pass(ctx))
      g["truncated"] = len(ctxs) > MAX_PASSES
      return g

    # ---- 1. run the snippet ----
    stage = "snippet"
    ns: dict = {"Tensor": Tensor, "dtypes": dtypes, "nn": nn, "UOp": UOp, "Ops": Ops, "Device": Device, "__name__": "__snippet__"}
    try:
      import numpy as np
      ns["np"] = np
    except ImportError: pass
    tree = ast.parse(code, "<snippet>")
    assigns_out = any(isinstance(n, ast.Name) and n.id == "out" and isinstance(n.ctx, ast.Store) for n in ast.walk(tree))
    if not assigns_out and tree.body and isinstance(tree.body[-1], ast.Expr):
      last = tree.body[-1]
      tree.body[-1] = ast.Assign(targets=[ast.Name("out", ast.Store())], value=last.value, lineno=last.lineno, col_offset=0)
      ast.fix_missing_locations(tree)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(stdout_buf): exec(compile(tree, "<snippet>", "exec"), ns)
    result["snippet_s"] = round(time.perf_counter() - t0, 4)
    if "out" not in ns: raise RuntimeError("the snippet must assign `out` (a Tensor or list/tuple/dict of Tensors), or end with an expression")
    out = ns["out"]
    if isinstance(out, dict): names, outs = list(map(str, out.keys())), list(out.values())
    elif isinstance(out, (list, tuple)): names, outs = [f"out[{i}]" for i in range(len(out))], list(out)
    else: names, outs = ["out"], [out]
    if not outs or not all(isinstance(t, Tensor) for t in outs): raise TypeError(f"`out` must be Tensor(s), got {type(out).__name__}")
    prelude_groups = len(O.tracked_keys)
    result["prelude"] = [{"name": O.tracked_keys[i].display_name, "n_passes": len(O.tracked_ctxs[i])} for i in range(prelude_groups)]
    result["outputs_meta"] = [{"name": n, "shape": list(t.shape), "dtype": str(t.dtype).replace("dtypes.", ""), "device": str(t.device)} for n, t in zip(names, outs)]

    # ---- 2. the lazy tensor graph ----
    stage = "tensor graph"
    tsink = UOp.sink(*[t.uop for t in outs])
    result["tensor_graph"] = ser(tsink)
    result["tensor_graph_text"] = _cap_text(_strip(render_uir(tsink)))

    # ---- 3. schedule (what Tensor.realize does, split into its two halves) ----
    stage = "schedule"
    to_realize = [t for t in outs if t.uop.base.needs_storage()]
    mark = len(O.tracked_keys)
    t0 = time.perf_counter()
    if to_realize:
      linear, var_vals = Tensor.linear_with_vars(*to_realize)
    else:
      linear, var_vals = UOp(Ops.LINEAR, src=()), {}
    result["schedule_s"] = round(time.perf_counter() - t0, 4)
    sched_groups_end = len(O.tracked_keys)
    result["linear"] = ser(linear)
    renderer = Device["CPU"].renderer
    calls = []
    for i, call in enumerate(linear.src):
      c = call.without_after if hasattr(call, "without_after") else call
      body = c.body
      ent: dict = {"i": i, "call": nid(c), "body_op": body.op.name}
      if body.op is Ops.SINK and isinstance(body.arg, KernelInfo):
        ent.update(kind="kernel", name=str(getattr(c.arg, "name", None) or body.arg.name), ast=ser(body), ast_text=_cap_text(_strip(render_uir(body))))
        try: ent["prg_key"] = to_program_key(body, renderer)
        except Exception: pass
      elif body.op is Ops.STORE: ent.update(kind="copy", name="copy")
      else: ent.update(kind=body.op.name.lower(), name=body.op.name.lower(), ast=ser(body))
      bufs = []
      for s in c.src[1:]:
        b = {"id": nid(s), "op": s.op.name, "dtype": str(s.dtype).replace("dtypes.", "")}
        try: b["numel"] = int(s.max_numel())
        except Exception: pass
        try: b["device"] = str(s.device)
        except Exception: pass
        try: b["text"] = _strip(_render_arg(s))[:90]
        except Exception: pass
        bufs.append(b)
      ent["bufs"] = bufs
      calls.append(ent)

    # ---- 4. compile + run ----
    stage = "compile+run"
    GlobalCounters.reset()
    runs.clear()   # drop the prelude's calls
    t0 = time.perf_counter()
    if len(linear.src): R.run_linear(linear, var_vals, wait=True)
    result["run_s"] = round(time.perf_counter() - t0, 4)
    result["global_counters"] = {"kernel_count": GlobalCounters.kernel_count, "global_ops": _int(GlobalCounters.global_ops),
                                 "global_mem": _int(GlobalCounters.global_mem), "time_sum_us": round(GlobalCounters.time_sum_s*1e6, 2)}
    run_end = len(O.tracked_keys)

    # per kernel: the compiled PROGRAM (from tinygrad's to_program_cache)
    for ent in calls:
      if ent["kind"] != "kernel" or "prg_key" not in ent: continue
      prg = to_program_cache.get(ent.pop("prg_key"))
      if prg is None: continue
      ent["fn"] = ent["name"] = prg.src[0].arg.function_name
      ent["prg_hash"] = prg.key.hex()
      sink = prg.src[0]
      ent["applied_opts"] = [str(o) for o in sink.arg.applied_opts]
      if sink.arg.estimates is not None:
        e = sink.arg.estimates
        ent["estimates"] = {"ops": _int(e.ops), "lds": _int(e.lds), "mem": _int(e.mem)}
      gs, ls = prg.arg.global_size, prg.arg.local_size
      ent["global_size"], ent["local_size"] = [_int(x) for x in gs], [_int(x) for x in ls]
      ent["final_sink"] = ser(sink)
      lin = next((s for s in prg.src if s.op is Ops.LINEAR), None)
      if lin is not None:
        pos = {u: j for j, u in enumerate(lin.src)}
        rows = []
        for j, u in enumerate(lin.src):
          try: arg = _strip(_render_arg(u))
          except Exception: arg = str(u.arg)
          rows.append({"op": u.op.name, "dt": str(u.dtype).replace("dtypes.", ""), "src": [pos.get(s, -1) for s in u.src],
                       "arg": arg[:100], "id": nid(u)})
        ent["uops"] = rows
      srcu = next((s for s in prg.src if s.op is Ops.SOURCE), None)
      if srcu is not None: ent["source"] = srcu.arg
      binu = next((s for s in prg.src if s.op is Ops.BINARY), None)
      if binu is not None: ent["binary_bytes"] = len(binu.arg)

    # ---- 5. outputs ----
    stage = "outputs"
    vals = []
    for n, t in zip(names, outs):
      lst = t.tolist()
      flat = _flatten(lst)
      vals.append({"name": n, "shape": list(t.shape), "dtype": str(t.dtype).replace("dtypes.", ""), "values": _jsonable(flat[:MAX_VALUES]),
                   "truncated": len(flat) > MAX_VALUES, "text": _fmt_nested(lst)})
    result["outputs"] = vals

    # ---- 6. serialize the tracked rewrite groups ----
    stage = "serialize"
    result["schedule_groups"] = [ser_group(i) for i in range(mark, sched_groups_end)]
    run_groups = [ser_group(i) for i in range(sched_groups_end, run_end)]
    result["kernel_groups"] = [g for g in run_groups if g["kind"] == "kernel"]
    result["runtime_groups"] = [g for g in run_groups if g["kind"] != "kernel"]
    # attach each kernel group to its schedule entry
    for ent in calls:
      if ent["kind"] == "kernel":
        ent["group"] = next((k for k, g in enumerate(result["kernel_groups"]) if g.get("prg_key") == ent.get("prg_hash")), None)
    result["calls"] = calls
    result["runs"] = runs
    result["nodes"] = table
    result["truncated"] = truncated
    result["stdout"] = stdout_buf.getvalue()[-5000:]
    result["ok"] = True
  except BaseException as e:
    result["error"] = f"{type(e).__name__}: {e}"
    result["stage"] = stage
    result["traceback"] = traceback.format_exc()[-6000:]
    result["stdout"] = stdout_buf.getvalue()[-5000:]
    # best effort: still return the rewrite groups recorded before the failure (useful to see where it broke)
    try:
      lo = locals().get("mark", locals().get("prelude_groups", 0))
      result["partial_groups"] = [ser_group(i) for i in range(lo, len(O.tracked_keys))][:12]
      result["nodes"] = table
    except Exception: pass
  with open(out_path, "w") as f: json.dump(result, f, default=str, separators=(",", ":"))

def _rel(path: str) -> str:
  p = path.replace("\\", "/")
  i = p.rfind("/tinygrad/tinygrad/")
  if i >= 0: return p[i + len("/tinygrad/"):]
  return p.split("/")[-1]

def _int(x):
  try: return int(x)
  except Exception: return str(x)

def _strip(s: str) -> str:
  import re
  return re.sub(r"\x1b\[[0-9;]*m", "", s)

def _cap_text(s: str, n: int = 400) -> str:
  lines = s.split("\n")
  return s if len(lines) <= n else "\n".join(lines[:n] + [f"... ({len(lines)-n} more lines)"])

def _flatten(x):
  if isinstance(x, (list, tuple)): return [y for e in x for y in _flatten(e)]
  return [x]

def _fmt_nested(x, depth=0, budget=[0]) -> str:
  if depth == 0: budget[0] = 0
  if isinstance(x, (list, tuple)):
    if budget[0] > 400: return "..."
    inner = [_fmt_nested(e, depth+1) for e in x]
    if x and isinstance(x[0], (list, tuple)): return "[" + (",\n" + " "*(depth+1)).join(inner) + "]"
    return "[" + ", ".join(inner) + "]"
  budget[0] += 1
  return f"{x:.5g}" if isinstance(x, float) else str(x)

def _jsonable(v):
  import math
  if isinstance(v, list): return [_jsonable(x) for x in v]
  if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return str(v)
  return v

# ---------------------------------------------------------------------------------------------------------------------

def main():
  import argparse
  ap = argparse.ArgumentParser(description="trace a tinygrad snippet through the whole pipeline (JSON)")
  ap.add_argument("snippet", help="path to a .py snippet, or - for stdin")
  ap.add_argument("--noopt", action="store_true")
  ap.add_argument("--beam", type=int, default=0)
  ap.add_argument("--timeout", type=float, default=60)
  ap.add_argument("--out", help="write JSON here (default: stdout)")
  a = ap.parse_args()
  code = sys.stdin.read() if a.snippet == "-" else open(a.snippet).read()
  res = run_trace(code, {"noopt": a.noopt, "beam": a.beam}, a.timeout)
  s = json.dumps(res, separators=(",", ":"))
  if a.out:
    with open(a.out, "w") as f: f.write(s)
    print(f"{'ok' if res.get('ok') else 'ERROR'}: {len(s)/1024:.1f} KB -> {a.out}" + ("" if res.get("ok") else f"\n{res.get('error')}\n{res.get('traceback','')}"))
  else: print(s)
  sys.exit(0 if res.get("ok") else 1)

if __name__ == "__main__":
  if len(sys.argv) == 4 and sys.argv[1] == "--worker": worker(sys.argv[2], sys.argv[3])
  else: main()
