#!/usr/bin/env python3
"""Chapter 22: the end-to-end chain in your terminal.

Runs a tinygrad snippet through tools/trace.py (a subprocess with DEV=CPU TRACK_MATCH_STATS=2) and prints every stage:
  1. lazy tensor graph   2. scheduling rewrite passes   3. the schedule (LINEAR of CALLs)
  4. per kernel: AST, applied opts, codegen passes, linearized UOps, C source   5. run: timings + output values

Usage:
  python3 examples/22_trace_cli.py                       # default snippet (a*b+1).sum()
  python3 examples/22_trace_cli.py -c "out = Tensor([1.,2,3]).exp2()"
  python3 examples/22_trace_cli.py my_snippet.py --noopt
  python3 examples/22_trace_cli.py --preset softmax      # any id from data/ch22_presets.json
  add --full to print every pass (default hides passes that matched nothing) and the whole AST / uop list
"""
import os, sys, json, argparse, importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("tg_trace", os.path.join(ROOT, "tools", "trace.py"))
tg_trace = importlib.util.module_from_spec(spec); spec.loader.exec_module(tg_trace)

DEFAULT = """\
a = Tensor([1., 2, 3, 4]).realize()
b = Tensor([5., 6, 7, 8]).realize()
out = (a * b + 1).sum()
"""

def hr(title): print(f"\n{'═'*100}\n  {title}\n{'═'*100}")

def head(text, n):
  lines = text.rstrip("\n").split("\n")
  return "\n".join(lines[:n] + ([f"  ... ({len(lines)-n} more lines, use --full)"] if len(lines) > n else []))

def print_passes(group, full):
  hidden = 0
  print(f"  {'pass':44s} {'matches':>7s} {'nodes in→out':>13s}  call site / top rules")
  for p in group["passes"]:
    if not full and p["matches"] == 0 and p.get("noop", True):
      hidden += 1; continue
    io = f"{p['in']['n']}→{p['out']['n']}" if "in" in p else "(group)"
    print(f"  {'  '*p['depth'] + p['name']:44.44s} {p['matches']:7d} {io:>13s}  {p['loc']}")
    for r in p["rules"][:2 if not full else 99]: print(f"  {'':44s} {'':7s} {'':13s}    ×{r['count']:<3d} {r['loc']}: {r['code'][:70]}")
  if hidden: print(f"  ({hidden} passes matched nothing and are hidden; --full shows them)")

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("snippet", nargs="?")
  ap.add_argument("-c", dest="code")
  ap.add_argument("--preset")
  ap.add_argument("--noopt", action="store_true")
  ap.add_argument("--beam", type=int, default=0)
  ap.add_argument("--full", action="store_true")
  a = ap.parse_args()
  if a.code: code = a.code
  elif a.preset: code = next(p["code"] for p in json.load(open(os.path.join(ROOT, "data", "ch22_presets.json")))["presets"] if p["id"] == a.preset)
  elif a.snippet: code = open(a.snippet).read()
  else: code = DEFAULT

  hr("0. the program")
  print(code.rstrip())
  t = tg_trace.run_trace(code, {"noopt": a.noopt, "beam": a.beam})
  if not t.get("ok"):
    print(f"\nTRACE FAILED in stage {t.get('stage')}: {t.get('error')}\n{t.get('traceback', t.get('stderr', ''))}"); sys.exit(1)
  if t["prelude"]: print(f"\n  (prelude: the snippet itself realized {len(t['prelude'])} group(s): {', '.join(g['name'] for g in t['prelude'])})")

  hr(f"1. lazy tensor graph — {t['tensor_graph']['n']} UOps, nothing computed yet  (render_uir of UOp.sink(*outs))")
  print(head(t["tensor_graph_text"], 60 if not a.full else 10**6))

  hr("2. scheduling: every tracked graph_rewrite inside Tensor.linear_with_vars")
  for g in t["schedule_groups"]:
    print(f"\n  [{g['name']}]  {g['n_passes']} rewrites")
    print_passes(g, a.full)

  hr(f"3. the schedule: LINEAR with {len(t['calls'])} CALL(s), in execution order")
  for c in t["calls"]:
    bufs = ", ".join(f"{b['op']}({b.get('dtype')}, {b.get('numel', '?')}, {b.get('device', '?')})" for b in c["bufs"])
    print(f"  #{c['i']}  {c['kind']:6s} {c.get('fn', c['name']):20s} args: {bufs}")

  for k, c in enumerate(c for c in t["calls"] if c["kind"] == "kernel"):
    hr(f"4.{k+1} kernel {c.get('fn')}  —  global {c.get('global_size')} local {c.get('local_size')}  opts {c.get('applied_opts')}")
    print("  ── kernel AST (input to codegen) ──")
    print(head(c["ast_text"], 40 if not a.full else 10**6))
    if c.get("group") is not None:
      g = t["kernel_groups"][c["group"]]
      print(f"\n  ── codegen passes: [{g['name']}] {g['n_passes']} rewrites (do_to_program → full_rewrite_to_sink → linearize/render) ──")
      print_passes(g, a.full)
    print(f"\n  ── linearized: {len(c.get('uops', []))} UOps in program order ──")
    rows = c.get("uops", [])
    for j, u in enumerate(rows if a.full else rows[:45]):
      print(f"  {j:4d} {u['op']:12s} {u['dt']:14.14s} {str(u['src']):18s} {u['arg'][:50]}")
    if not a.full and len(rows) > 45: print(f"  ... ({len(rows)-45} more, use --full)")
    print("\n  ── rendered C source (ClangRenderer) ──")
    print(c.get("source", "").rstrip())
    if "estimates" in c: print(f"\n  estimates: {c['estimates']}   binary: {c.get('binary_bytes')} bytes")

  hr("5. run (run_linear, wait=True)")
  for r in t["runs"]:
    tm = f"{r['time_us']:9.2f} us (kernel)" if r.get("time_us") is not None else f"{r['wall_us']:9.2f} us (wall)"
    print(f"  {r['kind']:6s} {r['name']:28s} {tm}   ops {r['ops']}  mem {r['mem']} B")
  for o in t["outputs"]: print(f"\n  {o['name']} {o['dtype']} {tuple(o['shape'])} =\n" + "\n".join("    " + l for l in o["text"].split("\n")))
  print(f"\n  wall time of the whole trace: {t['wall_s']} s")

if __name__ == "__main__": main()
