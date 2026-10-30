# Chapter 21 — VIZ=-1 (save traces, no web server) + `python -m tinygrad.viz.cli` to read them in a terminal
# Run: DEV=CPU python3 examples/21_viz_cli.py
#
# VIZ=-1 makes helpers.PROFILE and uop/ops.TRACK_MATCH_STATS=2 on, and at exit pickles
#   rewrites.pkl.<user> (every graph_rewrite + every pattern match) and profile.pkl.<user> (timeline events)
# into tempfile.gettempdir(). We point TMPDIR at a private directory so concurrent runs don't overwrite each other.
import os, re, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp(prefix="ch21viz_")
env = {**os.environ, "DEV": "CPU", "NO_COLOR": "1", "TMPDIR": tmp}

def run(title:str, args:list[str], extra_env:dict|None=None, max_lines=40, grep:str|None=None):
  p = subprocess.run(args, env={**env, **(extra_env or {})}, capture_output=True, text=True, cwd=HERE)
  out = (p.stdout + p.stderr).replace(tmp, "$TMPDIR")
  lines = [l for l in out.splitlines() if grep is None or re.search(grep, l)]
  print(f"$ {' '.join((f'{k}={v} ' for k, v in (extra_env or {}).items()))}{title}")
  print("\n".join(lines[:max_lines]) + (f"\n... ({len(lines) - max_lines} more lines)" if len(lines) > max_lines else ""))
  print()

cli = [sys.executable, "-m", "tinygrad.viz.cli"]
run("python3 21_debug_target.py", [sys.executable, "21_debug_target.py"], {"VIZ": "-1"})
run("python -m tinygrad.viz.cli --list", cli + ["--list"])
run("python -m tinygrad.viz.cli -t 8", cli + ["-t", "8"])
run("python -m tinygrad.viz.cli -s TINY   (first 24 events)", cli + ["-s", "TINY"], max_lines=24)
run("python -m tinygrad.viz.cli -s CPU", cli + ["-s", "CPU"])
run("python -m tinygrad.viz.cli -s CPU   (DEBUG=5 adds every rewrite step, '- N' = matches)", cli + ["-s", "CPU"], {"DEBUG": "5"}, max_lines=80)
run("python -m tinygrad.viz.cli -q 'kernel split'", cli + ["-q", "kernel split"], max_lines=30)
run("python -m tinygrad.viz.cli -s CPU r_2_4 'postopt symbolic'   (the matches of one rewrite)", cli + ["-s", "CPU", "r_2_4", "postopt symbolic"], max_lines=40)
run("python -m tinygrad.viz.cli --json -s CPU", cli + ["--json", "-s", "CPU"], max_lines=3)
shutil.rmtree(tmp)
