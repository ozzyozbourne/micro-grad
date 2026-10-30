# Chapter 21 — what does every DEBUG level print? Run the same program at DEBUG=0..8 and capture it.
# Run: DEV=CPU python3 examples/21_debug_levels.py
#
# Each level is a separate process (DEBUG is read once, at import, into a ContextVar in helpers.py).
# NO_COLOR=1 strips the ANSI colors so the text is easy to store; the chapter's slider replays data/ch21_debug_levels.json.
import json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(HERE, "21_debug_target.py")
levels = {}
KINDS = [  # (regex, short name) -- the same buckets the chapter's slider uses for its hover annotations
  (r"^compiling ", "tqdm"), (r"^opened device", "device"), (r"^loading ", "dlopen"), (r"^scheduled ", "schedule"),
  (r"^\*\*\* ", "kernel-run"), (r"^\S+ +opts: ", "opts"), (r"^%\d+ = ", "uir"), (r"^buffer: ", "alloc"),
  (r"^system: ", "system"), (r"^SELECT |^INSERT |^CREATE |^BEGIN|^COMMIT", "sqlite"), (r"^\[", "result")]
def kind(l:str) -> str: return next((n for r, n in KINDS if re.search(r, l)), "code/asm")
for d in range(0, 9):
  env = {**os.environ, "DEV": "CPU", "DEBUG": str(d), "NO_COLOR": "1"}
  p = subprocess.run([sys.executable, target], env=env, capture_output=True, text=True, cwd=HERE)
  # tqdm's "compiling" bar goes to stderr with carriage returns: keep only its final state
  err = "\n".join(l.split("\r")[-1] for l in p.stderr.splitlines() if l.strip())
  out = (err + "\n" if err else "") + p.stdout
  out = re.sub(r"pid:\d+", "pid:N", out).replace(HERE, "examples")
  levels[d] = out.rstrip("\n")
  kinds = {}
  for l in levels[d].splitlines(): kinds[kind(l)] = kinds.get(kind(l), 0) + 1
  print(f"DEBUG={d}: {len(levels[d].splitlines()):3d} lines  " + ", ".join(f"{k}:{v}" for k, v in kinds.items()))
json.dump(levels, open(os.path.join(HERE, "..", "data", "ch21_debug_levels.json"), "w"), indent=1)
print("wrote data/ch21_debug_levels.json")
