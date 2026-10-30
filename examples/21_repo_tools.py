# Chapter 21 — the test suite and the line counter, run from the tinygrad checkout
# Run: DEV=CPU python3 examples/21_repo_tools.py      (needs: pip install pytest tabulate)
#
# PYTHONDONTWRITEBYTECODE=1 and -p no:cacheprovider keep pytest from writing __pycache__/.pytest_cache into the repo.
import os, subprocess, sys
import tinygrad
REPO = os.path.dirname(os.path.dirname(os.path.abspath(tinygrad.__file__)))   # the editable checkout
env = {**os.environ, "DEV": "CPU", "PYTHONDONTWRITEBYTECODE": "1", "NO_COLOR": "1"}
def run(cmd:list[str], tail:int):
  p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
  print("$ python " + " ".join(c for c in cmd[1:] if c not in ("-p", "no:cacheprovider")))
  lines = (p.stdout + p.stderr).strip().splitlines()
  print("\n".join([l for l in lines if "ModuleNotFoundError" in l][:1] + lines[-tail:]) + "\n")
py = sys.executable
run([py, "-m", "pytest", "-p", "no:cacheprovider", "test/null/test_pattern_matcher.py", "-q"], 3)
run([py, "-m", "pytest", "-p", "no:cacheprovider", "test/null/test_memory_planner.py", "-v", "-k", "pinned"], 6)
# most test files import test/helpers.py, which needs numpy (pip install -e '.[testing_minimal]'); without it collection fails:
run([py, "-m", "pytest", "-p", "no:cacheprovider", "test/null/test_schedule.py", "-q", "-x"], 3)
run([py, "sz.py"], 18)
