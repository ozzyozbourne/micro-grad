# Chapter 18 — measure the Python overhead the JIT removes, for a small MLP on the CPU.
# run: DEV=CPU python3 examples/18_jit_timing.py      (writes data/ch18_timings.json)
import time, json, pathlib
from tinygrad import Tensor, TinyJit, nn
from tinygrad.engine.realize import run_linear
from tinygrad.helpers import GlobalCounters

N = 40
def med(xs): return sorted(xs)[len(xs)//2]

def run_config(inp, hidden, out, batch):
  Tensor.manual_seed(0)
  l1, l2 = nn.Linear(inp, hidden), nn.Linear(hidden, out)
  def forward(x): return l2(l1(x).relu()).log_softmax()
  def step(x): return forward(x).realize()
  jstep = TinyJit(step)
  xs = [Tensor.randn(batch, inp).realize() for _ in range(N)]

  def bench(f):
    ts = []
    for x in xs:
      st = time.perf_counter(); f(x); ts.append((time.perf_counter() - st) * 1e3)
    return ts
  plain, jit = bench(step), bench(jstep)

  # where does the non-JIT time go? build the lazy graph / schedule it / run it (compile+link+exec)
  phases = {"build graph": [], "schedule (linear_with_vars)": [], "run_linear (compile+link+exec)": []}
  for x in xs:
    t0 = time.perf_counter(); y = forward(x)
    t1 = time.perf_counter(); lin, vv = Tensor.linear_with_vars(y)
    t2 = time.perf_counter(); run_linear(lin, vv)
    t3 = time.perf_counter()
    for k, d in zip(phases, (t1 - t0, t2 - t1, t3 - t2)): phases[k].append(d * 1e3)
  ncalls = len(lin.src)

  # how many kernels does one JIT replay launch?
  GlobalCounters.reset(); jstep(xs[0]); kcount = GlobalCounters.kernel_count
  return {"shape": f"{inp}->{hidden}->{out}, batch {batch}", "plain_ms": plain, "jit_ms": jit,
          "phases_ms": {k: med(v[5:]) for k, v in phases.items()}, "calls": ncalls, "kernels_per_replay": kcount,
          "plain_median": med(plain[5:]), "jit_median": med(jit[5:])}

results = [run_config(64, 128, 10, 16), run_config(784, 1024, 10, 64)]
for r in results:
  print(f"MLP {r['shape']}: {r['calls']} CALLs per step")
  print(f"  without JIT : first calls {[round(t, 2) for t in r['plain_ms'][:3]]} ms, median {r['plain_median']:.3f} ms")
  print(f"  with TinyJit: first calls {[round(t, 2) for t in r['jit_ms'][:3]]} ms, median {r['jit_median']:.3f} ms"
        f"  -> {r['plain_median']/r['jit_median']:.1f}x faster")
  print("  non-JIT breakdown (median):", {k: round(v, 3) for k, v in r["phases_ms"].items()})
path = pathlib.Path(__file__).resolve().parent.parent / "data" / "ch18_timings.json"
path.write_text(json.dumps(results))
print("wrote", path.name)
