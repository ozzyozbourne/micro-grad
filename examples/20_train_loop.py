# Chapter 20 — the same training step, without and with TinyJit, then trained to convergence
# Run: DEV=CPU python3 examples/20_train_loop.py
#
# Part 1: time the plain step (python builds the graph, schedule cache, run) vs a @TinyJit step (replays the captured LINEAR).
# Part 2: train to convergence on the spiral and record, every few steps, the loss and the model's predictions on a
#         32x32 grid of the input plane -> data/ch20_train.json (replayed by the chapter's decision-boundary demo).
import json, math, os, random, time
from tinygrad import Tensor, nn, Context, TinyJit, GlobalCounters
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def spiral(n_per_class=64, seed=0):   # identical to examples/20_train_step.py
  rnd = random.Random(seed)
  pts, labels = [], []
  for c in range(2):
    for i in range(n_per_class):
      r = 0.15 + 0.85 * i / n_per_class
      t = 1.6 * math.pi * i / n_per_class + c * math.pi + rnd.gauss(0, 0.12)
      pts.append([r * math.cos(t), r * math.sin(t)]); labels.append(c)
  return pts, labels

class MLP:
  def __init__(self, hidden=32): self.l1, self.l2 = nn.Linear(2, hidden), nn.Linear(hidden, 2)
  def __call__(self, x:Tensor) -> Tensor: return self.l2(self.l1(x).relu())

def make(seed=0):
  Tensor.manual_seed(seed)
  model = MLP()
  opt = nn.optim.Adam(nn.state.get_parameters(model), lr=0.02)
  return model, opt

pts, labels = spiral()
X, Y = Tensor(pts).realize(), Tensor(labels).realize()

def step(model, opt):
  with Context(TRAINING=1):
    opt.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    return loss.realize(*opt.schedule_step())

def timed(fn, n):
  ts = []
  for _ in range(n):
    GlobalCounters.reset()
    st = time.perf_counter(); out = fn(); out.item(); ts.append((time.perf_counter() - st) * 1e3)
  return ts, GlobalCounters.kernel_count

print("=== Part 1: plain step vs TinyJit step ===")
model, opt = make()
plain_ms, kc = timed(lambda: step(model, opt), 12)
print(f"plain : {kc} kernels/step, step 1 {plain_ms[0]:7.1f} ms (cold: schedule+codegen+compile), "
      f"step 2 {plain_ms[1]:6.1f} ms, median of steps 3-12 {sorted(plain_ms[2:])[5]:6.2f} ms")

model, opt = make()
@TinyJit
def jit_step(): return step(model, opt)
jit_ms, kc = timed(jit_step, 12)
print(f"jit   : {kc} kernels/step, step 1 {jit_ms[0]:7.1f} ms (runs normally), step 2 {jit_ms[1]:6.1f} ms (capture), "
      f"median of steps 3-12 {sorted(jit_ms[2:])[5]:6.2f} ms (replay)")
print(f"jit captured {len(jit_step.captured._linear.src)} calls in one LINEAR")

print("\n=== Part 2: train to convergence ===")
model, opt = make()
@TinyJit
def train_step(): return step(model, opt)
G = 32
grid = Tensor([[-1.1 + 2.2 * c / (G - 1), 1.1 - 2.2 * r / (G - 1)] for r in range(G) for c in range(G)]).realize()
@TinyJit
def predict(): return model(grid).softmax(-1)[:, 1].realize()     # P(class 1) for every grid point (TRAINING=0: plain forward)
@TinyJit
def accuracy(): return (model(X).argmax(-1) == Y).mean().realize()

losses, frames = [], []
STEPS, EVERY = 400, 10
st = time.perf_counter()
for i in range(STEPS + 1):
  if i % EVERY == 0:
    frames.append({"step": i, "acc": round(accuracy().item(), 4), "p": [round(v, 2) for v in predict().tolist()]})
  if i == STEPS: break
  losses.append(round(train_step().item(), 5))
el = time.perf_counter() - st
for i in (0, 1, 10, 50, 100, 200, 300, 399): print(f"step {i:3d}  loss {losses[i]:.4f}")
print(f"final accuracy {frames[-1]['acc']*100:.1f}%  ({STEPS} steps + {len(frames)} grid evals in {el:.2f} s)")

json.dump({"points": [[round(a, 4), round(b, 4)] for a, b in pts], "labels": labels, "grid": G, "extent": 1.1, "every": EVERY,
           "loss": losses, "frames": frames, "plain_ms": [round(t, 3) for t in plain_ms], "jit_ms": [round(t, 3) for t in jit_ms]},
          open(os.path.join(DATA, "ch20_train.json"), "w"))
print("wrote data/ch20_train.json")
