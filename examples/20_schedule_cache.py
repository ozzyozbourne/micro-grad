# Chapter 20 — why the first two training steps are slow and the rest are not (no JIT)
# Run: DEV=CPU DEBUG=1 python3 examples/20_schedule_cache.py
#
# DEBUG=1 prints one "scheduled N kernels ... CACHE MISS / cache hit <key>" line per realize (schedule/__init__.py,
# lower_sink_to_linear). The key is the hash of the tensor graph *with buffers replaced by numbered PARAMs*, so the
# same step on new data/weights hits the schedule cache.
import importlib.util, os
spec = importlib.util.spec_from_file_location("ts", os.path.join(os.path.dirname(os.path.abspath(__file__)), "20_train_step.py"))
ts = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts)     # reuse spiral() and MLP (its __main__ part does not run)
from tinygrad import Tensor, nn, Context
Tensor.manual_seed(0)
pts, labels = ts.spiral()
X, Y = Tensor(pts).realize(), Tensor(labels).realize()
model = ts.MLP()
opt = nn.optim.Adam(nn.state.get_parameters(model), lr=0.02)   # note: params, m, v, lr are all still *lazy* here
for i in range(4):
  print("step", i)
  with Context(TRAINING=1):
    opt.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    loss.realize(*opt.schedule_step())
print("opt.lr is still lazy:", opt.lr.uop.op, "<- a COPY from the PYTHON device, re-run every step (the 31st kernel)")
