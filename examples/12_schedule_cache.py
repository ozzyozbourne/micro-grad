"""examples/12_schedule_cache.py — what the schedule cache keys on, and what it stores

Run:  DEV=CPU DEBUG=1 python3 examples/12_schedule_cache.py 2>/dev/null

DEBUG>=1 prints one line per scheduled graph with >1 kernel (see lower_sink_to_linear in tinygrad/schedule/__init__.py):
    scheduled     2 kernels in  8.49 ms | CACHE MISS eb417c35 | ...
The hex is the first 8 hex digits of the cache key = the .key (structural hash) of the CALL body built by
transform_to_call, where every input BUFFER was replaced by PARAM(slot) and every bound Variable by a value-free PARAM.
"""
from tinygrad import Tensor, Variable
import tinygrad.schedule as S
from tinygrad.uop.ops import Ops

def f(x): return x / x.sum(1, keepdim=True)

print("--- call 1: x is a fresh (4, 8) buffer")
f(Tensor.empty(4, 8)).realize()
print("--- call 2: a DIFFERENT (4, 8) buffer -> same key")
f(Tensor.empty(4, 8)).realize()
print("--- call 3: shape (5, 8) -> different graph, different key")
f(Tensor.empty(5, 8)).realize()
print("--- calls 4,5: a symbolic batch size; the value 3 / 11 is NOT part of the key")
xs = Tensor.empty(16, 8).realize()
bs = Variable("bs", 1, 16)
f(xs[:bs.bind(3)]).realize()
f(xs[:bs.bind(11)]).realize()

print(f"\nschedule_cache now holds {len(S.schedule_cache)} entries:")
for k, v in S.schedule_cache.items(): print(f"  {k.hex()[:8]} -> LINEAR of {len(v.src)} CALL(s)")
key, cached = list(S.schedule_cache.items())[0]
print(f"entry {key.hex()[:8]}: a {cached.op} of {len(cached.src)} CALLs. The cached CALLs are NOT bound to real memory:")
for i, call in enumerate(cached.src):
  args = [f"{s.op.name}(slot={s.arg.slot})" for s in call.src[1:]]
  print(f"  CALL {i}: body={call.body.op.name}  args={args}")
print("PARAM(slot=i) = 'the i-th argument of the outer CALL' (output buffer, x, ...); ALLOC = an intermediate buffer")
print("the scheduler created in bufferize_to_store — it gets fresh memory every time (create_new_buffer).")
print("resolve_linear_call (pm_post_sched_cache) substitutes both every time the schedule is used, hit or miss.")
