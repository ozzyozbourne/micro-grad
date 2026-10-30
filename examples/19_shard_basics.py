# Chapter 19 — what a sharded Tensor is: UOps, devices, axis, and the per-device MultiBuffer.
# run: DEV=CPU python3 examples/19_shard_basics.py
import struct
from tinygrad import Tensor
from tinygrad.uop.ops import Ops

devs = ("CPU:0", "CPU:1")                       # two CPU "devices": CPU (":0" is canonicalized away) and CPU:1
t = Tensor([[float(4*i + j) for j in range(4)] for i in range(4)]).realize()

# 1) shard = copy_to_device(devices) -> SHRINK by the DEVICE range -> UNSHARD(axis)
s = t.shard(devs, axis=0)
u = s.uop
print("1 device:", s.device, "| shape (global):", s.shape, "| uop.axis:", u.axis, "| shard_shape:", u.shard_shape)
print("  uop chain:", u.op.name, "<-", u.src[0].op.name, "<-", u.src[0].src[0].op.name, "(arg", u.src[0].src[0].arg, ")")
drange = u.src[1]
print("  sharding:", [(ax, f"RANGE({r.arg[0].name}, {int(r.vmax)+1})") for ax, r in u.sharding])
print("  SHRINK bounds per axis (symbolic in the DEVICE range r):", [tuple(x.render() if hasattr(x, 'render') else str(x) for x in b) for b in u.src[0].marg])

# 2) realize: one MultiBuffer = one Buffer per device, each holding its shard
s.realize()
mb = s.uop.src[0].base.buffer if s.uop.op is Ops.UNSHARD else s.uop.buffer
print("2 realized storage:", mb)
for b in mb.bufs: print(f"   {b.device:6s} {b.nbytes}B ->", struct.unpack(f"{b.nbytes//4}f", b.as_memoryview()))

# 3) axis=1 splits columns; axis=None replicates the whole tensor on every device
for ax in (1, None):
  r = t.shard(devs, axis=ax).realize()
  mb = (r.uop.src[0].base if r.uop.op is Ops.UNSHARD else r.uop.base).buffer
  print(f"3 axis={ax}:", [struct.unpack(f"{b.nbytes//4}f", b.as_memoryview()) for b in mb.bufs])

# 4) uneven splits are rejected
try: Tensor.ones(3, 4).contiguous().shard(devs, axis=0)
except RuntimeError as e: print("4 uneven:", e)

# 5) elementwise on a sharded tensor stays sharded; reading it (.tolist) gathers it to one device with COPYs
y = (s * 10 + 1)
print("5 y.device:", y.device, "axis", y.uop.axis, "->", y.tolist())

# 6) .to() a single device = gather; shard_like copies another tensor's sharding
g = s.to("CPU")
print("6 gathered:", g.device, g.shape, "| shard_like:", Tensor.ones(4, 4).contiguous().shard_like(s).uop.axis)
