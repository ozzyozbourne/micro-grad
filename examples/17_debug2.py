# Chapter 17 — reading the DEBUG=2 per-kernel line, and the GlobalCounters behind it.
# run: DEV=CPU DEBUG=2 python3 examples/17_debug2.py
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters

a = Tensor.rand(256, 256).realize()
b = Tensor.rand(256, 256).realize()

GlobalCounters.reset()            # kernel_count / global_ops / global_mem / time_sum_s back to 0 (mem_used is never reset)
c = (a @ b).relu().sum(axis=1)    # one fused matmul+relu+sum kernel
c.realize()
print(f"GlobalCounters: kernel_count={GlobalCounters.kernel_count} global_ops={GlobalCounters.global_ops:,} "
      f"global_mem={GlobalCounters.global_mem:,} B  time_sum_s={GlobalCounters.time_sum_s*1e3:.3f} ms  mem_used={GlobalCounters.mem_used:,} B")
print(f"=> {GlobalCounters.global_ops/GlobalCounters.time_sum_s*1e-9:.1f} GFLOPS, 256*256*256*2 = {256*256*256*2:,} FLOPs for the matmul alone")

c2 = (a @ b).relu().sum(axis=1)   # same graph again: schedule cache hit, compile cache hit, the line is no longer green
c2.realize()
