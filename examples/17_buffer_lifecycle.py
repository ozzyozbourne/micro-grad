# Chapter 17 — the life of a Buffer: lazy -> allocated -> filled -> read -> freed into the LRU cache -> reused.
# run: DEV=CPU python3 examples/17_buffer_lifecycle.py
import struct
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.helpers import GlobalCounters

cpu = Device["CPU"]
py = Device["PYTHON"]
# every device owns a 16B timeline + 8B error buffer (lazily created); touch them now so the counters below are clean
_ = cpu.timeline, cpu.error_state, py.timeline, py.error_state
base_mem = GlobalCounters.mem_used

# 1) Creating a Buffer allocates NOTHING. It is just (device, nbytes, options).
b = Buffer("CPU", 16)
print("1 created   :", b, "| is_allocated:", b.is_allocated())

# 2) allocate() asks the device's Allocator for storage (HostAllocator: an anonymous mmap)
b.allocate()
st = b.get_storage()
print("2 allocated :", b, "| addr", hex(st.buf), "| host view:", type(st.host).__name__,
      "| mem_used +", GlobalCounters.mem_used - base_mem)

# 3) "copyin": there is no Buffer.copyin any more. Data enters through a source Buffer (here on the PYTHON
#    device, built from bytes via initial_value) and Buffer.copy_from(), which runs a 1-call LINEAR (a STORE CALL)
src = Buffer("PYTHON", 16, initial_value=struct.pack("4f", 1, 2, 3, 4))
b.copy_from(src)
print("3 filled    :", struct.unpack("4f", b.as_memoryview()))
del src   # the PYTHON staging buffer is freed (into the PYTHON allocator's cache)

# 4) views share storage: a view is (base, offset, nbytes) and allocates only an offset pointer
v = b.view(8, 4)                         # bytes 4..12 = float elements [1] and [2]
print("4 view      :", v, "| base is b:", v.base is b, "| allocated:", v.is_allocated())
v.ensure_allocated()
print("  view addr :", hex(v._buf), "= base +", v._buf - b._buf, "->", struct.unpack("2f", v.as_memoryview()))

# 5) "copyout": as_memoryview() synchronizes the device and copies the bytes out (zero-copy if asked)
print("5 copyout   :", struct.unpack("4f", b.as_memoryview()), "| zero-copy view type:", type(b.as_memoryview(allow_zero_copy=True)).__name__)

# 6) free: deallocate() hands the storage to Allocator.free -> it goes into the LRU cache keyed by (size, options)
addr = b._buf
del v
b.deallocate()
cache = {k[0]: len(lst) for k, lst in cpu.allocator.cache.items() if lst}
print("6 freed     :", b, "| allocator.cache sizes->count:", cache, "| mem_used +", GlobalCounters.mem_used - base_mem)

# 7) the next alloc of the same (size, options) pops the cached storage: same address, no mmap
c = Buffer("CPU", 16).allocate()
print("7 realloc   : same address as before?", c._buf == addr, "| cache now:", {k[0]: len(l) for k, l in cpu.allocator.cache.items() if l})

# 8) a different size misses the cache and really allocates
d = Buffer("CPU", 32).allocate()
print("8 other size: new address?", d._buf != addr)
print("GlobalCounters.mem_used counts live bytes (cached ones are not counted):", GlobalCounters.mem_used - base_mem)
