# Chapter 17 — the Device registry and what a Compiled device is made of.
# run: DEV=CPU python3 examples/17_devices.py
from tinygrad import Device
from tinygrad.device import Compiled, canonicalize_device
from tinygrad.helpers import Target

# 1) The registry is built by listing tinygrad/runtime/ops_*.py — nothing is imported yet.
print("backends found on disk:", sorted(Device._devices))

# 2) Device names are canonicalized: upper-case the backend, drop a trailing ':0'.
for s in ["cpu", "CPU:0", "cpu:1", "amd:2", ("CPU:0", "cpu:1")]:
  print(f"canonicalize({s!r:22}) -> {canonicalize_device(s)!r}")

# 3) DEV is parsed into a Target(device, renderer, arch, interface, indices)
for s in ["CPU", "CPU:LLVM", "AMD:LLVM:gfx1100", "PCI+AMD"]:
  print(f"Target.parse({s!r:20}) -> {Target.parse(s)!r:20} fields={Target.parse(s).__dict__}")
print("Device.DEFAULT =", Device.DEFAULT)

# 4) Device[...] imports runtime/ops_cpu.py, finds the class named CPUDevice, instantiates it ONCE (functools.cache)
cpu = Device["CPU"]
print("\nDevice['CPU'] is", type(cpu).__name__, "| same object again?", Device["cpu:0"] is cpu)
print("isinstance Compiled:", isinstance(cpu, Compiled))

# 5) A Compiled device = allocator + renderer(s) (each renderer owns its compiler) + a Program runtime class
print("  allocator :", type(cpu.allocator).__name__, "->", [c.__name__ for c in type(cpu.allocator).__mro__[:3]])
print("  renderers :", [r.__name__ for r in cpu.renderers])
print("  renderer  :", type(cpu.renderer).__name__, " (picked from DEV's renderer field, default = first that initializes)")
print("  compiler  :", type(cpu.compiler).__name__, "cachekey =", cpu.compiler.cachekey)
print("  runtime_t :", cpu.runtime_t.__name__)
print("  arch      :", cpu.arch)

# 6) Other host devices
py = Device["PYTHON"]
print("\nDevice['PYTHON']:", type(py).__name__, type(py.allocator).__name__, type(py.renderer).__name__,
      type(py.compiler).__name__, py.runtime_t.__name__)
null = Device["NULL"]
print("Device['NULL']  :", type(null).__name__, type(null.allocator).__name__, type(null.renderer).__name__, null.runtime_t.__name__)
print("opened so far:", sorted(Device._opened_devices))
