# Chapter 17 — HCQ ("hardware command queue") lowering, observed on the NULL device (no GPU needed).
# run: DEV=NULL python3 examples/17_hcq_null.py
# On HCQ devices (AMD, NV, QCOM, CUDA, HIP, NULL, METAL, CL) a run of CALLs is batched into ONE host program,
# "hcq_submit", that fences, writes command buffers, rings the doorbell and bumps the timeline. That host program
# is itself a UOp kernel, compiled for HCQ_RUNTIME_DEV (default CPU:LLVM) and called like any other kernel.
from tinygrad import Tensor, Context
from tinygrad.uop.ops import Ops
from tinygrad.device import HCQ_DEVS
from tinygrad.engine.realize import compile_linear

print("HCQ_DEVS =", sorted(HCQ_DEVS))
with Context(DEV="NULL"):
  a, b = Tensor.empty(16), Tensor.empty(16)
  lin = ((a + b) * 2).sum().schedule_linear()
  print("scheduled:", [(c.body.op.name, c.body.arg.name) for c in lin.src])

  cl = compile_linear(lin)          # lower_and_compile, then hcq_compile batches the calls
  for x in cl.src:
    call = x.without_after
    info = call.arg.aux
    print("compiled entry:", call.body.op.name, call.body.src[0].arg.name, "| aux =", type(info).__name__)
    print("  queues on devices:", info.device)
    for dev, name, est, stamps, key, bufs, io in info.kernels: print(f"  enqueues kernel {name!r} on {dev}  (outs,ins)={io}")
    print("  host program target:", call.body.arg.target)
    print("  host program source (LLVM IR, first 30 lines):")
    print("\n".join("    " + l for l in call.body.src[2].arg.splitlines()[:30]))
