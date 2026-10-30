# Chapter 17 — from a scheduled CALL to machine code running on the CPU, by hand.
# run: DEV=CPU python3 examples/17_compile_and_run.py
import struct, time
from tinygrad import Tensor, Device
from tinygrad.uop.ops import Ops
from tinygrad.device import Buffer
from tinygrad.engine.realize import compile_linear, link_linear, get_runtime, run_linear
from tinygrad.runtime.support.elf import elf_loader

a = Tensor([0., 1, 2, 3, 4, 5, 6, 7]).realize()
c = (a * 3 + 1).sqrt()

# 1) the schedule: a LINEAR of CALLs. The body of the CALL is still a kernel AST (a SINK)
lin = c.schedule_linear()
call = lin.src[0]
print("1 scheduled :", lin.op, "with", len(lin.src), "CALL; body =", call.body.op, call.body.arg.name,
      "| args:", [(s.op.name, s.device) for s in call.src[1:]])

# 2) compile_linear = lower_and_compile (+ hcq_compile, a no-op for CPU): every SINK body becomes a PROGRAM
cl = compile_linear(lin)
prg = cl.src[0].body
print("2 compiled  :", cl.src[0].body.op, "srcs =", [s.op.name for s in prg.src])
print("  ProgramInfo: global", prg.arg.global_size, "local", prg.arg.local_size, "globals", prg.arg.globals,
      "outs", prg.arg.outs, "ins", prg.arg.ins, "target", prg.arg.target)
print("  SOURCE:\n" + prg.src[2].arg)
lib = prg.src[3].arg
print("  BINARY:", len(lib), "bytes, magic", lib[:4])

# 3) the ELF object is not a shared library: tinygrad links it itself (runtime/support/elf.py)
image, sections, relocs = elf_loader(lib)
print("3 elf       : sections", [s.name for s in sections if s.name], "| image", len(image), "bytes |", len(relocs), "relocations")

# 4) to_elf() packages (lib, name, target, signature); Device.runtime() builds the Program (CPUProgram)
elf = prg.to_elf()
print("4 TinyELF   :", elf.name, "signature", [(n, a.name, str(d)) for n, a, d, _ in elf.signature])
rt = get_runtime("CPU", prg)
print("  runtime   :", type(rt).__name__, "| code mapped at", hex(rt.buf.buf), "| ctypes fn", rt.fxn)

# 5) call it ourselves with two raw Buffers: args are buffer addresses, packed into a struct (CPUProgram.__call__)
out, inp = Buffer("CPU", 32).allocate(), Buffer("CPU", 32).allocate()
inp.copy_from(Buffer("PYTHON", 32, initial_value=struct.pack("8f", *[float(i) for i in range(8)])))
et = rt(out._buf, inp._buf, global_size=(1, 1, 1), local_size=(1, 1, 1), wait=True)
print("5 manual run:", struct.unpack("8f", out.as_memoryview()), f"| took {et*1e6:.1f} us")

# 6) the normal path does exactly this: link_linear(compile_linear(...)) then pm_exec dispatches each CALL
run_linear(lin)
print("6 run_linear:", c.tolist())

# 7) compile_cached: the clang output is stored in the sqlite disk cache under the compiler's cachekey
comp = Device["CPU"].compiler
st = time.perf_counter(); comp.compile(prg.src[2].arg); t_raw = time.perf_counter() - st
st = time.perf_counter(); lib2 = comp.compile_cached(prg.src[2].arg); t_cached = time.perf_counter() - st
print(f"7 clang -c  : {t_raw*1e3:.1f} ms | compile_cached (disk hit): {t_cached*1e3:.2f} ms | same bytes: {lib2 == lib}")
