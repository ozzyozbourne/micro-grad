# Chapter 16 — the end of the line on the CPU: SOURCE -> clang -> ELF object (BINARY) -> disassembly -> run it.
#
# do_compile (codegen/__init__.py) calls renderer.compiler.compile_cached(source); at DEBUG>=7 it also calls
# compiler.disassemble(lib), which for the CPU runs objdump on the object file. We do the same steps by hand here.
#
# run:   DEV=CPU python3 examples/16_compile_run.py
from dataclasses import replace
from tinygrad import Tensor, Device
from tinygrad.codegen import to_program

x = Tensor.empty(4, 8)
ast = (x.sum(1) + x.max(1)).schedule_linear().src[-1].src[0]
ast = ast.replace(arg=replace(ast.arg, opts_to_apply=()))
ren = Device["CPU"].renderer
prg = to_program(ast, ren)
sink, lin, source, binary = prg.src
print(f"renderer={type(ren).__name__} compiler={type(ren.compiler).__name__} args={ren.compiler.args}")
print(f"BINARY: {len(binary.arg)} bytes, starts with {binary.arg[:4]!r} (an ELF relocatable object)")
print("--- disassembly (what DEBUG=7 prints) ---", flush=True)
ren.compiler.disassemble(binary.arg)

# now run the real thing end to end and check it against plain python
vals = [[float((r * 8 + c) % 7 - 3) for c in range(8)] for r in range(4)]
t = Tensor(vals)
got = (t.sum(1) + t.max(1)).tolist()
want = [sum(row) + max(row) for row in vals]
print("tinygrad:", got)
print("python  :", want, "-> match" if got == want else "-> MISMATCH")
