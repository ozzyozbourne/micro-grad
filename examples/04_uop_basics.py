# Chapter 4 — UOp: the one IR.  Run with:  DEV=CPU python3 examples/04_uop_basics.py
# Everything here builds UOps by hand (no Tensor), so you can see the raw IR object.
import gc
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.ops import UOp, UOpMetaClass
from tinygrad.dtype import dtypes

def section(t): print(f"\n=== {t} ===")

section("1. a UOp is (op, src, arg); dtype is DERIVED, not stored")
x = UOp.variable("x", 0, 10)          # a Variable is a scalar PARAM that carries a value range
e = x * 2 + 1                          # operator overloading builds MUL and ADD nodes
print("e.op  =", e.op)
print("e.src =", [s.op for s in e.src])
print("e.arg =", e.arg)
print("e.dtype =", e.dtype, "  (computed by dtype_from_uop, cached in e.__dict__)")
print("fields stored on the dataclass:", UOp.__dataclass_fields__.keys())

section("2. CONST dtype comes from the Python value; a typed const is CAST(CONST)")
print(UOp.const(3).dtype, UOp.const(3.0).dtype, UOp.const(True).dtype)
print(repr(UOp.const(3, dtypes.int32)))

section("3. hash-consing: same (op, src, arg) -> the SAME Python object")
a = x * 2 + 1
b = x * 2 + 1
print("a is b:", a is b, "  id(a)==id(b):", id(a) == id(b))
print("UOp.const(1) is UOp.const(1):   ", UOp.const(1) is UOp.const(1))
print("UOp.const(1) is UOp.const(True):", UOp.const(1) is UOp.const(True), " (type(arg) is part of the cache key)")
print("UOp.const(1) is UOp.const(1.0): ", UOp.const(1) is UOp.const(1.0))

section("4. the cache holds WEAK references: nodes vanish when nobody uses them")
gc.collect()
n0 = len(UOpMetaClass.ucache)
tmp = UOp.variable("q", 0, 5) * 7 + 3      # PARAM q, CONST 7, MUL, CONST 3, ADD (the empty STACK already exists: shared!)
print("new cache entries after building q*7+3:", len(UOpMetaClass.ucache) - n0)
del tmp; gc.collect()
print("after `del` + gc:                       ", len(UOpMetaClass.ucache) - n0, "(UOp.__del__ removed them)")

section("5. key: a sha256 content hash, stable across processes")
print("a.key =", a.key.hex())
print("same expression, same key:", (x * 2 + 1).key == a.key, "| different expression:", (x * 3 + 1).key == a.key)

section("6. toposort: every node after all of its sources")
s = UOp.sink(a, x + 5)
for i, u in enumerate(s.toposort()): print(f"  {i}: {u.op}")

section("7. replace / substitute: you never mutate a UOp, you build a new one")
print("a.replace(op=Ops.MUL).render(simplify=False):", a.replace(op=Ops.MUL).render(simplify=False))
four = a.substitute({x: UOp.const(4)})
print("a.substitute({x: 4}).render(simplify=False):  ", four.render(simplify=False))
print("...then .simplify():                         ", four.simplify())
print("a is unchanged:", a.render(simplify=False))

section("8. three ways to print the same UOp")
print("-- repr (pretty_print): shared nodes get a name like x3:= --")
print(repr(x + x))
print("-- render_uir (SSA wire format) --")
print((x * 2 + 1).render_uir())
print("-- render (infix expression) --")
print((x * 2 + 1).render())

section("9. Ops enum order is semantic: tuplize puts CONSTs last")
print("Ops.PARAM =", Ops.PARAM.value, " Ops.ADD =", Ops.ADD.value, " Ops.CONST =", Ops.CONST.value)
two_x = UOp.const(2) * x
print("2*x as built:", two_x.render(simplify=False), "-> after symbolic:", two_x.simplify().render(simplify=False))

section("10. shapes are inferred too (_shape), and movement args are srcs")
buf = UOp.param(0, dtypes.float, 12)                  # a flat 12-element float buffer argument
r = buf.reshape((3, 4))
print("RESHAPE srcs:", [u.op for u in r.src], " src[1] is a STACK of CONSTs:", [c.arg for c in r.src[1].src])
print("shapes:", buf.shape, "->", r.shape, "->", r.permute((1, 0)).shape, "->", r.permute((1, 0)).expand((2, 4, 3)).shape)
print("sum over axis 1 (REDUCE):", r._rop(Ops.ADD, (1,)).shape)

section("11. vmin / vmax (range analysis, chapter 6)")
print(f"x in [{x.vmin}, {x.vmax}]  ->  x*2+1 in [{a.vmin}, {a.vmax}]  ->  (x*2+1)//4 in [{(a//4).vmin}, {(a//4).vmax}]")

section("12. GroupOp: named sets of Ops used by patterns")
for name in ["Unary", "Binary", "Ternary", "Movement", "Commutative", "Associative", "Idempotent", "Comparison", "Defines", "Irreducible"]:
  print(f"  {name:12s}", sorted(o.name for o in getattr(GroupOp, name)))
