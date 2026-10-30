#!/usr/bin/env python3
# Chapter 3 — dtypes: the DType object, the promotion lattice, weak types, casting vs bitcasting.
#
#   run:  DEV=CPU python3 examples/03_dtypes.py           (add --dump to write data/ch03_dtypes.json for the lattice demo)
import sys, json, pathlib, dataclasses, itertools
from tinygrad import Tensor, dtypes, Context
from tinygrad.dtype import DType, least_upper_dtype, promo_lattice, can_lossless_cast, sum_acc_dtype, strong_dtype, commit_int
from tinygrad.uop.ops import UOp, Ops

def section(s): print(f"\n### {s}")

section("1. a DType is a tiny frozen dataclass, created once (DTypeMetaClass caches it)")
print("fields:", [f.name for f in dataclasses.fields(DType)])
print(f"{'name':12s} {'prio':>4s} {'bits':>4s} {'itemsize':>8s} {'fmt':>4s}  min .. max")
for dt in [dtypes.bool, dtypes.weakint, *dtypes.ints, dtypes.weakfloat, *dtypes.floats]:
  rng = f"{dt.min} .. {dt.max}" if dt not in dtypes.weaks else "(no width: weak)"
  print(f"{dt.name:12s} {dt.priority:4d} {dt.bitsize:4d} {dt.itemsize:8d} {str(dt.fmt):>4s}  {rng}")
print("DType(14, 32, 'f32', 'f') is dtypes.float32 ->", DType(14, 32, "f32", "f") is dtypes.float32)
print("aliases: dtypes.float is dtypes.f32 ->", dtypes.float is dtypes.f32, "| dtypes.half is dtypes.float16 ->", dtypes.half is dtypes.float16)
print("finfo (exponent, mantissa) bits:", {dt.name: dtypes.finfo(dt) for dt in (dtypes.f16, dtypes.bf16, dtypes.f32, dtypes.f64, dtypes.fp8e4m3, dtypes.fp8e5m2)})

section("2. the promotion lattice and least_upper_dtype")
for k, v in promo_lattice.items(): print(f"  {k.name:12s} -> {', '.join(x.name for x in v)}")
for a, b in [(dtypes.int8, dtypes.uint8), (dtypes.int32, dtypes.uint32), (dtypes.int64, dtypes.uint64), (dtypes.float16, dtypes.bfloat16),
             (dtypes.int32, dtypes.float16), (dtypes.bool, dtypes.weakint), (dtypes.uint8, dtypes.weakfloat), (dtypes.fp8e4m3, dtypes.fp8e5m2)]:
  print(f"  least_upper_dtype({a.name}, {b.name}) = {least_upper_dtype(a, b).name}")

section("3. weak dtypes: Python literals have a kind (int/float) but no width")
for v in [3, 3.0, True, [1, 2], [1, 2**40], [1.0, 2]]: print(f"  dtypes.from_py({v!r:12}) = {dtypes.from_py(v)}")
print("  UOp.const(3).dtype =", UOp.const(3).dtype, "| UOp.const(3.0).dtype =", UOp.const(3.0).dtype)
print("  (Tensor([1, 2], dtype='int8') + 1).dtype   =", (Tensor([1, 2], dtype=dtypes.int8) + 1).dtype, "  <- the literal adopted int8")
print("  (Tensor([1, 2], dtype='half') * 0.5).dtype =", (Tensor([1., 2], dtype=dtypes.half) * 0.5).dtype)
print("  (Tensor([1, 2]) + 1.5).dtype               =", (Tensor([1, 2]) + 1.5).dtype, "  <- int32 meets weakfloat: still weak!")
w = Tensor([1, 2]) + 1.5
print("  ... its .tolist() commits it:", w.tolist(), " commit_dtype() =", w.commit_dtype())
print("  commit_int ladder: commit_int(0, 100) =", commit_int(0, 100), "| commit_int(0, 2**40) =", commit_int(0, 2**40), "| commit_int(0, 2**63) =", commit_int(0, 2**63))
print("  strong_dtype(weakfloat) =", strong_dtype(dtypes.weakfloat), "| strong_dtype(weakint) =", strong_dtype(dtypes.weakint))
try: Tensor.empty(2, dtype=dtypes.weakfloat)
except RuntimeError as e: print("  Tensor.empty(2, dtype=weakfloat) ->", e)

section("4. DEFAULT_FLOAT / DEFAULT_INT decide what weak values commit to")
with Context(DEFAULT_FLOAT="float16", DEFAULT_INT="int64"):
  print("  with DEFAULT_FLOAT=float16 DEFAULT_INT=int64:", Tensor([1.0, 2.0]).dtype, Tensor([1, 2]).dtype, Tensor.empty(2).dtype)
print("  defaults:", Tensor([1.0, 2.0]).dtype, Tensor([1, 2]).dtype, Tensor.empty(2).dtype)

section("5. dtype is DERIVED from the UOp (dtype_from_uop), not stored")
print("  UOp dataclass fields:", [f.name for f in dataclasses.fields(UOp)])
x = UOp.const(2) + UOp.const(1.5)
print("  CONST(2) + CONST(1.5) ->", x.op.name, x.dtype)
y = Tensor([1, 2], dtype=dtypes.int8).uop
for expr, u in [("y < 1", y < 1), ("y.sqrt()", y.sqrt()), ("y / 2", y / 2), ("y.cast(f16)", y.cast(dtypes.f16)), ("y.bitcast(u8)", y.bitcast(dtypes.u8))]:
  print(f"  {expr:14s} -> {u.op.name:7s} dtype={u.dtype}")

section("6. cast vs bitcast: value conversion vs reinterpreting the bits")
f = Tensor([1.0, -2.5, 0.1])
print("  f                    =", f.tolist())
print("  f.cast(int32)        =", f.cast(dtypes.int32).tolist(), "  (value converted, truncated toward 0)")
print("  f.bitcast(int32)     =", f.bitcast(dtypes.int32).tolist(), "  (same 32 bits read as int)")
print("  f.bitcast(uint32) hex=", [hex(v) for v in f.bitcast(dtypes.uint32).tolist()])
print("  f.cast(half).bitcast(uint16) hex =", [hex(v) for v in f.cast(dtypes.half).bitcast(dtypes.uint16).tolist()])
print("  shape change on bitcast: Tensor.empty(4, dtype=f32).bitcast(u8).shape =", Tensor.empty(4).bitcast(dtypes.uint8).shape)
# the same in pure Python (dtype.truncate rounds a value to a dtype, dtype.bitcast reinterprets): no kernel, works for bf16 on any machine
from tinygrad.dtype import truncate, bitcast
print(f"  {'x':>8s} | {'f32 bits':>10s} | {'f16 bits':>8s} {'f16 value':>16s} | {'bf16 bits':>9s} {'bf16 value':>22s}")
for x in [1.0, -2.5, 0.1, 3.14159, 65504.0, 1e-8]:
  h, b = truncate[dtypes.half](x), truncate[dtypes.bfloat16](x)
  print(f"  {x:8g} | {bitcast(truncate[dtypes.float](x), dtypes.float, dtypes.uint32):#010x} | "
        f"{bitcast(h, dtypes.half, dtypes.uint16):#06x} {h!r:>16} | {bitcast(b, dtypes.bfloat16, dtypes.uint16):#06x} {b!r:>22}")

section("7. accumulation dtypes and lossless casts")
for dt in (dtypes.bool, dtypes.int8, dtypes.uint8, dtypes.half, dtypes.float32): print(f"  sum_acc_dtype({dt.name}) = {sum_acc_dtype(dt).name}")
for a, b in [(dtypes.int16, dtypes.float32), (dtypes.int32, dtypes.float32), (dtypes.half, dtypes.bfloat16), (dtypes.uint8, dtypes.int16)]:
  print(f"  can_lossless_cast({a.name}, {b.name}) = {can_lossless_cast(a, b)}")

if "--dump" in sys.argv:
  lattice_dts = [dtypes.bool, dtypes.weakint, *dtypes.sints, *dtypes.uints, dtypes.weakfloat, *dtypes.fp8s, dtypes.f16, dtypes.bf16, dtypes.f32, dtypes.f64]
  out = {"dtypes": [{"name": d.name, "priority": d.priority, "bitsize": d.bitsize, "fmt": d.fmt} for d in lattice_dts],
         "lattice": {k.name: [x.name for x in v] for k, v in promo_lattice.items()},
         "lub": {f"{a.name}|{b.name}": least_upper_dtype(a, b).name for a, b in itertools.product(lattice_dts, repeat=2)}}
  p = pathlib.Path(__file__).resolve().parent.parent / "data" / "ch03_dtypes.json"
  p.write_text(json.dumps(out))
  print("wrote", p)
