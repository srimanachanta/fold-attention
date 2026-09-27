"""Inline PTX shared by the kernels.

`cute.arch.inline_ptx` cannot mark an instruction as having side effects, and
several of these must not be hoisted past a barrier or folded, so they are
built on `llvm.inline_asm` directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import cutlass
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

_IR = {Float32: "f32", Int32: "i32", Uint32: "i32", Int64: "i64", cutlass.Uint64: "i64"}


def asm(
    template: str,
    constraints: str,
    args: Sequence,
    results: Sequence = (),
    *,
    side_effects: bool = False,
    loc=None,
    ip=None,
):
    """`template` over DSL scalars `args`, returning one value per type in
    `results`: None for none, the value for one, a list otherwise."""
    names = [_IR[t] for t in results]
    if not names:
        res_ty = None
    elif len(names) == 1:
        res_ty = getattr(T, names[0])()
    else:
        res_ty = ir.Type.parse("!llvm.struct<(" + ", ".join(names) + ")>")
    res = llvm.inline_asm(
        res_ty,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        template,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if not names:
        return None
    if len(names) == 1:
        return results[0](res)
    return [
        t(llvm.extractvalue(getattr(T, n)(), res, [i], loc=loc, ip=ip))
        for i, (t, n) in enumerate(zip(results, names))
    ]


@dsl_user_op
def f32_rn(op: str, *args, loc=None, ip=None) -> Float32:
    """One fp32 instruction `op` with round-to-nearest. Inline PTX is never
    contracted into an FMA, so the result is the one torch rounds."""
    ops = ", ".join(f"${i + 1}" for i in range(len(args)))
    return asm(
        f"{op} $0, {ops};",
        "=f," + ",".join(["f"] * len(args)),
        [Float32(a) for a in args],
        [Float32],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cvt_rni_s32_f32(x, *, loc=None, ip=None) -> Int32:
    """Round to nearest even, saturating. Rounding toward zero would bias a sum
    of n terms by up to n/2 grid units, and saturation (not wrapping) means an
    out-of-range term is not order-free, so a grid must keep every term in
    range."""
    return asm("cvt.rni.s32.f32 $0, $1;", "=r,f", [Float32(x)], [Int32], loc=loc, ip=ip)


@dsl_user_op
def cvt_rni_s64_f32(x, *, loc=None, ip=None) -> Int64:
    return asm("cvt.rni.s64.f32 $0, $1;", "=l,f", [Float32(x)], [Int64], loc=loc, ip=ip)


# The last-arriver protocol of the backward's dK/dV combine. Volatile inline
# PTX rather than the equivalent `cute.arch` atomics: those leave ptxas free
# to move them, and it then reschedules the mainloop they share a function
# with.
@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None):
    asm("fence.acq_rel.gpu;", "", [], side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def atomic_add_acq_rel_gpu(addr, val, *, loc=None, ip=None) -> Int32:
    """`atom.add` at a global address with acquire-release GPU scope,
    returning the old value."""
    return asm(
        "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
        "=r,l,r",
        [Int64(addr), Int32(val)],
        [Int32],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_release_gpu(addr, val, *, loc=None, ip=None):
    asm(
        "st.release.gpu.global.s32 [$0], $1;",
        "l,r",
        [Int64(addr), Int32(val)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )
