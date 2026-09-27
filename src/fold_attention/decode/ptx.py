"""Inline PTX the decode kernels need and the DSL does not spell.

The writers' arithmetic goes through `f32_rn`, one uncontracted `.rn`
instruction per operation torch rounds on its own, so the device quantisers
reproduce `decode.cache`'s torch quantisers to the bit.
"""

from cutlass import Float32, Int32, Int64, Uint16, Uint32, cute
from cutlass.base_dsl.typing import Vector
from cutlass.cutlass_dsl import dsl_user_op

from ..ptx import asm, f32_rn

__all__ = ["f32_rn"]

# Adding 1.5 * 2^23 to a float in [-2^22, 2^22] rounds it to an integer, ties
# to even as `cvt.rni` does, and leaves that integer in the low mantissa bits.
MAGIC = 12582912.0


@dsl_user_op
def ex2(x, *, loc=None, ip=None) -> Float32:
    """`ex2.approx.ftz.f32`, the weight's exponential. `cute.math.exp2` goes
    through LLVM's fast-math lowering, which does not promise to flush a
    subnormal weight to zero the same way."""
    return asm("ex2.approx.ftz.f32 $0, $1;", "=f,f", [Float32(x)], [Float32], loc=loc, ip=ip)


@dsl_user_op
def ex2_if(keep, x, *, loc=None, ip=None) -> Float32:
    """`ex2(x)` where `keep` is non-zero and +0 elsewhere, as one predicated
    instruction: a select would issue the exponential for every pair, and it
    shares its pipe with the e4m3 conversions."""
    return asm(
        "{\n.reg .pred p;\nsetp.ne.b32 p, $1, 0;\nmov.b32 $0, 0;\n@p ex2.approx.ftz.f32 $0, $2;\n}",
        "=f,r,f",
        [Int32(keep), Float32(x)],
        [Float32],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def opaque_i32(x, *, loc=None, ip=None) -> Int32:
    """`x` through a `mov` the compiler may neither fold nor hoist.

    A shared tile's `wgmma` descriptors are loop-invariant, so they are built
    once before the loop and held across it, in GPRs once the uniform file is
    full. Built from this value inside the loop, each is a few uniform adds
    just before its matmul.
    """
    return asm("mov.b32 $0, $1;", "=r,r", [Int32(x)], [Int32], side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def movmatrix_t(x, *, loc=None, ip=None) -> Uint32:
    """The transpose of one 8x8 16-bit matrix held as a warp's fragment: lane
    `4 g + t` holds row `g`, columns `2t` and `2t + 1`, before and after.
    Warp-collective, so it is marked as a side effect and never sunk into a
    branch."""
    return asm(
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        "=r,r",
        [Uint32(x)],
        [Uint32],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def pack_weights(lo, hi, *, loc=None, ip=None):
    """Two f32 weights rounded into one bf16 pair, `lo` in the low half, and
    the two rounded values back as f32 for the denominator, which must sum
    what the matmul sees."""
    return asm(
        "cvt.rn.bf16x2.f32 $0, $4, $3;\nshl.b32 $1, $0, 16;\nand.b32 $2, $0, 0xffff0000;",
        "=r,=f,=f,f,f",
        [Float32(lo), Float32(hi)],
        [Uint32, Float32, Float32],
        loc=loc,
        ip=ip,
    )


# A byte b under the high byte 0x64 is the fp16 1024 + b, and under 0x44 it
# is 4 + b / 256. With the sign bit flipped, b = x + 128, so both planes come
# out exact after one subtraction each and only their sum rounds.
@dsl_user_op
def k16_pairs(wa, wb, *, loc=None, ip=None):
    """Four channels' plane-A and plane-B bytes as two fp16 pairs of
    `Ka + Kb / 256`, rounded once, channels 0 and 1 in the first."""
    return asm(
        "{ .reg .b32 xa, xb, a0, a1, b0, b1, ma, mb, ca, cb;\n"
        "xor.b32 xa, $2, 0x80808080; xor.b32 xb, $3, 0x80808080;\n"
        "mov.b32 ma, 0x64646464; mov.b32 mb, 0x44444444;\n"
        "mov.b32 ca, 0x64806480; mov.b32 cb, 0x44804480;\n"
        "prmt.b32 a0, xa, ma, 0x4140; prmt.b32 a1, xa, ma, 0x4342;\n"
        "prmt.b32 b0, xb, mb, 0x4140; prmt.b32 b1, xb, mb, 0x4342;\n"
        "sub.rn.f16x2 a0, a0, ca; sub.rn.f16x2 a1, a1, ca;\n"
        "sub.rn.f16x2 b0, b0, cb; sub.rn.f16x2 b1, b1, cb;\n"
        "add.rn.f16x2 $0, a0, b0; add.rn.f16x2 $1, a1, b1; }",
        "=r,=r,r,r",
        [Uint32(wa), Uint32(wb)],
        [Uint32, Uint32],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_f32(addr, vals, *, loc=None, ip=None) -> None:
    """One store of 1, 2 or 4 floats to a global address aligned to their
    width, rather than through a tensor whose strides are runtime values."""
    if len(vals) == 1:
        value = Float32(vals[0])
    else:
        value = Vector.from_elements(
            tuple(Float32(v) for v in vals), Float32, loc=loc, ip=ip
        ).ir_value(loc=loc, ip=ip)
    cute.arch.store(addr, value, loc=loc, ip=ip)


@dsl_user_op
def zero16(dst, *, loc=None, ip=None) -> None:
    """Sixteen zero bytes at a shared address. The DSL's `st_bulk` spelling
    fails NVVM compilation for SM90."""
    asm(
        "{\n    .reg .b32 z;\n    mov.b32 z, 0;\n    st.shared.v4.b32 [$0], {z, z, z, z};\n}",
        "r",
        [Int32(dst)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


# An e4m3 byte `b` read as the bf16 bits `sign(b) << 15 | (b & 0x7f) << 4` is
# `b * 2^-120` exactly, subnormals included: each format's subnormals sit one
# binade under its least normal exponent, 1 - 7 and 1 - 127. A `prmt` that
# replicates each byte's sign into the byte above it, a shift and a mask
# convert two bytes, where the hardware's e4m3 -> f16 -> f32 -> bf16 takes
# four instructions and two f32 temporaries. The weights carry `E4W` of the
# 2^120 back and the epilogue `E4O`: all of it on the weights would overflow
# a weight 2^8 over its reference, and all of it on the accumulator would
# push its sums toward f32's subnormals.
E4W = 2.0**64
E4O = 2.0**56
_E4M = "0x87F087F0"
# bf16 0.125, V's second plane's weight
_E8 = "0x3e003e00"


def _e4sel(lo: int, hi: int) -> str:
    """The selector putting source bytes `lo` and `hi` at bytes 0 and 2, each
    under its sign replicated."""
    return hex(((8 | hi) << 12) | (hi << 8) | ((8 | lo) << 4) | lo)


def _e4(dst, a, b, lo, hi, t):
    return (
        f"    prmt.b32 {t}, {a}, {b}, {_e4sel(lo, hi)};\n"
        f"    shl.b32 {t}, {t}, 4;\n"
        f"    and.b32 {dst}, {t}, {_E4M};\n"
    )


def _widen(refine):
    """Sixteen e4m3 bytes out of a staging tile into sixteen scaled bf16,
    optionally plus V's second plane at 2^-3, stored at the descriptor's XOR
    swizzle: the tile's 16-byte units are swizzled by the key, so the second
    half of a 32-byte write is at `dst ^ 16`, not `dst + 16`."""
    body = [
        "{",
        "    .reg .b32 a0,a1,a2,a3,f0,f1,f2,f3,f4,f5,f6,f7,t,d1;",
        "    ld.shared.v4.b32 {a0,a1,a2,a3}, [$0];",
    ]
    for i in range(8):
        body.append(_e4(f"f{i}", f"a{i // 2}", f"a{i // 2}", 2 * (i % 2), 2 * (i % 2) + 1, "t"))
    if refine:
        body += [
            "    .reg .b32 b0,b1,b2,b3,g,e8;",
            f"    mov.b32 e8, {_E8};",
            "    ld.shared.v4.b32 {b0,b1,b2,b3}, [$2];",
        ]
        for i in range(8):
            body.append(_e4("g", f"b{i // 2}", f"b{i // 2}", 2 * (i % 2), 2 * (i % 2) + 1, "t"))
            body.append(f"    fma.rn.bf16x2 f{i}, g, e8, f{i};")
    body += [
        "    xor.b32 d1, $1, 16;",
        "    st.shared.v4.b32 [$1], {f0,f1,f2,f3};",
        "    st.shared.v4.b32 [d1], {f4,f5,f6,f7};",
        "}",
    ]
    return "\n".join(body)


@dsl_user_op
def widen16_bf16(src, dst, *, loc=None, ip=None) -> None:
    asm(_widen(False), "r,r", [Int32(src), Int32(dst)], side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def widen16_refine_bf16(src, dst, src2, *, loc=None, ip=None) -> None:
    asm(
        _widen(True),
        "r,r,r",
        [Int32(src), Int32(dst), Int32(src2)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


# A lane owns `nc = D // 32` channels of the register value operand. A
# register pairs two *keys* of one channel, so channel `c` takes byte `c` of
# each key's word.
_VF_LD = {4: "ld.shared.b32", 2: "ld.shared.u16"}


def _vf(off0, off1, nc, ap, refine=None):
    ld = _VF_LD[nc]
    body = [
        "{",
        "    .reg .b32 x0, x1, t;",
        f"    {ld} x0, [${ap}+{off0}];",
        f"    {ld} x1, [${ap}+{off1}];",
    ]
    if refine is None:
        body += [_e4(f"${c}", "x0", "x1", c, 4 + c, "t") for c in range(nc)]
    else:
        roff0, roff1, ap2, am0, am1 = refine
        body += [
            "    .reg .b32 y0, y1, f, g, e8;",
            f"    mov.b32 e8, {_E8};",
            f"    {ld} y0, [${ap2}+{roff0}];",
            f"    {ld} y1, [${ap2}+{roff1}];",
            # a masked key's second plane is +0
            f"    and.b32 y0, y0, ${am0};",
            f"    and.b32 y1, y1, ${am1};",
        ]
        for c in range(nc):
            body.append(_e4("f", "x0", "x1", c, 4 + c, "t"))
            body.append(_e4("g", "y0", "y1", c, 4 + c, "t"))
            body.append(f"    fma.rn.bf16x2 ${c}, g, e8, f;")
    return "\n".join(body) + "\n}"


@dsl_user_op
def vfrag_bf16(addr, off=0, stride=128, nc=4, *, loc=None, ip=None):
    """A lane's `nc` channels of two keys of the register value operand, out
    of the e4m3 staging tile, as `nc` scaled bf16 pairs. Side effects on:
    these read shared memory other threads wrote and must not be hoisted
    above the barrier that publishes it."""
    return asm(
        _vf(off, off + stride, nc, nc),
        ",".join(["=r"] * nc) + ",r",
        [Int32(addr)],
        [Uint32] * nc,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def vfrag_refine_bf16(
    addr, addr2, m0, m1, off=0, off2=0, stride=128, stride2=128, nc=4, *, loc=None, ip=None
):
    """`vfrag_bf16` plus V's second plane for the keys the masks `m0`, `m1`
    (all ones or all zeros) keep."""
    return asm(
        _vf(off, off + stride, nc, nc, (off2, off2 + stride2, nc + 1, nc + 2, nc + 3)),
        ",".join(["=r"] * nc) + ",r,r,r,r",
        [Int32(x) for x in (addr, addr2, m0, m1)],
        [Uint32] * nc,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_v4(addr, nc, *, loc=None, ip=None):
    """Four words at a 16-byte-aligned global address. `nc` reads through the
    non-coherent path, for data nothing writes while the kernel runs; the
    coherent form stays in program order with this kernel's own stores."""
    op = "ld.global.nc.v4.b32" if nc else "ld.global.v4.b32"
    return asm(
        op + " {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        [Int64(addr)],
        [Uint32] * 4,
        side_effects=not nc,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_nc(addr, n, *, loc=None, ip=None):
    """`n` read-only words (1, 2, 4 or 8) at a global address aligned to
    their width, as a list."""
    if n == 8:
        return ld_global_v4(addr, True, loc=loc, ip=ip) + ld_global_v4(
            addr + 16, True, loc=loc, ip=ip
        )
    if n == 4:
        return ld_global_v4(addr, True, loc=loc, ip=ip)
    if n == 2:
        return asm(
            "ld.global.nc.v2.b32 {$0, $1}, [$2];",
            "=r,=r,l",
            [Int64(addr)],
            [Uint32] * 2,
            loc=loc,
            ip=ip,
        )
    return [ld_global_nc_b32(addr, loc=loc, ip=ip)]


@dsl_user_op
def ld_global_nc_b32(addr, *, loc=None, ip=None) -> Uint32:
    """A word nothing writes while the kernel runs, free to schedule early."""
    return asm("ld.global.nc.b32 $0, [$1];", "=r,l", [Int64(addr)], [Uint32], loc=loc, ip=ip)


@dsl_user_op
def ld_scale_cg(addr, *, loc=None, ip=None) -> Float32:
    """A bf16 key scale at `addr` as its f32 value, through L2 only: the
    front may have written it in this step, after the reader's L1 last saw
    the line."""
    return asm(
        "{\n.reg .b16 h, z;\nmov.b16 z, 0;\nld.global.cg.b16 h, [$1];\nmov.b32 $0, {z, h};\n}",
        "=f,l",
        [Int64(addr)],
        [Float32],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_b32(addr, *, loc=None, ip=None) -> Uint32:
    """A word this kernel may also write, so it stays in program order."""
    ptr = cute.make_ptr(Uint32, addr, cute.AddressSpace.gmem, assumed_align=4, loc=loc, ip=ip)
    return Uint32(cute.arch.load(ptr, Uint32, loc=loc, ip=ip))


def _vector(words, loc, ip):
    return Vector.from_elements(tuple(Uint32(x) for x in words), Uint32, loc=loc, ip=ip).ir_value(
        loc=loc, ip=ip
    )


@dsl_user_op
def st_global_v4(addr, words, *, loc=None, ip=None):
    ptr = cute.make_ptr(Uint32, addr, cute.AddressSpace.gmem, assumed_align=16, loc=loc, ip=ip)
    cute.arch.store(ptr, _vector(words, loc, ip), loc=loc, ip=ip)


@dsl_user_op
def st_global_b32(addr, word, *, loc=None, ip=None):
    ptr = cute.make_ptr(Uint32, addr, cute.AddressSpace.gmem, assumed_align=4, loc=loc, ip=ip)
    cute.arch.store(ptr, Uint32(word), loc=loc, ip=ip)


@dsl_user_op
def st_global_b16(addr, word, *, loc=None, ip=None):
    """The low half of the word `word`, stored as one 16-bit value."""
    ptr = cute.make_ptr(Uint16, addr, cute.AddressSpace.gmem, assumed_align=2, loc=loc, ip=ip)
    cute.arch.store(ptr, Uint16(word), loc=loc, ip=ip)


@dsl_user_op
def st_shared(addr, words, *, loc=None, ip=None):
    """1, 2 or 4 words at a 32-bit shared address aligned to their width."""
    ptr = cute.make_ptr(
        Uint32, Int32(addr), cute.AddressSpace.smem, assumed_align=4 * len(words), loc=loc, ip=ip
    )
    value = Uint32(words[0]) if len(words) == 1 else _vector(words, loc, ip)
    cute.arch.store(ptr, value, loc=loc, ip=ip)


@dsl_user_op
def st_shared_b16(addr, word, *, loc=None, ip=None):
    """The low half of the word `word` at a 32-bit shared address."""
    ptr = cute.make_ptr(
        Uint16, Int32(addr), cute.AddressSpace.smem, assumed_align=2, loc=loc, ip=ip
    )
    cute.arch.store(ptr, Uint16(word), loc=loc, ip=ip)


@dsl_user_op
def half_to_f32(w, hi, f16, *, loc=None, ip=None) -> Float32:
    """Half `hi` of a word of two 16-bit floats (fp16 if `f16`, else bf16),
    widened exactly."""
    if f16:
        src = (
            "{ .reg .b16 a, b; mov.b32 {a, b}, $1; cvt.f32.f16 $0, " + ("b" if hi else "a") + "; }"
        )
    elif hi:
        src = "and.b32 $0, $1, 0xffff0000;"
    else:
        src = "shl.b32 $0, $1, 16;"
    return asm(src, "=f,r", [Uint32(w)], [Float32], loc=loc, ip=ip)


@dsl_user_op
def f32_to_half2(x, f16, *, loc=None, ip=None) -> Uint32:
    """One fp32 rounded to a 16-bit float, in both halves of a word."""
    op = "cvt.rn.f16.f32" if f16 else "cvt.rn.bf16.f32"
    return asm(
        "{ .reg .b16 h; " + op + " h, $1; mov.b32 $0, {h, h}; }",
        "=r,f",
        [Float32(x)],
        [Uint32],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def butterfly(x, p, upper, *, loc=None, ip=None) -> Float32:
    """One butterfly stage across lanes: `x + p` on the pair's lower lane and
    `p - x` on its upper one."""
    return asm(
        "{ .reg .pred q; setp.ne.s32 q, $3, 0; @q sub.rn.f32 $0, $2, $1; "
        "@!q add.rn.f32 $0, $1, $2; }",
        "=f,f,f,r",
        [Float32(x), Float32(p), Int32(upper)],
        [Float32],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def pack_int8x4(a, b, c, d, *, loc=None, ip=None) -> Uint32:
    """Four fp32 `MAGIC + n`, n integral in [-128, 127], as the bytes n of one
    word, `a` lowest: n's two's-complement byte is the float's low byte."""
    return asm(
        "{ .reg .b32 a, b, c, d; mov.b32 a, $1; mov.b32 b, $2; mov.b32 c, $3; "
        "mov.b32 d, $4; prmt.b32 a, a, b, 0x0040; prmt.b32 c, c, d, 0x0040; "
        "prmt.b32 $0, a, c, 0x5410; }",
        "=r,f,f,f,f",
        [Float32(x) for x in (a, b, c, d)],
        [Uint32],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def pack_int8x2(a, b, *, loc=None, ip=None) -> Uint32:
    """`pack_int8x4` for two values, in the low half of the word."""
    return asm(
        "{ .reg .b32 a, b; mov.b32 a, $1; mov.b32 b, $2; prmt.b32 $0, a, b, 0x0040; }",
        "=r,f,f",
        [Float32(a), Float32(b)],
        [Uint32],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def pack_e4m3x4(a, b, c, d, *, loc=None, ip=None) -> Uint32:
    """Four fp32 already in [-448, 448] as e4m3fn bytes, round to nearest even."""
    return asm(
        "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; "
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi}; }",
        "=r,f,f,f,f",
        [Float32(x) for x in (a, b, c, d)],
        [Uint32],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def unpack_e4m3x4(w, *, loc=None, ip=None):
    """A word of four e4m3fn bytes as four exact fp32."""
    return asm(
        "{ .reg .b16 lo, hi, a, b, c, d; .reg .b32 h0, h1; mov.b32 {lo, hi}, $4; "
        "cvt.rn.f16x2.e4m3x2 h0, lo; cvt.rn.f16x2.e4m3x2 h1, hi; "
        "mov.b32 {a, b}, h0; mov.b32 {c, d}, h1; cvt.f32.f16 $0, a; "
        "cvt.f32.f16 $1, b; cvt.f32.f16 $2, c; cvt.f32.f16 $3, d; }",
        "=f,=f,=f,=f,r",
        [Uint32(w)],
        [Float32] * 4,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def int8_to_f32(w, shift, *, loc=None, ip=None) -> Float32:
    """The signed byte at bit `shift` of a word, as an exact fp32."""
    return asm(
        "{ .reg .s32 t; bfe.s32 t, $1, " + str(shift) + ", 8; cvt.rn.f32.s32 $0, t; }",
        "=f,r",
        [Uint32(w)],
        [Float32],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def pow2_ceil(x, *, loc=None, ip=None) -> Float32:
    """The least power of two at or above a positive normal `x`, from its bits."""
    return asm(
        "{ .reg .b32 b, e, m; .reg .pred q; mov.b32 b, $1; and.b32 m, b, 0x7fffff; "
        "and.b32 e, b, 0x7f800000; setp.ne.u32 q, m, 0; @q add.u32 e, e, 0x800000; "
        "mov.b32 $0, e; }",
        "=f,f",
        [Float32(x)],
        [Float32],
        loc=loc,
        ip=ip,
    )
