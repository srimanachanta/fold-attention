"""The device quantisers every cache writer shares.

A row is split over `lanes` threads, each holding a power-of-two run of its
elements. The Hadamard is `utils.hadamard`'s butterfly, the same adds in the
same order: the stages inside a thread, then the rest across the row's lanes
by shuffle. Every multiply and divide torch rounds on its own is an `.rn`
instruction (`f32_rn`), so the planes are `decode.cache`'s torch quantisers to
the bit, and a row's bits do not depend on how the row is split or batched.
"""

from __future__ import annotations

import math

import cutlass
from cutlass import Float32, Uint32, cute

from ..utils import LOG2E
from .cache import INV127, swizzle_of
from .ptx import (
    MAGIC,
    butterfly,
    f32_rn,
    half_to_f32,
    ld_global_v4,
    pack_e4m3x4,
    pack_int8x2,
    pack_int8x4,
    st_global_b16,
    st_global_b32,
    st_global_v4,
    st_shared,
    st_shared_b16,
    unpack_e4m3x4,
)


def unpack(words, f16):
    """Words of two 16-bit floats as their fp32 values, low half first."""
    x = []
    for w in words:
        x.append(half_to_f32(w, False, f16))
        x.append(half_to_f32(w, True, f16))
    return x


def load16(addr, f16):
    """Sixteen 16-bit floats at a 32-byte-aligned global address, as fp32
    values and as their raw words."""
    w = ld_global_v4(addr, True) + ld_global_v4(addr + 16, True)
    return unpack(w, f16), w


def rotate(x, lane, lanes, hscale):
    """The Hadamard of a row split over `lanes` threads, this thread's part
    `x`: its own stages in the thread, the rest across the lanes."""
    n = len(x)
    h = 1
    while h < n:
        y = list(x)
        for i in range(n):
            if (i & h) == 0:
                y[i] = x[i] + x[i + h]
                y[i + h] = x[i] - x[i + h]
        x = y
        h *= 2
    off = 1
    while off < lanes:
        upper = lane & off
        x = [butterfly(v, cute.arch.shuffle_sync_bfly(v, off), upper) for v in x]
        off *= 2
    return [f32_rn("mul.rn.f32", v, hscale) for v in x]


def row_amax(x, lanes):
    """The row's largest magnitude, a tree in the thread and then across the
    lanes: the max is exact in any order."""
    m = [cute.arch.fmax(v, -v) for v in x]
    while len(m) > 1:
        m = [cute.arch.fmax(m[2 * i], m[2 * i + 1]) for i in range(len(m) // 2)]
    m = m[0]
    off = 1
    while off < lanes:
        m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, off))
        off *= 2
    return m


def key_scale(x, lanes):
    """A key's scale from this thread's part of its rotated row: `cache.key_scale`,
    the row's `amax / 127` rounded up to the next bf16, as its f32 value and
    its bf16 bits in the low half of a word."""
    e = f32_rn("mul.rn.f32", cute.arch.fmax(row_amax(x, lanes), 1e-30), INV127)
    hi = (e.bitcast(Uint32) + Uint32(0xFFFF)) >> 16
    return (hi << 16).bitcast(Float32), hi


def int8_planes(x, e, second=True):
    """`x = e (a + b / 256)` as `cache._int8_planes` computes it, each plane's
    integers as `MAGIC + n` floats (whose low bytes are the int8s), and plane
    A's integers as floats.

    Adding `MAGIC` rounds to the nearest integer, ties to even, and leaves it
    in the low mantissa bits, so neither the rounding nor the byte costs a
    conversion, which issues at a quarter of the fp32 rate. Rounding commutes
    with the clamp to integer bounds, so the clamp goes first."""
    t = f32_rn("div.rn.f32", 256.0, e)
    ie = f32_rn("div.rn.f32", 1.0, e)
    ma, mb, a = [], [], []
    for v in x:
        u = cute.arch.fmax(cute.arch.fmin(f32_rn("mul.rn.f32", v, ie), 127.0), -128.0)
        m = f32_rn("add.rn.f32", u, MAGIC)
        ma.append(m)
        if second:
            pa = f32_rn("sub.rn.f32", m, MAGIC)
            r = f32_rn("sub.rn.f32", v, f32_rn("mul.rn.f32", pa, e))
            w = cute.arch.fmax(cute.arch.fmin(f32_rn("mul.rn.f32", r, t), 127.0), -128.0)
            a.append(pa)
            mb.append(f32_rn("add.rn.f32", w, MAGIC))
    return ma, mb, a


def pack_words(m):
    """A plane's `MAGIC + n` floats, four or more of them, as int8 words."""
    return [
        pack_int8x4(m[4 * j], m[4 * j + 1], m[4 * j + 2], m[4 * j + 3]) for j in range(len(m) // 4)
    ]


def quantize_q(words, lane, lanes, D, f16):
    """This thread's part of one query row, scaled by `LOG2E / sqrt(D)`,
    rotated and split into planes against the row's own amax: `(plane A,
    plane B, scale)`, the planes as `int8_planes` floats."""
    qscale = LOG2E / math.sqrt(D)
    x = [f32_rn("mul.rn.f32", v, qscale) for v in unpack(words, f16)]
    x = rotate(x, lane, lanes, 1.0 / math.sqrt(D))
    e = f32_rn("mul.rn.f32", cute.arch.fmax(row_amax(x, lanes), 1e-30), INV127)
    ma, mb, _ = int8_planes(x, e)
    return ma, mb, e


def store_plane_shared(addr, m):
    """A plane's `MAGIC + n` floats as int8s at a shared address, one store."""
    if len(m) >= 4:
        st_shared(addr, pack_words(m))
    else:
        st_shared_b16(addr, pack_int8x2(m[0], m[1]))


def store_plane_global(addr, m):
    """A plane's `MAGIC + n` floats as int8s at a global address."""
    if len(m) >= 4:
        w = pack_words(m)
        if len(w) == 4:
            st_global_v4(addr, w)
        else:
            for j in range(len(w)):
                st_global_b32(addr + 4 * j, w[j])
    else:
        st_global_b16(addr, pack_int8x2(m[0], m[1]))


@cute.jit
def put_k(xk, pKa, pKb, pEk, prow, off, lane, ok, D: cutlass.Constexpr):
    """A rotated key's two planes at cache row `prow`, swizzled by its slot
    `off` in the page, and its scale, from the `D / 16` lanes that each hold
    sixteen of its elements. Returns this lane's sixteen plane-A integers and
    the key's scale."""
    NU, SH, _ = swizzle_of(D)
    ek, eb = key_scale(xk, D // 16)
    ma, mb, a = int8_planes(xk, ek)
    wa, wb = pack_words(ma), pack_words(mb)
    kaddr = prow * D + ((lane ^ ((off >> SH) & (NU - 1))) * 16)
    if ok:
        st_global_v4(pKa + kaddr, wa)
        st_global_v4(pKb + kaddr, wb)
        if lane == 0:
            st_global_b16(pEk + prow * 2, eb)
    return a, ek


@cute.jit
def put_v(xv, wv, pVa, pVb, prow, lane, ok, IEVS, D: cutlass.Constexpr, V8: cutlass.Constexpr):
    """A value unit at cache row `prow`: two e4m3 planes in the scale's
    units, or its own 16-bit words."""
    vaddr = prow * D + lane * 16
    if cutlass.const_expr(V8):
        w = [
            cute.arch.fmax(cute.arch.fmin(f32_rn("mul.rn.f32", v, IEVS), 448.0), -448.0) for v in xv
        ]
        ca = [pack_e4m3x4(w[4 * j], w[4 * j + 1], w[4 * j + 2], w[4 * j + 3]) for j in range(4)]
        fa = []
        for j in cutlass.range_constexpr(4):
            fa = fa + unpack_e4m3x4(ca[j])
        r = [f32_rn("mul.rn.f32", f32_rn("sub.rn.f32", w[i], fa[i]), 8.0) for i in range(16)]
        r = [cute.arch.fmax(cute.arch.fmin(v, 448.0), -448.0) for v in r]
        cb = [pack_e4m3x4(r[4 * j], r[4 * j + 1], r[4 * j + 2], r[4 * j + 3]) for j in range(4)]
        if ok:
            st_global_v4(pVa + vaddr, ca)
            st_global_v4(pVb + vaddr, cb)
    else:
        if ok:
            st_global_v4(pVa + vaddr * 2, wv[:4])
            st_global_v4(pVa + vaddr * 2 + 16, wv[4:])
