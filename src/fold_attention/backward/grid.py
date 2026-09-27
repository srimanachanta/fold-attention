"""The backward's integer grids, derived from bounds the kernels compute.

Every kernel that quantises or reconstructs a partial evaluates the same bound
with the same rounded instructions, so each arrives at the same scale. The
inputs are the preprocess's maxima (`red.max` over float bits), which do not
depend on the order CTAs finish in or on anything outside the request.

dQ, per (request, KV head): `|sum_j dS_ij K_jd| <= (sum_j |dS_ij|) max|K|`
and, with P summing to one, `sum_j |dS_ij| <= |dO_i| max_j |V_j| + |D_i|`,
where `|V_j| <= C_D max|V|` and `C_D` is sqrt(D) rounded up to a power of two.
dV, on the canonical-record path: `|dV_jd| <= C max_i |dO_id|`, the largest
element of dO, since the bound is per output component. dK there:
`|dK_j| <= C (max|dO| C_D max|V| + max|D|) max|Q|`. `C` bounds the column
mass `sum_{h,i} P_hij`; the only data-free bound is the number of query rows
`G n`, which is why dK needs a 61-bit grid. Every bound is widened by 2^-6 for
the bf16 rounding of P and dS.
"""

import math

import cutlass
from cutlass import Float32, Int32

from ..ptx import f32_rn

# stats[b, h_kv, i], float bits, zero before the call: max |K|, max |V|, max
# ||dO_i|| over the group's rows, max |D_i| over them, and for the record
# path max |dO_id| over every element and max |Q| over the group
ST_K, ST_V, ST_DO, ST_DELTA, ST_DOC, ST_Q = 0, 1, 2, 3, 4, 5
N_STATS = 8

DQ_BITS = 30
DKV_BITS = 61


def root_d(D):
    """sqrt(D) rounded up to a power of two: exact, and a bound."""
    return float(2 ** math.ceil(math.log2(math.sqrt(D))))


def pow2_scale(bound, bits: int) -> Float32:
    """`2^(bits - ceil(log2 (bound (1 + 2^-6))))`, clamped to [2^-100, 2^100].
    A zero bound takes 2^100, since its partials are all zero."""
    b = f32_rn("mul.rn.f32", bound, Float32(1.0 + 2.0**-6))
    bb = b.bitcast(Int32)
    clog = (
        ((bb >> 23) & 0xFF) - 127 + Int32(cutlass.select_((bb & 0x7FFFFF) != 0, Int32(1), Int32(0)))
    )
    sexp = Int32(bits) - clog
    sexp = Int32(cutlass.select_(b > Float32(0.0), sexp, Int32(100)))
    sexp = Int32(cutlass.select_(sexp > 100, Int32(100), sexp))
    sexp = Int32(cutlass.select_(sexp < -100, Int32(-100), sexp))
    return ((sexp + 127) << 23).bitcast(Float32)


def inv_pow2(scale) -> Float32:
    """`1 / scale` for a power of two, from its bits."""
    return (Int32(0x7F000000) - scale.bitcast(Int32)).bitcast(Float32)


def log2_pow2(scale) -> Float32:
    """The exponent of a power of two, as a float."""
    return ((scale.bitcast(Int32) >> 23) - Int32(127)).to(Float32)


def load_stats(mStats, batch_idx, head_kv, which=range(N_STATS)):
    """The maxima `which` of one (request, KV head), by index."""
    return {i: mStats[batch_idx, head_kv, i].bitcast(Float32) for i in which}


def dq_scale(mStats, batch_idx, head_kv, cd: float) -> Float32:
    """dQ's scale for one (request, KV head).

    A constant exponent is free here: `P 2^s` is `exp2(c S - lse + s)`, so
    the softmax's own exponent carries it and the dQ partials leave the GEMM
    on the grid already."""
    st = load_stats(mStats, batch_idx, head_kv, (ST_K, ST_V, ST_DO, ST_DELTA))
    s = f32_rn(
        "add.rn.f32",
        f32_rn("mul.rn.f32", st[ST_DO], f32_rn("mul.rn.f32", st[ST_V], Float32(cd))),
        st[ST_DELTA],
    )
    return pow2_scale(f32_rn("mul.rn.f32", s, st[ST_K]), DQ_BITS)


def dv_scale(st, rows, bits: int) -> Float32:
    """dV's scale from `load_stats`, for `rows` query rows bounding the column
    mass."""
    return pow2_scale(f32_rn("mul.rn.f32", rows, st[ST_DOC]), bits)


def dk_scale(st, rows, cd: float, bits: int) -> Float32:
    """dK's scale from `load_stats`, for `rows` query rows bounding the column
    mass."""
    s = f32_rn(
        "add.rn.f32",
        f32_rn("mul.rn.f32", st[ST_DO], f32_rn("mul.rn.f32", st[ST_V], Float32(cd))),
        st[ST_DELTA],
    )
    return pow2_scale(f32_rn("mul.rn.f32", rows, f32_rn("mul.rn.f32", s, st[ST_Q])), bits)
