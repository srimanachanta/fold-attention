"""Device helpers shared by the decode kernels (`kernel`, `packed`, `wide`) and
the mass reference."""

import cutlass
from cutlass import cute
from cutlass.experimental import primitives
from cutlass.experimental.primitives.nvvm_wrapper import (
    cp_async_bulk_shared_cluster_global,
    nanosleep,
)

from .cache import KBR
from .ptx import st_global_f32, vfrag_bf16, vfrag_refine_bf16, zero16

# the sentinel logit a pair that does not exist carries; its weight is exactly +0
NEG = -3.0e38


@cute.jit
def logit_scale(eq, ek):
    """A query row's scale times a key's, one of them divided by the 256 that
    makes `256 c1 + c2 + ...` a logit. Dividing by 256 is exact, so whichever
    carries it, this one rounding is `(eq ek) / 256`'s, wherever a logit is
    formed: the mass reference must reproduce the kernel's logits to the
    bit."""
    return eq * ek


def load_acquire(ptr):
    """A word another CTA publishes with a release, read with acquire."""
    return cutlass.Int32(cute.arch.load(ptr, cutlass.Int32, sem="acquire", scope="gpu"))


def load_relaxed(ptr):
    """A word that is its own flag, read without ordering anything else."""
    return cutlass.Int32(cute.arch.load(ptr, cutlass.Int32, sem="relaxed", scope="gpu"))


def pidx(g, ky, BN: int):
    """Where query row `g`'s weight for key `ky` lives in the weight buffer.

    The buffer is the K-major swizzle atom a descriptor names for B. The
    swizzle has period eight, so a group wider than eight rows wraps.
    """
    return g * BN + (((ky >> 3) ^ (g & 7)) << 3) + (ky & 7)


def coarse_logit(c1, ng, i, NG: int):
    """Register `i` of row block `ng` of the stacked coarse logit as a float:
    `c1` holds Q plane A's NG blocks then plane B's, and plane B is 1/256 of
    plane A, so the integer logit is `256 c1 + c2`."""
    return cutlass.Float32(c1[ng * 4 + i] * 256 + c1[(NG + ng) * 4 + i])


def refine_logit(c3, ng, i, NG: int):
    """Register `i` of row block `ng` of the refine as a float: `c3` holds
    K plane B against Q plane A's NG blocks then plane B's, so the logit's
    refinement is `c3a + c3b / 256` in the coarse logit's units."""
    return cutlass.Float32(c3[ng * 4 + i]) + cutlass.Float32(c3[(NG + ng) * 4 + i]) * (1.0 / KBR)


def gather_dst(rr, uu, D: int, BN: int, EB: int, SW: int):
    """Where row `rr`'s 16-byte unit `uu` lands in a gathered tile.

    `SW = 0` is a plain row-major staging tile. `SW = 1` is the layout a
    descriptor names: 128-byte rows of 16-byte units, unit `u` of row `r` at
    `u ^ (r & 7)`, and a row wider than 128 bytes cut into blocks of `BN` rows.
    `SW = 2` is the staging tile the register value operand reads.
    """
    UEL = 16 // EB
    if not SW:
        return rr * D + uu * UEL
    if SW == 2:
        # A lane reads one word of sixteen rows, so the four lanes of a quad
        # would hit four rows on one bank. Those rows differ in (r >> 1) & 3,
        # which this XOR spreads, capped at the row's own unit count so it stays
        # a relabelling inside the row.
        NUR = (D * EB) // 16
        return rr * D + (uu ^ (((rr >> 1) & (NUR // 2 - 1)) << 1)) * UEL
    RWE = 128 // EB
    NUB = RWE // UEL
    return (uu // NUB) * (BN * RWE) + rr * RWE + ((uu % NUB) ^ (rr & 7)) * UEL


def contiguous_step(step: int, NSEG: int, PG: bool, SW: int) -> bool:
    """Whether rows `step` apart land a constant distance apart in both the
    cache and a gathered tile: the tile's rows are one run of the cache, and
    the tile's layout repeats its unit order every `step` rows."""
    return (NSEG == 1 or not PG) and SW in (0, 1) and step % 8 == 0


def row_elems(D: int, EB: int, SW: int) -> int:
    """How far apart two consecutive rows sit in a gathered tile, in elements."""
    return D if SW == 0 else 128 // EB


def smem_view(addr, dtype, swz, layout):
    """A swizzled shared tile at a 32-bit shared address, as a `wgmma`
    operand; the address's low ten bits must be zero."""
    return cute.make_tensor(
        cute.recast_ptr(
            cute.make_ptr(dtype, cutlass.Int64(addr), cute.AddressSpace.smem, assumed_align=1024),
            swz,
            dtype,
        ),
        layout,
    )


@cute.jit
def paged_row(mPgT, breq, hkv, key, PS: cutlass.Constexpr, HKV: cutlass.Constexpr):
    """The flat cache row of `key` under a page table laid out
    (page, kv head, slot)."""
    return (cutlass.Int64(mPgT[(breq, key // PS)]) * HKV + hkv) * PS + key % PS


@cute.jit
def cascade_row(w):
    """A stacked row's `(row group, row)` from its row-map word: slot in the
    high bits, row next, and in bit 0 whether the row exists."""
    return w >> 7, (w >> 1) & 63


@cute.jit
def tile_segments(
    sSeg,
    slot,
    base,
    n,
    mPgT,
    breq,
    hkv,
    PS: cutlass.Constexpr,
    SEGW: cutlass.Constexpr,
    NSEG: cutlass.Constexpr,
    HKV: cutlass.Constexpr,
    PG: cutlass.Constexpr,
):
    """The cache row each page segment of a tile's `n` rows starts at, for
    the thread that issues the tile's copy. A segment past the request's end
    is not looked up: its page index is past the table's row."""
    if cutlass.const_expr(PG):
        # Cache rows fit int32; multiplying them by D for byte addresses does not.
        for j in cutlass.range_constexpr(NSEG):
            if cutlass.const_expr(j == 0):
                sSeg[(slot, j)] = cutlass.Int32(paged_row(mPgT, breq, hkv, base, PS, HKV))
            elif j * SEGW < n:
                sSeg[(slot, j)] = cutlass.Int32(
                    paged_row(mPgT, breq, hkv, base + j * SEGW, PS, HKV)
                )


@cute.jit
def bulk_tile(
    sDst,
    gSrc,
    sSeg,
    slot,
    bar,
    base,
    n,
    D: cutlass.Constexpr,
    SEGW: cutlass.Constexpr,
    NSEG: cutlass.Constexpr,
    PG: cutlass.Constexpr,
):
    """The tile's dense stream: one bulk copy per page segment it covers."""
    if cutlass.const_expr(PG):
        for j in cutlass.range_constexpr(NSEG):
            nj = n - j * SEGW
            if nj > SEGW:
                nj = SEGW
            if nj > 0:
                cp_async_bulk_shared_cluster_global(
                    sDst.iterator + j * SEGW * D,
                    gSrc.iterator + cutlass.Int64(sSeg[(slot, j)]) * D,
                    bar,
                    nj * D,
                )
    else:
        cp_async_bulk_shared_cluster_global(
            sDst.iterator, gSrc.iterator + cutlass.Int64(base) * D, bar, n * D
        )


@cute.jit
def bulk_scales(
    pKs,
    gEk,
    sSeg,
    slot,
    bar,
    base,
    n,
    SEGW: cutlass.Constexpr,
    NSEG: cutlass.Constexpr,
    PG: cutlass.Constexpr,
):
    """The tile's key scales beside its plane A, into shared memory at `pKs`,
    one bulk copy per page segment. A copy is whole 16-byte units, eight
    scales, so a short segment reads up to seven rows past its end: rows of
    its own page, whose slots are whole multiples of sixteen, or the padding
    `quantize_k` leaves a contiguous cache's rows. `scale_bytes` is what they
    add to the tile."""
    if cutlass.const_expr(PG):
        for j in cutlass.range_constexpr(NSEG):
            nj = n - j * SEGW
            if nj > SEGW:
                nj = SEGW
            if nj > 0:
                cp_async_bulk_shared_cluster_global(
                    pKs + j * SEGW,
                    gEk.iterator + cutlass.Int64(sSeg[(slot, j)]),
                    bar,
                    (nj + 7) // 8 * 16,
                )
    else:
        cp_async_bulk_shared_cluster_global(
            pKs, gEk.iterator + cutlass.Int64(base), bar, (n + 7) // 8 * 16
        )


def scale_bytes(n):
    """The bytes `bulk_scales` moves for a tile of `n` rows: every segment
    but the last is whole units, so only the total rounds."""
    return (n + 7) // 8 * 16


@cute.jit
def key_scale(sKs, i):
    """Slot `i`'s `ek / 256`. A row past the tile's end holds whatever its
    copy's last 16-byte unit or an earlier tile left there, and the tail
    multiplies its projection before a zero weight meets it. `max` and `min`
    return a NaN's other operand, so the clamp makes any such value finite
    without naming the tile's end, which costs D = 64 a CTA per SM. A key
    whose scale reaches 2^100 overflows its logit anyway."""
    return cute.arch.fmin(
        cute.arch.fmax(cutlass.Float32(sKs[i]) * (1.0 / KBR), cutlass.Float32(0.0)),
        cutlass.Float32(2.0**100),
    )


def band_gate(slen, gate, bands, i: int):
    """A dense gate by the request's own length `slen`: `gate` (band 0's)
    up to `bands[0]` keys, then band 1's and past `bands[1]` band 2's.
    `bands` is `DecodeConfig.refine_bands`, `i` 0 for K's gate and 1 for
    V's."""
    return cutlass.Float32(
        cutlass.select_(
            slen > int(bands[1]),
            cutlass.Float32(-bands[3 + 2 * i]),
            cutlass.select_(slen > int(bands[0]), cutlass.Float32(-bands[2 + 2 * i]), gate),
        )
    )


@cute.jit
def split_bounds(
    slen,
    sp,
    SPLIT: cutlass.Constexpr,
    TILE: cutlass.Constexpr,
    INTERLEAVE: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr,
):
    """Split `sp`'s keys of a request of `slen`, as `(lo, hi)`. Under
    `INTERLEAVE` it walks tiles `sp`, `sp + SPLIT`, ... from `lo` to `hi`;
    otherwise it holds the contiguous chunk `[lo, hi)`, `CHUNK` keys when
    positive. A split past the request's end walks no tile and adds zeros,
    which leave every sum's bits as they were."""
    if cutlass.const_expr(INTERLEAVE):
        lo = sp * TILE
        hi = slen
    else:
        chunk = ((slen + SPLIT - 1) // SPLIT + TILE - 1) // TILE * TILE
        if cutlass.const_expr(CHUNK > 0):
            chunk = cutlass.Int32(CHUNK)
        lo = sp * chunk
        hi = lo + chunk
        if hi > slen:
            hi = slen
    return lo, hi


@cute.jit
def split_tiles(
    slen,
    sp,
    lo,
    hi,
    SPLIT: cutlass.Constexpr,
    TILE: cutlass.Constexpr,
    INTERLEAVE: cutlass.Constexpr,
):
    """The keys and tiles `split_bounds`'s split walks, as `(nrows,
    n_tiles)`, so an over-hanging split walks no empty tile."""
    nrows = hi - lo
    if nrows < 0:
        nrows = 0
    n_tiles = (nrows + TILE - 1) // TILE
    if cutlass.const_expr(INTERLEAVE):
        n_tiles = ((slen + TILE - 1) // TILE - sp + SPLIT - 1) // SPLIT
    return nrows, n_tiles


def cache_views(bh, S, SP, D: int, EKS: int, PAGED: bool, mKa, mKb, mEk, *mVs):
    """Row group `bh`'s K planes, key scales and V planes `mVs`, as
    `(gKa, gKb, gEk, *gVs)`. A paged cache is addressed by its flat row. The
    offsets are 64-bit: a contiguous cache's row groups run past 2 GiB."""
    koff = cutlass.Int64(bh) * SP * D
    voff = cutlass.Int64(bh) * S * D
    if PAGED:
        koff = 0
        voff = 0
    gKa = cute.make_tensor(mKa.iterator + koff, cute.make_layout((SP, D), stride=(D, 1)))
    eoff = bh * EKS
    if PAGED:
        eoff = 0
    gEk = cute.make_tensor(mEk.iterator + eoff, cute.make_layout(SP))
    gKb = cute.make_tensor(mKb.iterator + koff, cute.make_layout((SP, D), stride=(D, 1)))
    gVs = [
        cute.make_tensor(m.iterator + voff, cute.make_layout((S, D), stride=(D, 1))) for m in mVs
    ]
    return (gKa, gKb, gEk, *gVs)


@cute.jit
def zero_tile(sT, NBY: cutlass.Constexpr, tidx, NTH: cutlass.Constexpr):
    """The first `NBY` bytes of shared tile `sT` set to zero, 16 bytes a
    thread of `NTH`."""
    N = NBY // 16
    p = cute.recast_ptr(sT.iterator, None, cutlass.Uint8)
    for i in cutlass.range_constexpr((N + NTH - 1) // NTH):
        if i * NTH + tidx < N:
            zero16((p + (i * NTH + tidx) * 16).toint())


@cute.jit
def front_length(mRdy, sRdy, bh, tidx, G: cutlass.Constexpr, QK: cutlass.Constexpr):
    """Row group `bh`'s length, once the front has added it to the ready
    word, shifted past a count, with its Q and new token in the cache. Z and
    the cut are words of their own (`front_rows`).

    One thread polls, backing off, so the CTAs of a step do not crowd the
    word's L2 slice while the front writes it; the barrier carries its
    acquire to the rest, and the bulk copies read through the async proxy.
    `QK` is `heuristics.front_ownership`'s mode."""
    if tidx == 0:
        rdy = mRdy.iterator + bh * (1 + 2 * G)
        v = load_acquire(rdy)
        while (v >> 2) == 0 or (QK == 2 and (v & 3) == 0):
            nanosleep(256)
            v = load_acquire(rdy)
        cute.arch.fence_proxy("async.global")
        sRdy[0] = v >> 2
    cute.arch.sync_threads()
    return sRdy[0]


@cute.jit
def front_rows(
    mRdy,
    sZ,
    bh,
    tidx,
    G: cutlass.Constexpr,
    ROWS: cutlass.Constexpr,
    TAIL: cutlass.Constexpr,
    QK: cutlass.Constexpr,
):
    """Warp 0's wait for the front's reference: row `r`'s Z into `sZ[r]`
    and its cut into `sZ[ROWS + r]`. The count's acquire covers the V sums
    the loop and the epilogue read; Z and the cut are each their own flag,
    complemented so that zero means not yet written."""
    rdy = mRdy.iterator + bh * (1 + 2 * G)
    if cutlass.const_expr(TAIL or QK == 1):
        # The tail publishes its rows. With mass-owned Q/K, the append also
        # publishes V here rather than in the length.
        need = 2 if QK in (1, 2) and TAIL else 1
        v = load_acquire(rdy)
        while (v & 3) < need:
            nanosleep(64)
            v = load_acquire(rdy)
        cute.arch.fence_proxy("async.global")
    if tidx < G:
        zw = load_relaxed(rdy + 1 + tidx)
        while zw == 0:
            nanosleep(32)
            zw = load_relaxed(rdy + 1 + tidx)
        cw = load_relaxed(rdy + 1 + G + tidx)
        while cw == 0:
            nanosleep(32)
            cw = load_relaxed(rdy + 1 + G + tidx)
        sZ[tidx] = (zw ^ cutlass.Int32(-1)).bitcast(cutlass.Float32)
        sZ[ROWS + tidx] = (cw ^ cutlass.Int32(-1)).bitcast(cutlass.Float32)


@cute.jit
def front_sums(mRdy, bh, G: cutlass.Constexpr, QK: cutlass.Constexpr):
    """The wait for the front's V sums where only the epilogue reads them
    (V's mean): long out by then, one acquire each and no barrier."""
    rdy = mRdy.iterator + bh * (1 + 2 * G)
    v = load_acquire(rdy)
    need = 2 if QK == 2 else 1
    while (v & 3) < need:
        nanosleep(64)
        v = load_acquire(rdy)


@cute.jit
def issue_first(
    sKa,
    gKa,
    sKs,
    gEk,
    sSeg,
    sRed,
    fullb,
    mPgT,
    pid,
    bh,
    breq,
    hkv,
    lo,
    n0,
    D: cutlass.Constexpr,
    PAGE: cutlass.Constexpr,
    SEGW: cutlass.Constexpr,
    NSEG: cutlass.Constexpr,
    HKV: cutlass.Constexpr,
    NW: cutlass.Constexpr,
    PAGED: cutlass.Constexpr,
):
    """Thread 0's prologue: the block index and row group into `sRed`'s
    last row, where `issue_next` and the epilogue read them, `fullb`'s
    init, and the first tile's `n0` rows of plane A and key scales."""
    sRed[(2 * NW, 0)] = cutlass.Int32(pid).bitcast(cutlass.Float32)
    sRed[(2 * NW, 1)] = cutlass.Int32(bh).bitcast(cutlass.Float32)
    cute.arch.mbarrier_init(fullb, 1)
    cute.arch.mbarrier_init_fence()
    ntx0 = cutlass.Int32(0)
    if n0 > 0:
        ntx0 = n0 * D + scale_bytes(n0)
    cute.arch.mbarrier_arrive_and_expect_tx(fullb, ntx0)
    if n0 > 0:
        tile_segments(sSeg, 0, lo, n0, mPgT, breq, hkv, PAGE, SEGW, NSEG, HKV, PAGED)
        bulk_tile(sKa, gKa, sSeg, 0, fullb, lo, n0, D, SEGW, NSEG, PAGED)
        bulk_scales(sKs.iterator, gEk, sSeg, 0, fullb, lo, n0, SEGW, NSEG, PAGED)


@cute.jit
def issue_next(
    sKa,
    gKa,
    sKs,
    gEk,
    sSeg,
    sRed,
    fullb,
    mPgT,
    rPg,
    t,
    nb,
    nn,
    hi,
    D: cutlass.Constexpr,
    PAGE: cutlass.Constexpr,
    SEGW: cutlass.Constexpr,
    NSEG: cutlass.Constexpr,
    HKV: cutlass.Constexpr,
    NW: cutlass.Constexpr,
    PAGED: cutlass.Constexpr,
    KSS: cutlass.Constexpr,
    TSTEP: cutlass.Constexpr,
    PFP: cutlass.Constexpr,
):
    """Tile `t + 1`'s plane A and key scales, `nn` rows from `nb`, by the one
    thread that arrives on its phase of `fullb`. The scales land `KSS` slots
    in on an odd tile. Under `PFP`, a cache paged at least a tile wide, the
    tile's page entry is `rPg[0]`, loaded a tile ahead, which then loads the
    entry of the tile `TSTEP` keys on: the issuing thread lags its
    warpgroup by the copies' issue alone."""
    ntx = cutlass.Int32(0)
    if nn > 0:
        ntx = nn * D + scale_bytes(nn)
    cute.arch.mbarrier_arrive_and_expect_tx(fullb, ntx)
    if cutlass.const_expr(PFP):
        bhi = cutlass.Float32(sRed[(2 * NW, 1)]).bitcast(cutlass.Int32)
    if nn > 0:
        if cutlass.const_expr(PFP):
            sSeg[((t + 1) % 2, 0)] = (rPg[0] * HKV + bhi % HKV) * PAGE + nb % PAGE
        else:
            bhn = cutlass.Float32(sRed[(2 * NW, 1)]).bitcast(cutlass.Int32)
            tile_segments(
                sSeg, (t + 1) % 2, nb, nn, mPgT, bhn // HKV, bhn % HKV, PAGE, SEGW, NSEG, HKV, PAGED
            )
        bulk_tile(sKa, gKa, sSeg, (t + 1) % 2, fullb, nb, nn, D, SEGW, NSEG, PAGED)
        bulk_scales(
            sKs.iterator + ((t + 1) % 2) * KSS,
            gEk,
            sSeg,
            (t + 1) % 2,
            fullb,
            nb,
            nn,
            SEGW,
            NSEG,
            PAGED,
        )
    if cutlass.const_expr(PFP):
        n2 = nb + TSTEP
        if n2 < hi:
            rPg[0] = mPgT[(bhi // HKV, n2 // PAGE)]


@cute.jit
def gather_row(
    sSeg, slot, base, rr, SEGW: cutlass.Constexpr, NSEG: cutlass.Constexpr, PG: cutlass.Constexpr
):
    """The cache row of tile row `rr`."""
    if cutlass.const_expr(PG):
        if cutlass.const_expr(NSEG == 1):
            # one page covers the tile, so `slot` already holds its base
            return slot + rr
        return sSeg[(slot, rr // SEGW)] + (rr % SEGW)
    return base + rr


@cute.jit
def mask_word(msk, k0, NWD: cutlass.Constexpr):
    """A tile's mask from key `k0` on, at bit 0. `k0` is a warp's first key,
    a multiple of 16, so its sixteen keys sit in one word."""
    m = msk[0]
    for j in cutlass.range_constexpr(1, NWD):
        if k0 >= j * 32:
            m = msk[j]
    return m >> (k0 % 32)


@cute.jit
def mask_any(msk, NWD: cutlass.Constexpr):
    m = msk[0]
    for j in cutlass.range_constexpr(1, NWD):
        m = m | msk[j]
    return m != 0


@cute.jit
def gather_rows(
    tidx,
    base,
    sDst,
    gSrc,
    msk,
    BN: cutlass.Constexpr,
    D: cutlass.Constexpr,
    NT: cutlass.Constexpr,
    EB: cutlass.Constexpr,
    NWD: cutlass.Constexpr,
    CM: cutlass.Constexpr,
    SW: cutlass.Constexpr,
    sSeg,
    sslot,
    SEGW: cutlass.Constexpr,
    NSEG: cutlass.Constexpr,
    PG: cutlass.Constexpr,
):
    """The rows a register bitmask keeps, into shared, 16 bytes a thread.

    A declined row is not issued. Its slot keeps whatever an earlier tile left
    there, which is finite, and meets a zero weight.

    A thread's rows are `r0` plus multiples of `NROW`, which divides 32, so
    its bit of each row is at a constant offset once the words are shifted
    down by `r0`: one shift per word and tile rather than a mask register per
    row.
    """
    LPR = (D * EB) // 16
    NROW = NT // LPR
    uu = tidx % LPR
    c0 = uu * (16 // EB)
    r0 = tidx // LPR
    NIT = (BN + NROW - 1) // NROW
    ms = [msk[j] >> r0 for j in range(NWD)]
    # Contiguous rows and a row step that keeps the swizzle's row phase make
    # every address the first one plus a constant, so the loop issues copies
    # rather than rebuilding a 64-bit address for each.
    step = contiguous_step(NROW, NSEG, PG, SW)
    if cutlass.const_expr(step):
        src0 = gSrc.iterator + (
            cutlass.Int64(gather_row(sSeg, sslot, base, r0, SEGW, NSEG, PG)) * D + c0
        )
        dst0 = sDst.iterator + gather_dst(r0, uu, D, BN, EB, SW)
    for it in cutlass.range_constexpr(NIT):
        rr = it * NROW + r0
        if (ms[(it * NROW) // 32] >> ((it * NROW) % 32)) & 1 != 0:
            if cutlass.const_expr(step):
                primitives.cp_async_shared_global(
                    dst0 + it * NROW * row_elems(D, EB, SW), src0 + it * NROW * D, 16, CM
                )
            else:
                primitives.cp_async_shared_global(
                    sDst.iterator + gather_dst(rr, uu, D, BN, EB, SW),
                    gSrc.iterator
                    + (cutlass.Int64(gather_row(sSeg, sslot, base, rr, SEGW, NSEG, PG)) * D + c0),
                    16,
                    CM,
                )


@cute.jit
def gather_warp_rows(
    lane,
    kw0,
    base,
    sDst,
    gSrc,
    msk,
    BNW: cutlass.Constexpr,
    BN: cutlass.Constexpr,
    D: cutlass.Constexpr,
    EB: cutlass.Constexpr,
    NWD: cutlass.Constexpr,
    CM: cutlass.Constexpr,
    SW: cutlass.Constexpr,
    sSeg,
    sslot,
    SEGW: cutlass.Constexpr,
    NSEG: cutlass.Constexpr,
    PG: cutlass.Constexpr,
):
    """As `gather_rows`, but a warp gathers only its own `BNW` keys from
    `kw0`, which are the rows of a matmul's A its share of the `wgmma`
    reads. The rows are then the warp's alone: a warp barrier publishes them,
    and the warp may overwrite them once its own matmul is done."""
    LPR = (D * EB) // 16
    RPI = 32 // LPR
    NIT = BNW // RPI
    uu = lane % LPR
    c0 = uu * (16 // EB)
    r0 = lane // LPR
    mw = mask_word(msk, kw0, NWD) >> r0
    step = contiguous_step(RPI, NSEG, PG, SW)
    if cutlass.const_expr(step):
        src0 = gSrc.iterator + (
            cutlass.Int64(gather_row(sSeg, sslot, base, kw0 + r0, SEGW, NSEG, PG)) * D + c0
        )
        dst0 = sDst.iterator + gather_dst(kw0 + r0, uu, D, BN, EB, SW)
    for it in cutlass.range_constexpr(NIT):
        rr = kw0 + it * RPI + r0
        if (mw >> (it * RPI)) & 1 != 0:
            if cutlass.const_expr(step):
                primitives.cp_async_shared_global(
                    dst0 + it * RPI * row_elems(D, EB, SW), src0 + it * RPI * D, 16, CM
                )
            else:
                primitives.cp_async_shared_global(
                    sDst.iterator + gather_dst(rr, uu, D, BN, EB, SW),
                    gSrc.iterator
                    + (cutlass.Int64(gather_row(sSeg, sslot, base, rr, SEGW, NSEG, PG)) * D + c0),
                    16,
                    CM,
                )


@cute.jit
def value_fragment(
    fV4, ra, rb_addr, mRq, kk, h, rfv, off, off2, NVB: int = 2, LDA: int = 128, LDB: int = 128
):
    """Keys `16 kk + 8 h` to `+ 7` of the register value operand, out of the
    e4m3 staging tile, with V's second plane for the keys whose gate it
    earned. A lane owns `2 NVB` channels of two keys; block `jb` takes the pair
    `(2 jb, 2 jb + 1)`. `mRq` is the refine mask already shifted down by
    `2 tq`, the lane's first key of each eight."""
    u = 4 * kk + 2 * h
    nc = 2 * NVB
    # the writes are spelled once per arm: `rfv` is dynamic, and a tuple
    # assigned on both sides of a dynamic branch is not a name the DSL can join
    if rfv:
        rb = (mRq[kk // 2] >> (16 * (kk % 2) + 8 * h)) & 3
        fr = vfrag_refine_bf16(
            ra, rb_addr, 0 - (rb & 1), 0 - ((rb >> 1) & 1), off, off2, LDA, LDB, nc
        )
        for jb in cutlass.range_constexpr(NVB):
            fV4[jb][u] = fr[2 * jb]
            fV4[jb][u + 1] = fr[2 * jb + 1]
    else:
        fp = vfrag_bf16(ra, off, LDA, nc)
        for jb in cutlass.range_constexpr(NVB):
            fV4[jb][u] = fp[2 * jb]
            fV4[jb][u + 1] = fp[2 * jb + 1]


def gmem_vec(t, off, n: int):
    """`n` consecutive f32 of global tensor `t` from element `off`, as a view
    CuTe copies with one vector access: `off` is a multiple of `n` and the
    tensor starts 16-byte aligned, which the pointer is told."""
    p = cute.make_ptr(
        cutlass.Float32, (t.iterator + off).toint(), cute.AddressSpace.gmem, assumed_align=4 * n
    )
    return cute.make_tensor(p, cute.make_layout(n))


def smem_vec(t, off, n: int):
    """`gmem_vec` for `n` f32 at element `off` of a shared tile of any type,
    `off` a multiple of four. It claims 16 bytes, the widest shared load, even
    where `n` floats are aligned further: the wider claim picks a copy that
    costs D = 128's tail builds 22 registers."""
    p = cute.make_ptr(
        cutlass.Float32, t.iterator.toint() + off * 4, cute.AddressSpace.smem, assumed_align=16
    )
    return cute.make_tensor(p, cute.make_layout(n))


def partial_slot(pid, bh, sp, SPLIT: int, SLOTS: int, SLOT0: int, ORDERED: bool, DIRECT: bool):
    """The partial slot a CTA writes. With one level it is the block index,
    and saying so is worth 2.2% truncated: `bh * SLOTS + sp` would keep two
    values live across the tile loop."""
    if SLOTS != SPLIT or SLOT0:
        return bh * SLOTS + SLOT0 + sp
    if ORDERED and DIRECT:
        return bh
    return pid


@cute.jit
def load_basis(
    sV, mVr, bh, tidx, D: cutlass.Constexpr, R: cutlass.Constexpr, NT: cutlass.Constexpr
):
    """Row group `bh`'s rank-`R` basis, `(D, R)` f32, into V's tile once the
    last value matmul has released it, as one `cp.async` group. Read from
    global memory, the expansion had its loads hoisted as far as registers
    allowed: 64 of them at D = 128, which cost the tail builds a CTA per SM."""
    NVRU = D * R // 4
    for vi in cutlass.range_constexpr((NVRU + NT - 1) // NT):
        vu = vi * NT + tidx
        if vu < NVRU:
            primitives.cp_async_shared_global(
                cute.recast_ptr(sV.iterator, None, cutlass.Float32) + vu * 4,
                mVr.iterator + (bh * D * R + vu * 4),
                16,
                "cg",
            )
    cute.arch.cp_async_commit_group()


def add_rank_term(xs, ccs, sYd, sV, gq, R: int):
    """Row `gq`'s dropped keys' K-predicted V added to its channels `ccs`
    in `xs`: its rank sums in `sYd` expanded by the basis `load_basis`
    staged, once per row."""
    yr = [sYd[gq * R + j] for j in range(R)]
    vrc = cute.make_rmem_tensor((R,), cutlass.Float32)
    for c in range(len(xs)):
        cute.autovec_copy(smem_vec(sV, ccs[c] * R, R), vrc)
        acc = yr[0] * vrc[0]
        for j in range(1, R):
            acc = acc + yr[j] * vrc[j]
        xs[c] = xs[c] + acc


@cute.jit
def store_rank_sums(
    mVr,
    sYd,
    pix,
    tidx,
    NYG: cutlass.Constexpr,
    NYD: cutlass.Constexpr,
    NW: cutlass.Constexpr,
    NT: cutlass.Constexpr,
):
    """The CTA's `NYG` rank sums, each warp's `NYD`-word slots of `sYd`
    added in warp order, to partial slot `pix` of the combine's
    `(slot, row, rank)` operand, which the combine expands once per row."""
    for iy in cutlass.range_constexpr((NYG + NT - 1) // NT):
        ey = iy * NT + tidx
        if ey < NYG:
            ys = sYd[ey]
            for w in cutlass.range_constexpr(1, NW):
                ys = ys + sYd[w * NYD + ey]
            st_global_f32(mVr.iterator + (pix * NYG + ey), [ys])
