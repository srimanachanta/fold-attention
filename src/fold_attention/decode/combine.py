"""The split combine, launched as the decode's programmatic dependent."""

import cutlass
from cutlass import cute

from .device import gmem_vec, smem_vec


@cute.kernel
def combine(
    mO: cute.Tensor,
    mL: cute.Tensor,
    mCnt: cute.Tensor,
    mOo: cute.Tensor,
    mLo: cute.Tensor,
    mCo: cute.Tensor,
    mYd: cute.Tensor,
    mVr: cute.Tensor,
    mRdy: cute.Tensor,
    mOrd: cute.Tensor,
    EVS: cutlass.Float32,
    G: cutlass.Constexpr,
    D: cutlass.Constexpr,
    SLOTS: cutlass.Constexpr,
    FRONT: cutlass.Constexpr,
    CLEAR: cutlass.Constexpr,
    ORD: cutlass.Constexpr,
    WARPS: cutlass.Constexpr,
    OUT16: cutlass.Constexpr,
    TR: cutlass.Constexpr,
):
    """Sum the partial slots and divide, one warp per query row and `D / 32`
    consecutive channels a lane, so every partial is one vector load. One-warp
    blocks spread a small batch's rows over more SMs; larger batches pack
    independent rows into a block.

    With the tail at rank `TR` each split leaves its rank sums in `mYd`, and
    the combine adds them over the splits and expands them by the basis
    `mVr` once per row, in the partials' units.

    The reference is static, so the partials are plain sums: there is no
    per-split rescale. `SLOTS` is the partial slots a row group owns, over
    every split of every cascade level. `EVS` is V's scale, a runtime value so
    that a new cache scale does not compile a new combine.
    """
    block, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    row = block * WARPS + tidx // 32
    lane = cute.arch.lane_idx()
    CPL = D // 32
    NPL = (SLOTS + 31) // 32
    slot = row // G
    g = row % G
    bh = slot
    if cutlass.const_expr(ORD):
        bh = mOrd[slot]
    c0 = lane * CPL
    VS = CPL * TR + 4
    sVr = None
    sb = 0
    if cutlass.const_expr(TR):
        # The basis waits in shared memory, not registers: held across the grid
        # wait it would leave too few for the partials' loads to share one
        # round trip. A lane's rows are padded by four words, so a quarter
        # warp's 16-byte reads land on distinct banks.
        smem = cutlass.memory.SmemAllocator()
        sVr = smem.allocate_tensor(cutlass.Float32, cute.make_layout(WARPS * 32 * VS), 16)
        sb = tidx * VS
        for q in cutlass.range_constexpr(CPL * TR // 4):
            cute.arch.cp_async_shared_global(
                sVr.iterator + (sb + 4 * q), mVr.iterator + ((bh * D + c0) * TR + 4 * q), 16, "cg"
            )
        cute.arch.cp_async_commit_group()
    # everything above is indices and the basis, which run under the decode's tail
    cute.arch.griddepcontrol_wait()
    # Every lane sums the denominator itself, in slot order, and the partials
    # beside it. All of a row's words are loaded before any is used, so they
    # share one round trip.
    xl = [mL[(slot * SLOTS + sp, g)] for sp in range(SLOTS)]
    xo = []
    for sp in cutlass.range_constexpr(SLOTS):
        w = cute.make_rmem_tensor((CPL,), cutlass.Float32)
        cute.autovec_copy(gmem_vec(mO, ((slot * SLOTS + sp) * G + g) * D + c0, CPL), w)
        xo.append(w)
    if cutlass.const_expr(TR):
        # Lane j + TR k holds rank j of slots k, k + 32 / TR, ...: slot s always
        # lands on the same lane and position, so slots past a request's end add
        # zeros and change no bits. Loaded with the partials, ahead of the
        # stores below, which the compiler will not move them past.
        SUB = 32 // TR
        jl = lane % TR
        cls = lane // TR
        yl = []
        for k in cutlass.range_constexpr((SLOTS + SUB - 1) // SUB):
            spy = k * SUB + cls
            yk = cutlass.Float32(0.0)
            if cutlass.const_expr((k + 1) * SUB <= SLOTS):
                yk = mYd[(slot * SLOTS + spy, g, jl)]
            else:
                if spy < SLOTS:
                    yk = mYd[(slot * SLOTS + spy, g, jl)]
            yl.append(yk)
    t = cutlass.Float32(0.0)
    for sp in cutlass.range_constexpr(SLOTS):
        t = t + xl[sp]
    if g == 0:
        # the byte counts, a split a lane, exact in any order
        c = cutlass.Int32(0)
        cr = cutlass.Int32(0)
        for k in cutlass.range_constexpr(NPL):
            i = k * 32 + lane
            if i < SLOTS:
                c = c + mCnt[(slot * SLOTS + i, 0)]
                cr = cr + mCnt[(slot * SLOTS + i, 1)]
        c = cute.arch.warp_redux_sync(c, "add")
        cr = cute.arch.warp_redux_sync(cr, "add")
        if lane == 0:
            mCo[(bh, 0)] = c
            mCo[(bh, 1)] = cr
    if lane == 0:
        mLo[(bh, g)] = t
    y = [cutlass.Float32(0.0) for _ in range(CPL)]
    for sp in cutlass.range_constexpr(SLOTS):
        for c in cutlass.range_constexpr(CPL):
            y[c] = y[c] + xo[sp][c]
    if cutlass.const_expr(TR):
        # a lane's slots in order, then a butterfly over k: a fixed order, so the
        # bits do not depend on which split finished first
        yv = yl[0]
        for k in cutlass.range_constexpr(1, len(yl)):
            yv = yv + yl[k]
        for b in cutlass.range_constexpr(SUB.bit_length() - 1):
            yv = yv + cute.arch.shuffle_sync_bfly(yv, TR << b)
        yjs = [cute.arch.shuffle_sync(yv, j) for j in range(TR)]
        cute.arch.cp_async_wait_group(0)
        for c in cutlass.range_constexpr(CPL):
            for q in cutlass.range_constexpr(TR // 4):
                v4 = cute.make_rmem_tensor((4,), cutlass.Float32)
                cute.autovec_copy(smem_vec(sVr, sb + c * TR + 4 * q, 4), v4)
                for e in cutlass.range_constexpr(4):
                    y[c] = y[c] + v4[e] * yjs[4 * q + e]
    for c in cutlass.range_constexpr(CPL):
        y[c] = y[c] * EVS / t
    if cutlass.const_expr(OUT16 == 0):
        xs = cute.make_rmem_tensor((CPL,), cutlass.Float32)
        for c in cutlass.range_constexpr(CPL):
            xs[c] = y[c]
        cute.autovec_copy(xs, gmem_vec(mOo, (bh * G + g) * D + c0, CPL))
    elif cutlass.const_expr(OUT16 == 1):
        for c in cutlass.range_constexpr(CPL):
            mOo[(bh, g, c0 + c)] = cutlass.BFloat16(y[c])
    else:
        for c in cutlass.range_constexpr(CPL):
            mOo[(bh, g, c0 + c)] = cutlass.Float16(y[c])
    if cutlass.const_expr(FRONT and CLEAR):
        # every decode CTA has read the front's words, so the step clears them
        if lane == 0:
            mRdy[bh * (1 + 2 * G) + 1 + g] = 0
            mRdy[bh * (1 + 2 * G) + 1 + G + g] = 0
            if g == 0:
                mRdy[bh * (1 + 2 * G)] = 0


@cute.jit
def launch_combine(
    mO,
    mL,
    mCnt,
    mOo,
    mLo,
    mCo,
    mYd,
    mVr,
    mRdy,
    mOrd,
    EVS: cutlass.Float32,
    NBH: cutlass.Constexpr,
    G: cutlass.Constexpr,
    D: cutlass.Constexpr,
    SLOTS: cutlass.Constexpr,
    FRONT: cutlass.Constexpr,
    CLEAR: cutlass.Constexpr,
    ORD: cutlass.Constexpr,
    WARPS: cutlass.Constexpr,
    OUT16: cutlass.Constexpr,
    TR: cutlass.Constexpr,
    stream,
):
    combine(
        mO,
        mL,
        mCnt,
        mOo,
        mLo,
        mCo,
        mYd,
        mVr,
        mRdy,
        mOrd,
        EVS,
        G,
        D,
        SLOTS,
        FRONT,
        CLEAR,
        ORD,
        WARPS,
        OUT16,
        TR,
    ).launch(grid=[NBH * G // WARPS, 1, 1], block=[32 * WARPS, 1, 1], stream=stream, use_pdl=True)
