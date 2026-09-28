"""The wide decode kernel: query rows along M, keys along N.

The decode kernel puts keys on `wgmma`'s M so that a group of 8 rows can use
the tensor cores at all, and pays for it per (row, key) pair. When many query
rows read one KV head's cache, a cascade's stacked requests or a draft's
positions past 64 rows, the rows fill M on their own and the pass is compute
bound, so this kernel is laid out as a prefill kernel is: `S = Q K^T` with
M = 64 query rows a warpgroup and N = 64 keys, then `O += P V` with P
straight from the logit's registers.

The logit is one fp16 matmul. Both K planes and the key scales arrive by
bulk copy, and the producer warpgroup rounds each tile once to
`fp16(Ka + Kb / 256)` for every row that reads it; each consumer rounds its
rows' Q to `fp16(eq (Qa + Qb / 256))` once. The key's scale multiplies the
accumulator in the weight pass, so a weight is `ex2(ek acc - Z)`: two
instructions and half a pack a pair. The operands keep 11 bits, which puts a
logit within ~2^-11 of its value relative to `|q| |k|`, below the bf16
rounding of the weight itself.

Two consumer warpgroups share each tile and take turns at the tensor cores:
a turn issues one tile's logit matmul and the previous tile's value matmul,
so one warpgroup's weight pass runs under the other's matmuls and under its
own value matmul. The denominator is the weights' own `wgmma` against a
column of ones, so it sums exactly what the value matmul consumed. The
reference is static, so nothing is rescaled and the partials add in the
combine as the decode kernel's do.
"""

from __future__ import annotations

import cutlass
import cutlass.utils.hopper_helpers as sm90
from cutlass import cute
from cutlass.cute.nvgpu import warpgroup
from cutlass.experimental import primitives
from cutlass.experimental.primitives.nvvm_wrapper import cp_async_bulk_shared_cluster_global

from .cache import KBR, swizzle_of
from .config import BN, MR, NWG, PRODUCER_REGS, WIDE_RAW_STAGES, WIDE_STAGES
from .device import (
    bulk_scales,
    bulk_tile,
    cache_views,
    cascade_row,
    gather_dst,
    scale_bytes,
    smem_view,
    split_bounds,
    split_tiles,
    tile_segments,
    zero_tile,
)
from .ptx import ex2, k16_pairs, pack_weights, st_global_f32

I8 = cutlass.Int8
F16 = cutlass.Float16
PT = cutlass.BFloat16


@cute.kernel
def wide_kernel(
    mQa: cute.Tensor,
    mQb: cute.Tensor,
    mEq: cute.Tensor,
    mKa: cute.Tensor,
    mKb: cute.Tensor,
    mEk: cute.Tensor,
    mV: cute.Tensor,
    mVm: cute.Tensor,
    mZ: cute.Tensor,
    mCut: cute.Tensor,
    mO: cute.Tensor,
    mL: cute.Tensor,
    mCnt: cute.Tensor,
    mPgT: cute.Tensor,
    mSql: cute.Tensor,
    mTm: cute.Tensor,
    mRi: cute.Tensor,
    S: cutlass.Int32,
    SP: cutlass.Int32,
    EVS: cutlass.Float32,
    cfg: cutlass.Constexpr,
):
    D = cfg.head_dim
    SPLIT = cfg.split
    NRT = cfg.row_tiles
    NST = WIDE_STAGES
    NRAW = WIDE_RAW_STAGES
    PROD = 4 * NWG
    PAGED = cfg.paged
    PAGE = cfg.page_size
    HKV = cfg.kv_heads
    DRAFT = cfg.draft
    SEGW = min(PAGE, BN) if PAGED else BN
    NSEG = BN // SEGW
    NTH = cfg.threads

    if cutlass.const_expr(not cfg.direct):
        cute.arch.griddepcontrol_launch_dependents()
    tidx, _, _ = cute.arch.thread_idx()
    pid, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.make_warp_uniform(tidx // 32)
    # Row tiles of one row group and split are adjacent in the grid, so they
    # run together and read their K and V tiles once from HBM between them.
    rt = pid % NRT
    rest = pid // NRT
    sp = rest % SPLIT
    bh = rest // SPLIT
    breq = bh // HKV
    hkv = bh % HKV
    slen = S
    if cutlass.const_expr(PAGED):
        slen = mSql[breq]
    tbase = slen
    if cutlass.const_expr(DRAFT):
        tbase = slen - DRAFT
    lo, hi = split_bounds(slen, sp, SPLIT, BN, False, 0)
    nrows, n_tiles = split_tiles(slen, sp, lo, hi, SPLIT, BN, False)

    IMG = cfg.image
    smem = cutlass.memory.SmemAllocator()
    if cutlass.const_expr(IMG):
        sKa = None
        sKb = None
    else:
        sKa = smem.allocate_tensor(I8, cute.make_layout(NRAW * BN * D), 1024)
        sKb = smem.allocate_tensor(I8, cute.make_layout(NRAW * BN * D), 1024)
    sK = smem.allocate_tensor(F16, cute.make_layout(NST * BN * D), 1024)
    sV = smem.allocate_tensor(PT, cute.make_layout(NST * BN * D), 1024)
    sQ = smem.allocate_tensor(F16, cute.make_layout(NWG * MR * D), 1024)
    sOne = smem.allocate_tensor(PT, cute.make_layout(8 * BN), 1024)
    sKs = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(NRAW * BN), 16)
    sE = smem.allocate_tensor(cutlass.Float32, cute.make_layout(NST * BN), 16)
    sSeg = smem.allocate_tensor(cutlass.Int32, cute.make_layout((NRAW, NSEG)), 16)
    sRaw = smem.allocate_tensor(cutlass.Int64, cute.make_layout(NRAW), 8)
    sFull = smem.allocate_tensor(cutlass.Int64, cute.make_layout(NST), 8)
    sEmpty = smem.allocate_tensor(cutlass.Int64, cute.make_layout(NST), 8)

    if tidx == 0:
        for s in cutlass.range_constexpr(NRAW):
            cute.arch.mbarrier_init(sRaw.iterator + s, 1)
        for s in cutlass.range_constexpr(NST):
            # an image tile is three bulk copies from one thread; otherwise each
            # producer thread twice, its V copies landing and its share of the
            # tile's fp16 K written
            cute.arch.mbarrier_init(sFull.iterator + s, 1 if IMG else 256)
            cute.arch.mbarrier_init(sEmpty.iterator + s, 4 if cfg.key_split else 4 * NWG)
        cute.arch.mbarrier_init_fence()

    # Tiles start finite, V's and the scales: a zero weight times a NaN left
    # in shared memory would still be a NaN, and a tile's last rows are
    # whatever an earlier tile left there. An image's tiles are whole.
    if cutlass.const_expr(not IMG):
        zero_tile(sV, NST * BN * D * 2, tidx, NTH)
        zero_tile(sKs, NRAW * BN * 2, tidx, NTH)
    # the denominator's column of ones, eight wide for the smallest N
    if tidx < BN:
        for j in cutlass.range_constexpr(8):
            sOne[tidx * 8 + j] = PT(1.0)
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()

    # The producer warpgroup gives its registers to the consumers.
    if warp >= PROD:
        cute.arch.setmaxregister_decrease(PRODUCER_REGS)
        if cutlass.const_expr(IMG):
            _producer_image(
                sK, sV, sE, sFull, sEmpty, mKa, mV, mEk, bh, lo, n_tiles, tidx - 128 * NWG, SP, cfg
            )
        else:
            _producer(
                sKa,
                sKb,
                sKs,
                sK,
                sE,
                sV,
                sSeg,
                sRaw,
                sFull,
                sEmpty,
                mKa,
                mKb,
                mEk,
                mV,
                mPgT,
                bh,
                breq,
                hkv,
                lo,
                hi,
                n_tiles,
                tidx - 128 * NWG,
                S,
                SP,
                cfg,
            )
    else:
        cute.arch.setmaxregister_increase(cfg.consumer_regs)
        _consumer(
            sK,
            sV,
            sE,
            sQ,
            sOne,
            sFull,
            sEmpty,
            mQa,
            mQb,
            mZ,
            mCut,
            mEq,
            mTm,
            mRi,
            mVm,
            mO,
            mL,
            mCnt,
            EVS,
            bh,
            sp,
            rt,
            lo,
            hi,
            tbase,
            n_tiles,
            nrows,
            tidx,
            lane,
            warp,
            cfg,
        )


@cute.jit
def _producer(
    sKa,
    sKb,
    sKs,
    sK,
    sE,
    sV,
    sSeg,
    sRaw,
    sFull,
    sEmpty,
    mKa,
    mKb,
    mEk,
    mV,
    mPgT,
    bh,
    breq,
    hkv,
    lo,
    hi,
    n_tiles,
    pt,
    S,
    SP,
    cfg: cutlass.Constexpr,
):
    """The producer warpgroup: both K planes and the scales by bulk copy a
    tile ahead of their conversion, V's rows by `cp.async` into the
    descriptor's swizzled layout, and each tile's K rounded once to fp16 for
    every row that reads it. A stage is refilled once every consumer warp has
    released it."""
    D = cfg.head_dim
    NST = WIDE_STAGES
    NRAW = WIDE_RAW_STAGES
    PAGED = cfg.paged
    PAGE = cfg.page_size
    SEGW = min(PAGE, BN) if PAGED else BN
    SWU, SWRS, _ = swizzle_of(D)
    SWM = SWU - 1
    LEAD = NRAW - 1
    gKa, gKb, gEk, gV = cache_views(bh, S, SP, D, cfg.ek_stride, PAGED, mKa, mKb, mEk, mV)
    LPR = D // 8
    VROW = 128 // LPR
    vuu = pt % LPR
    vr0 = pt // LPR
    # the conversion: 16 channels, one 16-byte unit of each plane, an item
    NU = D // 16
    NIT = BN * NU // 128
    for j in cutlass.range_constexpr(LEAD):
        if j < n_tiles:
            _raw(sKa, sKb, sKs, sSeg, sRaw, gKa, gKb, gEk, mPgT, breq, hkv, lo, hi, j, j, pt, cfg)
    # the lead tiles' page bases, before any thread's V copies read them
    cute.arch.barrier(barrier_id=5, number_of_threads=128)
    for t in cutlass.range(0, n_tiles, 1, unroll=1):
        st = t % NST
        sr = t % NRAW
        # the raw stage the lead tile takes was converted last iteration, and
        # the producer's barrier below closed that
        if t + LEAD < n_tiles:
            _raw(
                sKa,
                sKb,
                sKs,
                sSeg,
                sRaw,
                gKa,
                gKb,
                gEk,
                mPgT,
                breq,
                hkv,
                lo,
                hi,
                t + LEAD,
                (t + LEAD) % NRAW,
                pt,
                cfg,
            )
        if t >= NST:
            cute.arch.mbarrier_wait(sEmpty.iterator + st, (t // NST - 1) % 2)
        base = lo + t * BN
        n = hi - base
        if n > BN:
            n = BN
        # a row past the tile's end keeps an earlier tile's finite bytes and
        # meets a zero weight
        for i in cutlass.range_constexpr(BN // VROW):
            rr = vr0 + i * VROW
            if rr < n:
                row = base + rr
                if cutlass.const_expr(PAGED):
                    row = sSeg[(sr, rr // SEGW)] + rr % SEGW
                primitives.cp_async_shared_global(
                    sV.iterator + (st * BN * D + gather_dst(rr, vuu, D, BN, 2, 1)),
                    gV.iterator + (cutlass.Int64(row) * D + vuu * 8),
                    16,
                    "cg",
                )
        cute.arch.cp_async_mbarrier_arrive_noinc(sFull.iterator + st)
        cute.arch.mbarrier_wait(sRaw.iterator + sr, (t // NRAW) % 2)
        # K rounded once to fp16 as `Ka + Kb / 256`, into 64-channel blocks
        # of the descriptor's 128-byte swizzle; the consumers apply the key's
        # scale to the logit, so no multiply rounds here
        for it in cutlass.range_constexpr(NIT):
            item = it * 128 + pt
            r = item // NU
            u = item % NU
            src = sr * BN * D + r * D + ((u ^ ((r >> SWRS) & SWM)) << 4)
            wa = cute.make_tensor(
                cute.recast_ptr(sKa.iterator, None, cutlass.Uint32) + src // 4, cute.make_layout(4)
            ).load()
            wb = cute.make_tensor(
                cute.recast_ptr(sKb.iterator, None, cutlass.Uint32) + src // 4, cute.make_layout(4)
            ).load()
            kw = cute.make_rmem_tensor((8,), cutlass.Uint32)
            for c in cutlass.range_constexpr(4):
                kw[2 * c], kw[2 * c + 1] = k16_pairs(wa[c], wb[c])
            kh = cute.make_tensor(kw.iterator, cute.make_layout((4, 2)))
            for h in cutlass.range_constexpr(2):
                cu = 2 * u + h
                dst = st * BN * D + (cu // 8) * BN * 64 + r * 64 + (((cu % 8) ^ (r & 7)) << 3)
                cute.autovec_copy(
                    kh[(None, h)],
                    cute.make_tensor(
                        cute.recast_ptr(sK.iterator, None, cutlass.Uint32) + dst // 2,
                        cute.make_layout(4),
                    ),
                )
            if u == 0:
                sE[st * BN + r] = cutlass.Float32(sKs[sr * BN + r])
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(sFull.iterator + st)
        # every producer thread is done with this raw stage and its page bases
        cute.arch.barrier(barrier_id=5, number_of_threads=128)


@cute.jit
def _producer_image(
    sK, sV, sE, sFull, sEmpty, mK, mV, mE, bh, lo, n_tiles, pt, NTI, cfg: cutlass.Constexpr
):
    """The producer for a `prefix_image`: one thread, three bulk copies a
    tile. `NTI` is the image's tiles per row group."""
    D = cfg.head_dim
    NST = WIDE_STAGES
    if pt == 0:
        t0 = cutlass.Int64(bh) * NTI + lo // BN
        for t in cutlass.range(0, n_tiles, 1, unroll=1):
            st = t % NST
            if t >= NST:
                cute.arch.mbarrier_wait(sEmpty.iterator + st, (t // NST - 1) % 2)
            bar = sFull.iterator + st
            cute.arch.mbarrier_arrive_and_expect_tx(bar, 4 * BN * D + 4 * BN)
            src = (t0 + t) * (BN * D)
            cp_async_bulk_shared_cluster_global(
                sK.iterator + st * BN * D, mK.iterator + src, bar, 2 * BN * D
            )
            cp_async_bulk_shared_cluster_global(
                sV.iterator + st * BN * D, mV.iterator + src, bar, 2 * BN * D
            )
            cp_async_bulk_shared_cluster_global(
                sE.iterator + st * BN, mE.iterator + (t0 + t) * BN, bar, 4 * BN
            )


@cute.jit
def _raw(
    sKa,
    sKb,
    sKs,
    sSeg,
    sRaw,
    gKa,
    gKb,
    gEk,
    mPgT,
    breq,
    hkv,
    lo,
    hi,
    t,
    sr,
    pt,
    cfg: cutlass.Constexpr,
):
    """Tile `t`'s K planes and scales into raw stage `sr`, by one thread,
    with its page bases for the V copies that follow."""
    D = cfg.head_dim
    PAGED = cfg.paged
    PAGE = cfg.page_size
    HKV = cfg.kv_heads
    SEGW = min(PAGE, BN) if PAGED else BN
    NSEG = BN // SEGW
    if pt == 0:
        base = lo + t * BN
        n = hi - base
        if n > BN:
            n = BN
        tile_segments(sSeg, sr, base, n, mPgT, breq, hkv, PAGE, SEGW, NSEG, HKV, PAGED)
        bar = sRaw.iterator + sr
        cute.arch.mbarrier_arrive_and_expect_tx(bar, 2 * n * D + scale_bytes(n))
        bulk_tile(
            cute.make_tensor(sKa.iterator + sr * BN * D, cute.make_layout(BN * D)),
            gKa,
            sSeg,
            sr,
            bar,
            base,
            n,
            D,
            SEGW,
            NSEG,
            PAGED,
        )
        bulk_tile(
            cute.make_tensor(sKb.iterator + sr * BN * D, cute.make_layout(BN * D)),
            gKb,
            sSeg,
            sr,
            bar,
            base,
            n,
            D,
            SEGW,
            NSEG,
            PAGED,
        )
        bulk_scales(sKs.iterator + sr * BN, gEk, sSeg, sr, bar, base, n, SEGW, NSEG, PAGED)


@cute.jit
def _weights(
    acc,
    ev,
    rM,
    mz,
    cutd,
    tmw,
    tq,
    nrow,
    base,
    tbase,
    TRUNCATE: cutlass.Constexpr,
    DROPPED: cutlass.Constexpr,
    DRAFT: cutlass.Constexpr,
    MASKED: cutlass.Constexpr,
):
    """The logits in `acc`, before their keys' scales `ev`, into weights, in
    place. Register `i` is row `16 warp + gid + 8 ((i / 2) % 2)` against key
    `8 (i / 4) + 2 tq + i % 2`, whose scale is `ev[2 (i / 4) + i % 2]`.
    A weight below its row's cut is dropped mass; `MASKED` tiles hold keys
    past the range's end or draft keys."""
    for i in cutlass.range_constexpr(cute.size(acc)):
        r = (i // 2) % 2
        s = acc[i] * ev[2 * (i // 4) + i % 2] + mz[r]
        if cutlass.const_expr(MASKED):
            # a key past the tile's end, or a draft key the row's node does
            # not descend from, carries no weight and no mass
            kc = 8 * (i // 4) + 2 * tq + (i % 2)
            vis = kc < nrow
            if cutlass.const_expr(DRAFT):
                jd = base + kc - tbase
                vis = vis and (jd < 0 or ((tmw[r] >> jd) & 1) != 0)
            s = cutlass.Float32(cutlass.select_(vis, s, cutlass.Float32(-3.0e38)))
        e = ex2(s)
        if cutlass.const_expr(TRUNCATE):
            keep = s >= cutd[r]
            acc[i] = cutlass.Float32(cutlass.select_(keep, e, 0.0))
            if cutlass.const_expr(DROPPED):
                rM[r] = rM[r] + cutlass.Float32(cutlass.select_(keep, 0.0, e))
        else:
            acc[i] = e


@cute.jit
def _pv(
    tmv, tmo, vgs, rO, rDen, fP, fPr, fOne, sV, st, swz, D: cutlass.Constexpr, W2: cutlass.Constexpr
):
    """Stage `st`'s value matmul and the denominator's column of ones, with
    the low weight term under `W2`."""
    vbo = sV.iterator.toint() + st * BN * D * 2
    for j in cutlass.range_constexpr(D // 64):
        fV = vgs.make_fragment_B(
            vgs.partition_B(
                smem_view(
                    vbo + j * BN * 64 * 2, PT, swz, cute.make_layout((64, BN), stride=(1, 64))
                )
            )
        )
        cute.gemm(tmv, rO[j], fP, fV, rO[j])
        if cutlass.const_expr(W2):
            cute.gemm(tmv, rO[j], fPr, fV, rO[j])
    cute.gemm(tmo, rDen, fP, fOne, rDen)
    if cutlass.const_expr(W2):
        cute.gemm(tmo, rDen, fPr, fOne, rDen)


@cute.jit
def _consumer(
    sK,
    sV,
    sE,
    sQ,
    sOne,
    sFull,
    sEmpty,
    mQa,
    mQb,
    mZ,
    mCut,
    mEq,
    mTm,
    mRi,
    mVm,
    mO,
    mL,
    mCnt,
    EVS,
    bh,
    sp,
    rt,
    lo,
    hi,
    tbase,
    n_tiles,
    nrows,
    tidx,
    lane,
    warp,
    cfg: cutlass.Constexpr,
):
    """One warpgroup's 64 rows against every tile of the split."""
    G = cfg.group
    D = cfg.head_dim
    SPLIT = cfg.split
    NST = WIDE_STAGES
    MRC = MR * NWG
    DRAFT = cfg.draft
    SHARED = cfg.shared
    DIRECT = cfg.direct
    SLOTS = cfg.slots
    SLOT0 = cfg.slot0
    UNIQUE_G = cfg.unique_group
    TRUNCATE = cfg.truncate
    DROPPED = cfg.dropped_mass
    W2 = cfg.weight_terms == 2
    NVB = D // 64
    NH = D // 64
    wg = warp // 4
    wq = warp % 4
    gid = lane >> 2
    tq = lane & 3
    KS = cfg.key_split
    row0 = rt * MRC + wg * MR
    if cutlass.const_expr(KS):
        row0 = rt * MR
    wt = tidx % 128
    PING = not KS
    # a warpgroup's tiles, and the partial slot it writes
    TSTEP = 2 if KS else 1
    tfirst = wg if KS else 0
    osp = sp * cfg.slots_per_split + (wg if KS else 0)

    # This warpgroup's rows of Q, `eq (Qa + Qb / 256)` rounded once to fp16,
    # into 64-channel blocks of the descriptor's 128-byte swizzle, each from
    # the (row group, row) a cascade's row-map word names. A row past the
    # group reads row 0 and is masked by its reference below.
    NQU = MR * (D // 8)
    for it in cutlass.range_constexpr(NQU // 128):
        u = it * 128 + wt
        r = u // (D // 8)
        cu = u % (D // 8)
        gq = row0 + r
        gsrc = gq
        if gq >= G:
            gsrc = cutlass.Int32(0)
        bhs = bh
        if cutlass.const_expr(SHARED):
            bhs, gsrc = cascade_row(mRi[(bh, gsrc)])
        qo = (bhs * UNIQUE_G + gsrc) * D + cu * 8
        qa = cute.make_tensor(mQa.iterator + qo, cute.make_layout(8)).load().to(cutlass.Float32)
        qb = cute.make_tensor(mQb.iterator + qo, cute.make_layout(8)).load().to(cutlass.Float32)
        e = mEq[(bhs, gsrc)] * (1.0 / KBR)
        q16 = ((qa * KBR + qb) * e).to(F16)
        dst = wg * MR * D + (cu // 8) * MR * 64 + r * 64 + (((cu % 8) ^ (r & 7)) << 3)
        cute.make_tensor(sQ.iterator + dst, cute.make_layout(8)).store(q16)
    cute.arch.fence_view_async_shared()
    cute.arch.barrier(barrier_id=1 + wg, number_of_threads=128)

    # The rows this thread's accumulators hold: `16 wq + gid` and 8 below.
    # Each carries -Z and its cut as a depth below Z.
    mz = []
    cutd = []
    tmw = []
    rok = []
    for r in cutlass.range_constexpr(2):
        gq = row0 + 16 * wq + gid + 8 * r
        z0 = cutlass.Float32(-3.0e38)
        c0 = cutlass.Float32(3.0e38)
        t0 = cutlass.Int32(0)
        ok = gq < G
        bhq = bh
        gsq = gq
        if gq >= G:
            gsq = cutlass.Int32(0)
        if cutlass.const_expr(SHARED):
            w = mRi[(bh, gsq)]
            bhq, gsq = cascade_row(w)
            ok = ok and (w & 1) != 0
        if ok:
            zz = mZ[(bhq, gsq)]
            z0 = -zz
            c0 = mCut[(bhq, gsq)] - zz
            if cutlass.const_expr(DRAFT):
                t0 = mTm[(bh, gsq)]
        mz.append(z0)
        cutd.append(c0)
        tmw.append(t0)
        rok.append(ok)

    swz = cute.make_swizzle(3, 4, 3)
    tmq = sm90.make_trivial_tiled_mma(
        F16,
        F16,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        cutlass.Float32,
        (1, 1, 1),
        (MR, BN),
    )
    wgq = tmq.get_slice(0)
    qbase = sQ.iterator.toint() + wg * MR * D * 2
    fQ = [
        wgq.make_fragment_A(
            wgq.partition_A(
                smem_view(
                    qbase + h * MR * 128, F16, swz, cute.make_layout((MR, 64), stride=(64, 1))
                )
            )
        )
        for h in range(NH)
    ]
    tmv = sm90.make_trivial_tiled_mma(
        PT,
        PT,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.MN,
        cutlass.Float32,
        (1, 1, 1),
        (MR, 64),
        warpgroup.OperandSource.RMEM,
    )
    tmv.set(warpgroup.Field.ACCUMULATE, True)
    vgs = tmv.get_slice(0)
    tmo = sm90.make_trivial_tiled_mma(
        PT,
        PT,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        cutlass.Float32,
        (1, 1, 1),
        (MR, 8),
        warpgroup.OperandSource.RMEM,
    )
    tmo.set(warpgroup.Field.ACCUMULATE, True)
    ogs = tmo.get_slice(0)
    fOne = ogs.make_fragment_B(
        ogs.partition_B(
            smem_view(sOne.iterator.toint(), PT, swz, cute.make_layout((8, BN), stride=(BN, 1)))
        )
    )

    acc = cute.make_rmem_tensor(tmq.partition_shape_C((MR, BN)), cutlass.Float32)
    fP = cute.make_rmem_tensor(tmv.partition_shape_A((MR, BN)), PT)
    fP4 = cute.recast_tensor(fP, cutlass.Uint32)
    fPr = fP
    fPr4 = fP4
    if cutlass.const_expr(W2):
        fPr = cute.make_rmem_tensor(tmv.partition_shape_A((MR, BN)), PT)
        fPr4 = cute.recast_tensor(fPr, cutlass.Uint32)
    rO = [
        cute.make_rmem_tensor(tmv.partition_shape_C((MR, 64)), cutlass.Float32) for _ in range(NVB)
    ]
    for j in cutlass.range_constexpr(NVB):
        rO[j].fill(0.0)
    rDen = cute.make_rmem_tensor(tmo.partition_shape_C((MR, 8)), cutlass.Float32)
    rDen.fill(0.0)
    rM = cute.make_rmem_tensor((2,), cutlass.Float32)
    rM.fill(0.0)
    NCI = cute.size(acc)

    # The two warpgroups take turns at the tensor cores: each issues a tile's
    # logit matmul together with the previous tile's value matmul, so one's
    # weight pass runs under the other's matmuls. Warpgroup 0 goes first.
    if cutlass.const_expr(PING):
        if wg == 1:
            cute.arch.barrier_arrive(barrier_id=3, number_of_threads=256)

    # the previous tile's stage, whose value matmul the next tile issues
    pst = cutlass.Int32(0)
    for t in cutlass.range(tfirst, n_tiles, TSTEP, unroll=1):
        st = t % NST
        base = lo + t * BN
        cute.arch.mbarrier_wait(sFull.iterator + st, (t // NST) % 2)
        # V's rows and the fp16 K were written through the generic proxy; the
        # matmuls read through the async one. An image arrives through the
        # async proxy already.
        if cutlass.const_expr(not cfg.image):
            cute.arch.fence_view_async_shared()
        kbase = sK.iterator.toint() + st * BN * D * 2
        if cutlass.const_expr(PING):
            cute.arch.barrier(barrier_id=3 + wg, number_of_threads=256)
        warpgroup.fence()
        tmq.set(warpgroup.Field.ACCUMULATE, False)
        for h in cutlass.range_constexpr(NH):
            fK = wgq.make_fragment_B(
                wgq.partition_B(
                    smem_view(
                        kbase + h * BN * 128, F16, swz, cute.make_layout((BN, 64), stride=(64, 1))
                    )
                )
            )
            for kb in cutlass.range_constexpr(4):
                cute.gemm(tmq, acc, fQ[h][(None, None, kb)], fK[(None, None, kb)], acc)
                tmq.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        if t > tfirst:
            _pv(tmv, tmo, vgs, rO, rDen, fP, fPr, fOne, sV, pst, swz, D, W2)
        warpgroup.commit_group()
        if cutlass.const_expr(PING):
            cute.arch.barrier_arrive(barrier_id=4 - wg, number_of_threads=256)
        ev = cute.make_rmem_tensor((BN // 4,), cutlass.Float32)
        for j in cutlass.range_constexpr(BN // 8):
            cute.autovec_copy(
                cute.make_tensor(sE.iterator + st * BN + 8 * j + 2 * tq, cute.make_layout(2)),
                cute.make_tensor(ev.iterator + 2 * j, cute.make_layout(2)),
            )
        # this tile's logits; the previous tile's value matmul may still run
        warpgroup.wait_group(1)

        nrow = hi - base
        if nrow > BN:
            nrow = BN
        edge = nrow < BN
        if cutlass.const_expr(DRAFT):
            edge = edge or base + BN > tbase
        if edge:
            _weights(
                acc, ev, rM, mz, cutd, tmw, tq, nrow, base, tbase, TRUNCATE, DROPPED, DRAFT, True
            )
        else:
            _weights(
                acc, ev, rM, mz, cutd, tmw, tq, nrow, base, tbase, TRUNCATE, DROPPED, DRAFT, False
            )
        # the previous value matmul is done with the weight registers and its
        # stage goes back to the producer
        warpgroup.wait_group(0)
        if t > tfirst:
            if lane == 0:
                cute.arch.mbarrier_arrive(sEmpty.iterator + pst)
        # the weights as the value matmul's A: register pairs in C order are
        # the A fragment's words in order
        for u in cutlass.range_constexpr(NCI // 2):
            pw, f0, f1 = pack_weights(acc[2 * u], acc[2 * u + 1])
            fP4[u] = pw
            if cutlass.const_expr(W2):
                pr, _, _ = pack_weights(acc[2 * u] - f0, acc[2 * u + 1] - f1)
                fPr4[u] = pr
        pst = st

    if n_tiles > tfirst:
        warpgroup.fence()
        _pv(tmv, tmo, vgs, rO, rDen, fP, fPr, fOne, sV, pst, swz, D, W2)
        warpgroup.commit_group()
    warpgroup.wait_group(0)
    # A row's dropped mass is spread over the four lanes of its quad.
    for r in cutlass.range_constexpr(2):
        m = rM[r]
        for stp in cutlass.range_constexpr(2):
            m = m + cute.arch.shuffle_sync_bfly(m, 1 << stp)
        rM[r] = m
    for r in cutlass.range_constexpr(2):
        gq = row0 + 16 * wq + gid + 8 * r
        if rok[r]:
            oslot = bh * SLOTS + SLOT0 + osp
            orow = gq
            if cutlass.const_expr(SHARED):
                ob, orow = cascade_row(mRi[(bh, gq)])
                oslot = ob * SLOTS + SLOT0 + osp
            den = rDen[2 * r] + rM[r]
            if tq == 0:
                mL[(oslot, orow)] = den
            ob_ = (oslot * UNIQUE_G + orow) * D
            for j in cutlass.range_constexpr(NVB):
                for cj in cutlass.range_constexpr(8):
                    ch = 64 * j + 8 * cj + 2 * tq
                    xs = [rO[j][4 * cj + 2 * r], rO[j][4 * cj + 2 * r + 1]]
                    for c in cutlass.range_constexpr(2):
                        x = xs[c]
                        if cutlass.const_expr(DROPPED):
                            if cutlass.const_expr(cfg.row_vmean):
                                x = x + rM[r] * mVm[(bh, gq, ch + c)]
                            else:
                                x = x + rM[r] * mVm[(bh, ch + c)]
                        if cutlass.const_expr(DIRECT):
                            x = x * EVS / den
                        xs[c] = x
                    st_global_f32(mO.iterator + (ob_ + ch), xs)
    if wt == 0 and wg == 0 and rt == 0:
        cnt = bh * SPLIT + sp
        mCnt[(cnt, 0)] = nrows
        mCnt[(cnt, 1)] = nrows


@cute.jit
def launch_wide(
    mQa,
    mQb,
    mEq,
    mKa,
    mKb,
    mEk,
    mV,
    mVm,
    mZ,
    mCut,
    mO,
    mL,
    mCnt,
    mPgT,
    mSql,
    mTm,
    mRi,
    S: cutlass.Int32,
    SP: cutlass.Int32,
    EVS: cutlass.Float32,
    cfg: cutlass.Constexpr,
    stream,
):
    wide_kernel(
        mQa,
        mQb,
        mEq,
        mKa,
        mKb,
        mEk,
        mV,
        mVm,
        mZ,
        mCut,
        mO,
        mL,
        mCnt,
        mPgT,
        mSql,
        mTm,
        mRi,
        S,
        SP,
        EVS,
        cfg,
    ).launch(
        grid=[cfg.n_groups * cfg.split * cfg.row_tiles, 1, 1],
        block=[cfg.threads, 1, 1],
        stream=stream,
        min_blocks_per_mp=1,
    )
