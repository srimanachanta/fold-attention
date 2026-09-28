"""The mass reference: each row's log-sum-exp, estimated before the decode.

Every split of a row group cuts against the same Z, so it is estimated once
per row group, with the decode's own coarse-logit matmul, on a tile of the
sink, the most recent keys and stratum centres of the rest. It reads the
cache and the step's Q and nothing a step before handed over. `mass_z` is the
standalone prepass; `front.mass_front` fuses the same estimate with the
step's Q quantisation and append.
"""

import cutlass
import cutlass.utils.hopper_helpers as sm90
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import warpgroup
from cutlass.experimental import primitives

from .cache import KBR, swizzle_of
from .config import BN
from .device import NEG, coarse_logit, logit_scale, paged_row
from .ptx import ex2, ld_scale_cg

I8 = cutlass.Int8
NW = 4
NT = 128


@cute.jit
def mass_key(r, slen, W: cutlass.Constexpr, NS: cutlass.Constexpr):
    """The key mass-tile row `r` scores: the sink, the `W` most recent keys
    newest first, then `NS` stratum centres, clamped into the request.
    Whether a row counts is `mass_estimate`'s question."""
    key = cutlass.Int32(0)
    if r >= 1 and r <= W:
        key = slen - r
    if r > W and r <= W + NS:
        j = r - (W + 1)
        key = ((2 * j + 1) * slen) // (2 * NS)
    if key < 0:
        key = cutlass.Int32(0)
    if key >= slen:
        key = cutlass.Int32(0)
    return key


def mass_smem(smem, G, D, NPR, NS):
    """The tile, both Q planes stacked in N, the tile's key scales and the
    estimate's scratch."""
    NG = (G + 7) // 8
    NSL = (NS + 31) // 32
    _, _, ALN = swizzle_of(D)
    f32 = cutlass.Float32
    sK = smem.allocate_tensor(I8, cute.make_layout((NPR, D), stride=(D, 1)), ALN)
    sQ = smem.allocate_tensor(I8, cute.make_layout((2 * NG * 8, D), stride=(D, 1)), ALN)
    sM = smem.allocate_tensor(f32, cute.make_layout((NW, NG * 8), stride=(NG * 8, 1)), 16)
    sE = smem.allocate_tensor(f32, cute.make_layout((NW, NG * 8), stride=(NG * 8, 1)), 16)
    # a row's samples padded by four words, so the four row pairs a warp
    # writes at once land on different banks
    sD = smem.allocate_tensor(
        f32, cute.make_layout((NG * 8, NSL * 32), stride=(NSL * 32 + 4, 1)), 16
    )
    sKey = smem.allocate_tensor(cutlass.Int32, cute.make_layout(NPR), 16)
    sCls = smem.allocate_tensor(cutlass.Int32, cute.make_layout(NPR), 16)
    sCnt = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2, NW), stride=(NW, 1)), 16)
    # each row's `ek / 256`
    sKs = smem.allocate_tensor(f32, cute.make_layout(NPR), 16)
    return sK, sQ, sM, sE, sD, sKey, sCls, sCnt, sKs


@cute.jit
def _mass_rows(
    sK,
    sKey,
    gKa,
    pEk,
    mPgT,
    slen,
    breq,
    hkv,
    skip,
    W: cutlass.Constexpr,
    NS: cutlass.Constexpr,
    NPR: cutlass.Constexpr,
    PG: cutlass.Constexpr,
    PS: cutlass.Constexpr,
    HKV: cutlass.Constexpr,
    D: cutlass.Constexpr,
):
    """The tile's rows by cp.async, each key's unit re-phased from the cache's
    swizzle to its tile row's, and each row's scale loaded into a register of
    the thread that names its key: `mass_estimate` stores them once the work
    it runs under their latency is done. Row `skip` only has its key named:
    its bytes and scale come from elsewhere."""
    tidx, _, _ = cute.arch.thread_idx()
    SWU, SWRS, _ = swizzle_of(D)
    SWM = SWU - 1
    NU = D // 16
    kscales = []
    for it in cutlass.range_constexpr((NPR * NU + NT - 1) // NT):
        pr = it * (NT // NU) + tidx // NU
        pu = tidx % NU
        pkey = mass_key(pr, slen, W, NS)
        prow = pkey
        if cutlass.const_expr(PG):
            prow = paged_row(mPgT, breq, hkv, pkey, PS, HKV)
        kscales.append(ld_scale_cg(pEk + cutlass.Int64(prow) * 2))
        if pr < NPR and pr != skip:
            primitives.cp_async_shared_global(
                sK.iterator
                + (pr * D + (((pu ^ ((pkey >> SWRS) & SWM)) ^ ((pr >> SWRS) & SWM)) << 4)),
                gKa.iterator + (prow * D + pu * 16),
                16,
                "cg",
            )
        if pr < NPR:
            if pu == 0:
                sKey[pr] = pkey
    return kscales


@cute.jit
def mass_estimate(
    sK,
    sQ,
    sM,
    sE,
    sD,
    sKey,
    sCls,
    sCnt,
    sKs,
    rEq,
    rTm,
    slen,
    bh,
    mZ,
    mDep,
    mCut,
    kscales,
    skip,
    G: cutlass.Constexpr,
    D: cutlass.Constexpr,
    W: cutlass.Constexpr,
    NS: cutlass.Constexpr,
    NPR: cutlass.Constexpr,
    TRIM: cutlass.Constexpr,
    TREE: cutlass.Constexpr,
    pZw=None,
    side=None,
    sEq=None,
):
    """The estimate from a tile whose rows and Q are in flight or in place:
    which rows count, the logits, each row's exact mass and its trimmed
    stratified rest, and the reference and cut it makes.

    `mDep` holds each row's depth; `mZ` and `mCut` receive the reference and
    `Z - depth`. `pZw` also publishes both as words that are their own flags.
    `kscales` are the rows' scales as `_mass_rows` loaded them; row `skip`'s
    is written to `sKs` by whoever fills the row. `side`, a function and its
    arguments, runs between the rows' classes and the wait on the gathers,
    and may put Q into the tile, in which case `sEq` holds Q's scales and
    `rEq` is made from them."""
    tidx, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.make_warp_uniform(tidx // 32)
    gid = lane >> 2
    tq = lane & 3
    NG = (G + 7) // 8
    NTL = NPR // BN
    SWU, _, _ = swizzle_of(D)
    S0 = W + 1
    NSL = (NS + 31) // 32
    # the depth each of this warp's rows is cut at, loaded now so the cut's
    # store at the end is not a round trip behind the estimate
    rDep = cute.make_rmem_tensor(((G + NW - 1) // NW,), cutlass.Float32)
    for rsel in cutlass.range_constexpr((G + NW - 1) // NW):
        gq = warp + NW * rsel
        rDep[rsel] = cutlass.Float32(0.0)
        if gq < G:
            rDep[rsel] = mDep[(bh, gq)]
    cute.arch.sync_threads()
    # Which rows count, under the gathers' latency. Class 1, the sink and the
    # window, is scored exactly. Class 2 is a stratum sample of the keys the
    # window does not hold.
    lo = slen - W
    n1 = cutlass.Int32(0)
    n2 = cutlass.Int32(0)
    for it in cutlass.range_constexpr((NPR + NT - 1) // NT):
        r = it * NT + tidx
        cls = cutlass.Int32(0)
        if r < NPR:
            key = sKey[r]
            inmid = key >= 1 and key < lo
            if r == 0:
                cls = cutlass.Int32(1)
            if r >= 1 and r <= W:
                if slen - r >= 1:
                    cls = cutlass.Int32(1)
            if r >= S0 and r < S0 + NS:
                # centres are distinct once the request is 2 NS keys long
                if inmid:
                    cls = cutlass.Int32(2)
                    if r > S0:
                        if sKey[r - 1] == key:
                            cls = cutlass.Int32(0)
            sCls[r] = cls
        n1 += cute.arch.popc(cute.arch.vote_ballot_sync(cls == 1))
        n2 += cute.arch.popc(cute.arch.vote_ballot_sync(cls == 2))
    if lane == 0:
        sCnt[(0, warp)] = n1
        sCnt[(1, warp)] = n2
    if cutlass.const_expr(side is not None):
        side[0](*side[1])
    NU = D // 16
    for it in cutlass.range_constexpr(len(kscales)):
        pr = it * (NT // NU) + tidx // NU
        if pr < NPR and pr != skip and tidx % NU == 0:
            sKs[pr] = kscales[it] * (1.0 / KBR)
    cute.arch.cp_async_wait_group(0)
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()
    if cutlass.const_expr(sEq is not None):
        for ng in cutlass.range_constexpr(NG):
            for j in cutlass.range_constexpr(2):
                gq = ng * 8 + 2 * tq + j
                e0 = cutlass.Float32(0.0)
                if gq < G:
                    e0 = sEq[gq]
                rEq[(ng, j)] = e0

    swz = cute.make_swizzle(SWU.bit_length() - 1, 4, 3)
    tmmq = sm90.make_trivial_tiled_mma(
        I8,
        I8,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        cutlass.Int32,
        (1, 1, 1),
        (BN, 2 * NG * 8),
    )
    wgq = tmmq.get_slice(0)
    wQab = cute.make_tensor(
        cute.recast_ptr(sQ.iterator, swz, I8), cute.make_layout((2 * NG * 8, D), stride=(D, 1))
    )
    fQab = wgq.make_fragment_B(wgq.partition_B(wQab))
    # The tiles' matmuls go out together and are waited on once, unless their
    # accumulators would cost more than 64 registers.
    PIPE = NG * NTL <= 8
    acc = [
        cute.make_rmem_tensor(tmmq.partition_shape_C((BN, 2 * NG * 8)), cutlass.Int32)
        for _ in range(NTL if PIPE else 1)
    ]
    fK = [
        wgq.make_fragment_A(
            wgq.partition_A(
                cute.make_tensor(
                    cute.recast_ptr(sK.iterator + tl * BN * D, swz, I8),
                    cute.make_layout((BN, D), stride=(D, 1)),
                )
            )
        )
        for tl in range(NTL)
    ]
    if cutlass.const_expr(PIPE):
        warpgroup.fence()
        for tl in cutlass.range_constexpr(NTL):
            cute.gemm(tmmq, acc[tl], fK[tl], fQab, acc[tl])
        warpgroup.commit_group()
        warpgroup.wait_group(0)
    # every pair's logit, NEG where the row does not count or a draft row
    # does not see the key
    rL = cute.make_rmem_tensor((NTL, 2, NG, 2), cutlass.Float32)
    rC = cute.make_rmem_tensor((NTL, 2), cutlass.Int32)
    rMx = cute.make_rmem_tensor((NG, 2), cutlass.Float32)
    for ng in cutlass.range_constexpr(NG):
        for j in cutlass.range_constexpr(2):
            rMx[(ng, j)] = cutlass.Float32(NEG)
    for tl in cutlass.range_constexpr(NTL):
        c1 = acc[tl] if PIPE else acc[0]
        if cutlass.const_expr(not PIPE):
            warpgroup.fence()
            cute.gemm(tmmq, c1, fK[tl], fQab, c1)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
        for hf in cutlass.range_constexpr(2):
            pr = tl * BN + warp * 16 + gid + 8 * hf
            cls = sCls[pr]
            ksc = sKs[pr]
            rC[(tl, hf)] = cls
            zkey = cutlass.Int32(-1)
            if cutlass.const_expr(TREE):
                zkey = sKey[pr] - (slen - TREE)
            for ng in cutlass.range_constexpr(NG):
                for j in cutlass.range_constexpr(2):
                    gq = ng * 8 + 2 * tq + j
                    ps = coarse_logit(c1, ng, 2 * hf + j, NG) * logit_scale(rEq[(ng, j)], ksc)
                    see = cls != 0 and gq < G
                    if cutlass.const_expr(TREE):
                        see = see and (zkey < 0 or ((rTm[(ng, j)] >> zkey) & 1) != 0)
                    ps = cutlass.Float32(cutlass.select_(see, ps, cutlass.Float32(NEG)))
                    rL[(tl, hf, ng, j)] = ps
                    rMx[(ng, j)] = cute.arch.fmax(rMx[(ng, j)], ps)
    # Each warp's exact mass against its own max, which the row's warp
    # rescales, so one barrier publishes both. The samples go to shared as
    # logits: the trim ranks them, and the row's max is not known yet.
    for ng in cutlass.range_constexpr(NG):
        for j in cutlass.range_constexpr(2):
            gq = ng * 8 + 2 * tq + j
            zm = rMx[(ng, j)]
            for st in cutlass.range_constexpr(3):
                zm = cute.arch.fmax(zm, cute.arch.shuffle_sync_bfly(zm, 4 << st))
            ex = cutlass.Float32(0.0)
            for tl in cutlass.range_constexpr(NTL):
                for hf in cutlass.range_constexpr(2):
                    pr = tl * BN + warp * 16 + gid + 8 * hf
                    ps = rL[(tl, hf, ng, j)]
                    e = ex2(ps - zm)
                    ex = ex + cutlass.Float32(cutlass.select_(rC[(tl, hf)] == 1, e, 0.0))
                    if cutlass.const_expr(tl * BN + BN > S0 and tl * BN < S0 + NS):
                        if pr >= S0 and pr < S0 + NS:
                            sD[(gq, pr - S0)] = cutlass.Float32(
                                cutlass.select_(rC[(tl, hf)] == 2, ps, cutlass.Float32(NEG))
                            )
            for st in cutlass.range_constexpr(3):
                ex = ex + cute.arch.shuffle_sync_bfly(ex, 4 << st)
            if gid == 0:
                sM[(warp, gq)] = zm
                sE[(warp, gq)] = ex
    cute.arch.sync_threads()
    nex = sCnt[(0, 0)]
    nsv = sCnt[(1, 0)]
    for ww in cutlass.range_constexpr(1, NW):
        nex = nex + sCnt[(0, ww)]
        nsv = nsv + sCnt[(1, ww)]
    unscored = slen - nex - nsv
    scale = cutlass.Float32(0.0)
    if nsv > TRIM and unscored > 0:
        scale = cutlass.Float32(unscored) / cutlass.Float32(nsv - TRIM)
    # One warp per row. A weight is non-negative, so its bits order as a
    # signed int does, and a trim round is one `redux.sync` rather than a
    # ten-shuffle argmax; the lowest lane holding the max drops it. Tied
    # samples drop in either order to the same sum. A warp's rows are
    # straight-line code, stage by stage, so their serial chains of shuffles
    # and reductions overlap; a row past G reads row G - 1 and stores nothing.
    NR = (G + NW - 1) // NW
    rows = []
    for rsel in cutlass.range_constexpr(NR):
        gq = warp + NW * rsel
        gr = gq
        if cutlass.const_expr(G % NW):
            gr = cutlass.Int32(cutlass.select_(gq < G, gq, G - 1))
        lz = sM[(0, gr)]
        for ww in cutlass.range_constexpr(1, NW):
            lz = cute.arch.fmax(lz, sM[(ww, gr)])
        ex = cutlass.Float32(0.0)
        for ww in cutlass.range_constexpr(NW):
            ex = ex + sE[(ww, gr)] * ex2(sM[(ww, gr)] - lz)
        vals = cute.make_rmem_tensor((NSL,), cutlass.Float32)
        tot = cutlass.Float32(0.0)
        for k in cutlass.range_constexpr(NSL):
            x = cutlass.Float32(0.0)
            if cutlass.const_expr((k + 1) * 32 <= NS):
                x = ex2(sD[(gr, k * 32 + lane)] - lz)
            else:
                if k * 32 + lane < NS:
                    x = ex2(sD[(gr, k * 32 + lane)] - lz)
            vals[k] = x
            tot = tot + x
        rows.append([gq, lz, ex, vals, tot])
    for st in cutlass.range_constexpr(5):
        for r in rows:
            r[4] = r[4] + cute.arch.shuffle_sync_bfly(r[4], 1 << st)
    heads = [cutlass.Float32(0.0) for _ in range(NR)]
    for _ in cutlass.range_constexpr(TRIM):
        for ri in cutlass.range_constexpr(NR):
            vals = rows[ri][3]
            lb = vals[0].bitcast(Int32)
            li = cutlass.Int32(0)
            for k in cutlass.range_constexpr(1, NSL):
                b = vals[k].bitcast(Int32)
                li = cutlass.Int32(cutlass.select_(b > lb, k, li))
                lb = cutlass.Int32(cutlass.select_(b > lb, b, lb))
            wb = cute.arch.warp_redux_sync(lb, "max")
            win = cute.arch.vote_ballot_sync(lb == wb)
            first = (win & ((cutlass.Int32(1) << lane) - 1)) == 0 and lb == wb
            for k in cutlass.range_constexpr(NSL):
                vals[k] = cutlass.Float32(cutlass.select_(first and li == k, 0.0, vals[k]))
            heads[ri] = heads[ri] + wb.bitcast(Float32)
    for ri in cutlass.range_constexpr(NR):
        gq, lz, ex, _, tot = rows[ri]
        rest = cute.arch.fmax(tot - heads[ri], cutlass.Float32(0.0))
        lz = lz + cute.math.log2(ex + tot + rest * scale, fastmath=True)
        keep = lane == 0
        if cutlass.const_expr(G % NW):
            keep = keep and gq < G
        if keep:
            cz = lz - rDep[ri]
            mZ[(bh, gq)] = lz
            mCut[(bh, gq)] = cz
            if cutlass.const_expr(pZw is not None):
                # Each word is its own flag: complemented, zero is the one
                # value no arithmetic result takes, so the decode needs no
                # release here and does not wait on this CTA's other stores.
                zw = lz.bitcast(Int32) ^ Int32(-1)
                cw = cz.bitcast(Int32) ^ Int32(-1)
                cute.arch.store(pZw + gq, zw, sem="relaxed", scope="gpu")
                cute.arch.store(pZw + G + gq, cw, sem="relaxed", scope="gpu")


@cute.kernel
def mass_z(
    mQa: cute.Tensor,
    mQb: cute.Tensor,
    mEq: cute.Tensor,
    mKa: cute.Tensor,
    mEk: cute.Tensor,
    mPgT: cute.Tensor,
    mSql: cute.Tensor,
    mZ: cute.Tensor,
    mDep: cute.Tensor,
    mCut: cute.Tensor,
    mTm: cute.Tensor,
    S: cutlass.Int32,
    SP: cutlass.Int32,
    G: cutlass.Constexpr,
    D: cutlass.Constexpr,
    W: cutlass.Constexpr,
    NS: cutlass.Constexpr,
    NPR: cutlass.Constexpr,
    TRIM: cutlass.Constexpr,
    PG: cutlass.Constexpr,
    PS: cutlass.Constexpr,
    HKV: cutlass.Constexpr,
    TREE: cutlass.Constexpr,
    ESTR: cutlass.Constexpr,
):
    """Each row's log-sum-exp, estimated once per row group before the decode.

    The tile scores the sink and the recent window exactly, and `NS` stratum
    centres of what is left. The unscored keys' mass is the stratified mean
    with the `TRIM` largest samples dropped: those still count once, but a
    sample that lands on a head key is not multiplied by the stride, which is
    the only way this estimate lands above the true log-sum-exp. Every gate
    the decode hangs off the reference is then a claim about a key's share of
    its row's mass.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bh, _, _ = cute.arch.block_idx()
    tq = cute.arch.lane_idx() & 3
    breq = bh // HKV
    hkv = bh % HKV
    # Launched as its predecessor's programmatic dependent, so the launch
    # overlaps that kernel's tail; everything read below it may have written.
    # The decode is released only past this wait: it gathers Q and its first
    # K tile before waiting on this grid, and those are the predecessor's.
    cute.arch.griddepcontrol_wait()
    cute.arch.griddepcontrol_launch_dependents()
    slen = S
    if cutlass.const_expr(PG):
        slen = mSql[breq]
    NG = (G + 7) // 8
    SWU, SWRS, _ = swizzle_of(D)
    SWM = SWU - 1
    NU = D // 16
    smem = cutlass.memory.SmemAllocator()
    sK, sQ, sM, sE, sD, sKey, sCls, sCnt, sKs = mass_smem(smem, G, D, NPR, NS)
    koff = cutlass.Int64(bh) * SP * D
    eoff = bh * ESTR
    if cutlass.const_expr(PG):
        koff = 0
        eoff = 0
    gKa = cute.make_tensor(mKa.iterator + koff, cute.make_layout((SP, D), stride=(D, 1)))
    pEk = mEk.iterator.toint() + cutlass.Int64(eoff) * 2
    none = cutlass.Int32(-1)
    kscales = _mass_rows(
        sK, sKey, gKa, pEk, mPgT, slen, breq, hkv, none, W, NS, NPR, PG, PS, HKV, D
    )
    NQU = NG * 8 * NU
    for it in cutlass.range_constexpr((2 * NQU + NT - 1) // NT):
        u = it * NT + tidx
        uq = u % NQU
        gq8 = uq // NU
        cu = uq % NU
        gsrc = gq8
        if gq8 >= G:
            gsrc = cutlass.Int32(0)
        qo = (bh * G + gsrc) * D + cu * 16
        so = gq8 * D + ((cu ^ ((gq8 >> SWRS) & SWM)) << 4)
        if u < NQU:
            primitives.cp_async_shared_global(sQ.iterator + so, mQa.iterator + qo, 16, "cg")
        else:
            if u < 2 * NQU:
                primitives.cp_async_shared_global(
                    sQ.iterator + NQU * 16 + so, mQb.iterator + qo, 16, "cg"
                )
    cute.arch.cp_async_commit_group()
    rEq = cute.make_rmem_tensor((NG, 2), cutlass.Float32)
    rTm = cute.make_rmem_tensor((NG, 2), cutlass.Int32)
    for ng in cutlass.range_constexpr(NG):
        for j in cutlass.range_constexpr(2):
            gq = ng * 8 + 2 * tq + j
            e0 = cutlass.Float32(0.0)
            t0 = cutlass.Int32(0)
            if gq < G:
                e0 = mEq[(bh, gq)]
                if cutlass.const_expr(TREE):
                    t0 = mTm[(bh, gq)]
            rEq[(ng, j)] = e0
            rTm[(ng, j)] = t0
    mass_estimate(
        sK,
        sQ,
        sM,
        sE,
        sD,
        sKey,
        sCls,
        sCnt,
        sKs,
        rEq,
        rTm,
        slen,
        bh,
        mZ,
        mDep,
        mCut,
        kscales,
        none,
        G,
        D,
        W,
        NS,
        NPR,
        TRIM,
        TREE,
    )


@cute.jit
def launch_mass(
    mQa,
    mQb,
    mEq,
    mKa,
    mEk,
    mPgT,
    mSql,
    mZ,
    mDep,
    mCut,
    mTm,
    S: cutlass.Int32,
    SP: cutlass.Int32,
    NBH: cutlass.Constexpr,
    G: cutlass.Constexpr,
    D: cutlass.Constexpr,
    W: cutlass.Constexpr,
    NS: cutlass.Constexpr,
    NPR: cutlass.Constexpr,
    TRIM: cutlass.Constexpr,
    PG: cutlass.Constexpr,
    PS: cutlass.Constexpr,
    HKV: cutlass.Constexpr,
    TREE: cutlass.Constexpr,
    ESTR: cutlass.Constexpr,
    stream,
):
    mass_z(
        mQa,
        mQb,
        mEq,
        mKa,
        mEk,
        mPgT,
        mSql,
        mZ,
        mDep,
        mCut,
        mTm,
        S,
        SP,
        G,
        D,
        W,
        NS,
        NPR,
        TRIM,
        PG,
        PS,
        HKV,
        TREE,
        ESTR,
    ).launch(grid=[NBH, 1, 1], block=[NT, 1, 1], stream=stream, use_pdl=True)
