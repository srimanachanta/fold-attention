"""The decode kernel over 128-key tiles, for query groups of at most four rows
and a bf16 V: `decode.kernel`'s loop with two keys in every logical row of
the logit's M.

Row `8 g + j` of the logit's A holds key `16 g + j` in the first half of K and
key `16 g + j + 8` in the second, over plane A's tile exactly as the cache
stores it: within a 16-key group each half of eight keys is one core-matrix
block, and the second starts a whole swizzle repeat in. A query takes two
columns, `[q | 0]` against the first key of a row and `[0 | q]` against the
second, which lands in the N padding a narrow group leaves idle. Every
product is the one the 64-key kernel forms, so the logits are the same bits.

A lane owns one query row of the group. At G <= 2 both Q planes share one
eight-column block: column `2 t + p` is plane `p` of query `t & 1` against
key half `t >> 1`, and at G = 1 the second query is a copy of the first, so
each lane of a quad takes a key of its own. At G = 3 or 4 plane A's block
and plane B's are separate, column `2 t + h` query `t` against half `h`.

Warp `w` owns keys `32 w` to `32 w + 31`, and a tile takes one plane-A copy,
one plane-B round trip, one V round trip and one CTA barrier for 128 keys.
The verdict needs no barrier: each warp gathers its own keys, its live and
refine words come from its own lanes, the refine runs on every tile, and the
next plane A goes out as soon as the logit, one warpgroup-wide operation,
has retired.
"""

import cutlass
import cutlass.utils.hopper_helpers as sm90
from cutlass import cute
from cutlass.cute.nvgpu import warpgroup
from cutlass.experimental.primitives.nvvm_wrapper import (
    MMALayout,
    stmatrix,
)

from .cache import KBR, swizzle_of
from .config import NT, TK, weights_in_plane_b
from .device import (
    NEG,
    add_rank_term,
    band_gate,
    cache_views,
    front_length,
    front_rows,
    front_sums,
    gather_dst,
    gather_row,
    gather_warp_rows,
    issue_first,
    issue_next,
    key_scale,
    load_basis,
    logit_scale,
    partial_slot,
    smem_view,
    split_bounds,
    split_tiles,
    store_rank_sums,
    zero_tile,
)
from .ptx import (
    ex2,
    ex2_if,
    movmatrix_t,
    opaque_i32,
    pack_weights,
    st_global_f32,
    zero16,
)

I8 = cutlass.Int8
PT = cutlass.BFloat16


def pidx2(q, ky):
    """Query row `q`'s weight for key `ky` in the weight buffer: two 64-key
    blocks, each the K-major swizzle atom of eight query rows."""
    return (ky >> 6) * (8 * 64) + q * 64 + ((((ky & 63) >> 3) ^ (q & 7)) << 3) + (ky & 7)


def qtile_off(n, u, NR: int):
    """Byte offset of 16-byte unit `u` of row `n` of a packed Q tile of `NR`
    rows: K in blocks of 128 bytes, each block its rows' swizzle atom."""
    return (u >> 3) * (NR * 128) + n * 128 + (((u & 7) ^ (n & 7)) << 4)


def pair_logit(c, k, tq, G: int):
    """Pair `k`'s two plane products in accumulator `c`, plane A's then
    plane B's. At G = 1 the lane's key is in group `tq & 1`."""
    if G == 1:
        sel = (tq & 1) != 0
        return (
            cutlass.Int32(cutlass.select_(sel, c[2], c[0])),
            cutlass.Int32(cutlass.select_(sel, c[3], c[1])),
        )
    if G <= 2:
        return c[2 * k], c[2 * k + 1]
    return c[k], c[4 + k]


def warp_mass(pd, G: int):
    """The sum of a warp's weights `pd` (this lane's pairs) per query row, on
    every lane of that row."""
    m = pd[0]
    for k in range(1, len(pd)):
        m = m + pd[k]
    for st in range(0 if G == 1 else (1 if G == 2 else 2), 5):
        m = m + cute.arch.shuffle_sync_bfly(m, 1 << st)
    return m


def rank_sums(rY, cy, pdk, lane, G: int, R: int):
    """`rY[q] += p y` over this lane's share: it holds y for the four keys
    (s, h) of its gid at ranks 8 bb + 2 tq + e, and each key's dropped weight
    per query row, times its scale (`pdk`), comes from the lane that owns
    that pair."""
    yf = [cutlass.Float32(cy[x]) for x in range(R)]
    for sh in range(4):
        ss = sh & 1
        hh = sh >> 1
        for q in range(G):
            if G == 1:
                src, kk = ss + 2 * hh, 0
            elif G == 2:
                src, kk = 2 * hh + q, ss
            else:
                src, kk = q, 2 * ss + hh
            pq = cute.arch.shuffle_sync(pdk[kk], (lane & ~3) | src)
            for bb in range(R // 8):
                for e in range(2):
                    rY[(q, bb, e)] = rY[(q, bb, e)] + pq * yf[(hh * (R // 8) + bb) * 4 + 2 * ss + e]


def rank_mma(wmma, rYm, cy, pd, ksr, lane, tq, R: int):
    """`rYm += y^T p` as warp matmuls, for G >= 3, where a lane's query is its
    tq and its four pairs are the four keys (s, h) of its gid: per 16 keys s
    and per 16 ranks, A is y^T (ranks x keys) and B the dropped weights
    (keys x rows). A C block of y (keys x ranks) and a pair of lanes' weights
    (keys x two rows) are each one `movmatrix` from those fragments."""
    fA = cute.make_rmem_tensor(wmma.partition_shape_A((16, 16)), cutlass.BFloat16)
    fA4 = cute.recast_tensor(fA, cutlass.Uint32)
    fB = cute.make_rmem_tensor(wmma.partition_shape_B((8, 16)), cutlass.BFloat16)
    fB4 = cute.recast_tensor(fB, cutlass.Uint32)
    src = (lane & ~3) | ((2 * tq) & 3)
    for ss in range(2):
        for kh in range(2):
            i = 2 * ss + kh
            # rows 2 tq and 2 tq + 1 of the group; tq >= 2 are rows past four
            p0 = cute.arch.shuffle_sync(pd[i], src)
            p1 = cute.arch.shuffle_sync(pd[i], src | 1)
            p0 = cutlass.Float32(cutlass.select_(tq < 2, p0, 0.0))
            p1 = cutlass.Float32(cutlass.select_(tq < 2, p1, 0.0))
            pw, _, _ = pack_weights(p0, p1)
            fB4[kh] = movmatrix_t(pw)
        for jt in range(R // 16):
            for kh in range(2):
                for rh in range(2):
                    b = kh * (R // 8) + 2 * jt + rh
                    ks = ksr[2 * ss + kh]
                    yw, _, _ = pack_weights(
                        cutlass.Float32(cy[b * 4 + 2 * ss]) * ks,
                        cutlass.Float32(cy[b * 4 + 2 * ss + 1]) * ks,
                    )
                    fA4[2 * kh + rh] = movmatrix_t(yw)
            cute.gemm(wmma, rYm[jt], fA, fB, rYm[jt])


@cute.kernel
def packed_kernel(
    mQa: cute.Tensor,
    mQb: cute.Tensor,
    mEq: cute.Tensor,
    mKa: cute.Tensor,
    mKb: cute.Tensor,
    mEk: cute.Tensor,
    mV: cute.Tensor,
    mVb: cute.Tensor,
    mVm: cute.Tensor,
    mVbk: cute.Tensor,
    mU: cute.Tensor,
    mVr: cute.Tensor,
    mZ: cute.Tensor,
    mCut: cute.Tensor,
    mO: cute.Tensor,
    mL: cute.Tensor,
    mCnt: cute.Tensor,
    mPgT: cute.Tensor,
    mSql: cute.Tensor,
    mTm: cute.Tensor,
    mRi: cute.Tensor,
    mRdy: cute.Tensor,
    mOrd: cute.Tensor,
    S: cutlass.Int32,
    SP: cutlass.Int32,
    EVS: cutlass.Float32,
    RK: cutlass.Float32,
    RV: cutlass.Float32,
    cfg: cutlass.Constexpr,
):
    G = cfg.group
    D = cfg.head_dim
    SPLIT = cfg.split
    TRUNCATE = cfg.truncate
    DROPPED = cfg.dropped_mass
    ROW_VMEAN = cfg.row_vmean
    PAGED = cfg.paged
    PAGE = cfg.page_size
    HKV = cfg.kv_heads
    Z_PREPASS = cfg.z_prepass
    KEEP_ALL = cfg.keep_all
    DRAFT = cfg.draft
    SLOTS = cfg.slots
    SLOT0 = cfg.slot0
    UNIQUE_G = cfg.unique_group
    DIRECT = cfg.direct
    ORDERED = cfg.order
    SOUND = cfg.sound
    PFP = PAGED and PAGE >= TK
    TAIL = cfg.tail_blocks
    if cutlass.const_expr(TAIL and not KEEP_ALL):
        raise ValueError("the tail's virtual key needs the group's truncation")
    R = cfg.tail_rank
    if cutlass.const_expr(cfg.v8 or cfg.shared or G > 4):
        raise ValueError("the packed kernel takes G <= 4 on a bf16 V at one level")

    CM = "cg"
    NW = NT // 32
    KPW = TK // NW
    NVB = D // 64
    SWU, SWRS, ALN = swizzle_of(D)
    SWB = SWU.bit_length() - 1
    SWM = SWU - 1
    W2 = cfg.weight_terms == 2
    SEGW = min(PAGE, TK) if PAGED else TK
    NSEG = TK // SEGW
    # G <= 2 folds both Q planes into one eight-column block
    MERGE = G <= 2
    NQB = 1 if MERGE else 2
    NQR = 8 * NQB
    # (key, query) pairs a lane owns a tile
    NP = 1 if G == 1 else (2 if G == 2 else 4)
    NQ = 8
    # query rows' records and sums: G <= 4
    NR = 4
    NQU = (2 * D) // 16

    if cutlass.const_expr(not DIRECT):
        cute.arch.griddepcontrol_launch_dependents()
    tidx, _, _ = cute.arch.thread_idx()
    pid, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.make_warp_uniform(tidx // 32)
    gid = lane >> 2
    tq = lane & 3
    slot = pid // SPLIT
    sp = pid % SPLIT
    bh = slot
    if cutlass.const_expr(ORDERED):
        bh = mOrd[slot]
    breq = bh // HKV
    hkv = bh % HKV
    slen = S
    if cutlass.const_expr(PAGED and not cfg.front):
        slen = mSql[breq]

    smem = cutlass.memory.SmemAllocator()
    sKa = smem.allocate_tensor(I8, cute.make_layout((TK, D), stride=(D, 1)), ALN)
    sKb = smem.allocate_tensor(I8, cute.make_layout((TK, D), stride=(D, 1)), ALN)
    sV = smem.allocate_tensor(PT, cute.make_layout(TK * D), 1024)
    VBY = TK * D * 2
    NPB = NQ * TK * (2 if W2 else 1)
    if cutlass.const_expr(weights_in_plane_b(D, 1, W2, False, pack=2)):
        # Plane B's tile is dead from the refine's retirement to the next
        # tile's gathers. The rows of queries past G are never written and
        # hold plane-B bytes: they reach only their own columns of the value
        # accumulator, which nothing reads.
        sPb = cute.make_tensor(cute.recast_ptr(sKb.iterator, None, PT), cute.make_layout(NPB))
    else:
        sPb = smem.allocate_tensor(PT, cute.make_layout(NPB), 1024)
    sQ = smem.allocate_tensor(I8, cute.make_layout(NQR * 2 * D), 1024)
    if cutlass.const_expr(R):
        # The rank's columns are [U | 0] against a row's first key and [0 | U]
        # against its second. Stored as U, zeros, U in rows of D bytes, the
        # second K half's view starts R rows in, a whole swizzle repeat, so
        # the two views share the zeros.
        sU = smem.allocate_tensor(I8, cute.make_layout(3 * R * D), ALN)
    sZ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((4 if SOUND else 3) * NR), 16)
    sTm = smem.allocate_tensor(cutlass.Int32, cute.make_layout(NR), 16)
    # per-warp dropped mass and denominator, then their sum; until the
    # epilogue, per query row the cut as a depth, Q's scale, -Z and the bound
    sRed = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((2 * NW + 1, NR), stride=(NR, 1)), 16
    )
    sRow = cute.make_tensor(sRed.iterator, cute.make_layout(4 * NR))
    # each warp's live and refined key counts, and the tiles' key scales,
    # double buffered: the next tile's copy goes out while this one's may
    # still be in a slow warp's registers only as a pending load
    sCw = smem.allocate_tensor(cutlass.Int32, cute.make_layout(2 * NW), 16)
    sKs = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(2 * TK), 16)
    sSeg = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2, NSEG)), 16)
    sBar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1), 8)
    fullb = sBar.iterator
    if cutlass.const_expr(cfg.smem_pad):
        smem.allocate_tensor(cutlass.Int8, cute.make_layout(cfg.smem_pad), 16)

    if cutlass.const_expr(cfg.front):
        sRdy = smem.allocate_tensor(cutlass.Int32, cute.make_layout(1), 4)
        slen = front_length(mRdy, sRdy, bh, tidx, G, cfg.front_qk)
    tbase = slen
    if cutlass.const_expr(DRAFT):
        tbase = slen - DRAFT
    rk = RK
    if cutlass.const_expr(len(cfg.refine_bands) > 0):
        rk = band_gate(slen, RK, cfg.refine_bands, 0)
    # the splits interleave by tile as `decode.kernel`'s do, except under a
    # fixed chunk or beside a cascade's shared levels
    INTERLEAVE = not (cfg.chunk_keys > 0 or SLOTS != SPLIT or SLOT0)
    TSTEP = SPLIT * TK if INTERLEAVE else TK
    lo, hi = split_bounds(slen, sp, SPLIT, TK, INTERLEAVE, cfg.chunk_keys)
    gKa, gKb, gEk, gV = cache_views(bh, S, SP, D, cfg.ek_stride, PAGED, mKa, mKb, mEk, mV)
    nrows, n_tiles = split_tiles(slen, sp, lo, hi, SPLIT, TK, INTERLEAVE)
    n0 = nrows
    if n0 > TK:
        n0 = TK
    if tidx == 0:
        issue_first(
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
            D,
            PAGE,
            SEGW,
            NSEG,
            HKV,
            NW,
            PAGED,
        )

    zero_tile(sV, VBY, tidx, NT)
    # The Q tile's other half of every row is zero; the copies below write
    # only the query's own half, so the two never meet.
    NQZ = NQR * 2 * D // 16
    qz = cute.recast_ptr(sQ.iterator, None, cutlass.Uint8)
    for i in cutlass.range_constexpr((NQZ + NT - 1) // NT):
        uz = i * NT + tidx
        if uz < NQZ:
            nz = uz // NQU
            uuz = uz % NQU
            if cutlass.const_expr(MERGE):
                hz = (nz >> 2) & 1
            else:
                hz = nz & 1
            # the unit is zero when it lies in the row's other half
            if (uuz // (D // 16)) != hz:
                zero16((qz + qtile_off(nz, uuz, NQR)).toint())
    if tidx < G:
        if cutlass.const_expr(not Z_PREPASS):
            sZ[tidx] = mZ[(bh, tidx)]
            sZ[NR + tidx] = mCut[(bh, tidx)]
        sZ[2 * NR + tidx] = mEq[(bh, tidx)]
        if cutlass.const_expr(DRAFT):
            sTm[tidx] = mTm[(bh, tidx)]
    nlive = cutlass.Int32(0)
    nref = cutlass.Int32(0)

    # Row n of the Q tile: merged, n = 2 t + p is plane p of query t & 1
    # against half t >> 1; otherwise n = 8 p + 2 q + h. A query past G is a
    # copy of query 0, which at G = 1 gives every lane of a quad a key.
    NCU = D // 16
    for it in cutlass.range_constexpr((NQR * NCU + NT - 1) // NT):
        uq = it * NT + tidx
        nq = uq // NCU
        cu = uq % NCU
        if cutlass.const_expr(MERGE):
            pq = nq & 1
            qq = (nq >> 1) & 1
            hq = (nq >> 2) & 1
        else:
            pq = nq >> 3
            qq = (nq >> 1) & 3
            hq = nq & 1
        gsrc = qq
        if qq >= G:
            gsrc = cutlass.Int32(0)
        qo = (bh * UNIQUE_G + gsrc) * D + cu * 16
        so = qtile_off(nq, hq * NCU + cu, NQR)
        if uq < NQR * NCU:
            if pq == 0:
                cute.arch.cp_async_shared_global(sQ.iterator + so, mQa.iterator + qo, 16, CM)
            else:
                cute.arch.cp_async_shared_global(sQ.iterator + so, mQb.iterator + qo, 16, CM)
    if cutlass.const_expr(R):
        NUU = R * NCU
        for it in cutlass.range_constexpr((3 * NUU + NT - 1) // NT):
            uu = it * NT + tidx
            ur = uu // NCU
            cuu = uu % NCU
            uo = ur * D + ((cuu ^ ((ur >> SWRS) & SWM)) << 4)
            if uu < 3 * NUU:
                if ur >= R and ur < 2 * R:
                    zero16((cute.recast_ptr(sU.iterator, None, cutlass.Uint8) + uo).toint())
                else:
                    cute.arch.cp_async_shared_global(
                        sU.iterator + uo,
                        mU.iterator + (bh * R * D + (ur % R) * D + cuu * 16),
                        16,
                        CM,
                    )
    cute.arch.cp_async_commit_group()
    if cutlass.const_expr(cfg.front):
        if warp == 0:
            front_rows(mRdy, sZ, bh, tidx, G, NR, TAIL, cfg.front_qk)
    elif cutlass.const_expr(Z_PREPASS):
        cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(Z_PREPASS and not cfg.front):
        if tidx < G:
            sZ[tidx] = mZ[(bh, tidx)]
            sZ[NR + tidx] = mCut[(bh, tidx)]
    cute.arch.cp_async_wait_group(0)
    cute.arch.fence_view_async_shared()

    # the packed operand over plane A's (and plane B's) tile as stored
    swz = cute.make_swizzle(SWB, 4, 3)
    lka = cute.make_layout(((8, 8), (D, 2)), stride=((D, 16 * D), (1, 8 * D)))
    swq = cute.make_swizzle(3, 4, 3)
    tmq = sm90.make_trivial_tiled_mma(
        I8,
        I8,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        cutlass.Int32,
        (1, 1, 1),
        (64, NQR),
    )
    wgq = tmq.get_slice(0)
    lq = cute.make_layout((NQR, (128, (2 * D) // 128)), stride=(128, (1, NQR * 128)))
    c1 = cute.make_rmem_tensor(tmq.partition_shape_C((64, NQR)), cutlass.Int32)
    c3 = cute.make_rmem_tensor(tmq.partition_shape_C((64, NQR)), cutlass.Int32)
    if cutlass.const_expr(R):
        # y = K U over the same A: register (b, i) is rank 8 (b % (R / 8)) +
        # 2 tq + i % 2 of key 16 (i >> 1) + 8 (b >= R / 8) + gid of the warp's 32
        tmy = sm90.make_trivial_tiled_mma(
            I8,
            I8,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            cutlass.Int32,
            (1, 1, 1),
            (64, 2 * R),
        )
        wgy = tmy.get_slice(0)
        lu = cute.make_layout((2 * R, (D, 2)), stride=(D, (1, R * D)))
        cy = cute.make_rmem_tensor(tmy.partition_shape_C((64, 2 * R)), cutlass.Int32)
        # this lane's share of every query row's sum over dropped keys of
        # p y: ranks 8 bb + 2 tq + e
        if cutlass.const_expr(G <= 2):
            rY = cute.make_rmem_tensor((G, R // 8, 2), cutlass.Float32)
            rY.fill(0.0)
        else:
            # per 16 ranks, (rank gid (+ 8), rows 2 tq and 2 tq + 1)
            wmma = cute.make_tiled_mma(
                cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
            )
            rYm = [
                cute.make_rmem_tensor(wmma.partition_shape_C((16, 8)), cutlass.Float32)
                for _ in range(R // 16)
            ]
            for jt in cutlass.range_constexpr(R // 16):
                rYm[jt].fill(0.0)
        NBK = (S + 63) // 64
    swv = cute.make_swizzle(3, 4, 3)
    tmv = sm90.make_trivial_tiled_mma(
        PT,
        PT,
        cute.nvgpu.OperandMajorMode.MN,
        cute.nvgpu.OperandMajorMode.K,
        cutlass.Float32,
        (1, 1, 1),
        (64, NQ),
    )
    tmv.set(warpgroup.Field.ACCUMULATE, True)
    vgs = tmv.get_slice(0)
    rD = [
        cute.make_rmem_tensor(tmv.partition_shape_C((64, NQ)), cutlass.Float32) for _ in range(NVB)
    ]
    rS = cute.make_rmem_tensor((NP,), cutlass.Float32)
    # this lane's query row's dropped mass and denominator
    rM = cutlass.Float32(0.0)
    rN = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(NVB):
        rD[j].fill(0.0)
    cute.arch.sync_threads()

    if cutlass.const_expr(SOUND):
        sQi32 = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, None, cutlass.Int32), cute.make_layout(NQR * D // 2)
        )
        # sum_d (128 |qa_d| + |qb_d| / 2) per query row, from the rows that
        # hold its first half; eight lanes a row
        for sli in cutlass.range_constexpr((G + NW - 1) // NW):
            slq = sli * NW + warp
            slw = lane
            sls = cutlass.Float32(0.0)
            if slq < G and slw < D // 4:
                for pp in cutlass.range_constexpr(2):
                    if cutlass.const_expr(MERGE):
                        nr = 2 * slq + pp
                    else:
                        nr = 8 * pp + 2 * slq
                    wd = sQi32[(qtile_off(nr, slw // 4, NQR) >> 2) + (slw % 4)]
                    ss = cutlass.Int32(0)
                    for slby in cutlass.range_constexpr(4):
                        ub = (wd >> (8 * slby)) & 0xFF
                        ss = ss + cutlass.Int32(cutlass.select_(ub >= 128, 256 - ub, ub))
                    sls = sls + cutlass.Float32(ss) * (128.0 if pp == 0 else 0.5)
            for slst in cutlass.range_constexpr(5):
                sls = sls + cute.arch.shuffle_sync_bfly(sls, 1 << slst)
            if lane == 0 and slq < G:
                sZ[3 * NR + slq] = sls
        cute.arch.sync_threads()

    # one record per query row: the cut as a depth, Q's scale, -Z, the bound
    if tidx < NR:
        z0 = cutlass.Float32(0.0)
        cd0 = cutlass.Float32(1e30)
        e0 = cutlass.Float32(0.0)
        sl0 = cutlass.Float32(0.0)
        if tidx < G:
            z0 = sZ[tidx]
            e0 = sZ[2 * NR + tidx]
            cd0 = sZ[NR + tidx] - z0
            if cutlass.const_expr(SOUND):
                sl0 = e0 * sZ[3 * NR + tidx]
        sRow[4 * tidx] = cd0
        sRow[4 * tidx + 1] = e0
        sRow[4 * tidx + 2] = -z0
        sRow[4 * tidx + 3] = sl0

    kw0 = warp * KPW
    # this lane's query row, and its keys' offsets in the warp's 32
    if cutlass.const_expr(G == 1):
        qrow = cutlass.Int32(0)
        koffs = [16 * (tq & 1) + 8 * (tq >> 1) + gid]
    elif cutlass.const_expr(G == 2):
        qrow = tq & 1
        koffs = [16 * k + 8 * (tq >> 1) + gid for k in range(2)]
    else:
        qrow = tq
        koffs = [16 * (i >> 1) + 8 * (i & 1) + gid for i in range(4)]
    qok = qrow < G
    # The weights as `stmatrix.trans` sources: a pair of C columns per lane
    # and eight C rows per matrix, so destination row `c` is source column `c`
    # of every row, eight consecutive keys of one query. Lanes 0-15 name the
    # destination rows.
    cst = lane & 7
    if cutlass.const_expr(G == 1):
        # column 2 t holds key 16 (t & 1) + 8 (t >> 1) + gid of query 0, and
        # column 2 t + 1 a zero, sent to a pad row
        tt = cst >> 1
        pbs = sPb.iterator + pidx2(cst & 1, kw0 + 16 * (tt & 1) + 8 * (tt >> 1))
    elif cutlass.const_expr(G == 2):
        pbs = sPb.iterator + pidx2((cst >> 1) & 1, kw0 + 16 * (cst & 1) + 8 * (cst >> 2))
    else:
        pbs = sPb.iterator + pidx2(cst >> 1, kw0 + 16 * ((lane >> 3) & 1) + 8 * (cst & 1))
    if cutlass.const_expr(W2):
        prs = pbs + NQ * TK
    cute.arch.sync_threads()
    HOLD = D < 128
    rref = cute.make_rmem_tensor((4,), cutlass.Float32)
    if cutlass.const_expr(HOLD):
        for x in cutlass.range_constexpr(4):
            rref[x] = sRow[4 * qrow + x]

    rPg = cute.make_rmem_tensor((1,), cutlass.Int32)
    rPg[0] = 0
    if cutlass.const_expr(PFP):
        if tidx == 0 and lo + TSTEP < hi:
            rPg[0] = mPgT[(breq, (lo + TSTEP) // PAGE)]
    # Lane l reads key kw0 + l's verdict from a lane that owns it, so the
    # warp's live and refine words come out in key order from one shuffle
    # and two ballots, with no exchange between warps.
    if cutlass.const_expr(G == 1):
        osrc = 4 * (lane & 7) + ((lane >> 4) & 1) + 2 * ((lane >> 3) & 1)
        osh = 0
    elif cutlass.const_expr(G == 2):
        osrc = 4 * (lane & 7) + 2 * ((lane >> 3) & 1)
        osh = 3 * ((lane >> 4) & 1)
    else:
        osrc = 4 * (lane & 7)
        osh = 3 * (2 * ((lane >> 4) & 1) + ((lane >> 3) & 1))
    mLw = cute.make_rmem_tensor((1,), cutlass.Int32)
    mRw = cute.make_rmem_tensor((1,), cutlass.Int32)
    for t in cutlass.range(0, n_tiles, 1, unroll=1):
        base = lo + t * TSTEP
        nrow = hi - base
        if nrow > TK:
            nrow = TK
        nb = lo + (t + 1) * TSTEP
        nn = hi - nb
        if nn > TK:
            nn = TK
        if nn < 0:
            nn = 0
        pmsk = cutlass.Int32(0)
        if cutlass.const_expr(DRAFT):
            if base + nrow > tbase:
                for k in cutlass.range_constexpr(NP):
                    jd = base + kw0 + koffs[k] - tbase
                    hid = cutlass.Int32(0)
                    if jd >= 0 and qok:
                        hid = 1 - ((sTm[qrow] >> jd) & 1)
                    pmsk = pmsk | (hid << k)
        c1.fill(0)
        c3.fill(0)
        cute.arch.mbarrier_wait(fullb, t % 2)
        ksr = [key_scale(sKs, (t % 2) * TK + kw0 + koffs[k]) for k in range(NP)]
        sgb = t % 2
        if cutlass.const_expr(PAGED and NSEG == 1):
            sgb = sSeg[(t % 2, 0)]
        # K = 2 D doubles every operand's descriptors, which held across the
        # loop overflow the uniform file into 30-40 registers a thread: each
        # tile builds them from a base the compiler can neither fold nor hoist
        kab = opaque_i32(sKa.iterator.toint())
        qbo = opaque_i32(sQ.iterator.toint())
        fKa = wgq.make_fragment_A(wgq.partition_A(smem_view(kab, I8, swz, lka)))
        fQ = wgq.make_fragment_B(wgq.partition_B(smem_view(qbo, I8, swq, lq)))
        warpgroup.fence()
        cute.gemm(tmq, c1, fKa, fQ, c1)
        if cutlass.const_expr(R):
            fKy = wgy.make_fragment_A(wgy.partition_A(smem_view(kab, I8, swz, lka)))
            fU = wgy.make_fragment_B(
                wgy.partition_B(smem_view(opaque_i32(sU.iterator.toint()), I8, swz, lu))
            )
            cute.gemm(tmy, cy, fKy, fU, cy)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
        # the logit is one warpgroup-wide operation, so its retirement in any
        # warp means every warp's share has read plane A
        if tidx == 0:
            issue_next(
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
                D,
                PAGE,
                SEGW,
                NSEG,
                HKV,
                NW,
                PAGED,
                TK,
                TSTEP,
                PFP,
            )

        if cutlass.const_expr(not HOLD):
            for x in cutlass.range_constexpr(4):
                rref[x] = sRow[4 * qrow + x]
        # Pair k's verdict bits at shift 3 k: live 1, refine 2. The lanes
        # holding one key's other queries are ORed in below.
        pk = cutlass.Int32(0)
        if cutlass.const_expr(not TRUNCATE):
            # every key of the tile, and none past its end, which would gather
            # rows past the request
            for k in cutlass.range_constexpr(NP):
                if kw0 + koffs[k] < nrow:
                    pk = pk | (1 << (3 * k))
        tk0 = rref[1]
        for k in cutlass.range_constexpr(NP):
            rS[k] = cutlass.Float32(NEG)
            tok = cutlass.Int32(0)
            if cutlass.const_expr(DRAFT):
                tok = (pmsk >> k) & 1
            if qok and kw0 + koffs[k] < nrow and tok == 0:
                ca, cb = pair_logit(c1, k, tq, G)
                s1 = cutlass.Float32(ca * 256 + cb) * logit_scale(tk0, ksr[k]) + rref[2]
                rS[k] = s1
                if cutlass.const_expr(TRUNCATE):
                    s1c = s1
                    if cutlass.const_expr(SOUND):
                        s1c = s1 + rref[3] * ksr[k]
                    if s1c >= rref[0]:
                        pk = pk | (1 << (3 * k))
                if s1 >= rk:
                    pk = pk | (2 << (3 * k))
        if cutlass.const_expr(G == 2):
            pk = pk | cute.arch.shuffle_sync_bfly(pk, 1)
        elif cutlass.const_expr(G > 2):
            for st in cutlass.range_constexpr(2):
                pk = pk | cute.arch.shuffle_sync_bfly(pk, 1 << st)
        # a key no row keeps is neither gathered nor refined
        for k in cutlass.range_constexpr(NP):
            bk = (pk >> (3 * k)) & 7
            bk = cutlass.Int32(cutlass.select_((bk & 1) != 0, bk, 0))
            pk = (pk & ~(7 << (3 * k))) | (bk << (3 * k))
        vw = (cute.arch.shuffle_sync(pk, osrc) >> osh) & 7
        mLw[0] = cute.arch.vote_ballot_sync((vw & 1) != 0)
        mRw[0] = cute.arch.vote_ballot_sync((vw & 2) != 0)
        nlive += cute.arch.popc(mLw[0])
        nref += cute.arch.popc(mRw[0])

        if cutlass.const_expr(TAIL):
            # A warp's dropped keys re-enter as one virtual key: the first of
            # its 32 that no row kept takes their 64-key block's row of V as
            # its V and the warp's dropped mass per row as its weight.
            nv = nrow - kw0
            vm = cutlass.Int32(
                cutlass.select_(
                    nv >= 32,
                    cutlass.Int32(-1),
                    cutlass.select_(nv > 0, (cutlass.Int32(1) << nv) - 1, cutlass.Int32(0)),
                )
            )
            dw = ~mLw[0] & vm
            slb = dw & (0 - dw)
            shas = dw != 0
            sidx = cute.arch.popc(slb - 1)
            vrow = bh * NBK + (base + kw0) // 64
            if cutlass.const_expr(PAGED):
                vrow = gather_row(sSeg, sgb, base, kw0, SEGW, NSEG, PAGED) // 64
        if mRw[0] != 0:
            gather_warp_rows(
                lane,
                kw0,
                base,
                sKb,
                gKb,
                mRw,
                KPW,
                TK,
                D,
                1,
                1,
                CM,
                0,
                sSeg,
                sgb,
                SEGW,
                NSEG,
                PAGED,
            )
        cute.arch.cp_async_commit_group()
        # each warp gathers its own keys' V, which only the value barrier
        # publishes
        gather_warp_rows(
            lane, kw0, base, sV, gV, mLw, KPW, TK, D, 2, 1, CM, 1, sSeg, sgb, SEGW, NSEG, PAGED
        )
        if cutlass.const_expr(TAIL):
            if shas and lane < D // 8:
                cute.arch.cp_async_shared_global(
                    sV.iterator + gather_dst(kw0 + sidx, lane, D, TK, 2, 1),
                    mVbk.iterator + (vrow * D + lane * 8),
                    16,
                    CM,
                )
        cute.arch.cp_async_commit_group()
        if cutlass.const_expr(TAIL):
            # The tail keeps every live key for every row, so a dropped
            # key is no row's and its weight is its coarse logit's, final
            # at the verdict: the tail's sums run here, under the gathers,
            # and y dies before the refine.
            pd = [
                cutlass.Float32(cutlass.select_(((pk >> (3 * k)) & 1) != 0, 0.0, ex2(rS[k])))
                for k in range(NP)
            ]
            msum = warp_mass(pd, G)
            if cutlass.const_expr(R):
                if cutlass.const_expr(G <= 2):
                    rank_sums(rY, cy, [pd[k] * ksr[k] for k in range(NP)], lane, G, R)
                else:
                    rank_mma(wmma, rYm, cy, pd, ksr, lane, tq, R)
        # the refine is one warpgroup-wide matmul, so it runs on every tile;
        # a row nobody refined meets its select
        cute.arch.cp_async_wait_group(1)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        warpgroup.fence()
        fKb = wgq.make_fragment_A(
            wgq.partition_A(smem_view(opaque_i32(sKb.iterator.toint()), I8, swz, lka))
        )
        fQb = wgq.make_fragment_B(
            wgq.partition_B(smem_view(opaque_i32(sQ.iterator.toint()), I8, swq, lq))
        )
        cute.gemm(tmq, c3, fKb, fQb, c3)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

        wv = []
        for k in cutlass.range_constexpr(NP):
            bk = pk >> (3 * k)
            lv = qok and kw0 + koffs[k] < nrow and (bk & 1) != 0
            rfk = (bk & 2) != 0
            if cutlass.const_expr(DROPPED):
                rfk = lv and rfk
            ra, rb = pair_logit(c3, k, tq, G)
            s = rS[k]
            s = cutlass.Float32(
                cutlass.select_(
                    rfk,
                    s
                    + (cutlass.Float32(ra) + cutlass.Float32(rb) * (1.0 / KBR))
                    * logit_scale(rref[1], ksr[k]),
                    s,
                )
            )
            if cutlass.const_expr(not DROPPED or TAIL):
                keep = lv
                if cutlass.const_expr(not KEEP_ALL):
                    keep = lv and s >= rref[0]
                w = ex2_if(cutlass.Int32(keep), s)
            else:
                e = ex2(s)
                if cutlass.const_expr(KEEP_ALL):
                    keep = lv
                else:
                    keep = s >= rref[0]
                w = cutlass.Float32(cutlass.select_(keep, e, 0.0))
                rM = rM + cutlass.Float32(cutlass.select_(keep, 0.0, e))
            wv.append(w)
        if cutlass.const_expr(TAIL):
            # the virtual key's lanes carry the warp's dropped mass
            vs = sidx >> 4
            vh = (sidx >> 3) & 1
            vg = gid == (sidx & 7)
            for k in cutlass.range_constexpr(NP):
                if cutlass.const_expr(G == 1):
                    own = vg and tq == vs + 2 * vh
                elif cutlass.const_expr(G == 2):
                    own = vg and (tq >> 1) == vh and vs == k
                else:
                    own = vg and 2 * vs + vh == k
                wv[k] = cutlass.Float32(cutlass.select_(shas and own, msum, wv[k]))
        if cutlass.const_expr(G == 1):
            wv.append(cutlass.Float32(0.0))
        # the denominator sums the rounded weight the matmul will see
        pws = []
        prl = []
        for hk in cutlass.range_constexpr(len(wv) // 2):
            w0 = wv[2 * hk]
            w1 = wv[2 * hk + 1]
            pw, f0, f1 = pack_weights(w0, w1)
            if cutlass.const_expr(W2):
                pr, _, _ = pack_weights(w0 - f0, w1 - f1)
                f0 = w0
                f1 = w1
                prl.append(pr)
            rN = rN + f0
            if cutlass.const_expr(G > 1):
                rN = rN + f1
            pws.append(pw)
        stmatrix(pbs, pws, MMALayout.COL)
        if cutlass.const_expr(W2):
            stmatrix(prs, prl, MMALayout.COL)
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        pbo = opaque_i32(sPb.iterator.toint())
        lpt = cute.make_layout((NQ, (64, 2)), stride=(64, (1, NQ * 64)))
        fP = vgs.make_fragment_B(vgs.partition_B(smem_view(pbo, PT, swv, lpt)))
        if cutlass.const_expr(W2):
            fPr = vgs.make_fragment_B(vgs.partition_B(smem_view(pbo + NQ * TK * 2, PT, swv, lpt)))
        vbo = opaque_i32(sV.iterator.toint())
        fVo = [
            vgs.make_fragment_A(
                vgs.partition_A(
                    smem_view(
                        vbo + j * TK * 64 * 2,
                        PT,
                        swv,
                        cute.make_layout((64, TK), stride=(1, 64)),
                    )
                )
            )
            for j in range(NVB)
        ]
        warpgroup.fence()
        for j in cutlass.range_constexpr(NVB):
            if cutlass.const_expr(W2):
                for kb in cutlass.range_constexpr(TK // 16):
                    cute.gemm(tmv, rD[j], fVo[j][(None, None, kb)], fP[(None, None, kb)], rD[j])
                    cute.gemm(tmv, rD[j], fVo[j][(None, None, kb)], fPr[(None, None, kb)], rD[j])
            else:
                cute.gemm(tmv, rD[j], fVo[j], fP, rD[j])
        warpgroup.commit_group()
    pid = cutlass.Float32(sRed[(2 * NW, 0)]).bitcast(cutlass.Int32)
    bh = cutlass.Float32(sRed[(2 * NW, 1)]).bitcast(cutlass.Int32)
    sp = pid % SPLIT
    if cutlass.const_expr(cfg.front and not TAIL and cfg.front_qk != 1):
        front_sums(mRdy, bh, G, cfg.front_qk)
    warpgroup.wait_group(0)
    RKC = cfg.rank_combine and R > 0
    if cutlass.const_expr(R):
        # Each warp's rank sums into plane A's tile, which the last logit read
        # before the last tile's value barrier; the combine or the expansion
        # below adds the warps in warp order.
        NYD = G * R
        sYd = cute.make_tensor(
            cute.recast_ptr(sKa.iterator, None, cutlass.Float32), cute.make_layout(NW * NYD)
        )
        if cutlass.const_expr(G <= 2):
            for q in cutlass.range_constexpr(G):
                for bb in cutlass.range_constexpr(R // 8):
                    for e_ in cutlass.range_constexpr(2):
                        yv = rY[(q, bb, e_)]
                        for st in cutlass.range_constexpr(2, 5):
                            yv = yv + cute.arch.shuffle_sync_bfly(yv, 1 << st)
                        if gid == 0:
                            sYd[warp * NYD + q * R + 8 * bb + 2 * tq + e_] = yv
        else:
            for jt in cutlass.range_constexpr(R // 16):
                for x in cutlass.range_constexpr(4):
                    qy = 2 * tq + (x % 2)
                    if qy < G:
                        sYd[warp * NYD + qy * R + 16 * jt + 8 * (x // 2) + gid] = rYm[jt][x]
        if cutlass.const_expr(not RKC):
            load_basis(sV, mVr, bh, tidx, D, R, NT)
            cute.arch.cp_async_wait_group(0)
    pix = partial_slot(pid, bh, sp, SPLIT, SLOTS, SLOT0, ORDERED, DIRECT)
    # the lanes sharing a query row sum over the warp, then shared over warps
    m = rM
    n = rN
    LO = 0 if G == 1 else (1 if G == 2 else 2)
    for st in cutlass.range_constexpr(LO, 5):
        m = m + cute.arch.shuffle_sync_bfly(m, 1 << st)
        n = n + cute.arch.shuffle_sync_bfly(n, 1 << st)
    if lane < (1 << LO) and qok:
        sRed[(warp, qrow)] = m
        sRed[(NW + warp, qrow)] = n
    # each warp counted its own keys
    if lane == 0:
        sCw[warp] = nlive
        sCw[NW + warp] = nref
    cute.arch.sync_threads()
    if tidx < G:
        m = cutlass.Float32(0.0)
        n = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(NW):
            m = m + sRed[(w, tidx)]
            n = n + sRed[(NW + w, tidx)]
        sRed[(2 * NW, tidx)] = m
        if cutlass.const_expr(DIRECT):
            sZ[tidx] = n + m
        mL[(pix, tidx)] = n + m
    if cutlass.const_expr(RKC):
        store_rank_sums(mVr, sYd, pix, tidx, G * R, NYD, NW, NT)
    elif cutlass.const_expr(R):
        # the warps' sums in warp order, once per (row, rank)
        if tidx < G * R:
            ys = sYd[tidx]
            for w in cutlass.range_constexpr(1, NW):
                ys = ys + sYd[w * NYD + tidx]
            sYd[tidx] = ys
    if cutlass.const_expr(DROPPED or DIRECT or (R and not RKC)):
        cute.arch.sync_threads()
    for r in cutlass.range_constexpr(2):
        gq = 2 * tq + r
        if gq < G:
            xs = []
            ccs = []
            for j in cutlass.range_constexpr(NVB):
                for hb in cutlass.range_constexpr(2):
                    ccs.append(64 * j + 16 * warp + gid + 8 * hb)
                    xs.append(rD[j][2 * hb + r])
            if cutlass.const_expr(R and not RKC):
                add_rank_term(xs, ccs, sYd, sV, gq, R)
            for c in cutlass.range_constexpr(len(xs)):
                x = xs[c]
                if cutlass.const_expr(DROPPED and not TAIL):
                    if cutlass.const_expr(ROW_VMEAN):
                        x = x + sRed[(2 * NW, gq)] * mVm[(bh, gq, ccs[c])]
                    else:
                        x = x + sRed[(2 * NW, gq)] * mVm[(bh, ccs[c])]
                if cutlass.const_expr(DIRECT):
                    x = x * EVS / sZ[gq]
                xs[c] = x
            ob = (pix * UNIQUE_G + gq) * D
            for c in cutlass.range_constexpr(len(xs)):
                st_global_f32(mO.iterator + (ob + ccs[c]), [xs[c]])
    if tidx == 0:
        nlive = sCw[0] + sCw[1] + sCw[2] + sCw[3]
        nref = sCw[NW] + sCw[NW + 1] + sCw[NW + 2] + sCw[NW + 3]
        mCnt[(pix, 0)] = nlive
        mCnt[(pix, 1)] = nref
        if cutlass.const_expr(cfg.front and cfg.front_clear and DIRECT):
            for k in cutlass.range_constexpr(1 + 2 * G):
                mRdy[bh * (1 + 2 * G) + k] = 0
    if cutlass.const_expr(cfg.front):
        cute.arch.griddepcontrol_wait()


@cute.jit
def launch_packed(
    mQa,
    mQb,
    mEq,
    mKa,
    mKb,
    mEk,
    mV,
    mVb,
    mVm,
    mVbk,
    mU,
    mVr,
    mZ,
    mCut,
    mO,
    mL,
    mCnt,
    mPgT,
    mSql,
    mTm,
    mRi,
    mRdy,
    mOrd,
    S: cutlass.Int32,
    SP: cutlass.Int32,
    EVS: cutlass.Float32,
    RK: cutlass.Float32,
    RV: cutlass.Float32,
    cfg: cutlass.Constexpr,
    stream,
):
    packed_kernel(
        mQa,
        mQb,
        mEq,
        mKa,
        mKb,
        mEk,
        mV,
        mVb,
        mVm,
        mVbk,
        mU,
        mVr,
        mZ,
        mCut,
        mO,
        mL,
        mCnt,
        mPgT,
        mSql,
        mTm,
        mRi,
        mRdy,
        mOrd,
        S,
        SP,
        EVS,
        RK,
        RV,
        cfg,
    ).launch(
        grid=[cfg.n_groups * cfg.split, 1, 1],
        block=[NT, 1, 1],
        stream=stream,
        min_blocks_per_mp=cfg.min_blocks,
        use_pdl=bool(cfg.z_prepass),
    )
