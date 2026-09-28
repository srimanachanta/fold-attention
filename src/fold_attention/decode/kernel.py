"""The decode kernel: one CTA per (row group, split), one warpgroup per CTA.

A row group is the `G` query rows that share one KV head.

Per 64-key tile:

1. Plane A of K arrives by one bulk copy. The coarse logit `S^T = K_a Q^T`
   stacks both Q planes in N, so one `wgmma` gives `256 c1 + c2`.
2. The verdict packs three bits per key, ORed over the query group: live
   (some row clears its cut), refine K (weight >= 2^-refine_k) and refine V.
3. The live keys' V rows and the refined keys' K plane-B rows are gathered,
   and plane B of K against both Q planes refines those keys' logits.
   The next tile's plane A goes out once the logit has retired (at G >= 8)
   or behind the verdict's barrier, its page entry loaded a tile ahead.
4. The weight pass writes `p = 2^(s - Z)` into the weight buffer, and
   `A^T += V^T P^T` runs as a `wgmma` with M = channels and N = G.

A tile takes two CTA barriers, one to publish the verdict and one to publish
V and the weights, plus a third under an 8-bit V that is widened in shared.

A group of at most four rows at D = 64 on a bf16 V runs `decode.packed`
instead (`config.pack_for`), two keys in each row of the logit's M over
128-key tiles.

`Z` is fixed before the loop, so a weight is final the moment its logit is,
and nothing is rescaled. The partial numerator and denominator are plain sums
that `combine` adds across splits.
"""

import cutlass
import cutlass.utils.hopper_helpers as sm90
from cutlass import cute
from cutlass.cute.nvgpu import warpgroup
from cutlass.experimental import primitives
from cutlass.experimental.primitives.nvvm_wrapper import MMALayout, stmatrix

from .cache import KBR, swizzle_of
from .config import BN, NT, weights_in_plane_b
from .device import (
    NEG,
    add_rank_term,
    band_gate,
    cache_views,
    cascade_row,
    coarse_logit,
    front_length,
    front_rows,
    front_sums,
    gather_dst,
    gather_rows,
    gather_warp_rows,
    issue_first,
    issue_next,
    key_scale,
    load_basis,
    logit_scale,
    mask_any,
    mask_word,
    partial_slot,
    pidx,
    refine_logit,
    smem_vec,
    smem_view,
    split_bounds,
    split_tiles,
    store_rank_sums,
    value_fragment,
    zero_tile,
)
from .ptx import (
    E4O,
    E4W,
    ex2,
    ex2_if,
    movmatrix_t,
    opaque_i32,
    pack_weights,
    st_global_f32,
    widen16_bf16,
    widen16_refine_bf16,
)

F8 = cutlass.Float8E4M3FN
I8 = cutlass.Int8


def row_ref(sRow, ng, tq, n: int):
    """The first `n` words of lanes `tq`'s record for row block `ng`, in one
    vector load: the cut then Q's scale, then -Z and the sound bound."""
    r = cute.make_rmem_tensor((n,), cutlass.Float32)
    cute.autovec_copy(smem_vec(sRow, (ng * 4 + tq) * 8, n), r)
    return r


@cute.kernel
def decode_kernel(
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
    V8 = cfg.v8
    V_REGS = cfg.v_regs
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
    SHARED = cfg.shared
    DIRECT = cfg.direct
    ORDERED = cfg.order
    TAIL = cfg.tail_blocks
    TAIL_RANK = cfg.tail_rank
    # The next tile's plane A goes out as soon as the logit, one
    # warpgroup-wide operation, has retired: every warp's share has read the
    # tile by then. A group of eight rows or more has a verdict long enough to
    # cover the issuing thread's work; a narrower one would wait on it at the
    # verdict's barrier (1-2% at D = 128), so it issues behind that barrier.
    EI = G >= 8

    CM = "cg"
    NW = NT // 32
    # keys per warp, which is one 16-key m-tile of the logit's C
    BNW = BN // NW
    NG = (G + 7) // 8
    NVB = D // 64
    # channels a lane owns of one 64-channel value-operand block
    CPT = D // 32
    NUV = D // 16
    # The cache stores plane A with 16-byte unit `u` of row `r` at
    # `u ^ ((r >> SWRS) & SWM)`, which is the descriptor's own swizzle atom, so
    # the shared tile needs no padding and arrives in one bulk copy.
    SWU, SWRS, ALN = swizzle_of(D)
    SWB = SWU.bit_length() - 1
    SWM = SWU - 1
    PT = cutlass.BFloat16
    # A bf16 weight carries its rounding error as a second bf16 weight in a
    # buffer of its own, and the value matmul runs over both into the one f32
    # accumulator: 16 bits of mantissa with f32's range, where f16 has 11 and
    # overflows 16 binades over the reference. The V operand is shared.
    W2 = cfg.weight_terms == 2
    # where the value matmul's descriptors overflow the uniform file: the
    # second term's four, or a wide group's
    PIN = W2 or NG >= 3
    # an 8-bit V lands in a staging tile swizzled for whichever pass reads it;
    # a 16-bit one is gathered straight into the descriptor's layout
    VSW = 2 if V_REGS else (0 if V8 else 1)
    EBV = 1 if V8 else 2
    # a tile is `BN` contiguous rows of one page when a page is at least a
    # tile wide, and `BN / PAGE` runs otherwise; the shared tile is the same
    SEGW = min(PAGE, BN) if PAGED else BN
    NSEG = BN // SEGW
    PFP = PAGED and NSEG == 1
    # U's rows follow both Q planes in one tile, so y = K U rides the logit's
    # matmul: rank j is column j % 8 of its C block 2 NG + j // 8
    NQR = 2 * NG * 8 + TAIL_RANK
    # 16-row tiles of the rank, the M of the warp matmul that sums y over the
    # dropped keys
    NJT = (TAIL_RANK + 15) // 16

    # the combine launches while this grid runs and waits for it to finish
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
        # the split is over this request's own length
        slen = mSql[breq]

    # Descriptor-read buffers first: they start on the swizzle period, and the
    # small ones pack behind them with no alignment gap. Shared memory is what
    # caps residency.
    smem = cutlass.memory.SmemAllocator()
    sKa = smem.allocate_tensor(I8, cute.make_layout((BN, D), stride=(D, 1)), ALN)
    # Plane B is gathered into its own buffer, which V's second plane reuses
    # once the refine matmul has read it.
    sKb = smem.allocate_tensor(I8, cute.make_layout((BN, D), stride=(D, 1)), ALN)
    if cutlass.const_expr(V_REGS):
        # the e4m3 staging tile is the only V tile: the register operand is
        # converted straight out of it
        sV = smem.allocate_tensor(F8, cute.make_layout((BN, D), stride=(D, 1)), 128)
        VBY = BN * D
    else:
        sV = smem.allocate_tensor(PT, cute.make_layout(BN * D), 1024)
        VBY = BN * D * 2
    # under W2 the weights' rounding errors follow them in a buffer of their own
    NPB = NG * 8 * BN * (2 if W2 else 1)
    if cutlass.const_expr(weights_in_plane_b(D, NG, W2, V8)):
        # Plane B's tile is dead from the refine matmul's retirement, before the
        # weight pass, to the next tile's verdict, after this tile's value
        # matmul is retired: the weights fit in that gap, and a buffer of their
        # own would cost D = 128 its last CTA per SM. A declined key's plane-B
        # row then holds weight bytes, which its select discards.
        sPb = cute.make_tensor(cute.recast_ptr(sKb.iterator, None, PT), cute.make_layout(NPB))
    else:
        sPb = smem.allocate_tensor(PT, cute.make_layout(NPB), 1024)
    # Both Q planes are B against the same A, so they are one tile stacked in N.
    sQa = smem.allocate_tensor(I8, cute.make_layout((NQR, D), stride=(D, 1)), ALN)
    sQb = cute.make_tensor(sQa.iterator + NG * 8 * D, cute.make_layout((NG * 8, D), stride=(D, 1)))
    # Z, the cut and Q's scale, one block of NG * 8 each
    sZ = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((4 if cfg.sound else 3) * NG * 8), 16
    )
    # one ancestor bitmask per query row under a draft; under `SHARED` (which
    # rules out a draft) the same array holds each stacked row's map word
    sTm = smem.allocate_tensor(cutlass.Int32, cute.make_layout(NG * 8), 16)
    sRi = sTm
    # per-warp dropped mass and denominator, then their sum
    sRed = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((2 * NW + 1, NG * 8), stride=(NG * 8, 1)), 16
    )
    # Two tenants of sRed until the epilogue's first write to it, which comes
    # after a CTA barrier that follows their last reads. Per (row block, tq),
    # the reference of the two rows those lanes own: the cut as a depth, Q's
    # scale, -Z and the sound bound, which D = 128 reads on every tile rather
    # than holding in registers across the loop, where they cost 6 per row
    # block. A shared buffer of their own would cost D = 64 and the tail
    # builds a CTA per SM.
    sRow = cute.make_tensor(sRed.iterator, cute.make_layout(NG * 32))
    # Then, at `sRed[(2 NW, 0)]` and `[(2 NW, 1)]`, past the table in the
    # row-major buffer and reached only by the epilogue's second write, the
    # block index and the row group: the loop needs them only for the issuing
    # thread's page lookups and the epilogue for its addresses, and held across
    # the loop they cost 2-4 registers, with the front's word.
    # BN packed verdicts
    sLive = smem.allocate_tensor(cutlass.Int32, cute.make_layout(BN), 16)
    # the tiles' key scales, which arrive with their plane A, double buffered
    # by the tile's parity so a tile's weight pass reads its own after the
    # next tile's copy is out, rather than holding them in registers across
    # the gathers
    sKs = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(2 * BN), 16)
    if cutlass.const_expr(V8 and not V_REGS):
        sVq = smem.allocate_tensor(F8, cute.make_layout((BN, D), stride=(D, 1)), 16)
    else:
        sVq = sV
    sVr = sKb
    # double buffered: thread 0 writes the next tile's page bases when it
    # issues that tile's copy, and the value barrier publishes them
    sSeg = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2, NSEG)), 16)
    sBar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1), 8)
    fullb = sBar.iterator
    if cutlass.const_expr(cfg.smem_pad):
        # holds an SM to the grid's share of CTAs (`heuristics.capped_footprint`)
        smem.allocate_tensor(cutlass.Int8, cute.make_layout(cfg.smem_pad), 16)

    if cutlass.const_expr(cfg.front):
        # Nothing here waits on the front's grid, so everything up to the first
        # verdict runs under its estimate, and the loop starts on this row
        # group's reference, not the step's last.
        sRdy = smem.allocate_tensor(cutlass.Int32, cute.make_layout(1), 4)
        slen = front_length(mRdy, sRdy, bh, tidx, G, cfg.front_qk)
    # The draft's keys are the last `DRAFT` of the cache. Everything below
    # `tbase` is context every row sees, and `tbase` is CTA-uniform, so only a
    # tile reaching past it pays for the mask.
    tbase = slen
    if cutlass.const_expr(DRAFT):
        tbase = slen - DRAFT
    # A request's gates follow its own length, not the batch's longest, so its
    # bits do not depend on what it is batched with.
    rk = RK
    rv = RV
    if cutlass.const_expr(len(cfg.refine_bands) > 0):
        rk = band_gate(slen, RK, cfg.refine_bands, 0)
        rv = band_gate(slen, RV, cfg.refine_bands, 1)
    # Live keys cluster in the most recent tiles, and contiguous chunks put all
    # of them, and the gathers and weights that follow them, on the last split,
    # whose CTA then ends the grid, so the splits interleave by tile. A fixed
    # chunk keeps its contiguous range, which is what keeps a request's sums
    # independent of its batch. A cascade's levels keep contiguous chunks too.
    INTERLEAVE = not (cfg.chunk_keys > 0 or SHARED or SLOTS != SPLIT or SLOT0)
    TSTEP = SPLIT * BN if INTERLEAVE else BN
    lo, hi = split_bounds(slen, sp, SPLIT, BN, INTERLEAVE, cfg.chunk_keys)
    gKa, gKb, gEk, gV, gVb = cache_views(bh, S, SP, D, cfg.ek_stride, PAGED, mKa, mKb, mEk, mV, mVb)
    nrows, n_tiles = split_tiles(slen, sp, lo, hi, SPLIT, BN, INTERLEAVE)
    n0 = nrows
    if n0 > BN:
        n0 = BN
    # the first tile goes out before the setup below, which runs under it
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

    # a declined row keeps what an earlier tile left in its slot, and the first
    # tile has to find something finite there
    zero_tile(sV, VBY, tidx, NT)
    # sKb needs no such fill: a stale plane-B row only reaches the logit of a
    # key the refine gate declined, and V's second plane is masked per key
    if tidx < G:
        # Under `SHARED` each stacked row names its own (row group, row), and Z,
        # the cut and Q's scale are the unique level's: one reference for both
        # levels is what makes their partials addends of one sum.
        bhq = bh
        gsq = tidx
        if cutlass.const_expr(SHARED):
            w = mRi[(bh, tidx)]
            bhq, gsq = cascade_row(w)
            sRi[tidx] = ((bhq * SLOTS + SLOT0 + sp) << 7) | (w & 127)
        # under `Z_PREPASS` the prepass (or the front) writes Z and the cut, and
        # they are read below
        if cutlass.const_expr(not Z_PREPASS):
            sZ[tidx] = mZ[(bhq, gsq)]
            sZ[NG * 8 + tidx] = mCut[(bhq, gsq)]
        sZ[2 * NG * 8 + tidx] = mEq[(bhq, gsq)]
        if cutlass.const_expr(DRAFT):
            sTm[tidx] = mTm[(bh, tidx)]
    nlive = cutlass.Int32(0)
    nref = cutlass.Int32(0)

    # The swizzle goes on the pointer, so each tile stays a row-major view of
    # bytes the cache already stores swizzled. The shift is 3 in every mode
    # wgmma accepts, which is why a 64-byte row XORs by `(r >> 1) & 3`.
    swz = cute.make_swizzle(SWB, 4, 3)
    wKa = cute.make_tensor(
        cute.recast_ptr(sKa.iterator, swz, I8), cute.make_layout((BN, D), stride=(D, 1))
    )
    wKb = cute.make_tensor(
        cute.recast_ptr(sKb.iterator, swz, I8), cute.make_layout((BN, D), stride=(D, 1))
    )
    wQ2 = cute.make_tensor(
        cute.recast_ptr(sQa.iterator, swz, I8), cute.make_layout((2 * NG * 8, D), stride=(D, 1))
    )
    wQab = cute.make_tensor(
        cute.recast_ptr(sQa.iterator, swz, I8), cute.make_layout((NQR, D), stride=(D, 1))
    )
    # one 16-byte copy per unit of both Q planes, the CTA's first global
    # round trip
    NQU = NG * 8 * (D // 16)
    for it in cutlass.range_constexpr((2 * NQU + NT - 1) // NT):
        u = it * NT + tidx
        uq = u % NQU
        gq8 = uq // (D // 16)
        cu = uq % (D // 16)
        # rows past G pad N to eight; the verdict masks them by `gq < G`
        gsrc = gq8
        if gq8 >= G:
            gsrc = cutlass.Int32(0)
        bhs = bh
        if cutlass.const_expr(SHARED):
            # from global rather than sRi, which is a barrier away
            bhs, gsrc = cascade_row(mRi[(bh, gsrc)])
        qo = (bhs * UNIQUE_G + gsrc) * D + cu * 16
        so = gq8 * D + ((cu ^ ((gq8 >> SWRS) & SWM)) << 4)
        if u < NQU:
            primitives.cp_async_shared_global(sQa.iterator + so, mQa.iterator + qo, 16, CM)
        else:
            if u < 2 * NQU:
                primitives.cp_async_shared_global(sQb.iterator + so, mQb.iterator + qo, 16, CM)
    if cutlass.const_expr(TAIL_RANK):
        # U's rows follow both Q planes, swizzled as a Q row is
        NUU = TAIL_RANK * (D // 16)
        for it in cutlass.range_constexpr((NUU + NT - 1) // NT):
            uu = it * NT + tidx
            ur = 2 * NG * 8 + uu // (D // 16)
            cuu = uu % (D // 16)
            if uu < NUU:
                primitives.cp_async_shared_global(
                    sQa.iterator + (ur * D + ((cuu ^ ((ur >> SWRS) & SWM)) << 4)),
                    mU.iterator + (bh * TAIL_RANK * D + uu * 16),
                    16,
                    CM,
                )
    cute.arch.cp_async_commit_group()
    if cutlass.const_expr(cfg.front):
        if warp == 0:
            front_rows(mRdy, sZ, bh, tidx, G, NG * 8, TAIL, cfg.front_qk)
    elif cutlass.const_expr(Z_PREPASS):
        # only Z and the cut are the prepass's output, so only they wait on it
        cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(Z_PREPASS and not cfg.front):
        # a cascade computes its reference before either level, so a shared
        # level never has a prepass of its own
        if tidx < G:
            sZ[tidx] = mZ[(bh, tidx)]
            sZ[NG * 8 + tidx] = mCut[(bh, tidx)]
    cute.arch.cp_async_wait_group(0)
    # `wgmma` reads through the async proxy and `st.shared` and `cp.async`
    # write through the generic one, and a barrier orders threads, not
    # proxies: without this fence before every descriptor read of a tile
    # written that way, a descriptor reads a stale tile about one block in a
    # hundred, and only when CTAs share an SM.
    cute.arch.fence_view_async_shared()
    # The refine takes both Q planes too: plane B of K against plane B of Q is
    # 2^-16 of the logit's scale per channel, the same size as K's own
    # rounding, and leaving it out doubles a refined logit's error.
    tmma = sm90.make_trivial_tiled_mma(
        I8,
        I8,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        cutlass.Int32,
        (1, 1, 1),
        (BN, 2 * NG * 8),
    )
    wgs = tmma.get_slice(0)
    fKb = wgs.make_fragment_A(wgs.partition_A(wKb))
    fQ2 = wgs.make_fragment_B(wgs.partition_B(wQ2))
    # the coarse logit takes both Q planes against one A in one N = 16 NG
    # matmul: half the instructions and half the serial latency of two
    tmmq = sm90.make_trivial_tiled_mma(
        I8,
        I8,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        cutlass.Int32,
        (1, 1, 1),
        (BN, NQR),
    )
    wgq = tmmq.get_slice(0)
    fKa = wgq.make_fragment_A(wgq.partition_A(wKa))
    fQab = wgq.make_fragment_B(wgq.partition_B(wQab))
    # C at N = 8 NG is NG blocks of the m64n8 pattern in order, so register
    # `ng * 4 + i` is key `gid (+ 8 for i >= 2)` of this warp's sixteen against
    # row `ng * 8 + 2 tq + i % 2`. c1 holds qa's blocks, qb's, then y's.
    c1 = cute.make_rmem_tensor(tmmq.partition_shape_C((BN, NQR)), cutlass.Int32)
    c3 = cute.make_rmem_tensor(tmma.partition_shape_C((BN, 2 * NG * 8)), cutlass.Int32)
    # A is V^T: channels are contiguous, so the operand is MN-major, one
    # 64-channel block a (64, BN) view. B is P^T, K-major.
    swv = cute.make_swizzle(3, 4, 3)
    if cutlass.const_expr(V_REGS):
        # A register A is K-major, and its M is a label: row 16w + gid (+ 8) of
        # block j stands for channel 4p + hi + 2j, p = 8w + gid, so one 32-bit
        # load feeds a lane's four rows.
        tmv = sm90.make_trivial_tiled_mma(
            PT,
            PT,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32,
            (1, 1, 1),
            (64, NG * 8),
            warpgroup.OperandSource.RMEM,
        )
    else:
        tmv = sm90.make_trivial_tiled_mma(
            PT,
            PT,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32,
            (1, 1, 1),
            (64, NG * 8),
        )
    # this accumulator carries the whole KV loop
    tmv.set(warpgroup.Field.ACCUMULATE, True)
    vgs = tmv.get_slice(0)
    if cutlass.const_expr(V_REGS):
        fV = [cute.make_rmem_tensor(tmv.partition_shape_A((64, BN)), PT) for _ in range(NVB)]
        fV4 = [cute.recast_tensor(x, cutlass.Uint32) for x in fV]
    rD = [
        cute.make_rmem_tensor(tmv.partition_shape_C((64, NG * 8)), cutlass.Float32)
        for _ in range(NVB)
    ]
    rS = cute.make_rmem_tensor((NG, 4), cutlass.Float32)
    rL = cute.make_rmem_tensor((2,), cutlass.Int32)
    rR = cute.make_rmem_tensor((2,), cutlass.Int32)
    # a tile's verdicts, one bit per key
    NWD = (BN + 31) // 32
    mLv = cute.make_rmem_tensor((NWD,), cutlass.Int32)
    mRf = cute.make_rmem_tensor((NWD,), cutlass.Int32)
    mRv = cute.make_rmem_tensor((NWD,), cutlass.Int32)
    # the dropped mass and the kept mass (the denominator), per row
    rM = cute.make_rmem_tensor((NG, 2), cutlass.Float32)
    rN = cute.make_rmem_tensor((NG, 2), cutlass.Float32)
    rM.fill(0.0)
    rN.fill(0.0)
    if cutlass.const_expr(TAIL):
        # the keys no row kept
        mDd = cute.make_rmem_tensor((NWD,), cutlass.Int32)
        # the tile's block row of V, whose rows are 64 keys of one row group
        NBK = (S + BN - 1) // BN
        # Warp matmuls over one warp's sixteen keys (K), rows in N. B is the
        # dropped weights, A is ones for their sum and y for its rank term.
        wmma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
        )
        fOne = cute.make_rmem_tensor(wmma.partition_shape_A((16, 16)), cutlass.BFloat16)
        fOne.fill(1.0)
        fBd = cute.make_rmem_tensor(wmma.partition_shape_B((8, 16)), cutlass.BFloat16)
        fBd4 = cute.recast_tensor(fBd, cutlass.Uint32)
        fDv = cute.make_rmem_tensor(fBd.layout, cutlass.Float32)
        fMm = cute.make_rmem_tensor(wmma.partition_shape_C((16, 8)), cutlass.Float32)
    if cutlass.const_expr(TAIL_RANK):
        # this warp's sum over its dropped keys of p y, per row block and 16
        # ranks: (rank gid (+ 8), row 2 tq (+ 1) of the block)
        rYd = [
            [
                cute.make_rmem_tensor(wmma.partition_shape_C((16, 8)), cutlass.Float32)
                for _ in range(NJT)
            ]
            for _ in range(NG)
        ]
        for ng in cutlass.range_constexpr(NG):
            for jt in cutlass.range_constexpr(NJT):
                rYd[ng][jt].fill(0.0)
        fY = [
            cute.make_rmem_tensor(wmma.partition_shape_A((16, 16)), cutlass.BFloat16)
            for _ in range(NJT)
        ]
        fY4 = [cute.recast_tensor(x, cutlass.Uint32) for x in fY]
        fYf = cute.make_rmem_tensor(fY[0].layout, cutlass.Float32)
    for j in cutlass.range_constexpr(NVB):
        rD[j].fill(0.0)
    cute.arch.sync_threads()

    # The verdict tests the plane-A estimate against the cut, but that estimate
    # omits K's second plane, which is worth up to `ek / 2` a channel, so the
    # screen as written can drop a key the full path would call live. The
    # omission is bounded by `(ek / 2) sum_d |q_d|`: a row's sum, held as
    # `sum_d (128 |qa_d| + |qb_d| / 2)`, which the row's `eq` and each key's
    # `ek / 256` then scale.
    SOUND = cfg.sound
    if cutlass.const_expr(SOUND):
        SLW = D // 4
        SLR = 32 // SLW
        sQi32 = cute.make_tensor(
            cute.recast_ptr(sQa.iterator, None, cutlass.Int32),
            cute.make_layout(NQR * (D // 4)),
        )
        for sli in cutlass.range_constexpr((NG * 8 + NW * SLR - 1) // (NW * SLR)):
            slrow = (sli * NW + warp) * SLR + lane // SLW
            slw = lane % SLW
            # the tile is swizzled in 16-byte units, which are four int32 words
            sloff = slrow * (D // 4) + (((slw // 4) ^ ((slrow >> SWRS) & SWM)) << 2) + (slw % 4)
            slsa = cutlass.Int32(0)
            slsb = cutlass.Int32(0)
            if slrow < NG * 8:
                slwa = sQi32[sloff]
                slwb = sQi32[NG * 8 * (D // 4) + sloff]
                for slby in cutlass.range_constexpr(4):
                    # masked after the shift, so its signedness cannot matter
                    slua = (slwa >> (8 * slby)) & 0xFF
                    slub = (slwb >> (8 * slby)) & 0xFF
                    slsa = slsa + cutlass.Int32(cutlass.select_(slua >= 128, 256 - slua, slua))
                    slsb = slsb + cutlass.Int32(cutlass.select_(slub >= 128, 256 - slub, slub))
            for slst in cutlass.range_constexpr(SLW.bit_length() - 1):
                slsa = slsa + cute.arch.shuffle_sync_bfly(slsa, 1 << slst)
                slsb = slsb + cute.arch.shuffle_sync_bfly(slsb, 1 << slst)
            if slw == 0 and slrow < NG * 8:
                sZ[3 * NG * 8 + slrow] = cutlass.Float32(slsa) * 128.0 + cutlass.Float32(slsb) * 0.5
        cute.arch.sync_threads()

    # The rows lane (w, gid, tq) owns are `ng * 8 + 2 tq (+ 1)`, the same in
    # every warp, so lanes 0-3 of warp 0 write each (row block, tq)'s record.
    # The loop's currency is `s - Z`: `-Z` lands the logit in one FFMA and every
    # gate is a compare against a literal; the cut is a depth in it.
    if warp == 0 and gid == 0:
        for ng in cutlass.range_constexpr(NG):
            for j in cutlass.range_constexpr(2):
                gq = ng * 8 + 2 * tq + j
                z0 = cutlass.Float32(0.0)
                # a row outside the group clears no gate
                cd0 = cutlass.Float32(1e30)
                e0 = cutlass.Float32(0.0)
                sl0 = cutlass.Float32(0.0)
                ing = gq < G
                if cutlass.const_expr(SHARED):
                    ing = ing and (sRi[gq] & 1) != 0
                if ing:
                    z0 = sZ[gq]
                    e0 = sZ[2 * NG * 8 + gq]
                    cd0 = sZ[NG * 8 + gq] - z0
                    if cutlass.const_expr(SOUND):
                        sl0 = e0 * sZ[3 * NG * 8 + gq]
                rec = (ng * 4 + tq) * 8
                sRow[rec + j] = cd0
                sRow[rec + 2 + j] = e0
                sRow[rec + 4 + j] = -z0
                sRow[rec + 6 + j] = sl0

    if cutlass.const_expr(V_REGS):
        # Lane (w, gid, tq) reads channels `CPT p ..` (p = 8w + gid) of keys
        # 2tq + {0, 1, 8, 9} of every sixteen. All of those rows have
        # (r >> 1) & 3 == tq, so the staging XOR is a per-lane constant.
        vby = CPT * (8 * warp + gid)
        vcol = (((vby >> 4) ^ ((tq & (NUV // 2 - 1)) << 1)) << 4) + (vby & 15)
        vsa = sVq.iterator.toint() + vcol + 2 * tq * D
        vsb = sVr.iterator.toint() + vcol + 2 * tq * D
    if cutlass.const_expr(V8 and not V_REGS):
        # the widening pass: lane `l` takes unit `l % (D / 16)` of its rows
        NCH_ = D // 16
        vqb = sVq.iterator.toint() + (lane % NCH_) * 16
        vrb = sVr.iterator.toint() + (lane % NCH_) * 16
        vvb = sV.iterator.toint() + ((lane % NCH_) >> 2) * (BN * 128)
        vu2 = ((lane % NCH_) & 3) * 2
    # this warp's first key in a tile
    kw0 = warp * BNW
    # A warp's weights for a pair of row blocks are four 8x8 matrices, (row
    # block, key half) in that order, stored by one `stmatrix`: lane `8m + j`
    # names row `j` of matrix `m`, and the next pair is 16 rows further on.
    lmx = lane >> 3
    pbs = sPb.iterator + pidx((lmx >> 1) * 8 + (lane & 7), kw0 + 8 * (lmx & 1), BN)
    if cutlass.const_expr(W2):
        prs = pbs + NG * 8 * BN
    cute.arch.sync_threads()
    # D = 64 is latency-bound, and shared memory rather than registers sets
    # most of its residency, so it holds the rows' records across the loop:
    # a reload puts a shared round trip on every tile's verdict, which D = 128
    # hides under its plane A
    HOLD = D < 128
    # One query row fills one lane of each quad: the logit's C puts key gid on
    # lane (gid, 0) as pair 0 and key gid + 8 on the same lane as pair 2. The pad
    # rows are row 0's copies, so lane (gid, 1) holds the same logits and takes
    # pair 2: each lane runs one pair's verdict and weight rather than two, and
    # every lane reads row 0's record.
    ONE = G == 1 and not (V8 or DRAFT or SHARED)
    rtq = 0 if ONE else tq
    rhold = [row_ref(sRow, ng, rtq, 8) for ng in range(NG)] if HOLD else None

    rPg = cute.make_rmem_tensor((1,), cutlass.Int32)
    rPg[0] = 0
    if cutlass.const_expr(PFP):
        if tidx == 0 and lo + TSTEP < hi:
            rPg[0] = mPgT[(breq, (lo + TSTEP) // PAGE)]
    for t in cutlass.range(0, n_tiles, 1, unroll=1):
        base = lo + t * TSTEP
        nrow = hi - base
        if nrow > BN:
            nrow = BN
        nb = lo + (t + 1) * TSTEP
        nn = hi - nb
        if nn > BN:
            nn = BN
        if nn < 0:
            nn = 0
        # The draft mask as one bit per (key, row) pair this lane owns. A draft
        # is at most 32 keys and a tile is 64, so at most one tile of one split
        # builds it; the verdict then tests a bit at a constant shift.
        pmsk = cutlass.Int32(0)
        if cutlass.const_expr(DRAFT):
            if base + nrow > tbase:
                for ngm in cutlass.range_constexpr(NG):
                    for im in cutlass.range_constexpr(4):
                        kym = kw0 + gid + (8 if im >= 2 else 0)
                        gqm = ngm * 8 + 2 * tq + (im % 2)
                        jd = base + kym - tbase
                        hid = cutlass.Int32(0)
                        if jd >= 0 and gqm < G:
                            hid = 1 - ((sTm[gqm] >> jd) & 1)
                        pmsk = pmsk | (hid << (ngm * 4 + im))
        # The logit's first k-block overwrites c1, so its zeroing is dead code.
        # c3's is not: a tile without a refine matmul still reads it, under a
        # select that discards it.
        c1.fill(0)
        c3.fill(0)
        # Nothing before the verdict's barrier writes a buffer the previous
        # tile still reads: that tile's last readers were behind its own
        # value-matmul barrier. So the full barrier alone admits the tile.
        cute.arch.mbarrier_wait(fullb, t % 2)
        ksb = (t % 2) * BN + kw0 + gid
        ksr = [key_scale(sKs, ksb + 8 * hk) for hk in range(2)]

        sgb = t % 2
        if cutlass.const_expr(PAGED and NSEG == 1):
            sgb = sSeg[(t % 2, 0)]
        warpgroup.fence()
        cute.gemm(tmmq, c1, fKa, fQab, c1)

        warpgroup.commit_group()
        warpgroup.wait_group(0)
        if cutlass.const_expr(EI):
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
                    BN,
                    TSTEP,
                    PFP,
                )
        if cutlass.const_expr(TAIL_RANK):
            # y as the warp matmul's A: C block (key half h, ranks 8 jb ..)
            # holds a key per lane group, and the transpose puts a rank there.
            # A's words are (rank half, key half) with the rank half minor.
            for jt in cutlass.range_constexpr(NJT):
                for h in cutlass.range_constexpr(2):
                    for jj in cutlass.range_constexpr(2):
                        jb = 2 * jt + jj
                        for e in cutlass.range_constexpr(2):
                            if cutlass.const_expr(8 * jb < TAIL_RANK):
                                fYf[4 * h + 2 * jj + e] = (
                                    cutlass.Float32(c1[(2 * NG + jb) * 4 + 2 * h + e]) * ksr[h]
                                )
                            else:
                                fYf[4 * h + 2 * jj + e] = cutlass.Float32(0.0)
                fY[jt].store(fYf.load().to(cutlass.BFloat16))
                for x in cutlass.range_constexpr(4):
                    fY4[jt][x] = movmatrix_t(fY4[jt][x])

        # The four lanes sharing a gid hold every row of one key, so the
        # group's verdict is a butterfly over them. All three verdicts of both
        # key halves are bits of one word (live 1, refine K 2, refine V 4, at
        # shift 0 for key gid and 3 for gid + 8), so it is one butterfly.
        pk = cutlass.Int32(0)
        if cutlass.const_expr(not TRUNCATE):
            pk = pk | (1 + (1 << 3))
        # the group's largest relative logit, per key half
        dg0 = cutlass.Float32(NEG)
        dg1 = cutlass.Float32(NEG)
        rref = rhold if HOLD else [row_ref(sRow, ng, rtq, 8) for ng in range(NG)]
        if cutlass.const_expr(ONE):
            for i in cutlass.range_constexpr(4):
                rS[(0, i)] = cutlass.Float32(NEG)
            # this lane's key: gid on tq 0, gid + 8 on tq 1; pair 0 holds it
            kyp = kw0 + gid + 8 * tq
            cip = cutlass.Int32(
                cutlass.select_(tq == 0, c1[0] * 256 + c1[NG * 4], c1[2] * 256 + c1[NG * 4 + 2])
            )
            ksp = cutlass.Float32(cutlass.select_(tq == 0, ksr[0], ksr[1]))
            if tq < 2 and kyp < nrow:
                s1 = cutlass.Float32(cip) * logit_scale(rref[0][2], ksp) + rref[0][4]
                rS[(0, 0)] = s1
                if cutlass.const_expr(TRUNCATE):
                    s1c = s1
                    if cutlass.const_expr(cfg.sound):
                        s1c = s1 + rref[0][6] * ksp
                    if s1c >= rref[0][0]:
                        pk = pk | (1 << (3 * tq))
                dg0 = s1
            # the gate at the key's own half's shift
            if dg0 >= rk:
                pk = pk | (2 << (3 * tq))
        else:
            for i in cutlass.range_constexpr(4):
                ky = kw0 + gid + (8 if i >= 2 else 0)
                kok = ky < nrow
                sh = 0 if i < 2 else 3
                for ng in cutlass.range_constexpr(NG):
                    gq = ng * 8 + 2 * tq + (i % 2)
                    # written on every tile: a dynamic branch around a register
                    # array makes it addressable, which costs registers or spills
                    rS[(ng, i)] = cutlass.Float32(NEG)
                    if cutlass.const_expr(G == 1 and i % 2 == 1):
                        # row 1 of a one-row group does not exist
                        continue
                    # a masked draft pair keeps the sentinel, which every consumer
                    # below treats as a hard zero
                    tok = cutlass.Int32(0)
                    if cutlass.const_expr(DRAFT):
                        tok = (pmsk >> (ng * 4 + i)) & 1
                    if gq < G and kok and tok == 0:
                        tk = logit_scale(rref[ng][2 + i % 2], ksr[i // 2])
                        s1 = coarse_logit(c1, ng, i, NG) * tk + rref[ng][4 + i % 2]
                        rS[(ng, i)] = s1
                        if cutlass.const_expr(TRUNCATE):
                            s1c = s1
                            if cutlass.const_expr(cfg.sound):
                                # plane A's omission, bounded per key by its scale
                                s1c = s1 + rref[ng][6 + i % 2] * ksr[i // 2]
                            if s1c >= rref[ng][i % 2]:
                                pk = pk | (1 << sh)
                        # A refine gate ORed over the group is a max over it, and
                        # its threshold is a literal shared by every row, so the
                        # compare happens once per key.
                        if cutlass.const_expr(i < 2):
                            dg0 = cute.arch.fmax(dg0, s1)
                        else:
                            dg1 = cute.arch.fmax(dg1, s1)
        for hf in cutlass.range_constexpr(0 if ONE else 2):
            dgh = dg0 if hf == 0 else dg1
            shf = 0 if hf == 0 else 3
            # a claim about precision, not liveness, so it holds with the
            # truncation off
            if dgh >= rk:
                pk = pk | (2 << shf)
            if cutlass.const_expr(V8):
                if dgh >= rv:
                    pk = pk | (4 << shf)
        for st in cutlass.range_constexpr(2):
            pk = pk | cute.arch.shuffle_sync_bfly(pk, 1 << st)
        b0 = pk & 7
        b1 = (pk >> 3) & 7
        # a key no row keeps is neither gathered nor refined: its mass is its
        # coarse logit's
        b0 = cutlass.Int32(cutlass.select_((b0 & 1) != 0, b0, 0))
        b1 = cutlass.Int32(cutlass.select_((b1 & 1) != 0, b1, 0))
        rL[0] = b0 & 1
        rL[1] = b1 & 1
        rR[0] = b0 & 2
        rR[1] = b1 & 2
        if tq == 0:
            if kw0 + gid < nrow:
                sLive[kw0 + gid] = b0
            if kw0 + gid + 8 < nrow:
                sLive[kw0 + gid + 8] = b1
        cute.arch.sync_threads()
        # Past the verdict's barrier every warp's share of the logit has read
        # sKa and every thread has passed its wait on this tile's phase, so
        # the next tile's plane A goes out now, under the gathers, without
        # racing either.
        if cutlass.const_expr(not EI):
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
                    BN,
                    TSTEP,
                    PFP,
                )
        # every consumer wants a verdict as a predicate, so lift them out of
        # shared once per tile as ballots
        for j in cutlass.range_constexpr(NWD):
            kk = j * 32 + lane
            lvb = cutlass.Int32(0)
            rfb = cutlass.Int32(0)
            rvb = cutlass.Int32(0)
            if kk < nrow:
                lvb = sLive[kk]
                rfb = lvb & 2
                rvb = lvb & 4
                lvb = lvb & 1
            mLv[j] = cute.arch.vote_ballot_sync(lvb != 0)
            mRf[j] = cute.arch.vote_ballot_sync(rfb != 0)
            if cutlass.const_expr(V8):
                mRv[j] = cute.arch.vote_ballot_sync(rvb != 0)
            # every thread holds the same ballots, so the counts need no
            # reduction at the end
            nlive += cute.arch.popc(mLv[j])
            nref += cute.arch.popc(mRf[j])
        if cutlass.const_expr(TAIL):
            # A warp's dropped keys re-enter as one virtual key: the first of
            # its sixteen that no row kept takes the tile's block row of V as
            # its V and the warp's dropped mass per row as its weight. Under
            # `KEEP_ALL` a key is kept for every row or for none, so that slot's
            # V is free. A key past the tile's end carries no mass.
            for j in cutlass.range_constexpr(NWD):
                nv = nrow - 32 * j
                vm = cutlass.Int32(
                    cutlass.select_(
                        nv >= 32,
                        cutlass.Int32(-1),
                        cutlass.select_(nv > 0, (cutlass.Int32(1) << nv) - 1, cutlass.Int32(0)),
                    )
                )
                mDd[j] = ~mLv[j] & vm
            dw = mask_word(mDd, kw0, NWD) & 0xFFFF
            slb = dw & (0 - dw)
            shas = dw != 0
            sidx = cute.arch.popc(slb - 1)
            skey = kw0 + sidx
            tsl0 = shas and gid == (sidx & 7) and sidx < 8
            tsl1 = shas and gid == (sidx & 7) and sidx >= 8
            vrow = bh * NBK + base // BN
            if cutlass.const_expr(PAGED):
                vrow = sgb // BN

        # A tile no row keeps a key of, with no block row to carry its dropped
        # mass, adds exact zeros to the accumulators and the denominator, so
        # the whole CTA (the ballots are the tile's) skips its V side. A deep
        # cut leaves most tiles so.
        live_tile = True
        if cutlass.const_expr(not TAIL):
            live_tile = mask_any(mLv, NWD)
        if live_tile:
            # Plane B is the refine matmul's A, and a warp's share of a `wgmma`
            # reads only its own sixteen rows of A, so each warp gathers its own
            # keys and a warp barrier publishes them.
            if mask_any(mRf, NWD):
                gather_warp_rows(
                    lane,
                    kw0,
                    base,
                    sKb,
                    gKb,
                    mRf,
                    BNW,
                    BN,
                    D,
                    1,
                    NWD,
                    CM,
                    0,
                    sSeg,
                    sgb,
                    SEGW,
                    NSEG,
                    PAGED,
                )
            cute.arch.cp_async_commit_group()
            gather_rows(
                tidx, base, sVq, gV, mLv, BN, D, NT, EBV, NWD, CM, VSW, sSeg, sgb, SEGW, NSEG, PAGED
            )
            if cutlass.const_expr(TAIL):
                # the virtual key's V row, by its own warp, in V's group
                if shas:
                    if lane < D // 8:
                        primitives.cp_async_shared_global(
                            sV.iterator + gather_dst(skey, lane, D, BN, 2, 1),
                            mVbk.iterator + (vrow * D + lane * 8),
                            16,
                            CM,
                        )
            cute.arch.cp_async_commit_group()
            # a warpgroup-uniform branch, which `wgmma` requires anyway
            if mask_any(mRf, NWD):
                cute.arch.cp_async_wait_group(1)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                warpgroup.fence()
                cute.gemm(tmma, c3, fKb, fQ2, c3)
                warpgroup.commit_group()
                warpgroup.wait_group(0)

            if cutlass.const_expr(V8):
                # V's second plane goes into the warp's own plane-B rows once its
                # share of the refine has read them, in a cp.async group of its
                # own, which the value matmul's wait covers
                if mask_any(mRv, NWD):
                    gather_warp_rows(
                        lane,
                        kw0,
                        base,
                        sVr,
                        gVb,
                        mRv,
                        BNW,
                        BN,
                        D,
                        1,
                        NWD,
                        CM,
                        VSW,
                        sSeg,
                        sgb,
                        SEGW,
                        NSEG,
                        PAGED,
                    )
                cute.arch.cp_async_commit_group()
            # The weight pass, one row block at a time so a pair of blocks is one
            # `stmatrix`. A pair that does not exist, including every row past G,
            # gets weight exactly zero, so the pad rows of the last block are
            # rewritten with zeros on every tile.
            # a row past the tile's end is never refined, so the select below
            # discards whatever its slot holds
            ksw = [cutlass.Float32(sKs[ksb + 8 * hk]) * (1.0 / KBR) for hk in range(2)]
            pws = []
            prl = []
            for ng in cutlass.range_constexpr(NG):
                wref = rhold[ng] if HOLD else row_ref(sRow, ng, rtq, 4)
                wv = []
                if cutlass.const_expr(ONE):
                    # this lane's pair, then key gid + 8's from lane (gid, 1) into
                    # pair 2 of lane (gid, 0), whose fragment holds row 0
                    kyw = kw0 + gid + 8 * tq
                    lvp = (
                        tq < 2
                        and kyw < nrow
                        and cutlass.Int32(cutlass.select_(tq == 0, rL[0], rL[1])) != 0
                    )
                    rfp = cutlass.Int32(cutlass.select_(tq == 0, rR[0], rR[1])) != 0
                    if cutlass.const_expr(DROPPED and not TAIL):
                        rfp = lvp and rfp
                    c3p = cutlass.Float32(
                        cutlass.select_(
                            tq == 0, refine_logit(c3, 0, 0, NG), refine_logit(c3, 0, 2, NG)
                        )
                    )
                    ksq = cutlass.Float32(cutlass.select_(tq == 0, ksw[0], ksw[1]))
                    sp_ = rS[(0, 0)]
                    sp_ = cutlass.Float32(
                        cutlass.select_(rfp, sp_ + c3p * logit_scale(wref[2], ksq), sp_)
                    )
                    fdp = cutlass.Float32(0.0)
                    if cutlass.const_expr(not DROPPED):
                        kpp = lvp
                        if cutlass.const_expr(not KEEP_ALL):
                            kpp = lvp and sp_ >= wref[0]
                        wp = ex2_if(cutlass.Int32(kpp), sp_)
                    else:
                        ep = ex2(sp_)
                        kpp = lvp
                        if cutlass.const_expr(not KEEP_ALL):
                            kpp = sp_ >= wref[0]
                        wp = cutlass.Float32(cutlass.select_(kpp, ep, 0.0))
                        fdp = cutlass.Float32(cutlass.select_(kpp, 0.0, ep))
                    src1 = (opaque_i32(lane) & ~3) | 1
                    wp8 = cute.arch.shuffle_sync(wp, src1)
                    wv = [
                        cutlass.Float32(cutlass.select_(tq == 0, wp, 0.0)),
                        cutlass.Float32(0.0),
                        cutlass.Float32(cutlass.select_(tq == 0, wp8, 0.0)),
                        cutlass.Float32(0.0),
                    ]
                    if cutlass.const_expr(DROPPED):
                        fdp8 = cute.arch.shuffle_sync(fdp, src1)
                        fd0 = cutlass.Float32(cutlass.select_(tq == 0, fdp, 0.0))
                        fd2 = cutlass.Float32(cutlass.select_(tq == 0, fdp8, 0.0))
                        if cutlass.const_expr(TAIL):
                            fDv[0] = fd0
                            fDv[1] = cutlass.Float32(0.0)
                            fDv[2] = fd2
                            fDv[3] = cutlass.Float32(0.0)
                        else:
                            rM[(0, 0)] = rM[(0, 0)] + fd0
                            rM[(0, 0)] = rM[(0, 0)] + fd2
                else:
                    for i in cutlass.range_constexpr(4):
                        if cutlass.const_expr(G == 1 and i % 2 == 1):
                            wv.append(cutlass.Float32(0.0))
                            if cutlass.const_expr(TAIL):
                                fDv[i] = cutlass.Float32(0.0)
                            continue
                        ky = kw0 + gid + (8 if i >= 2 else 0)
                        gq = ng * 8 + 2 * tq + (i % 2)
                        # Selects rather than branches: a branch per pair is a divergent
                        # region per pair, where the predicated form is the same
                        # instructions without the reconvergence.
                        lv = gq < G and ky < nrow and rL[i // 2] != 0
                        rfk = rR[i // 2] != 0
                        s = rS[(ng, i)]
                        if cutlass.const_expr(DROPPED and not TAIL):
                            rfk = lv and rfk
                        s = cutlass.Float32(
                            cutlass.select_(
                                rfk,
                                s
                                + refine_logit(c3, ng, i, NG)
                                * logit_scale(wref[2 + i % 2], ksw[i // 2]),
                                s,
                            )
                        )
                        if cutlass.const_expr(not DROPPED):
                            keep = lv
                            if cutlass.const_expr(not KEEP_ALL):
                                keep = lv and s >= wref[i % 2]
                            w = ex2_if(cutlass.Int32(keep), s)
                        else:
                            # The correction needs every pair's weight, so the cut
                            # only decides which accumulator it lands in; a pair that
                            # does not exist carries the sentinel, whose weight is
                            # zero. Adding +0 to a sum that starts at +0 is exact, so
                            # the other accumulator's select costs no bit.
                            e = ex2(s)
                            if cutlass.const_expr(KEEP_ALL):
                                # the gather is the group's, so a key this row's cut
                                # declines is already here for the rows that kept it:
                                # its exact weight costs no byte
                                keep = lv
                            else:
                                # `s` is finite, so `ge` and `lt` are exact complements,
                                # which ptxas cannot assume under NaN. `lv` is not
                                # needed: `rL` is the group's OR of this same compare.
                                keep = s >= wref[i % 2]
                            w = cutlass.Float32(cutlass.select_(keep, e, 0.0))
                            if cutlass.const_expr(TAIL):
                                fDv[i] = cutlass.Float32(cutlass.select_(keep, 0.0, e))
                            else:
                                rM[(ng, i % 2)] = rM[(ng, i % 2)] + cutlass.Float32(
                                    cutlass.select_(keep, 0.0, e)
                                )
                        wv.append(w)
                if cutlass.const_expr(TAIL):
                    # The dropped weights as B: the pair (key half, rows 2 tq ..)
                    # transposes into (row, keys 2 tq ..). Ones as A sum them per
                    # row, which the virtual key then carries; its rounded value
                    # is what the denominator and the value matmul both see.
                    fBd.store(fDv.load().to(cutlass.BFloat16))
                    for x in cutlass.range_constexpr(2):
                        fBd4[x] = movmatrix_t(fBd4[x])
                    fMm.fill(0.0)
                    cute.gemm(wmma, fMm, fOne, fBd, fMm)
                    if cutlass.const_expr(TAIL_RANK):
                        for jt in cutlass.range_constexpr(NJT):
                            cute.gemm(wmma, rYd[ng][jt], fY[jt], fBd, rYd[ng][jt])
                    for x in cutlass.range_constexpr(4):
                        tsx = tsl0 if x < 2 else tsl1
                        wv[x] = cutlass.Float32(cutlass.select_(tsx, fMm[x % 2], wv[x]))
                # the denominator sums the rounded weight the matmul will see
                for hk in cutlass.range_constexpr(2):
                    w0 = wv[2 * hk]
                    w1 = wv[2 * hk + 1]
                    if cutlass.const_expr(V8):
                        # an 8-bit V's operand is V * 2^-120, of which the
                        # weight carries `E4W` back, exactly
                        w0 = w0 * E4W
                        w1 = w1 * E4W
                    pw, f0, f1 = pack_weights(w0, w1)
                    if cutlass.const_expr(W2):
                        # The difference is exact in f32, and its own rounding
                        # is 2^-8 of it, so the two terms carry the weight to
                        # 2^-16 of it, unbiased: the denominator takes the f32
                        # weight, which leaves the second pack's unpacking dead.
                        pr, _, _ = pack_weights(w0 - f0, w1 - f1)
                        f0 = wv[2 * hk]
                        f1 = wv[2 * hk + 1]
                        prl.append(pr)
                    elif cutlass.const_expr(V8):
                        f0 = f0 * (1.0 / E4W)
                        f1 = f1 * (1.0 / E4W)
                    rN[(ng, 0)] = rN[(ng, 0)] + f0
                    rN[(ng, 1)] = rN[(ng, 1)] + f1
                    pws.append(pw)
                if cutlass.const_expr(ng % 2 == 1 or ng == NG - 1):
                    stmatrix(pbs + (ng // 2) * 16 * BN, pws, MMALayout.COL)
                    pws = []
                    if cutlass.const_expr(W2):
                        stmatrix(prs + (ng // 2) * 16 * BN, prl, MMALayout.COL)
                        prl = []
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()
            if cutlass.const_expr(V8 and not V_REGS):
                # e4m3 out of the staging tile, f16 into the descriptor's tile,
                # once per live row, by the warp whose share of the value matmul
                # reads those channels: a warp barrier publishes them
                NCH = D // 16
                NRW = 32 // NCH
                NITW = (BNW + NRW - 1) // NRW
                rw0 = lane // NCH
                wlv = mask_word(mLv, kw0, NWD) >> rw0
                wsv = mask_word(mRv, kw0, NWD) >> rw0
                for it in cutlass.range_constexpr(NITW):
                    rr = kw0 + it * NRW + rw0
                    do = vvb + (rr << 7) + ((vu2 ^ (rr & 7)) << 4)
                    if (wlv >> (it * NRW)) & 1 != 0:
                        if (wsv >> (it * NRW)) & 1 != 0:
                            widen16_refine_bf16(vqb + rr * D, do, vrb + rr * D)
                        else:
                            widen16_bf16(vqb + rr * D, do)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_threads()
            if cutlass.const_expr(V_REGS):
                # Every row is converted: a declined row's stale bytes are finite
                # and meet a zero weight. Skipping dead slices would keep the whole
                # fragment live across the loop, 32 registers under the six-CTA cap.
                mRc = [mRv[j] for j in range(NWD)]
                mro = mRc[0]
                for j in cutlass.range_constexpr(1, NWD):
                    mro = mro | mRc[j]
                rfv = mro != 0
                # this lane's first key of every eight is `2 tq`
                mRq = [mRc[j] >> (2 * tq) for j in range(NWD)]
                for kk in cutlass.range_constexpr(BN // 16):
                    for h in cutlass.range_constexpr(2):
                        value_fragment(
                            fV4,
                            vsa,
                            vsb,
                            mRq,
                            kk,
                            h,
                            rfv,
                            (16 * kk + 8 * h) * D,
                            (16 * kk + 8 * h) * D,
                            NVB,
                            D,
                            D,
                        )
            # A^T += V^T P^T. K is the whole tile, so no warp holds a partial sum.
            # The group is retired by the next tile's logit `wait_group`, which
            # comes before the verdict's barrier, the first point after which
            # anything overwrites sV or the weight buffer.
            #
            # Under `PIN` the operands are built from a base the compiler can
            # neither fold nor hoist. Built before the loop, their descriptors
            # are held across it, in GPRs once the uniform file is full: up to
            # 50 registers at G = 32, and a CTA per SM on a sixth of the
            # two-term builds. Where the file holds them, rebuilding them
            # every tile costs 1-1.5% at D = 64.
            pbo = sPb.iterator.toint()
            if cutlass.const_expr(PIN):
                pbo = opaque_i32(pbo)
            fP = vgs.make_fragment_B(
                vgs.partition_B(
                    smem_view(pbo, PT, swv, cute.make_layout((NG * 8, BN), stride=(BN, 1)))
                )
            )
            if cutlass.const_expr(W2):
                fPr = vgs.make_fragment_B(
                    vgs.partition_B(
                        smem_view(
                            pbo + NG * 8 * BN * 2,
                            PT,
                            swv,
                            cute.make_layout((NG * 8, BN), stride=(BN, 1)),
                        )
                    )
                )
            if cutlass.const_expr(V_REGS):
                fVo = fV
            else:
                vbo = sV.iterator.toint()
                if cutlass.const_expr(PIN):
                    vbo = opaque_i32(vbo)
                fVo = [
                    vgs.make_fragment_A(
                        vgs.partition_A(
                            smem_view(
                                vbo + j * BN * 64 * 2,
                                PT,
                                swv,
                                cute.make_layout((64, BN), stride=(1, 64)),
                            )
                        )
                    )
                    for j in range(NVB)
                ]
            warpgroup.fence()
            for j in cutlass.range_constexpr(NVB):
                if cutlass.const_expr(W2):
                    # A k-block's two terms back to back on one V operand: a
                    # second pass over the tile kept all of V's descriptors
                    # live across it.
                    for kb in cutlass.range_constexpr(BN // 16):
                        cute.gemm(tmv, rD[j], fVo[j][(None, None, kb)], fP[(None, None, kb)], rD[j])
                        cute.gemm(
                            tmv, rD[j], fVo[j][(None, None, kb)], fPr[(None, None, kb)], rD[j]
                        )
                else:
                    cute.gemm(tmv, rD[j], fVo[j], fP, rD[j])
            warpgroup.commit_group()

        else:
            if cutlass.const_expr(DROPPED):
                # no row keeps a key of this tile, so every pair's weight is
                # dropped mass, in the order the weight pass adds it
                if cutlass.const_expr(ONE):
                    edp = ex2(rS[(0, 0)])
                    edp8 = cute.arch.shuffle_sync(edp, (opaque_i32(lane) & ~3) | 1)
                    rM[(0, 0)] = rM[(0, 0)] + cutlass.Float32(cutlass.select_(tq == 0, edp, 0.0))
                    rM[(0, 0)] = rM[(0, 0)] + cutlass.Float32(cutlass.select_(tq == 0, edp8, 0.0))
                else:
                    for ng in cutlass.range_constexpr(NG):
                        for i in cutlass.range_constexpr(4):
                            rM[(ng, i % 2)] = rM[(ng, i % 2)] + ex2(rS[(ng, i)])

    pid = cutlass.Float32(sRed[(2 * NW, 0)]).bitcast(cutlass.Int32)
    bh = cutlass.Float32(sRed[(2 * NW, 1)]).bitcast(cutlass.Int32)
    sp = pid % SPLIT
    # the last tile's group has nobody to retire it
    if cutlass.const_expr(cfg.front and not TAIL and cfg.front_qk != 1):
        front_sums(mRdy, bh, G, cfg.front_qk)
    warpgroup.wait_group(0)
    RKC = cfg.rank_combine and TAIL_RANK > 0
    if cutlass.const_expr(RKC):
        # Each warp's rank sums into slots of its own in plane A's tile, which
        # every warp's last logit read before the last verdict's barrier. The
        # barrier after the row sums below publishes them, and the combine
        # expands the splits' sums once per row, so no CTA loads the basis.
        NYD = NG * 8 * TAIL_RANK
        if cutlass.const_expr(NW * NYD * 4 > BN * D):
            raise ValueError(f"the tail's {NW * NYD} sums do not fit plane A's tile")
        sYd = cute.make_tensor(
            cute.recast_ptr(sKa.iterator, None, cutlass.Float32), cute.make_layout(NW * NYD)
        )
        for ng in cutlass.range_constexpr(NG):
            for jt in cutlass.range_constexpr(NJT):
                for x in cutlass.range_constexpr(4):
                    if cutlass.const_expr(16 * jt + 8 * (x // 2) < TAIL_RANK):
                        sYd[
                            warp * NYD
                            + (ng * 8 + 2 * tq + (x % 2)) * TAIL_RANK
                            + 16 * jt
                            + 8 * (x // 2)
                            + gid
                        ] = rYd[ng][jt][x]
    elif cutlass.const_expr(TAIL_RANK):
        # The warps' sums over their own keys meet in plane A's tile, dead
        # once the last logit has read it, added in warp order so the bits do
        # not depend on which warp finished first. The barrier after the row
        # sums below publishes the last warp's, and the basis staged beside
        # them.
        NYD = NG * 8 * TAIL_RANK
        if cutlass.const_expr(NYD * 4 > BN * D):
            raise ValueError(f"the tail's {NYD} sums do not fit plane A's tile")
        sYd = cute.make_tensor(
            cute.recast_ptr(sKa.iterator, None, cutlass.Float32), cute.make_layout(NYD)
        )
        for ww in cutlass.range_constexpr(NW):
            cute.arch.sync_threads()
            if cutlass.const_expr(ww == 0):
                # the basis goes out under the sums
                load_basis(sV, mVr, bh, tidx, D, TAIL_RANK, NT)
            if warp == ww:
                for ng in cutlass.range_constexpr(NG):
                    for jt in cutlass.range_constexpr(NJT):
                        for x in cutlass.range_constexpr(4):
                            if cutlass.const_expr(16 * jt + 8 * (x // 2) < TAIL_RANK):
                                ix = (
                                    (ng * 8 + 2 * tq + (x % 2)) * TAIL_RANK
                                    + 16 * jt
                                    + 8 * (x // 2)
                                    + gid
                                )
                                if cutlass.const_expr(ww == 0):
                                    sYd[ix] = rYd[ng][jt][x]
                                else:
                                    sYd[ix] = sYd[ix] + rYd[ng][jt][x]
        cute.arch.cp_async_wait_group(0)
    pix = partial_slot(pid, bh, sp, SPLIT, SLOTS, SLOT0, ORDERED, DIRECT)
    # a shared level counts its bytes into its own buffer at its own index
    pcn = pix
    if cutlass.const_expr(SHARED):
        pcn = pid
    # The eight lanes of a `tq` group own the same two rows, so one butterfly
    # over the gid bits closes the warp, and shared closes the four warps.
    for ng in cutlass.range_constexpr(NG):
        for j in cutlass.range_constexpr(2):
            m = rM[(ng, j)]
            n = rN[(ng, j)]
            for st in cutlass.range_constexpr(3):
                m = m + cute.arch.shuffle_sync_bfly(m, 4 << st)
                n = n + cute.arch.shuffle_sync_bfly(n, 4 << st)
            if gid == 0:
                gq = ng * 8 + 2 * tq + j
                if gq < G:
                    sRed[(warp, gq)] = m
                    sRed[(NW + warp, gq)] = n
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
        if cutlass.const_expr(SHARED):
            wl = sRi[tidx]
            if (wl & 1) != 0:
                mL[((wl >> 7), (wl >> 1) & 63)] = n + m
        else:
            mL[(pix, tidx)] = n + m
    if cutlass.const_expr(RKC):
        store_rank_sums(mVr, sYd, pix, tidx, G * TAIL_RANK, NYD, NW, NT)
    if cutlass.const_expr(DROPPED or DIRECT):
        # the row's dropped mass and denominator, for every lane of the row
        cute.arch.sync_threads()
    # The output straight from the accumulator. Under `V_REGS` a lane's channels
    # of a row are four (two at D = 64) consecutive floats, one vector store;
    # otherwise eight lanes cover 32 contiguous bytes of a row, a whole sector.
    for ng in cutlass.range_constexpr(NG):
        for r in cutlass.range_constexpr(2):
            gq = ng * 8 + 2 * tq + r
            if gq < G:
                oslot = pix
                orow = gq
                ook = cutlass.Boolean(True)
                if cutlass.const_expr(SHARED):
                    wo = sRi[gq]
                    oslot, orow = cascade_row(wo)
                    ook = (wo & 1) != 0
                if ook:
                    xs = []
                    ccs = []
                    for j in cutlass.range_constexpr(NVB):
                        for hb in cutlass.range_constexpr(2):
                            if cutlass.const_expr(V_REGS):
                                ccs.append(CPT * (8 * warp + gid) + hb + 2 * j)
                            else:
                                ccs.append(64 * j + 16 * warp + gid + 8 * hb)
                            xs.append(rD[j][ng * 4 + 2 * hb + r])
                    if cutlass.const_expr(TAIL_RANK and not RKC):
                        add_rank_term(xs, ccs, sYd, sV, gq, TAIL_RANK)
                    for c in cutlass.range_constexpr(len(xs)):
                        x = xs[c]
                        if cutlass.const_expr(V8):
                            x = x * E4O
                        if cutlass.const_expr(DROPPED and not TAIL):
                            if cutlass.const_expr(ROW_VMEAN):
                                x = x + sRed[(2 * NW, gq)] * mVm[(bh, gq, ccs[c])]
                            else:
                                x = x + sRed[(2 * NW, gq)] * mVm[(bh, ccs[c])]
                        if cutlass.const_expr(DIRECT):
                            x = x * EVS / sZ[gq]
                        xs[c] = x
                    ob = (oslot * UNIQUE_G + orow) * D
                    if cutlass.const_expr(V_REGS):
                        st_global_f32(mO.iterator + (ob + ccs[0]), xs)
                    else:
                        for c in cutlass.range_constexpr(len(xs)):
                            st_global_f32(mO.iterator + (ob + ccs[c]), [xs[c]])
    if tidx == 0:
        mCnt[(pcn, 0)] = nlive
        mCnt[(pcn, 1)] = nref
        if cutlass.const_expr(cfg.front and cfg.front_clear and DIRECT):
            # the step's last reader of the words clears them for the next front
            for k in cutlass.range_constexpr(1 + 2 * G):
                mRdy[bh * (1 + 2 * G) + k] = 0
    if cutlass.const_expr(cfg.front):
        # Every front CTA has released its last write by now; the wait is
        # what hands the next kernel, which waits on this grid alone, the
        # front's completion as well. It returns at once.
        cute.arch.griddepcontrol_wait()


@cute.jit
def launch_decode(
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
    # under a prepass the decode is its programmatic dependent, so its
    # prologue overlaps the prepass
    decode_kernel(
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
