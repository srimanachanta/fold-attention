"""A decode step's front: Q's planes, the appended token and each row's mass
reference, in one launch, which the decode spins on word by word rather
than waiting on its grid."""

from __future__ import annotations

import math

import cutlass
import torch
from cutlass import Float32, Int32, Int64, Uint32, cute
from cutlass.cute.runtime import from_dlpack

from ..utils import Launch, compile_cached, current_stream, full_carveout
from .cache import swizzle_of
from .config import MASS_STRATA, MASS_TRIM, mass_tile
from .device import paged_row
from .ptx import (
    ld_global_b32,
    ld_global_nc,
    ld_global_nc_b32,
    ld_scale_cg,
    st_global_b16,
    st_global_b32,
    st_global_v4,
    st_shared,
)
from .quant import (
    int8_planes,
    key_scale,
    pack_words,
    quantize_q,
    rotate,
    store_plane_global,
    store_plane_shared,
    unpack,
)
from .reference import NT, mass_estimate, mass_key, mass_smem
from .write import (
    StepConfig,
    StepPointers,
    _pointers,
    _split_bound,
    append_planes,
    append_row,
    bind_pointers,
    move_length,
    tail_pointers,
)


def _q_words(p: StepPointers, bh, tidx, lq, q0, QT, NQP, G, D, TPR):
    """This thread's share of a row group's raw Q rows, one list of eight
    words per pass; a thread outside the share loads row 0 and uses none."""
    tq0 = Int32(cutlass.select_(tidx >= q0, tidx - q0, Int32(0)))
    out = []
    for it in range(NQP):
        g = (it * QT + tq0) // TPR
        gs = Int32(cutlass.select_(g < G, g, Int32(0)))
        out.append(ld_global_nc(p.q + Int64(bh * G + gs) * (D * 2) + lq * 32, 8))
    return out


@cute.jit
def _q_quant(
    wqs,
    p: StepPointers,
    bh,
    G: cutlass.Constexpr,
    D: cutlass.Constexpr,
    F16: cutlass.Constexpr,
    q0: cutlass.Constexpr,
    QT: cutlass.Constexpr,
):
    """A row group's Q planes into the cache's Q from the words `_q_words`
    loaded, by threads `q0` and up, as `write.step_kernel` makes them."""
    tidx, _, _ = cute.arch.thread_idx()
    TPR = D // 16
    lq = tidx % TPR
    if tidx >= q0:
        # every pass's arithmetic before any pass's stores, which branch, so
        # the passes' serial chains are one block ptxas can interleave
        outs = []
        for it in cutlass.range_constexpr(len(wqs)):
            ma, mb, e = quantize_q(wqs[it], lq, TPR, D, F16)
            outs.append((pack_words(ma), pack_words(mb), e))
        for it in cutlass.range_constexpr(len(wqs)):
            wa, wb, e = outs[it]
            g = (it * QT + tidx - q0) // TPR
            qr = Int64(bh * G + Int32(cutlass.select_(g < G, g, Int32(0))))
            if g < G:
                qaddr = qr * D + lq * 16
                st_global_v4(p.qa + qaddr, wa)
                st_global_v4(p.qb + qaddr, wb)
                if lq == 0:
                    st_global_b32(p.eq + qr * 4, e.bitcast(Uint32))


def front_smem(G, D, tail_rank=-1):
    """`mass_front`'s shared memory, allocation by allocation."""
    NPR, _ = mass_tile()
    NG = (G + 7) // 8
    NSL = (MASS_STRATA + 31) // 32
    ALN = swizzle_of(D)[2]
    sizes = [
        (NPR * D, ALN),
        (2 * NG * 8 * D, ALN),
        (16 * NG * 8, 16),
        (16 * NG * 8, 16),
        (4 * ((NG * 8 - 1) * (NSL * 32 + 4) + NSL * 32), 16),
        (4 * NPR, 16),
        (4 * NPR, 16),
        (32, 16),
        (4 * NPR, 16),
        (4 * NG * 8, 16),
    ]
    if tail_rank >= 0:
        sizes += [(4 * (D + 1), 16), (4 * D, 16), (4 * max(tail_rank, 1), 16)]
    off = 0
    for n, a in sizes:
        off = -(-off // a) * a + n
    return off


def _q_elems_per_thread(G, D):
    """Q's elements a thread quantises for the tile: the fewest, down to a
    row a warp wide, that still fit the group in one pass of the CTA, since
    a pass is one warp's serial chain and its length is what Z waits on."""
    e = D // 32
    while e < 16 and G * D // e > 128:
        e *= 2
    return e


def _tile_words(p: StepPointers, bh, tidx, G, D, has_kv):
    """The raw words of the tile's own quantisation: Q at
    `_q_elems_per_thread` elements a thread, and the new key at D / 32,
    which every warp loads."""
    EQ = _q_elems_per_thread(G, D)
    LQ = D // EQ
    RPP = NT // LQ
    NP = (G + RPP - 1) // RPP
    lq = tidx % LQ
    qw = []
    for it in range(NP):
        g = it * RPP + tidx // LQ
        gs = Int32(cutlass.select_(g < G, g, Int32(0)))
        qw.append(ld_global_nc(p.q + Int64(bh * G + gs) * (D * 2) + lq * (EQ * 2), EQ // 2))
    kw = [Uint32(0)]
    if has_kv:
        EK = D // 32
        kw = ld_global_nc(p.k + Int64(bh) * (D * 2) + (tidx % 32) * (EK * 2), EK // 2)
    return qw, kw


@cute.jit
def _tile_own_rows(
    qw,
    kw,
    sK,
    sKs,
    sQ,
    sEq,
    bh,
    G: cutlass.Constexpr,
    D: cutlass.Constexpr,
    F16: cutlass.Constexpr,
    HASKV: cutlass.Constexpr,
    own=None,
):
    """The tile's rows no gather fills, from `_tile_words`: Q's planes, the
    new key's plane A in row 1 and its scale, and zeros in the pad rows of
    Q's last block.
    Every thread quantises; the key's row is one warp's, repeated in each so
    its chain overlaps Q's, and warp 0 stores it.

    `own` makes this CTA the owner of Q (and, if its last field is set, of
    the new key's planes): `(pointers, cache row, page slot, ready word,
    release value, publish K)`. It then also writes them to the cache and
    releases the ready word."""
    tidx, _, _ = cute.arch.thread_idx()
    NG = (G + 7) // 8
    NU = D // 16
    SWU, SWRS, _ = swizzle_of(D)
    SWM = SWU - 1
    EQ = _q_elems_per_thread(G, D)
    LQ = D // EQ
    RPP = NT // LQ
    lq = tidx % LQ
    outs = []
    for it in cutlass.range_constexpr(len(qw)):
        outs.append(quantize_q(qw[it], lq, LQ, D, F16))
    mk = mkb = None
    ek = eb = None
    if cutlass.const_expr(HASKV):
        lk = tidx % 32
        xk = rotate(unpack(kw, F16), lk, 32, 1.0 / math.sqrt(D))
        ek, eb = key_scale(xk, 32)
        mk, mkb, _ = int8_planes(xk, ek, own is not None and own[5])
    sqa = sQ.iterator.toint()
    c0 = lq * EQ
    for it in cutlass.range_constexpr(len(qw)):
        ma, mb, e = outs[it]
        g = it * RPP + tidx // LQ
        if g < G:
            so = g * D + (((c0 >> 4) ^ ((g >> SWRS) & SWM)) << 4) + (c0 & 15)
            store_plane_shared(sqa + so, ma)
            store_plane_shared(sqa + NG * 8 * NU * 16 + so, mb)
            if lq == 0:
                sEq[g] = e
            if cutlass.const_expr(own is not None):
                p = own[0]
                qr = Int64(bh * G + g)
                qaddr = qr * D + c0
                store_plane_global(p.qa + qaddr, ma)
                store_plane_global(p.qb + qaddr, mb)
                if lq == 0:
                    st_global_b32(p.eq + qr * 4, e.bitcast(Uint32))
    if cutlass.const_expr(HASKV):
        if tidx < 32:
            EK = D // 32
            ck = (tidx % 32) * EK
            store_plane_shared(
                sK.iterator.toint() + D + (((ck >> 4) ^ ((1 >> SWRS) & SWM)) << 4) + (ck & 15), mk
            )
            if tidx == 0:
                sKs[1] = ek * (1.0 / 256.0)
            if cutlass.const_expr(own is not None and own[5]):
                p, prow, off = own[0], own[1], own[2]
                ko = prow * D + ((((ck >> 4) ^ ((off >> SWRS) & SWM)) << 4) + (ck & 15))
                store_plane_global(p.ka + ko, mk)
                store_plane_global(p.kb + ko, mkb)
                if tidx == 0:
                    st_global_b16(p.ek + prow * 2, eb)
    if cutlass.const_expr(NG * 8 > G):
        NQU = NG * 8 * NU
        NPU = (NG * 8 - G) * NU
        z4 = [Uint32(0)] * 4
        for it in cutlass.range_constexpr((2 * NPU + NT - 1) // NT):
            u = it * NT + tidx
            if u < 2 * NPU:
                up = u % NPU
                r = G + up // NU
                cu = up % NU
                st_shared(
                    sqa + (u // NPU) * (NQU * 16) + r * D + ((cu ^ ((r >> SWRS) & SWM)) << 4), z4
                )
    if cutlass.const_expr(own is not None):
        cute.arch.sync_threads()
        if tidx == 0:
            cute.arch.red(own[3], Int32(own[4]), op="add", dtype="u32", sem="release", scope="gpu")


@cute.kernel
def mass_front(
    mKa: cute.Tensor,
    mPgT: cute.Tensor,
    mZ: cute.Tensor,
    mDep: cute.Tensor,
    mCut: cute.Tensor,
    mRdy: cute.Tensor,
    p: StepPointers,
    ppr: Int32,
    EVS: Float32,
    IEVS: Float32,
    cfg: cutlass.Constexpr,
    W: cutlass.Constexpr,
    NS: cutlass.Constexpr,
    NPR: cutlass.Constexpr,
    TRIM: cutlass.Constexpr,
    QKOWN: cutlass.Constexpr,
    EARLY: cutlass.Constexpr,
    PAD: cutlass.Constexpr,
):
    """Q's planes, the appended token and each row's reference, in two or
    three CTAs a row group.

    The append CTA quantises the group's Q rows and the new key and value
    into the cache and adds the group's length, shifted past a count, to its
    word in `mRdy` with one release: that is all the decode's prologue reads.
    The sums CTA keeps the running sums (the tail's block row, V's sum and
    mean) and adds one to the count, which the decode's loop waits on. The
    mass CTA estimates the reference on the prepass's tile and writes Z and
    the cut as words that are their own flags, so no fence stands before
    them. It quantises Q and the new key too, the same code on the same
    inputs, straight into its tile. `QKOWN` = 1 lets it publish both rows
    instead of the append CTA; 2 publishes Q while the append CTA owns K, V
    and the tail, which removes the sums CTA. Its raw loads and the gathers
    go out before any arithmetic, and the rows' classes and quantisation run
    under the gathers. Without the tail the running sum is one row of V, and
    the append CTA keeps it after its first release and adds the count
    itself, as it does when there is no new token; the decode then reads the
    count only before its epilogue.

    The decode spins on these words and never on this grid. A request's
    length moves once every CTA of every head has read it. `PAD` bytes of
    shared memory make each CTA as large as a decode CTA, so the hole it
    leaves on an SM the decode fills takes one.
    """
    B, H, HKV, D, PS = cfg.batch, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.page_size
    F16, HASKV, TRK = cfg.f16, cfg.has_kv, cfg.tail_rank
    G = H // HKV
    NBH = B * HKV
    # the tail's sums take a CTA of their own; the plain V sum rides the append
    NCR = 3 if HASKV and TRK >= 0 and QKOWN != 2 else 2
    NTK = NCR * HKV
    RW = 1 + 2 * G
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    cute.arch.griddepcontrol_wait()
    # `EARLY` releases the decode now, so its prologue overlaps this grid;
    # otherwise it launches when this grid ends (`heuristics.front_early`)
    if cutlass.const_expr(EARLY):
        cute.arch.griddepcontrol_launch_dependents()
    NG = (G + 7) // 8
    TPR = D // 16
    SWU, SWRS, _ = swizzle_of(D)
    SWM = SWU - 1
    NU = D // 16
    lq = tidx % TPR
    # warp 0 takes the new token and warps 1-3 Q's rows, all four without one
    QT = NT - 32 if HASKV else NT
    q0 = 32 if HASKV else 0
    NQP = (G * TPR + QT - 1) // QT
    smem = cutlass.memory.SmemAllocator()
    sK, sQ, sM, sE, sD, sKey, sCls, sCnt, sKs = mass_smem(smem, G, D, NPR, NS)
    sEq = smem.allocate_tensor(Float32, cute.make_layout(NG * 8), 16)
    sA = sX = sY = None
    if cutlass.const_expr(HASKV and TRK >= 0):
        sA = smem.allocate_tensor(Float32, cute.make_layout(D + 1), 16)
        sX = smem.allocate_tensor(Float32, cute.make_layout(D), 16)
        sY = smem.allocate_tensor(Float32, cute.make_layout(max(TRK, 1)), 16)
    if cutlass.const_expr(PAD):
        smem.allocate_tensor(cutlass.Int8, cute.make_layout(PAD), 16)
    if cta < NBH:
        bh = cta
        breq = bh // HKV
        hkv = bh % HKV
        # Nothing the length decides is needed to load Q's rows and the new
        # key, so their round trip overlaps the length's and the page
        # table's. Every thread loads, so the words are defined outside the
        # branches that use them.
        qw, kw = _tile_words(p, bh, tidx, G, D, HASKV)
        pos = Int32(ld_global_b32(p.lens_in + Int64(breq) * 4))
        slen = pos
        if cutlass.const_expr(HASKV):
            slen = pos + 1
        skip = Int32(1) if HASKV else Int32(-1)
        pks = []
        prs = []
        kscales = []
        for it in cutlass.range_constexpr((NPR * NU + NT - 1) // NT):
            pr = it * (NT // NU) + tidx // NU
            pkey = mass_key(pr, slen, W, NS)
            pks.append(pkey)
            prs.append(paged_row(mPgT, breq, hkv, pkey, PS, HKV))
            kscales.append(ld_scale_cg(p.ek + prs[it] * 2))
        # row 1 is the new key, which this CTA fills from registers
        for it in cutlass.range_constexpr((NPR * NU + NT - 1) // NT):
            pr = it * (NT // NU) + tidx // NU
            pu = tidx % NU
            pkey = pks[it]
            if pr < NPR and pr != skip:
                cute.arch.cp_async_shared_global(
                    sK.iterator
                    + (pr * D + (((pu ^ ((pkey >> SWRS) & SWM)) ^ ((pr >> SWRS) & SWM)) << 4)),
                    mKa.iterator + (prs[it] * D + pu * 16),
                    16,
                    "cg",
                )
            if pr < NPR:
                if pu == 0:
                    sKey[pr] = pkey
        cute.arch.cp_async_commit_group()
        rEq = cute.make_rmem_tensor((NG, 2), Float32)
        rTm = cute.make_rmem_tensor((NG, 2), Int32)
        for ng in cutlass.range_constexpr(NG):
            for j in cutlass.range_constexpr(2):
                rTm[(ng, j)] = Int32(0)
        own = None
        if cutlass.const_expr(QKOWN != 0):
            pg = Int32(ld_global_nc_b32(p.page_table + (Int64(breq) * ppr + pos // PS) * 4))
            prow = (Int64(pg) * HKV + hkv) * PS + pos % PS
            release = (slen << 2) if QKOWN == 1 else Int32(1)
            own = (p, prow, pos % PS, mRdy.iterator + bh * RW, release, QKOWN == 1)
        side = (_tile_own_rows, (qw, kw, sK, sKs, sQ, sEq, bh, G, D, F16, HASKV, own))
        pzw = mRdy.iterator + bh * RW + 1
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
            skip,
            G,
            D,
            W,
            NS,
            NPR,
            TRIM,
            0,
            pZw=pzw,
            side=side,
            sEq=sEq,
        )
        # The decode never reads the length this moves, and its end waits on
        # this grid, so the ticket's round trip stays off the reference's path.
        if cutlass.const_expr(HASKV):
            move_length(breq, pos, p, NTK)
    elif cta < 2 * NBH:
        bh = cta - NBH
        breq = bh // HKV
        wqs = None
        if cutlass.const_expr(QKOWN == 0):
            wqs = _q_words(p, bh, tidx, lq, q0, QT, NQP, G, D, TPR)
        pos = Int32(ld_global_b32(p.lens_in + Int64(breq) * 4))
        slen = pos
        if cutlass.const_expr(HASKV):
            slen = pos + 1
        qjob = None
        if cutlass.const_expr(QKOWN == 0):
            qjob = (_q_quant, (wqs, p, bh, G, D, F16, q0, QT))
        rdy = mRdy.iterator + bh * RW
        # Q and the new row are all the decode's prologue reads, so their
        # release is the length and waits on nothing else
        if cutlass.const_expr(HASKV):
            if cutlass.const_expr(QKOWN == 2 and TRK >= 0):
                append_row(
                    sA, sX, sY, bh, p, ppr, EVS, IEVS, cfg, True, NTK, None, (rdy, slen << 2)
                )
            else:
                relv = (slen << 2) if QKOWN != 1 else (Int32(1) if TRK >= 0 else Int32(0))
                append_planes(
                    bh,
                    p,
                    ppr,
                    IEVS,
                    cfg,
                    NTK,
                    qjob,
                    (rdy, relv),
                    None if TRK >= 0 else EVS,
                    QKOWN != 1,
                )
        else:
            _q_quant(*qjob[1])
            cute.arch.sync_threads()
            if tidx == 0:
                cute.arch.red(
                    rdy, (slen << 2) + 1, op="add", dtype="u32", sem="release", scope="gpu"
                )
    else:
        # the running sums, which the decode's loop and epilogue read: their
        # release is the count, ahead of the length's ticket
        bh = cta - 2 * NBH
        rdy = mRdy.iterator + bh * RW
        if cutlass.const_expr(HASKV and TRK >= 0 and QKOWN != 2):
            append_row(sA, sX, sY, bh, p, ppr, EVS, IEVS, cfg, False, NTK, None, (rdy, None))


@cute.jit
def launch_front(
    mKa,
    mPgT,
    mZ,
    mDep,
    mCut,
    mRdy,
    q: Int64,
    qa: Int64,
    qb: Int64,
    eq: Int64,
    k: Int64,
    v: Int64,
    ek: Int64,
    ka: Int64,
    kb: Int64,
    va: Int64,
    vb: Int64,
    vsum: Int64,
    vmean: Int64,
    lens_in: Int64,
    lens_out: Int64,
    page_table: Int64,
    tail_u: Int64,
    tail_vr: Int64,
    tail_vsum: Int64,
    tail_ysum: Int64,
    tail_rows: Int64,
    tickets: Int64,
    ppr: Int32,
    EVS: Float32,
    IEVS: Float32,
    cfg: cutlass.Constexpr,
    W: cutlass.Constexpr,
    NS: cutlass.Constexpr,
    NPR: cutlass.Constexpr,
    TRIM: cutlass.Constexpr,
    QKOWN: cutlass.Constexpr,
    EARLY: cutlass.Constexpr,
    PAD: cutlass.Constexpr,
    stream,
):
    p = StepPointers(
        q,
        qa,
        qb,
        eq,
        k,
        v,
        ek,
        ka,
        kb,
        va,
        vb,
        vsum,
        vmean,
        lens_in,
        lens_out,
        page_table,
        tail_u,
        tail_vr,
        tail_vsum,
        tail_ysum,
        tail_rows,
        tickets,
    )
    ctas = 3 if cfg.has_kv and cfg.tail_rank >= 0 and QKOWN != 2 else 2
    mass_front(
        mKa, mPgT, mZ, mDep, mCut, mRdy, p, ppr, EVS, IEVS, cfg, W, NS, NPR, TRIM, QKOWN, EARLY, PAD
    ).launch(
        grid=[ctas * cfg.batch * cfg.kv_heads, 1, 1], block=[NT, 1, 1], stream=stream, use_pdl=True
    )


def prepare_front(
    qa,
    qb,
    eq,
    ek,
    page_table,
    page_size,
    lens,
    ka,
    kb,
    va,
    vb,
    v_scale,
    vsum,
    vmean,
    z,
    depth,
    cut,
    ready,
    *,
    heads,
    n_kv_heads,
    dtype,
    has_kv=True,
    tail=None,
    lens_out=None,
    qk_owner=0,
    early=True,
    footprint=0,
):
    """Bind a decode step's front (`mass_front`).

    `z` and `cut` receive the reference and `z - depth`; `ready`
    ((NBH, 1 + 2 G) int32, zero between steps) holds the words the decode
    spins on. The decode that follows reads them with `reference="front"`
    and must follow every front, since it or its combine clears `ready`.
    `qk_owner` is `heuristics.front_ownership`'s choice and `early`
    `heuristics.front_early`'s. `footprint` is the shared memory of a CTA of
    the decode that follows: a front CTA smaller than one is padded up to
    it, since an SM packed with decode CTAs cannot use a smaller hole. The
    rest are `write.prepare_step`'s; `lens_out` defaults to `lens`.

    Returns `run(q, k=None, v=None)`."""
    if lens_out is None:
        lens_out = lens
    B = lens.shape[0]
    HKV = int(n_kv_heads)
    D = ka.shape[-1]
    G = int(heads) // HKV
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("the step takes 16-bit Q, K and V")
    if ready.shape != (B * HKV, 1 + 2 * G) or not ready.is_contiguous():
        raise ValueError("ready is (NBH, 1 + 2 G) int32")
    if G * (D // 16) > 2 * NT or 32 % (D // 16):
        raise ValueError(
            f"G={G} at D={D}: the front quantises a row group's Q in at most two passes"
        )
    NPR, W = mass_tile()
    rank, tail_ptrs = tail_pointers(tail if has_kv else None, vb, page_size, D)
    tickets = torch.zeros(B, device=lens.device, dtype=torch.int32)
    cfg = StepConfig(
        B,
        int(heads),
        HKV,
        D,
        int(page_size),
        vb is not None,
        dtype == torch.float16,
        True,
        bool(has_kv),
        rank,
    )
    has_kv = bool(has_kv)
    qk_owner = int(qk_owner) if has_kv else 0
    if qk_owner not in (0, 1, 2):
        raise ValueError("qk_owner is 0, 1 or 2")
    held = (
        qa,
        qb,
        eq,
        ek,
        page_table,
        lens,
        lens_out,
        ka,
        kb,
        va,
        vb,
        vsum,
        vmean,
        z,
        depth,
        cut,
        ready,
        tail,
        tickets,
    )
    like = (ka, page_table, z, depth, cut, ready.view(-1))
    views = [from_dlpack(ka, assumed_align=16)] + [
        from_dlpack(t, assumed_align=4) for t in like[1:]
    ]
    bound = _split_bound(
        bind_pointers(
            qa,
            qb,
            eq,
            ek,
            ka,
            kb,
            va,
            vb,
            vsum,
            vmean,
            lens,
            lens_out,
            page_table,
            tail_ptrs,
            tickets,
        )
    )
    ppr = Int32(page_table.shape[1])
    scale, inv = Float32(v_scale), Float32(1.0 / v_scale)
    early = bool(early)
    pad = -(-max(0, int(footprint) - front_smem(G, D, rank if has_kv else -1)) // 16) * 16
    key = (
        "front",
        cfg,
        W,
        MASS_STRATA,
        NPR,
        MASS_TRIM,
        qk_owner,
        early,
        pad,
        tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in like),
    )

    # resolved on the first call and kept, so a step does not hash the key
    compiled = []

    def run(q, k=None, v=None):
        if (k is None) == has_kv or (v is None) == has_kv:
            raise ValueError("the prepared front's KV presence is fixed")
        if (
            q.dtype != dtype
            or (k is not None and k.dtype != dtype)
            or (v is not None and v.dtype != dtype)
        ):
            raise ValueError(f"the prepared front takes {dtype} inputs")
        args = (*views, *_pointers(bound, q, k, v), ppr, scale, inv)
        if compiled:
            compiled[0](*args, current_stream(q.device))
            return
        kernel = compile_cached(
            key,
            lambda: full_carveout(
                cute.compile(
                    launch_front,
                    *args,
                    cfg,
                    W,
                    MASS_STRATA,
                    NPR,
                    MASS_TRIM,
                    qk_owner,
                    early,
                    pad,
                    cutlass.cuda.default_stream(),
                )
            ),
        )
        compiled.append(kernel)
        kernel(*args, current_stream(q.device))

    return Launch(run, held)
