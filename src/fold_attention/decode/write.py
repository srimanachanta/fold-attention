"""Every write into the serving cache: a prompt's, and a decode step's Q and
new token.

A row is `D / 16` threads holding one sixteen-byte unit each, the unit every
plane is stored in, so a load and a store are one vector each and K's swizzle
moves a whole unit. The quantisers are `quant.py`'s, which reproduce
`cache.py`'s torch quantisers to the bit.

The kernels take raw addresses bound once per prepared step, rather than
tensors converted on every call, because the step runs once per token and its
host cost is on the critical path. A prepared step therefore holds the
tensors it was bound to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cutlass
import torch
from cutlass import Float32, Int32, Int64, Uint32, cute
from quack.cute_dsl_utils import ParamsBase

from ..utils import Launch, compile_cached, current_stream, full_carveout
from .cache import TAIL_BLOCK, TailState
from .ptx import (
    f32_rn,
    f32_to_half2,
    int8_to_f32,
    ld_global_b32,
    ld_global_nc_b32,
    ld_global_v4,
    pow2_ceil,
    st_global_b16,
    st_global_b32,
    st_global_v4,
)
from .quant import load16, pack_words, put_k, put_v, quantize_q, rotate

NT = 128


@dataclass(frozen=True)
class StepConfig:
    """What a decode step's writer is compiled for."""

    batch: int
    heads: int
    kv_heads: int
    head_dim: int
    page_size: int
    v8: bool  # V is two e4m3 planes rather than 16-bit
    f16: bool  # the step's Q, K and V are fp16 rather than bf16
    has_q: bool
    has_kv: bool
    tail_rank: int  # the decode tail's rank, -1 for none


@dataclass
class StepPointers(ParamsBase):
    """The addresses a step writes and reads, in kernel-parameter order."""

    q: Int64
    qa: Int64
    qb: Int64
    eq: Int64
    k: Int64
    v: Int64
    ek: Int64
    ka: Int64
    kb: Int64
    va: Int64
    vb: Int64
    vsum: Int64
    vmean: Int64
    lens_in: Int64
    lens_out: Int64
    page_table: Int64
    tail_u: Int64
    tail_vr: Int64
    tail_vsum: Int64
    tail_ysum: Int64
    tail_rows: Int64
    tickets: Int64


_BOUND = (
    "qa",
    "qb",
    "eq",
    "ek",
    "ka",
    "kb",
    "va",
    "vb",
    "vsum",
    "vmean",
    "lens_in",
    "lens_out",
    "page_table",
    "tail_u",
    "tail_vr",
    "tail_vsum",
    "tail_ysum",
    "tail_rows",
    "tickets",
)


def _split_bound(bound: dict):
    """The bound addresses around the step's own inputs, in parameter order:
    `(qa, qb, eq)` between `q` and `k`, and the rest after `v`."""
    fields = list(StepPointers.__dataclass_fields__)
    return (tuple(bound[f] for f in fields[1:4]), tuple(bound[f] for f in fields[6:]))


def _pointers(bound, q, k, v):
    """The kernel's pointer arguments for one call. The step runs once per
    token, so this is a tuple join rather than a lookup per field."""
    mid, rest = bound
    return (
        Int64(q.data_ptr() if q is not None else 0),
        *mid,
        Int64(k.data_ptr() if k is not None else 0),
        Int64(v.data_ptr() if v is not None else 0),
        *rest,
    )


@cute.jit
def _page_row(p: StepPointers, b, hh, pos, ppr, cfg: cutlass.Constexpr):
    """Request `b`'s cache row for head `hh` at position `pos`, and the slot
    in its page."""
    PS, HKV = cfg.page_size, cfg.kv_heads
    pg = Int32(ld_global_nc_b32(p.page_table + (Int64(b) * ppr + pos // PS) * 4))
    off = pos % PS
    return (Int64(pg) * HKV + hh) * PS + off, off


@cute.jit
def _update_vsum(sw, xv, pos, EVS, pVsum, pVmean, sbase, ok):
    """V's running sum over `pos + 1` keys and its mean in the cache's units,
    a thread's sixteen channels at `sbase`."""
    n = f32_rn("mul.rn.f32", Float32(pos + 1), EVS)
    for j in cutlass.range_constexpr(4):
        s4 = [f32_rn("add.rn.f32", sw[j][i].bitcast(Float32), xv[4 * j + i]) for i in range(4)]
        m4 = [f32_rn("div.rn.f32", s, n).bitcast(Uint32) for s in s4]
        if ok:
            st_global_v4(pVsum + sbase + j * 16, [s.bitcast(Uint32) for s in s4])
            st_global_v4(pVmean + sbase + j * 16, m4)


@cute.jit
def move_length(b, pos, p: StepPointers, n: cutlass.Constexpr, rel=None):
    """`pos + 1` into request `b`'s length once all `n` CTAs that read it have
    taken a ticket; the last one resets the counter. `rel` is an address
    this CTA's stores are released to, with 1, ahead of the ticket."""
    tidx, _, _ = cute.arch.thread_idx()
    # every thread of this CTA has read the length before one moves it
    cute.arch.sync_threads()
    if tidx == 0:
        if cutlass.const_expr(rel is not None):
            cute.arch.red(rel, Int32(1), op="add", dtype="u32", sem="release", scope="gpu")
        if cutlass.const_expr(n == 1):
            st_global_b32(p.lens_out + Int64(b) * 4, Uint32(pos + 1))
        else:
            ticket = cute.make_ptr(
                Uint32, p.tickets + Int64(b) * 4, cute.AddressSpace.gmem, assumed_align=4
            )
            t = Int32(cute.arch.atomic_add(ticket, Uint32(1)))
            if t == n - 1:
                st_global_b32(p.tickets + Int64(b) * 4, Uint32(0))
                st_global_b32(p.lens_out + Int64(b) * 4, Uint32(pos + 1))


@cute.jit
def append_row(
    sA,
    sX,
    sY,
    kr,
    p: StepPointers,
    ppr: Int32,
    EVS: Float32,
    IEVS: Float32,
    cfg: cutlass.Constexpr,
    planes: cutlass.Constexpr = True,
    n_tickets: cutlass.Constexpr = 0,
    side=None,
    rel=None,
):
    """One appended row, a CTA to itself, and its block's tail.

    Warp 0 writes the key's and value's planes. Then every thread takes a
    rank's share of `y = kint U` (the row's lanes and a butterfly, integers,
    exact in any order) and a channel of the block's V sum and its row
    `vbar - ybar Vr` (`cache.tail_model`), in V's 16-bit format. Every word
    this launch rewrites is loaded before anything is stored, so the stores
    wait on one round trip rather than one per rank. A block's first key
    restarts its sums, so a page reused from an earlier prompt carries
    nothing over.

    A request's heads are CTAs of their own, so its length moves once all of
    them have read it (`move_length`, `n_tickets` CTAs, default `H_KV`).
    `planes=False` leaves the key's and value's planes to another CTA.
    `side`, a function and its arguments, runs beside warp 0's row, on the
    warps it leaves idle. `rel`, an address and a value, is released once the
    row and `side`'s stores are out (unless the value is None), and released
    again with 1 once the sums are.
    """
    HKV, D, TRK = cfg.kv_heads, cfg.head_dim, cfg.tail_rank
    TPR = D // 16
    tidx, _, _ = cute.arch.thread_idx()
    b = kr // HKV
    hh = kr % HKV
    lane = tidx % TPR
    c = tidx % D
    cok = tidx < D
    j = tidx // TPR
    jr = j % max(TRK, 1)
    sv = ld_global_b32(p.vsum + (Int64(kr) * D + c) * 4)
    pos = Int32(ld_global_b32(p.lens_in + Int64(b) * 4))
    prow, off = _page_row(p, b, hh, pos, ppr, cfg)
    blk = prow // TAIL_BLOCK
    first = (off % TAIL_BLOCK) == 0
    nk = Float32(off % TAIL_BLOCK + 1)
    tv = ld_global_b32(p.tail_vsum + (blk * D + c) * 4)
    ty = Uint32(0)
    if cutlass.const_expr(TRK > 0):
        ty = ld_global_b32(p.tail_ysum + (blk * TRK + jr) * 4)
    uw = [Uint32(0)] * 4
    if cutlass.const_expr(TRK > 0):
        uw = ld_global_v4(p.tail_u + (Int64(kr) * TRK + jr) * D + lane * 16, True)
    vr = []
    for jq in cutlass.range_constexpr(TRK // 4):
        vr = vr + ld_global_v4(p.tail_vr + ((Int64(kr) * D + c) * TRK + 4 * jq) * 4, True)
    if tidx < 32:
        # the whole warp runs the row so its shuffles have partners; lanes
        # past the row store nothing
        ok = tidx < TPR
        ubase = Int64(kr) * (D * 2) + lane * 32
        xk, _ = load16(p.k + ubase, cfg.f16)
        xv, wv = load16(p.v + ubase, cfg.f16)
        kst = ok
        if cutlass.const_expr(not planes):
            kst = cutlass.Boolean(False)
        a, ek = put_k(
            rotate(xk, lane, TPR, 1.0 / math.sqrt(D)), p.ka, p.kb, p.ek, prow, off, lane, kst, D
        )
        if cutlass.const_expr(planes):
            put_v(xv, wv, p.va, p.vb, prow, lane, ok, IEVS, D, cfg.v8)
        if ok:
            for i in cutlass.range_constexpr(16):
                sA[lane * 16 + i] = a[i]
                sX[lane * 16 + i] = xv[i]
            if lane == 0:
                # the key's `ek / 256`, which scales its projection
                sA[D] = ek * (1.0 / 256.0)
    if cutlass.const_expr(side is not None):
        side[0](*side[1])
    cute.arch.sync_threads()
    if cutlass.const_expr(rel is not None and rel[1] is not None):
        if tidx == 0:
            cute.arch.red(rel[0], Int32(rel[1]), op="add", dtype="u32", sem="release", scope="gpu")
    if cutlass.const_expr(TRK > 0):
        yj = Float32(0.0)
        for q in cutlass.range_constexpr(4):
            for bb in cutlass.range_constexpr(4):
                yj = yj + sA[lane * 16 + 4 * q + bb] * int8_to_f32(uw[q], 8 * bb)
        for st in cutlass.range_constexpr(TPR.bit_length() - 1):
            yj = yj + cute.arch.shuffle_sync_bfly(yj, 1 << st)
        # the integer projection is exact; the key's scale rounds it once
        yj = f32_rn("mul.rn.f32", yj, sA[D])
        if j < TRK:
            if lane == 0:
                ys = f32_rn(
                    "add.rn.f32",
                    Float32(cutlass.select_(first, Float32(0.0), ty.bitcast(Float32))),
                    yj,
                )
                st_global_b32(p.tail_ysum + (blk * TRK + j) * 4, ys.bitcast(Uint32))
                sY[j] = f32_rn("div.rn.f32", ys, nk)
        cute.arch.sync_threads()
    xc = sX[c]
    s = f32_rn("add.rn.f32", Float32(cutlass.select_(first, Float32(0.0), tv.bitcast(Float32))), xc)
    w = f32_rn("div.rn.f32", s, nk)
    for jq in cutlass.range_constexpr(TRK // 4):
        for jj in cutlass.range_constexpr(4):
            w = w - sY[4 * jq + jj] * vr[4 * jq + jj].bitcast(Float32)
    if cok:
        st_global_b32(p.tail_vsum + (blk * D + c) * 4, s.bitcast(Uint32))
        st_global_b16(p.tail_rows + (blk * D + c) * 2, f32_to_half2(w, cfg.f16))
    n = f32_rn("mul.rn.f32", Float32(pos + 1), EVS)
    s2 = f32_rn("add.rn.f32", sv.bitcast(Float32), xc)
    if cok:
        st_global_b32(p.vsum + (Int64(kr) * D + c) * 4, s2.bitcast(Uint32))
        st_global_b32(
            p.vmean + (Int64(kr) * D + c) * 4, f32_rn("div.rn.f32", s2, n).bitcast(Uint32)
        )
    move_length(b, pos, p, n_tickets if n_tickets else HKV, None if rel is None else rel[0])


@cute.jit
def append_planes(
    kr,
    p: StepPointers,
    ppr: Int32,
    IEVS: Float32,
    cfg: cutlass.Constexpr,
    n_tickets: cutlass.Constexpr,
    side=None,
    rel=None,
    sums=None,
    put_key: cutlass.Constexpr = True,
):
    """An appended row's key and value planes, a CTA to itself: warp 0
    writes them while `side`, a function and its arguments, runs on the
    rest. `rel`, an address and a value, is released once all of it is out,
    and the length then moves as in `append_row`. `sums`, V's scale, adds
    the tail-less running sums between the two, and a second release of 1
    after them."""
    HKV, D = cfg.kv_heads, cfg.head_dim
    TPR = D // 16
    tidx, _, _ = cute.arch.thread_idx()
    b = kr // HKV
    hh = kr % HKV
    lane = tidx % TPR
    pos = Int32(ld_global_b32(p.lens_in + Int64(b) * 4))
    if tidx < 32:
        ok = tidx < TPR
        ubase = Int64(kr) * (D * 2) + lane * 32
        xk = None
        if cutlass.const_expr(put_key):
            xk, _ = load16(p.k + ubase, cfg.f16)
        xv, wv = load16(p.v + ubase, cfg.f16)
        prow, off = _page_row(p, b, hh, pos, ppr, cfg)
        if cutlass.const_expr(put_key):
            put_k(
                rotate(xk, lane, TPR, 1.0 / math.sqrt(D)), p.ka, p.kb, p.ek, prow, off, lane, ok, D
            )
        put_v(xv, wv, p.va, p.vb, prow, lane, ok, IEVS, D, cfg.v8)
    if cutlass.const_expr(side is not None):
        side[0](*side[1])
    cute.arch.sync_threads()
    if cutlass.const_expr(rel is not None):
        if tidx == 0:
            cute.arch.red(rel[0], Int32(rel[1]), op="add", dtype="u32", sem="release", scope="gpu")
    if cutlass.const_expr(sums is not None):
        _append_sums(kr, p, sums, cfg, n_tickets, None if rel is None else rel[0])
    else:
        move_length(b, pos, p, n_tickets)


@cute.jit
def _append_sums(
    kr,
    p: StepPointers,
    EVS: Float32,
    cfg: cutlass.Constexpr,
    n_tickets: cutlass.Constexpr,
    rel=None,
):
    """An appended row's V sum and mean without the tail; its planes are
    another CTA's. The length moves as in `append_row`."""
    HKV, D = cfg.kv_heads, cfg.head_dim
    TPR = D // 16
    tidx, _, _ = cute.arch.thread_idx()
    b = kr // HKV
    lane = tidx % TPR
    pos = Int32(ld_global_b32(p.lens_in + Int64(b) * 4))
    if tidx < TPR:
        sbase = Int64(kr) * (D * 4) + lane * 64
        sw = [ld_global_v4(p.vsum + sbase + j * 16, False) for j in range(4)]
        xv, _ = load16(p.v + Int64(kr) * (D * 2) + lane * 32, cfg.f16)
        _update_vsum(sw, xv, pos, EVS, p.vsum, p.vmean, sbase, cutlass.Boolean(True))
    move_length(b, pos, p, n_tickets, rel)


@cute.kernel
def step_kernel(p: StepPointers, ppr: Int32, EVS: Float32, IEVS: Float32, cfg: cutlass.Constexpr):
    B, H, HKV, D = cfg.batch, cfg.heads, cfg.kv_heads, cfg.head_dim
    TPR = D // 16
    n_kv_ctas = _kv_ctas(cfg)
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    lane = tidx % TPR
    sA = sX = sY = None
    if cutlass.const_expr(cfg.has_kv and cfg.tail_rank >= 0):
        smem = cutlass.memory.SmemAllocator()
        sA = smem.allocate_tensor(Float32, cute.make_layout(D + 1), 16)
        sX = smem.allocate_tensor(Float32, cute.make_layout(D), 16)
        sY = smem.allocate_tensor(Float32, cute.make_layout(max(cfg.tail_rank, 1)), 16)
    if cta < n_kv_ctas:
        if cutlass.const_expr(cfg.tail_rank >= 0):
            append_row(sA, sX, sY, cta, p, ppr, EVS, IEVS, cfg)
        else:
            g = cta * NT + tidx
            kr = g // TPR
            ok = kr < B * HKV
            # a lane past the batch works on a real row so its loads stay in
            # bounds and its shuffles have partners, and stores nothing
            kr = kr % (B * HKV)
            b = kr // HKV
            hh = kr % HKV
            # the running sum is read before the length, so the two loads
            # share one round trip rather than one each
            sbase = Int64(kr) * (D * 4) + lane * 64
            sw = [ld_global_v4(p.vsum + sbase + j * 16, False) for j in range(4)]
            ubase = Int64(kr) * (D * 2) + lane * 32
            xk, _ = load16(p.k + ubase, cfg.f16)
            xv, wv = load16(p.v + ubase, cfg.f16)
            pos = Int32(ld_global_b32(p.lens_in + Int64(b) * 4))
            prow, off = _page_row(p, b, hh, pos, ppr, cfg)
            put_k(
                rotate(xk, lane, TPR, 1.0 / math.sqrt(D)), p.ka, p.kb, p.ek, prow, off, lane, ok, D
            )
            put_v(xv, wv, p.va, p.vb, prow, lane, ok, IEVS, D, cfg.v8)
            _update_vsum(sw, xv, pos, EVS, p.vsum, p.vmean, sbase, ok)
            # every lane of the request has read its length before one moves it
            cute.arch.sync_threads()
            if ok:
                if hh + lane == 0:
                    st_global_b32(p.lens_out + Int64(b) * 4, Uint32(pos + 1))
    else:
        g = (cta - n_kv_ctas) * NT + tidx
        qr = g // TPR
        ok = qr < B * H
        qr = qr % (B * H)
        words = ld_global_v4(p.q + Int64(qr) * (D * 2) + lane * 32, True) + ld_global_v4(
            p.q + Int64(qr) * (D * 2) + lane * 32 + 16, True
        )
        ma, mb, e = quantize_q(words, lane, TPR, D, cfg.f16)
        wa, wb = pack_words(ma), pack_words(mb)
        qaddr = Int64(qr) * D + lane * 16
        if ok:
            st_global_v4(p.qa + qaddr, wa)
            st_global_v4(p.qb + qaddr, wb)
            if lane == 0:
                st_global_b32(p.eq + Int64(qr) * 4, e.bitcast(Uint32))


def _kv_ctas(cfg: StepConfig):
    """The append's CTAs: a row each when the tail is kept, else rows packed
    `NT / (D / 16)` to a CTA."""
    if not cfg.has_kv:
        return 0
    rows = cfg.batch * cfg.kv_heads
    return rows if cfg.tail_rank >= 0 else -(-rows * (cfg.head_dim // 16) // NT)


@cute.jit
def launch_step(
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
    n = _kv_ctas(cfg) + (-(-cfg.batch * cfg.heads * (cfg.head_dim // 16) // NT) if cfg.has_q else 0)
    step_kernel(p, ppr, EVS, IEVS, cfg).launch(grid=[n, 1, 1], block=[NT, 1, 1], stream=stream)


def tail_pointers(tail: TailState | None, vb, page_size, D):
    """`(rank or -1, pointers)` for the step's tail operands."""
    if tail is None:
        return -1, dict.fromkeys(
            ("tail_u", "tail_vr", "tail_vsum", "tail_ysum", "tail_rows"), Int64(0)
        )
    if vb is not None:
        raise ValueError("the decode tail needs a 16-bit V cache")
    if page_size % TAIL_BLOCK:
        raise ValueError(f"page_size={page_size} does not hold whole {TAIL_BLOCK}-key blocks")
    rank = tail.rank
    if rank % 4:
        raise ValueError(f"tail rank {rank} is not a multiple of 4")
    if rank * (D // 16) > NT:
        raise ValueError(f"tail rank {rank} at D={D}: a rank's lanes must fit one {NT}-thread CTA")
    return rank, {
        f"tail_{n}": Int64(t.data_ptr() if t is not None else 0)
        for n, t in (
            ("u", tail.u),
            ("vr", tail.vr),
            ("vsum", tail.vsum),
            ("ysum", tail.ysum),
            ("rows", tail.rows),
        )
    }


def bind_pointers(
    qa, qb, eq, ek, ka, kb, va, vb, vsum, vmean, lens, lens_out, page_table, tail_ptrs, tickets
):
    """The addresses a prepared step or front binds once."""
    bound = {
        n: Int64(t.data_ptr() if t is not None else 0)
        for n, t in (
            ("qa", qa),
            ("qb", qb),
            ("eq", eq),
            ("ek", ek),
            ("ka", ka),
            ("kb", kb),
            ("va", va),
            ("vb", vb),
            ("vsum", vsum),
            ("vmean", vmean),
            ("lens_in", lens),
            ("lens_out", lens_out),
            ("page_table", page_table),
            ("tickets", tickets),
        )
    }
    bound.update(tail_ptrs)
    assert set(bound) == set(_BOUND)
    return bound


def prepare_step(
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
    lens_out,
    *,
    heads,
    n_kv_heads,
    dtype,
    has_q=True,
    has_kv=True,
    tail=None,
):
    """Bind a decode step's writer: Q `(B H, D)` into its planes, and one
    token per request `(B, H_KV, D)` appended at `lens` with `lens + 1`
    written to `lens_out`, which may be `lens` itself. Each row group's running
    V sum is updated and its mean over `lens + 1` keys written in the V
    cache's units. `tail` keeps the decode tail's block rows current as tokens
    append (`tail_pointers`).

    Returns `run(q=None, k=None, v=None)`; which of them it takes is fixed
    here by `has_q` and `has_kv`."""
    B = lens.shape[0]
    HKV = int(n_kv_heads)
    D = ka.shape[-1]
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("the step takes 16-bit Q, K and V")
    if (HKV * (D // 16)) & (HKV * (D // 16) - 1) or HKV * (D // 16) > NT:
        raise ValueError(f"H_KV={HKV}: a request's lanes must tile a {NT}-thread CTA")
    rank, tail_ptrs = tail_pointers(tail if has_kv else None, vb, page_size, D)
    # a tail build appends each head in a CTA of its own, and the last of a
    # request's heads to finish moves its length (`append_row`)
    tickets = torch.zeros(B, device=lens.device, dtype=torch.int32) if rank >= 0 else None
    held = (qa, qb, eq, ek, page_table, lens, ka, kb, va, vb, vsum, vmean, lens_out, tail, tickets)
    has_q, has_kv = bool(has_q), bool(has_kv)
    cfg = StepConfig(
        B,
        int(heads),
        HKV,
        D,
        int(page_size),
        vb is not None,
        dtype == torch.float16,
        has_q,
        has_kv,
        rank,
    )
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
    scale = Float32(v_scale)
    inv_scale = Float32(1.0 / v_scale)

    # resolved on the first call and kept, so a step does not hash the key
    compiled = []

    def run(q=None, k=None, v=None):
        if (q is None) == has_q or (k is None) == has_kv or (v is None) == has_kv:
            raise ValueError("the prepared step's Q/KV presence is fixed")
        if (
            (q is not None and q.dtype != dtype)
            or (k is not None and k.dtype != dtype)
            or (v is not None and v.dtype != dtype)
        ):
            raise ValueError(f"the prepared step takes {dtype} inputs")
        x = q if q is not None else k
        assert x is not None
        args = (*_pointers(bound, q, k, v), ppr, scale, inv_scale)
        if compiled:
            compiled[0](*args, current_stream(x.device))
            return
        kernel = compile_cached(
            ("step", cfg),
            lambda: full_carveout(
                cute.compile(launch_step, *args, cfg, cutlass.cuda.default_stream())
            ),
        )
        compiled.append(kernel)
        kernel(*args, current_stream(x.device))

    return Launch(run, held)


def step(
    q,
    k,
    v,
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
    lens_out,
):
    """`prepare_step` and one launch. `q=None` launches only the append,
    `k=None` only the quantisation."""
    B = lens.shape[0]
    # K is (B, H_KV, D); a Q-only step's state is the batch's own
    HKV = k.shape[1] if k is not None else vsum.shape[0] // B
    x = q if q is not None else k
    heads = q.shape[0] // B if q is not None else HKV
    prepare_step(
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
        lens_out,
        heads=heads,
        n_kv_heads=HKV,
        dtype=x.dtype,
        has_q=q is not None,
        has_kv=k is not None,
    )(q, k, v)


# A prompt's reductions run over fixed chunks counted from its request's start
# and are summed in order, so its V scale input and V sum are its own bits
# whatever else is in the batch.
CHUNK = 256


@cute.kernel
def _stats_kernel(
    pV: Int64,
    pCh: Int64,
    pVmax: Int64,
    pPart: Int64,
    cfg: cutlass.Constexpr,
):
    """Per chunk and head: V's amax and V's sum."""
    HKV, D, F16 = cfg
    TPR = D // 16
    RPB = NT // TPR
    NW = NT // 32
    smem = cutlass.memory.SmemAllocator()
    sPart = smem.allocate_tensor(Float32, cute.make_layout((RPB, D), stride=(D, 1)), 16)
    sMax = smem.allocate_tensor(Float32, cute.make_layout(NW), 16)
    tidx, _, _ = cute.arch.thread_idx()
    ch, h, _ = cute.arch.block_idx()
    slot = tidx // TPR
    lane = tidx % TPR
    t0 = Int32(ld_global_nc_b32(pCh + Int64(ch) * 16 + 4))
    cnt = Int32(ld_global_nc_b32(pCh + Int64(ch) * 16 + 12))
    vm = Float32(0.0)
    vs = [Float32(0.0) for _ in range(16)]
    for it in cutlass.range_constexpr(CHUNK // RPB):
        r = it * RPB + slot
        # a row past the chunk reads a real one and weighs it zero
        okf = Float32(Int32(r < cnt))
        base = (Int64(t0 + r % cnt) * HKV + h) * (D * 2) + lane * 32
        xv, _ = load16(pV + base, F16)
        for i in cutlass.range_constexpr(16):
            vm = cute.arch.fmax(vm, cute.arch.fmax(xv[i], -xv[i]) * okf)
            vs[i] = f32_rn("add.rn.f32", vs[i], f32_rn("mul.rn.f32", xv[i], okf))
    for i in cutlass.range_constexpr(16):
        sPart[(slot, lane * 16 + i)] = vs[i]
    for s in cutlass.range_constexpr(5):
        vm = cute.arch.fmax(vm, cute.arch.shuffle_sync_bfly(vm, 1 << s))
    if tidx % 32 == 0:
        sMax[tidx // 32] = vm
    cute.arch.sync_threads()
    if tidx < D:
        acc = Float32(0.0)
        for j in cutlass.range_constexpr(RPB):
            acc = f32_rn("add.rn.f32", acc, sPart[(j, tidx)])
        st_global_b32(pPart + (Int64(ch) * HKV + h) * (D * 4) + tidx * 4, acc.bitcast(Uint32))
    if tidx == 0:
        v2 = sMax[0]
        for j in cutlass.range_constexpr(1, NW):
            v2 = cute.arch.fmax(v2, sMax[j])
        st_global_b32(pVmax + (Int64(ch) * HKV + h) * 4, v2.bitcast(Uint32))


@cute.kernel
def _reduce_kernel(
    pCh: Int64,
    pFirst: Int64,
    pVmax: Int64,
    pPart: Int64,
    pVsum: Int64,
    pVmean: Int64,
    pEvs: Int64,
    nc: Int32,
    VH: Float32,
    VFIX: Float32,
    cfg: cutlass.Constexpr,
):
    """Per request and head, its chunks in order: V's sum and mean.

    V's scale is the layer's, so every CTA derives it from every chunk's amax
    alike, and the first also stores it: `VFIX`, or the least power of two at
    or above the amax times `VH` over e4m3's 448."""
    HKV, D = cfg
    NW = NT // 32
    smem = cutlass.memory.SmemAllocator()
    sMax = smem.allocate_tensor(Float32, cute.make_layout(NW), 16)
    tidx, _, _ = cute.arch.thread_idx()
    b, h, _ = cute.arch.block_idx()
    vm = Float32(0.0)
    for i in cutlass.range(tidx, nc * HKV, NT, unroll=8):
        vm = cute.arch.fmax(vm, ld_global_nc_b32(pVmax + Int64(i) * 4).bitcast(Float32))
    for s in cutlass.range_constexpr(5):
        vm = cute.arch.fmax(vm, cute.arch.shuffle_sync_bfly(vm, 1 << s))
    if tidx % 32 == 0:
        sMax[tidx // 32] = vm
    cute.arch.sync_threads()
    vm = sMax[0]
    for j in cutlass.range_constexpr(1, NW):
        vm = cute.arch.fmax(vm, sMax[j])
    evs = VFIX
    if VFIX == 0.0:
        evs = pow2_ceil(
            cute.arch.fmax(f32_rn("mul.rn.f32", f32_rn("mul.rn.f32", vm, VH), 1.0 / 448.0), 1e-30)
        )
    lo = Int32(ld_global_nc_b32(pFirst + Int64(b) * 4))
    hi = Int32(ld_global_nc_b32(pFirst + Int64(b) * 4 + 4))
    c = tidx % D
    vs = Float32(0.0)
    n = Int32(0)
    # unrolled so the loads issue ahead; the adds stay in chunk order
    for ch in cutlass.range(lo, hi, 1, unroll=8):
        vs = f32_rn(
            "add.rn.f32",
            vs,
            ld_global_nc_b32(pPart + (Int64(ch) * HKV + h) * (D * 4) + c * 4).bitcast(Float32),
        )
        n = n + Int32(ld_global_nc_b32(pCh + Int64(ch) * 16 + 12))
    if tidx < D:
        row = Int64(b) * HKV + h
        m = f32_rn("div.rn.f32", vs, f32_rn("mul.rn.f32", Float32(n), evs))
        st_global_b32(pVsum + row * (D * 4) + c * 4, vs.bitcast(Uint32))
        st_global_b32(pVmean + row * (D * 4) + c * 4, m.bitcast(Uint32))
        if tidx == 0 and b + h == 0:
            st_global_b32(pEvs, evs.bitcast(Uint32))


@cute.kernel
def _prompt_kernel(
    pK: Int64,
    pV: Int64,
    pCh: Int64,
    pEk: Int64,
    pPt: Int64,
    pKa: Int64,
    pKb: Int64,
    pVa: Int64,
    pVb: Int64,
    pEvs: Int64,
    ppr: Int32,
    cfg: cutlass.Constexpr,
):
    """One CTA per `NT / (D / 16)` rows of a chunk and head: each row rotated,
    quantised and scattered to its page slot."""
    HKV, D, PS, V8, F16 = cfg
    TPR = D // 16
    RPB = NT // TPR
    hscale = 1.0 / math.sqrt(D)
    tidx, _, _ = cute.arch.thread_idx()
    cid, h, _ = cute.arch.block_idx()
    ch = cid // (CHUNK // RPB)
    it = cid % (CHUNK // RPB)
    slot = tidx // TPR
    lane = tidx % TPR
    b = Int32(ld_global_nc_b32(pCh + Int64(ch) * 16))
    t0 = Int32(ld_global_nc_b32(pCh + Int64(ch) * 16 + 4))
    p0 = Int32(ld_global_nc_b32(pCh + Int64(ch) * 16 + 8))
    cnt = Int32(ld_global_nc_b32(pCh + Int64(ch) * 16 + 12))
    ievs = f32_rn("div.rn.f32", 1.0, ld_global_nc_b32(pEvs).bitcast(Float32))
    r = it * RPB + slot
    ok = r < cnt
    rr = r % cnt
    base = (Int64(t0 + rr) * HKV + h) * (D * 2) + lane * 32
    xk, _ = load16(pK + base, F16)
    xv, wv = load16(pV + base, F16)
    pos = p0 + rr
    pg = Int32(ld_global_nc_b32(pPt + (Int64(b) * ppr + pos // PS) * 4))
    off = pos % PS
    prow = (Int64(pg) * HKV + h) * PS + off
    put_k(rotate(xk, lane, TPR, hscale), pKa, pKb, pEk, prow, off, lane, ok, D)
    put_v(xv, wv, pVa, pVb, prow, lane, ok, ievs, D, V8)


@cute.jit
def _launch_prompt(
    pK,
    pV,
    pCh,
    pFirst,
    pVmax,
    pPart,
    pEk,
    pVsum,
    pVmean,
    pEvs,
    pPt,
    pKa,
    pKb,
    pVa,
    pVb,
    nc: Int32,
    B: Int32,
    ppr: Int32,
    VH: Float32,
    VFIX: Float32,
    cfg: cutlass.Constexpr,
    stream,
):
    HKV, D, _, _, F16 = cfg
    _stats_kernel(pV, pCh, pVmax, pPart, (HKV, D, F16)).launch(
        grid=[nc, HKV, 1], block=[NT, 1, 1], stream=stream
    )
    _reduce_kernel(pCh, pFirst, pVmax, pPart, pVsum, pVmean, pEvs, nc, VH, VFIX, (HKV, D)).launch(
        grid=[B, HKV, 1], block=[NT, 1, 1], stream=stream
    )
    _prompt_kernel(pK, pV, pCh, pEk, pPt, pKa, pKb, pVa, pVb, pEvs, ppr, cfg).launch(
        grid=[nc * (CHUNK // (NT // (D // 16))), HKV, 1], block=[NT, 1, 1], stream=stream
    )


def write_prompt(
    k,
    v,
    lens,
    page_table,
    page_size,
    ka,
    kb,
    va,
    vb,
    ek,
    vsum,
    vmean,
    *,
    v_headroom,
    v_scale,
):
    """Quantise packed prompts `(total, H_KV, D)` of host lengths `lens` into the
    paged cache, each key's scale into the pool `ek`, and each row group's V
    sum and mean.

    Returns V's scale as a one-element device tensor: an 8-bit V's power of two
    is `v_scale`, or the least one at or above the prompts' largest value times
    `v_headroom` over 448, and a bf16 V's is one. Nothing here waits on the
    device: the chunk table is one upload and every scale is computed there.
    """
    _, HKV, D = k.shape
    B = len(lens)
    first, rows, t0 = [0], [], 0
    for b, n in enumerate(lens):
        for p0 in range(0, n, CHUNK):
            rows += (b, t0 + p0, p0, min(CHUNK, n - p0))
        first.append(len(rows) // 4)
        t0 += n
    dev = k.device
    meta = torch.tensor(first + rows, dtype=torch.int32).pin_memory().to(dev, non_blocking=True)
    nc = len(rows) // 4
    scratch = torch.empty(nc * HKV * (D + 1) + 1, device=dev)
    vmax = scratch[: nc * HKV]
    part = scratch[nc * HKV : -1]
    evs = scratch[-1:]
    v8 = vb is not None
    cfg = (HKV, D, int(page_size), v8, k.dtype == torch.float16)
    vfix = (float(v_scale) if v_scale is not None else 0.0) if v8 else 1.0
    base = meta.data_ptr()
    ptrs = [
        Int64(x)
        for x in (
            k.data_ptr(),
            v.data_ptr(),
            base + 4 * (B + 1),
            base,
            vmax.data_ptr(),
            part.data_ptr(),
            ek.data_ptr(),
            vsum.data_ptr(),
            vmean.data_ptr(),
            evs.data_ptr(),
            page_table.data_ptr(),
            ka.data_ptr(),
            kb.data_ptr(),
            va.data_ptr(),
            vb.data_ptr() if v8 else 0,
        )
    ]
    args = (
        *ptrs,
        Int32(nc),
        Int32(B),
        Int32(page_table.shape[1]),
        Float32(v_headroom),
        Float32(vfix),
    )
    kernel = compile_cached(
        ("prompt", cfg),
        lambda: cute.compile(_launch_prompt, *args, cfg, cutlass.cuda.default_stream()),
    )
    kernel(*args, current_stream(dev))
    return evs
