"""The backward's host side: plan, preprocess, mainloop, postprocess.

dQ's per-tile partials, and dK/dV's wherever more than one CTA reaches a key
block, are rounded onto integer grids and summed as integers, so every
cross-CTA sum is order-free. The grids are exponents the kernels derive from
the preprocess's maxima (`grid.py`): the accumulators hold `dQ / sm * 2^s_dq`,
`dK / sm * 2^s_dk` and `dV * 2^s_dv`, and the postprocess multiplies back.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import cutlass
import torch
from cutlass import Float32, cute
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.interface import (
    _blocks_to_batch_size,
    _compute_blocks_to_batch,
    _compute_tile_cumsum,
)

from ..utils import Launch, compile_cached
from .grid import N_STATS
from .kernel import FoldBackwardSm90
from .plan import Plan, plan_dense, plan_varlen, tile_config, work_list
from .postprocess import FoldBackwardPostprocess
from .preprocess import FoldBackwardPreprocess

_OPTIONS = "--enable-tvm-ffi"
# From this many packed requests the varlen schedulers read FA-4's cumulative
# tile counts and O(1) block-to-request index instead of scanning.
_VARLEN_INDEX_MIN_BATCH = 512


@dataclass(eq=False)
class BackwardLaunch(Launch):
    """`prepare_backward`'s result: calling it returns `(dq, dk, dv)`."""

    plan: Plan | None = None


def _stream():
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


def _round_up(x, m):
    return (x + m - 1) // m * m


def _t(x, align=16):
    return None if x is None else to_cute_tensor(x, assumed_align=align)


_WORK_LISTS: dict = {}


def _work_list(B, H, H_KV, S, D, causal, cfg, plan, sms, dev):
    """The dense work list on the device, built once per shape.

    Building it in numpy and copying it over is most of an eager call's host
    time, about 0.4 ms, which is the whole backward at small shapes. The
    kernels only read it, so every launch of one shape shares one copy."""
    key = (B, H, H_KV, S, D, causal, cfg, plan, sms, str(dev))
    wl = _WORK_LISTS.get(key)
    if wl is None:
        wl = torch.from_numpy(work_list(B, H, H_KV, S, D, causal, cfg, plan, sms)).to(dev)
        _WORK_LISTS[key] = wl
    return wl


def prepare_backward(
    q,
    k,
    v,
    o,
    do,
    lse,
    *,
    causal,
    softmax_scale=None,
    cu_seqlens=None,
    max_seqlen=None,
    out_dtype=None,
    plan: Plan | None = None,
    varlen_index: bool | None = None,
):
    """Bind dQ, dK and dV of causal or full self-attention to one call's
    tensors, and return the launch that computes them.

    Dense: `(B, H, S, D)` views and `lse` `(B, H, S)`. Varlen: `(total, H, D)`
    packed by `cu_seqlens` `(B + 1,)` int32, and `lse` `(H, total)`, which is
    what FA-4's varlen forward returns. `lse` is in nats. `plan` overrides the
    decomposition `plan_dense`/`plan_varlen` choose and `varlen_index` the
    packed scheduler's batch lookup; neither changes the gradient's accuracy.

    The launch returns `(dq, dk, dv)`, the same tensors on every call, and
    owns every workspace, so two captured graphs never share one.
    """
    return _prepare(
        q,
        k,
        v,
        o,
        do,
        lse,
        causal=causal,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        out_dtype=out_dtype,
        plan=plan,
        varlen_index=varlen_index,
    )


def _prepare(
    q,
    k,
    v,
    o,
    do,
    lse,
    *,
    causal,
    softmax_scale=None,
    cu_seqlens=None,
    max_seqlen=None,
    out_dtype=None,
    plan: Plan | None = None,
    varlen_index: bool | None = None,
    dq_fp32: bool = False,
    persistent: bool = True,
):
    """`prepare_backward` with the variants `backward.ablation` measures:
    `dq_fp32` is `FoldBackwardSm90`'s, and `persistent=False` runs a dense
    call one CTA per work tile, whose tiles carry no canonical record."""
    varlen = cu_seqlens is not None
    if varlen:
        total, H, D = q.shape
        H_KV = k.shape[1]
        B = cu_seqlens.shape[0] - 1
        if max_seqlen is None:
            max_seqlen = int(torch.diff(cu_seqlens).max())
        S = int(max_seqlen)
    else:
        B, H, S, D = q.shape
        H_KV = k.shape[1]
        if k.shape[2] != S:
            raise NotImplementedError("the integer grids assume self-attention")
    if v.shape[-1] != D:
        raise NotImplementedError("the value head dim must equal the key head dim")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("q, k and v must be bf16 or fp16")
    if H % H_KV:
        raise ValueError("the query heads must be a multiple of the KV heads")
    dev = q.device
    G = H // H_KV
    sm = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(D)
    out_dtype = q.dtype if out_dtype is None else out_dtype
    dt = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}[q.dtype]
    cfg = tile_config(D)
    if plan is None:
        plan = (
            plan_varlen(H, H_KV, cfg) if varlen else plan_dense(B, H, H_KV, S, D, bool(causal), cfg)
        )
    if not persistent and not varlen:
        plan = dataclasses.replace(plan, record=0)
    # Heads split one to a tile, or canonical records, give many short dK/dV
    # partials: rounded onto integer grids and reduced asynchronously.
    # Subgroups of two or more heads over the whole range give a few long ones,
    # combined in fp32 by the last to arrive, which costs 1.2-2x the fold when
    # the partials are short and many and wins when they are long.
    fold = plan.record > 0 or (G > 1 and plan.subgroup == 1)
    combine = 1 < plan.subgroup < G and not fold
    dk_bits = 64 if fold else 0
    # dV's bound carries only the column mass `G S`, so 30 bits hold FA-4's
    # accuracy up to `G S = 16384`
    dv_bits = (32 if G * S <= 16384 else 64) if fold else 0
    if fold and varlen:
        raise NotImplementedError("canonical records need the dense work list")

    dq, dk, dv = (torch.empty_like(x, dtype=out_dtype) for x in (q, k, v))
    if varlen:
        # every sequence starts a tile, so each may waste up to one tile
        s_q_r = _round_up(total + (B + 1) * cfg.tile_m - 1, cfg.tile_m)
        s_k_r = _round_up(total + (B + 1) * cfg.tile_n - 1, cfg.tile_n)
        lead_q = (H,)
    else:
        s_q_r, s_k_r = _round_up(S, cfg.tile_m), _round_up(S, cfg.tile_n)
        lead_q = (B, H)
    dq_accum = torch.zeros(
        *lead_q, s_q_r * D, device=dev, dtype=torch.float32 if dq_fp32 else torch.int32
    )
    # A key block's dK/dV partials, one slot per subgroup, each MMA thread's
    # fragment as 16-byte groups interleaved over the threads, and an arrival
    # counter per key block, zero on entry and reset by the last arriver.
    part = part_ctr = None
    if combine:
        floats = 2 * cfg.tile_n * D // 256
        lead_blocks = (H_KV, s_k_r // cfg.tile_n) if varlen else (B, H_KV, -(-S // cfg.tile_n))
        part = torch.empty(
            *lead_blocks, G // plan.subgroup, 256, floats, device=dev, dtype=torch.float32
        )
        part_ctr = torch.zeros(*lead_blocks, 1, device=dev, dtype=torch.int32)
    dk_accum = dv_accum = None
    if fold:
        acc = {32: torch.int32, 64: torch.int64}
        dk_accum = torch.zeros(B, H_KV, s_k_r * D, device=dev, dtype=acc[dk_bits])
        dv_accum = torch.zeros(B, H_KV, s_k_r * D, device=dev, dtype=acc[dv_bits])
    dpsum = torch.empty(*lead_q, s_q_r, device=dev, dtype=torch.float32)
    lse_log2 = torch.empty(*lead_q, s_q_r, device=dev, dtype=torch.float32)
    stats = torch.zeros(B, H_KV, N_STATS, device=dev, dtype=torch.int32)

    if varlen:
        qh, kh, vh, oh, doh, dqh, dkh, dvh = q, k, v, o, do, dq, dk, dv
        cu = cu_seqlens
    else:
        qh, kh, vh, oh, doh, dqh, dkh, dvh = (
            x.transpose(1, 2) for x in (q, k, v, o, do, dq, dk, dv)
        )
        cu = None
    use_index = varlen and (
        B >= _VARLEN_INDEX_MIN_BATCH if varlen_index is None else bool(varlen_index)
    )
    cum_q = cum_k = b2b_q = b2b_k = None
    if use_index:
        cum_q, _ = _compute_tile_cumsum(cu_seqlens=cu, tile_size=cfg.tile_m)
        cum_k, _ = _compute_tile_cumsum(cu_seqlens=cu, tile_size=cfg.tile_n)
        b2b_q = _compute_blocks_to_batch(
            cum_q, _blocks_to_batch_size(total, B, cfg.tile_m, 1, False), dev
        )
        b2b_k = _compute_blocks_to_batch(
            cum_k, _blocks_to_batch_size(total, B, cfg.tile_n, 1, False), dev
        )

    pre_key = (
        "bwd_pre",
        varlen,
        str(dt),
        B,
        H,
        S,
        D,
        cfg.tile_m,
        lse.stride(),
        o.stride(),
        do.stride(),
        q.stride(),
        k.stride(),
        v.stride(),
        G,
        fold,
        use_index,
        dq_fp32,
    )
    pre = compile_cached(
        pre_key,
        lambda: cute.compile(
            FoldBackwardPreprocess(dt, D, cfg.tile_m, G, record_stats=fold),
            _t(oh),
            _t(doh),
            _t(dpsum),
            _t(lse, 4),
            _t(lse_log2),
            _t(dq_accum),
            _t(stats, 4),
            _t(qh),
            _t(kh),
            _t(vh),
            _t(cu, 4),
            _t(cum_q, 4),
            _t(b2b_q, 4),
            _stream(),
            options=_OPTIONS,
        ),
    )

    wl = sched_counter = None
    if not varlen and persistent:
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        wl = _work_list(B, H, H_KV, S, D, bool(causal), cfg, plan, sms, dev)
        sched_counter = torch.zeros(2, device=dev, dtype=torch.int32)

    dk_main = dkh if dk_accum is None else dk_accum
    dv_main = dvh if dv_accum is None else dv_accum
    main_key = (
        "bwd_main",
        varlen,
        B,
        H,
        H_KV,
        S,
        D,
        str(dt),
        str(out_dtype),
        bool(causal),
        cfg,
        plan,
        dk_bits,
        dv_bits,
        q.stride(),
        k.stride(),
        v.stride(),
        do.stride(),
        use_index,
        dq_fp32,
        persistent,
    )
    main = compile_cached(
        main_key,
        lambda: cute.compile(
            FoldBackwardSm90(
                dt,
                D,
                G,
                is_causal=bool(causal),
                tile_m=cfg.tile_m,
                tile_n=cfg.tile_n,
                atom_layout_m_dq=cfg.atom_layout_m_dq,
                gqa_subgroup=plan.subgroup,
                record_width=plan.record,
                dk_accum_bits=dk_bits,
                dv_accum_bits=dv_bits,
                dq_fp32=dq_fp32,
            ),
            _t(qh),
            _t(kh),
            _t(vh),
            _t(doh),
            _t(lse_log2),
            _t(dpsum),
            _t(dq_accum),
            _t(dk_main),
            _t(dv_main),
            Float32(sm),
            _t(stats, 4),
            _t(cu, 4),
            _t(cum_k, 4),
            _t(b2b_k, 4),
            _t(wl, 4),
            _t(sched_counter, 4),
            _t(part),
            _t(part_ctr, 4),
            _stream(),
            options=_OPTIONS,
        ),
    )

    post_key = (
        "bwd_post_dq",
        varlen,
        str(dt),
        B,
        H,
        S,
        D,
        cfg,
        str(out_dtype),
        G,
        use_index,
        dq_fp32,
    )
    post = compile_cached(
        post_key,
        lambda: cute.compile(
            FoldBackwardPostprocess(
                dt, D, cfg.tile_m, 256, cfg.atom_layout_m_dq, G, "dq", 0 if dq_fp32 else 32
            ),
            _t(dq_accum),
            _t(dqh),
            Float32(sm),
            _t(stats, 4),
            _t(cu, 4),
            _t(cum_q, 4),
            _t(b2b_q, 4),
            _stream(),
            options=_OPTIONS,
        ),
    )
    posts = [(post, dq_accum, dqh, sm)]
    if fold:
        for accum, outh, scale, kind, bits in (
            (dk_accum, dkh, sm, "dk", dk_bits),
            (dv_accum, dvh, 1.0, "dv", dv_bits),
        ):
            key = ("bwd_post_kv", kind, str(dt), B, H_KV, S, D, cfg, str(out_dtype), G, bits)
            post = compile_cached(
                key,
                lambda accum=accum, outh=outh, scale=scale, kind=kind, bits=bits: cute.compile(
                    FoldBackwardPostprocess(dt, D, cfg.tile_n, 256, 2, G, kind, bits),
                    _t(accum),
                    _t(outh),
                    Float32(scale),
                    _t(stats, 4),
                    None,
                    None,
                    None,
                    _stream(),
                    options=_OPTIONS,
                ),
            )
            posts.append((post, accum, outh, scale))

    qd = qh.detach()

    def run():
        stats.zero_()
        pre(oh, doh, dpsum, lse, lse_log2, dq_accum, stats, qh, kh, vh, cu, cum_q, b2b_q)
        main(
            qd,
            kh,
            vh,
            doh,
            lse_log2,
            dpsum,
            dq_accum,
            dk_main,
            dv_main,
            Float32(sm),
            stats,
            cu,
            cum_k,
            b2b_k,
            wl,
            sched_counter,
            part,
            part_ctr,
        )
        for post, accum, outh, scale in posts:
            post(accum, outh, Float32(scale), stats, cu, cum_q, b2b_q)
        return dq, dk, dv

    held = (
        q,
        k,
        v,
        o,
        do,
        lse,
        dq,
        dk,
        dv,
        dq_accum,
        dk_accum,
        dv_accum,
        dpsum,
        lse_log2,
        stats,
        cum_q,
        cum_k,
        b2b_q,
        b2b_k,
        wl,
        sched_counter,
        part,
        part_ctr,
    )
    return BackwardLaunch(run, held, plan)
