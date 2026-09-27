"""The public entry points.

Training: `fold_attn_func` and `fold_attn_varlen_func` run FlashAttention-4's
forward and this package's backward, whose gradients are the same bits
however the GPU schedules them and whatever a request is batched with.

Serving: `fold_attn_with_kvcache` attends over a `FoldKVCache`, optionally
appending one token per request first.
"""

from __future__ import annotations

import math

import torch
from flash_attn.cute.interface import _flash_attn_fwd

from .backward.launch import prepare_backward
from .kv_cache import FoldKVCache


def _check_qkv(q, k, v, ndim):
    if q.ndim != ndim or k.ndim != ndim or v.ndim != ndim:
        raise ValueError(f"q, k and v must have {ndim} dimensions")
    if q.shape[0] != k.shape[0] or k.shape != v.shape:
        raise ValueError("q, k and v must have compatible shapes")
    if q.shape[-1] != k.shape[-1] or q.shape[-2] % k.shape[-2]:
        raise ValueError(
            "head dims must match and the query heads must be a multiple of the KV heads"
        )
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k and v must share a bf16 or fp16 dtype")
    if q.device != k.device or q.device != v.device or q.device.type != "cuda":
        raise ValueError("q, k and v must be on the same CUDA device")
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        raise ValueError("FoldAttention requires an SM90 GPU")


class _FoldAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, causal):
        out, lse, _, _ = _flash_attn_fwd(
            q, k, v, softmax_scale=softmax_scale, causal=causal, return_lse=True
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale, ctx.causal = softmax_scale, causal
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (dout,) = grad_outputs
        q, k, v, out, lse = ctx.saved_tensors
        run = prepare_backward(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            out.transpose(1, 2),
            dout.contiguous().transpose(1, 2),
            lse,
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
        )
        dq, dk, dv = run()
        return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), None, None


class _FoldAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, softmax_scale, causal):
        out, lse, _, _ = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=causal,
            softmax_scale=softmax_scale,
            return_lse=True,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
        ctx.max_seqlen, ctx.softmax_scale, ctx.causal = max_seqlen, softmax_scale, causal
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (dout,) = grad_outputs
        q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
        run = prepare_backward(
            q,
            k,
            v,
            out,
            dout.contiguous(),
            lse,
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
        )
        dq, dk, dv = run()
        return dq, dk, dv, None, None, None, None


def fold_attn_func(q, k, v, softmax_scale=None, causal=False):
    """Self-attention over `(batch, seqlen, nheads, headdim)` tensors.

    `k` and `v` may have fewer heads than `q` (GQA/MQA). Returns the output,
    shaped and typed as `q`. The forward is FlashAttention-4's; the backward
    reduces across CTAs in integers, so the gradients do not depend on the
    order work runs in. Head dims 64, 96 and 128, bf16 or fp16, SM90.
    """
    _check_qkv(q, k, v, 4)
    if q.shape[1] != k.shape[1]:
        raise ValueError("self-attention needs equal query and key lengths")
    scale = 1.0 / math.sqrt(q.shape[-1]) if softmax_scale is None else float(softmax_scale)
    return _FoldAttnFunc.apply(q, k, v, scale, bool(causal))


def fold_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    softmax_scale=None,
    causal=False,
):
    """Self-attention over sequences packed along the first axis.

    `q` is `(total, nheads, headdim)` and `k`, `v` `(total, nheads_k,
    headdim)`, split by `cu_seqlens_q`, a CUDA int32 `(batch + 1,)` tensor.
    The keys are the queries' own sequences, so `cu_seqlens_k` and
    `max_seqlen_k` may be left out; if given they must match. A request's
    gradients are the same bits alone and packed with others.
    """
    _check_qkv(q, k, v, 3)
    cu = cu_seqlens_q
    if cu.ndim != 1 or cu.dtype != torch.int32 or cu.device != q.device:
        raise ValueError("cu_seqlens_q must be a CUDA int32 vector on q's device")
    if cu_seqlens_k is not None and cu_seqlens_k is not cu and not torch.equal(cu_seqlens_k, cu):
        raise ValueError("self-attention needs cu_seqlens_k == cu_seqlens_q")
    if max_seqlen_q is None:
        max_seqlen_q = int(torch.diff(cu).max())
    if max_seqlen_k is not None and int(max_seqlen_k) != int(max_seqlen_q):
        raise ValueError("self-attention needs max_seqlen_k == max_seqlen_q")
    scale = 1.0 / math.sqrt(q.shape[-1]) if softmax_scale is None else float(softmax_scale)
    return _FoldAttnVarlenFunc.apply(q, k, v, cu, int(max_seqlen_q), scale, bool(causal))


def fold_attn_with_kvcache(q, cache: FoldKVCache, k=None, v=None):
    """Attention of one query token per request over `cache`.

    `q` is `(batch, nheads, headdim)` or `(batch, 1, nheads, headdim)`. With
    `k` and `v` (`(batch, nheads_k, headdim)`, or with the extra axis of 1)
    the token is appended first, so the query sees it. Returns the output in
    `q`'s shape and dtype.
    """
    squeeze = q.ndim == 4
    if squeeze and q.shape[1] != 1:
        raise ValueError("fold_attn_with_kvcache takes one query token per request")
    if squeeze:
        q = q[:, 0]
    if k is None and v is None:
        out = cache.attend(q)
    elif k is not None and v is not None:
        out = cache.decode(q, k[:, 0] if squeeze else k, v[:, 0] if squeeze else v)
    else:
        raise ValueError("k and v are both given or both left out")
    return out[:, None] if squeeze else out
