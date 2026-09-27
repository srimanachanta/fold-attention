"""The backward: is it FA's gradient, and do its bits depend on anything but
the request.

Every positive claim has a control that can fail: the accuracy tests compare
against FA-4's own backward error, and the invariance tests compare bits
across packings, repeats, dispatchers and layouts that a float accumulator
would move.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

cute_available = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
if cute_available:
    from exact_fold_attn import fold_attn_func, fold_attn_varlen_func
    from exact_fold_attn.backward import (
        Plan,
        plan_dense,
        plan_varlen,
        prepare_backward,
        tile_config,
    )
    from exact_fold_attn.backward import launch as backward_launch
    from exact_fold_attn.backward.ablation import prepare_backward_variant
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

pytestmark = pytest.mark.skipif(not cute_available, reason="needs an SM90 GPU")


def _packed(lens, H, HKV, D, seed=0):
    torch.manual_seed(seed)
    dev, bf = "cuda", torch.bfloat16
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32)
    T = sum(lens)
    q = torch.randn(T, H, D, device=dev, dtype=bf)
    k = torch.randn(T, HKV, D, device=dev, dtype=bf)
    v = torch.randn(T, HKV, D, device=dev, dtype=bf)
    do = torch.randn(T, H, D, device=dev, dtype=bf)
    return q, k, v, do, cu


def _fwd(q, k, v, cu, S):
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=S,
        max_seqlen_k=S,
        causal=True,
        return_lse=True,
    )


def _bwd(q, k, v, o, do, lse, cu, S, **kw):
    run = prepare_backward(q, k, v, o, do, lse, causal=True, cu_seqlens=cu, max_seqlen=S, **kw)
    return tuple(x.clone() for x in run())


def _dense(B, H, HKV, S, D, seed=0, causal=True):
    """Dense `(B, H, S, D)` views, FA-4's forward, and its LSE."""
    torch.manual_seed(seed)
    dev, bf = "cuda", torch.bfloat16
    q, do = (torch.randn(B, S, H, D, device=dev, dtype=bf) for _ in range(2))
    k, v = (torch.randn(B, S, HKV, D, device=dev, dtype=bf) for _ in range(2))
    o, lse = _flash_attn_fwd(q, k, v, softmax_scale=D**-0.5, causal=causal, return_lse=True)[:2]
    tr = lambda x: x.transpose(1, 2)
    return (tr(q), tr(k), tr(v), tr(o), tr(do)), lse.contiguous()


def _ref64(q, k, v, do, causal=True):
    """fp64 gradients of softmax attention for one request, (S, H, D)."""
    S, H, D = q.shape
    G = H // k.shape[1]
    qd = q.double().requires_grad_()
    kd = k.double().repeat_interleave(G, 1).requires_grad_()
    vd = v.double().repeat_interleave(G, 1).requires_grad_()
    s = torch.einsum("shd,thd->hst", qd, kd) / math.sqrt(D)
    if causal:
        s = s.masked_fill(
            torch.ones(S, S, device=q.device, dtype=torch.bool).triu(1), float("-inf")
        )
    torch.einsum("hst,thd->shd", s.softmax(-1), vd).backward(do.double())
    return (qd.grad, kd.grad.view(S, H // G, G, D).sum(2), vd.grad.view(S, H // G, G, D).sum(2))


def _rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize(
    "H,HKV,D", [(8, 8, 128), (8, 2, 128), (16, 2, 64), (32, 4, 128), (8, 8, 96)]
)
def test_the_gradient_is_as_accurate_as_fa4s(H, HKV, D):
    """Against fp64: no worse than FA-4's own fp32-atomic backward. The grids
    are derived from the inputs, so a bad one would show here."""
    q, k, v, do, cu = _packed([1536], H, HKV, D)
    S = 1536
    o, lse = _fwd(q, k, v, cu, S)
    ours = _bwd(q, k, v, o, do, lse, cu, S)
    fa4 = _flash_attn_bwd(
        q,
        k,
        v,
        o,
        do,
        lse,
        causal=True,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=S,
        max_seqlen_k=S,
        softmax_scale=1 / math.sqrt(D),
    )[:3]
    ref = _ref64(q, k, v, do)
    for name, a, b, r in zip("qkv", ours, fa4, ref):
        assert _rel(a, r) <= 1.05 * _rel(b, r) + 1e-5, (name, _rel(a, r), _rel(b, r))


@pytest.mark.parametrize(
    "B,H,HKV,S,D",
    [(1, 8, 2, 2048, 128), (2, 16, 2, 1024, 64), (1, 32, 8, 1024, 128), (4, 8, 2, 2048, 128)],
)
def test_every_dense_plan_is_as_accurate_as_fa4s(B, H, HKV, S, D):
    """The dense planner's split heads, subgroups, canonical records and
    private groups, each against fp64 beside FA-4."""
    (q, k, v, o, do), lse = _dense(B, H, HKV, S, D, seed=5)
    run = prepare_backward(q, k, v, o, do, lse, causal=True)
    ours = tuple(x.clone() for x in run())
    tr = lambda x: x.transpose(1, 2)
    fa4 = _flash_attn_bwd(
        tr(q), tr(k), tr(v), tr(o), tr(do), lse, causal=True, softmax_scale=1 / math.sqrt(D)
    )[:3]
    for b in range(B):
        ref = _ref64(tr(q)[b], tr(k)[b], tr(v)[b], tr(do)[b])
        for name, a, f, r in zip("qkv", ours, fa4, ref):
            a, f = tr(a)[b], f[b]
            assert _rel(a, r) <= 1.05 * _rel(f, r) + 1e-5, (run.plan, name)


@pytest.mark.parametrize(
    "B,H,HKV,S,D,causal",
    [
        (2, 8, 2, 1024, 128, False),
        (1, 16, 2, 1000, 64, False),
        (1, 8, 8, 1000, 96, False),
        (3, 8, 2, 777, 128, True),
    ],
)
def test_full_and_ragged_attention_are_as_accurate_as_fa4s(B, H, HKV, S, D, causal):
    """Masks are applied only where they can change P: the diagonal under
    causal, and a key block running past the sequence. Full attention, and
    lengths that end mid-tile, check the rest need none."""
    (q, k, v, o, do), lse = _dense(B, H, HKV, S, D, seed=7, causal=causal)
    run = prepare_backward(q, k, v, o, do, lse, causal=causal)
    ours = tuple(x.clone() for x in run())
    tr = lambda x: x.transpose(1, 2)
    fa4 = _flash_attn_bwd(
        tr(q), tr(k), tr(v), tr(o), tr(do), lse, causal=causal, softmax_scale=1 / math.sqrt(D)
    )[:3]
    for b in range(B):
        ref = _ref64(tr(q)[b], tr(k)[b], tr(v)[b], tr(do)[b], causal=causal)
        for name, a, f, r in zip("qkv", ours, fa4, ref):
            a, f = tr(a)[b], f[b]
            assert _rel(a, r) <= 1.05 * _rel(f, r) + 1e-5, (run.plan, name)
    assert _same(ours, run())


@pytest.mark.parametrize("H,HKV,D", [(8, 8, 128), (8, 2, 128), (16, 2, 64), (32, 4, 128)])
def test_a_request_does_not_see_its_batch(H, HKV, D):
    """A request's gradients alone and packed among others are the same bits."""
    lens = [1000, 777, 2048, 64, 1500]
    q, k, v, do, cu = _packed(lens, H, HKV, D)
    S = max(lens)
    o, lse = _fwd(q, k, v, cu, S)
    packed = _bwd(q, k, v, o, do, lse, cu, S)
    a, b = int(cu[2]), int(cu[3])
    sl = lambda x: x[a:b].contiguous()
    cu1 = torch.tensor([0, b - a], device="cuda", dtype=torch.int32)
    o1, lse1 = _fwd(sl(q), sl(k), sl(v), cu1, b - a)
    alone = _bwd(sl(q), sl(k), sl(v), o1, sl(do), lse1, cu1, b - a)
    assert all(torch.equal(x, sl(y)) for x, y in zip(alone, packed))


def test_the_packed_plan_reads_only_head_counts():
    """The packed planner has no batch, length or device argument to read, and
    across a growing batch the plan and the bits of one request stay put."""
    c128, c64 = tile_config(128), tile_config(64)
    assert plan_varlen(8, 2, c128).subgroup == 2
    assert plan_varlen(32, 4, c128).subgroup == 2
    assert plan_varlen(32, 8, c128).subgroup == 4
    assert plan_varlen(16, 4, c128).subgroup == 4
    assert plan_varlen(8, 8, c128).subgroup == 1
    # a group of eight is split at head_dim 128 and kept whole at 64, where
    # its tile is half as long; sixteen splits into four-head subgroups
    assert plan_varlen(64, 8, c128).subgroup == 2
    assert plan_varlen(64, 8, c64).subgroup == 8
    assert plan_varlen(64, 4, c64).subgroup == 4
    H, HKV, D, L = 32, 8, 128, 1024
    q1, k1, v1, do1, _ = _packed([L], H, HKV, D, seed=3)
    got, plans = [], []
    for B in (1, 3, 9):
        cu = torch.arange(B + 1, device="cuda", dtype=torch.int32) * L
        q, k, v, do = (x.repeat(B, 1, 1) for x in (q1, k1, v1, do1))
        o, lse = _fwd(q, k, v, cu, L)
        run = prepare_backward(q, k, v, o, do, lse, causal=True, cu_seqlens=cu, max_seqlen=L)
        plans.append(run.plan)
        got.append(tuple(x[:L].clone() for x in run()))
    assert plans[0] == plans[1] == plans[2]
    assert _same(got[0], got[1]) and _same(got[0], got[2])


def test_the_two_dkv_reductions_do_not_agree():
    """The control for the packing tests: a private group and two-head
    subgroups combined by the last arriver really are different bits, so a
    plan that followed the packing would show."""
    q, k, v, do, cu = _packed([1024, 1024], 32, 8, 128, seed=4)
    o, lse = _fwd(q, k, v, cu, 1024)
    private = plan_varlen(32, 8, tile_config(128))
    split = dataclasses.replace(private, subgroup=2)
    loop = _bwd(q, k, v, o, do, lse, cu, 1024, plan=private)
    fold = _bwd(q, k, v, o, do, lse, cu, 1024, plan=split)
    assert torch.equal(loop[0], fold[0]), "dQ is order-free and must not move"
    for name, a, b in zip("kv", loop[1:], fold[1:]):
        assert not torch.equal(a, b), f"d{name} now agrees across the modes"


def test_repeated_calls_are_the_same_bits():
    q, k, v, do, cu = _packed([2048, 1024], 8, 2, 128)
    o, lse = _fwd(q, k, v, cu, 2048)
    run = prepare_backward(q, k, v, o, do, lse, causal=True, cu_seqlens=cu, max_seqlen=2048)
    first = tuple(x.clone() for x in run())
    for _ in range(3):
        assert _same(first, run())


def test_the_varlen_batch_index_changes_no_bits():
    lens = [17, 0, 129, 65, 257, 1, 191]
    q, k, v, do, cu = _packed(lens, 8, 2, 128, seed=5)
    o, lse = _fwd(q, k, v, cu, max(lens))
    scan = _bwd(q, k, v, o, do, lse, cu, max(lens), varlen_index=False)
    run = prepare_backward(
        q, k, v, o, do, lse, causal=True, cu_seqlens=cu, max_seqlen=max(lens), varlen_index=True
    )
    indexed = tuple(x.clone() for x in run())
    assert _same(scan, indexed)
    assert _same(indexed, run())


def test_the_dense_plan_follows_the_request_shape():
    cfg = tile_config(128)
    # whole-group tiles B H_KV S/128 below 256: split heads, 16-block records
    p = plan_dense(1, 8, 2, 2048, 128, True, cfg)
    assert (p.subgroup, p.record) == (1, 16)
    # a long request with few KV heads: two-head subgroups over whole ranges
    p = plan_dense(1, 8, 2, 8192, 128, True, cfg)
    assert (p.subgroup, p.record) == (2, 0)
    # enough whole-group tiles: the group stays private and dK/dV never
    # leave the CTA
    for shape in ((2, 8, 2, 8192), (2, 32, 8, 8192), (1, 32, 8, 8192), (8, 32, 8, 2048)):
        p = plan_dense(*shape, 128, True, cfg)
        assert (p.subgroup, p.record) == (shape[1] // shape[2], 0), shape
    # a group of eight or more is too long a tile to balance: two-head
    # subgroups combined in fp32
    for shape in ((2, 32, 4, 8192), (2, 32, 2, 8192), (2, 32, 1, 8192), (8, 32, 4, 2048)):
        p = plan_dense(*shape, 128, True, cfg)
        assert (p.subgroup, p.record) == (2, 0), shape


@pytest.mark.parametrize(
    "B,H,HKV,S,D", [(1, 8, 2, 1024, 128), (1, 16, 2, 2048, 64), (3, 8, 8, 1536, 128)]
)
def test_dispatch_does_not_move_the_bits(B, H, HKV, S, D, monkeypatch):
    """The persistent grid claims tiles from a counter, so which CTA runs a
    tile, and when, changes from call to call. Reversing the work list moves
    every tile to another CTA and another point in the claim order, and the
    bits stay put. Canonical records are included: their rounding boundary is
    logical."""
    (q, k, v, o, do), lse = _dense(B, H, HKV, S, D, seed=2)
    run = prepare_backward(q, k, v, o, do, lse, causal=True)
    ref = tuple(x.clone() for x in run())
    for _ in range(3):
        assert _same(ref, run())
    forward_list = backward_launch.work_list
    monkeypatch.setattr(
        backward_launch, "work_list", lambda *a, **kw: forward_list(*a, **kw)[::-1].copy()
    )
    rev = prepare_backward(q, k, v, o, do, lse, causal=True)
    assert _same(ref, rev())


def test_a_partial_subgroup_maps_heads_and_replays():
    B, H, HKV, S, D = 1, 8, 2, 256, 64
    (q, k, v, o, do), lse = _dense(B, H, HKV, S, D, seed=13)
    base = Plan(subgroup=1, record=0)
    one = prepare_backward(q, k, v, o, do, lse, causal=True, plan=base)
    two = prepare_backward(
        q, k, v, o, do, lse, causal=True, plan=dataclasses.replace(base, subgroup=2)
    )
    ref = tuple(x.clone() for x in one())
    got = tuple(x.clone() for x in two())
    assert torch.equal(ref[0], got[0])
    assert _same(got, two())


def test_varlen_is_the_dense_layout():
    """A packed batch of equal lengths is the dense batch, bit for bit, under
    one plan. The planners may choose differently -- dense reads its batch,
    packed does not -- so comparing layouts means holding the plan fixed."""
    B, L, H, HKV, D = 4, 1024, 8, 2, 128
    q, k, v, do, cu = _packed([L] * B, H, HKV, D)
    o, lse = _fwd(q, k, v, cu, L)
    plan = plan_varlen(H, HKV, tile_config(D))
    vl = _bwd(q, k, v, o, do, lse, cu, L, plan=plan)
    dn = lambda x: x.view(B, L, *x.shape[1:]).transpose(1, 2)
    lse_d = lse.view(H, B, L).transpose(0, 1).contiguous()
    run = prepare_backward(dn(q), dn(k), dn(v), dn(o), dn(do), lse_d, causal=True, plan=plan)
    de = tuple(x.clone() for x in run())
    assert all(torch.equal(dn(x), y) for x, y in zip(vl, de))


def test_autograd_matches_the_prepared_call():
    q, k, v, do, cu = _packed([1024, 512], 8, 2, 128)
    qa, ka, va = (x.clone().requires_grad_() for x in (q, k, v))
    out = fold_attn_varlen_func(qa, ka, va, cu, causal=True)
    out.backward(do)
    o, lse = _fwd(q, k, v, cu, 1024)
    assert torch.equal(out.detach(), o)
    want = _bwd(q, k, v, o, do, lse, cu, 1024)
    assert all(torch.equal(x.grad, y) for x, y in zip((qa, ka, va), want))


@pytest.mark.parametrize("causal", [False, True])
def test_dense_public_autograd_matches_fa4_forward_and_prepared_backward(causal):
    B, S, H, HKV, D = 2, 256, 8, 2, 64
    torch.manual_seed(7)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    qa, ka, va = (x.clone().requires_grad_() for x in (q, k, v))
    got = fold_attn_func(qa, ka, va, causal=causal)
    got.backward(do)

    ref, lse = flash_attn_func(q, k, v, causal=causal, return_lse=True)
    assert torch.equal(got.detach(), ref)
    qh, kh, vh, oh, doh = (x.transpose(1, 2) for x in (q, k, v, ref, do))
    want = prepare_backward(qh, kh, vh, oh, doh, lse, causal=causal)()
    for x, y in zip((qa, ka, va), want):
        assert x.grad is not None and torch.equal(x.grad, y.transpose(1, 2))


_VARIANTS = [(True, True), (False, False), (True, False)]


@pytest.mark.parametrize("fp32_dq,persistent", _VARIANTS)
@pytest.mark.parametrize("B,H,HKV,S,D", [(2, 8, 2, 1024, 128), (2, 16, 2, 1024, 64)])
def test_each_ablation_is_as_accurate_as_fa4s(fp32_dq, persistent, B, H, HKV, S, D):
    """The measured variants still compute the gradient: fp32 dQ atomics and
    one CTA per work tile, each against fp64 beside FA-4."""
    (q, k, v, o, do), lse = _dense(B, H, HKV, S, D, seed=7)
    run = prepare_backward_variant(
        q, k, v, o, do, lse, causal=True, fp32_dq=fp32_dq, persistent=persistent
    )
    ours = tuple(x.clone() for x in run())
    tr = lambda x: x.transpose(1, 2)
    fa4 = _flash_attn_bwd(
        tr(q), tr(k), tr(v), tr(o), tr(do), lse, causal=True, softmax_scale=1 / math.sqrt(D)
    )[:3]
    for b in range(B):
        ref = _ref64(tr(q)[b], tr(k)[b], tr(v)[b], tr(do)[b])
        for name, a, f, r in zip("qkv", ours, fa4, ref):
            a, f = tr(a)[b], f[b]
            assert _rel(a, r) <= 1.05 * _rel(f, r) + 1e-5, (fp32_dq, persistent, name)


def test_the_ablation_defaults_are_the_shipped_kernel():
    """`prepare_backward_variant` with its defaults is `prepare_backward` to the
    bit. One CTA per tile repeats its bits and changes only the order of
    order-free sums, so nothing."""
    (q, k, v, o, do), lse = _dense(2, 16, 2, 2048, 128, seed=3)
    shipped = tuple(x.clone() for x in prepare_backward(q, k, v, o, do, lse, causal=True)())
    run = prepare_backward_variant(q, k, v, o, do, lse, causal=True)
    assert _same(tuple(x.clone() for x in run()), shipped)
    run = prepare_backward_variant(q, k, v, o, do, lse, causal=True, persistent=False)
    first = tuple(x.clone() for x in run())
    assert _same(tuple(x.clone() for x in run()), first)
    assert _same(first, shipped)
