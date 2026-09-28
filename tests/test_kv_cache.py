"""The serving cache: FA-4's prefill writes the cache the decode kernel reads.

The prefill is FA's function, so what is tested is the seam: that the cache a
prefill writes is the cache the kernel's own quantisers would have built, that
decode steps after it track the exact softmax over everything written so far,
and that a request's answer does not depend on the rest of its batch.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

cuda = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
if cuda:
    from flash_attn.cute import flash_attn_varlen_func

    from fold_attention import FoldKVCache, fold_attn_with_kvcache
    from fold_attention.decode import write
    from fold_attention.decode.launch import _prepare

pytestmark = pytest.mark.skipif(not cuda, reason="needs an SM90 GPU")


def test_one_launch_serves_every_refine_band():
    """Dense gates follow each request's own length inside the kernel, so the
    batch's lengths never re-key the launch. A fixed gate or a depth has no
    bands."""
    att = FoldKVCache(1, 8, 1, 128, max_len=32768)
    att.lens = [23170]
    assert att._refine() == (10.0, 8.0)
    mid = att._launch(split=2)
    assert mid.config.refine_bands == (5793.0, 23170.0, 10.0, 12.0, 8.0, 8.0)
    att.lens = [23171]
    assert att._launch(split=2) is mid
    fixed = FoldKVCache(1, 8, 1, 128, max_len=8192, refine_k=14.0)
    fixed.lens = [1024]
    assert fixed._refine()[0] == 14.0
    assert fixed._launch(split=2).config.refine_bands == ()


def test_decode_cache_address_crosses_two_gib():
    """The final request's K/V addresses exceed signed 32-bit byte offsets."""
    if torch.cuda.get_device_properties(0).total_memory < 32 << 30:
        pytest.skip("needs a GPU with at least 32 GiB")
    B, HKV, G, D, S = 128, 4, 8, 128, 32768
    q = torch.randn((1, HKV * G, D), device="cuda", dtype=torch.bfloat16)
    q = q.expand(B, -1, -1).contiguous()
    k = torch.randn((S - 1, HKV, D), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kp = k.repeat(B, 1, 1)
    vp = v.repeat(B, 1, 1)
    lens = [S - 1] * B
    cu = torch.arange(B + 1, device="cuda", dtype=torch.int32) * (S - 1)
    fa = FoldKVCache(B, HKV * G, HKV, D, max_len=S + 64, page_size=128)
    fa.write_prompt(kp, vp, cu, lens)
    del kp, vp
    kn = k[-1:].expand(B, -1, -1).contiguous()
    vn = v[-1:].expand(B, -1, -1).contiguous()
    out = fa.decode(q, kn, vn)
    assert torch.isfinite(out).all()
    assert torch.equal(fa.z[:HKV], fa.z[-HKV:])
    assert torch.equal(out[0], out[-1])


def _layer(n, H, HKV, D, seed=0, alpha=30.0, beta=4.0, vmu=3.6):
    """`n` tokens of one layer's q, k and v with a real cache's logit tail.

    As in `test_decode._case`, the spread comes from a few keys aligning with
    the direction the queries share, and V sits off the origin.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    G = H // HKV
    u = torch.randn(HKV, D, device="cuda", generator=g)
    u = u / u.norm(dim=-1, keepdim=True)
    q = torch.randn(n, H, D, device="cuda", generator=g) + beta * u.repeat_interleave(G, 0)
    w = torch.rand(n, HKV, 1, device="cuda", generator=g) ** 6
    k = torch.randn(n, HKV, D, device="cuda", generator=g) + alpha * w * u
    m = torch.randn(HKV, D, device="cuda", generator=g)
    v = torch.randn(n, HKV, D, device="cuda", generator=g) + vmu * m / m.norm(dim=-1, keepdim=True)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def _attend(q, k, v):
    """fp32 attention of `q` `(T, H, D)` over `k`, `v` `(S, H_KV, D)`, the last
    T keys being q's own positions."""
    T, H, D = q.shape
    S, HKV = k.shape[0], k.shape[1]
    kk = k.float().repeat_interleave(H // HKV, 1)
    vv = v.float().repeat_interleave(H // HKV, 1)
    s = torch.einsum("thd,shd->hts", q.float(), kk) / math.sqrt(D)
    last = S - T + torch.arange(T, device=q.device)
    mask = torch.arange(S, device=q.device)[None, :] > last[:, None]
    s = s.masked_fill(mask[None], float("-inf"))
    return torch.einsum("hts,shd->thd", torch.softmax(s, -1), vv)


def _l2(o, ref):
    return float((o.float() - ref).norm() / ref.norm())


def _batch(lens, H, HKV, D, extra, seed=0):
    """Per request: its full token stream, prompt then `extra` decode tokens."""
    return [_layer(n + extra, H, HKV, D, seed=seed + 7 * b) for b, n in enumerate(lens)]


def _prefill(att, reqs, lens):
    q = torch.cat([r[0][:n] for r, n in zip(reqs, lens)])
    k = torch.cat([r[1][:n] for r, n in zip(reqs, lens)])
    v = torch.cat([r[2][:n] for r, n in zip(reqs, lens)])
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda", dtype=torch.int32)
    return att.prefill(q, k, v, cu), (q, k, v, cu)


def _step(att, reqs, lens, t):
    q = torch.stack([r[0][n + t] for r, n in zip(reqs, lens)])
    k = torch.stack([r[1][n + t] for r, n in zip(reqs, lens)])
    v = torch.stack([r[2][n + t] for r, n in zip(reqs, lens)])
    o = att.decode(q, k, v)
    # an f16 weight overflows into NaN, and NaN slips through every `max`
    assert torch.isfinite(o).all()
    return o


def test_decode_cast_fusion_preserves_output_and_ownership():
    lens, H, HKV, D = [65, 81], 16, 2, 64
    reqs = _batch(lens, H, HKV, D, 2)
    att = FoldKVCache(2, H, HKV, D, 128)
    _prefill(att, reqs, lens)
    q = torch.stack([r[0][n] for r, n in zip(reqs, lens)])
    k = torch.stack([r[1][n] for r, n in zip(reqs, lens)])
    v = torch.stack([r[2][n] for r, n in zip(reqs, lens)])
    want = att.prepare_replay_step(q, k, v)()[0].view(2, H, D).to(q.dtype)
    first = att.decode(q, k, v)
    assert torch.equal(first, want)
    saved = first.clone()
    second = _step(att, reqs, lens, 1)
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, saved)


def test_the_prefill_is_fa4_to_the_bit():
    lens, H, HKV, D = [300, 1000, 64], 16, 2, 128
    reqs = _batch(lens, H, HKV, D, 0)
    att = FoldKVCache(len(lens), H, HKV, D, 2048)
    out, (q, k, v, cu) = _prefill(att, reqs, lens)
    want = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(lens),
        max_seqlen_k=max(lens),
        causal=True,
    )
    want = want[0] if isinstance(want, tuple) else want
    assert torch.equal(out, want)
    for b, n in enumerate(lens):
        lo = int(cu[b])
        assert _l2(out[lo : lo + n], _attend(*(x[:n] for x in reqs[b]))) < 1e-2


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("v8", [False, True])
def test_decode_after_prefill_is_the_softmax_over_everything_written(D, v8):
    """Every step reads the prompt the prefill wrote and the tokens appended
    since, at the precision the kernel reaches on a cache it built itself."""
    lens, H, HKV, steps = [777, 1500, 130], 16, 2, 6
    reqs = _batch(lens, H, HKV, D, steps)
    att = FoldKVCache(len(lens), H, HKV, D, 2048, v8=v8)
    _prefill(att, reqs, lens)
    worst = 0.0
    for t in range(steps):
        o = _step(att, reqs, lens, t)
        for b, n in enumerate(lens):
            r = reqs[b]
            ref = _attend(r[0][n + t : n + t + 1], r[1][: n + t + 1], r[2][: n + t + 1])[0]
            worst = max(worst, _l2(o[b], ref))
    assert worst < 4e-3, worst


def test_control_decode_misses_a_token_it_did_not_append():
    """The control: attending without appending must not match the reference
    that includes the new token, or the test above cannot fail."""
    lens, H, HKV, D = [512], 8, 1, 128
    reqs = _batch(lens, H, HKV, D, 1)
    att = FoldKVCache(1, H, HKV, D, 1024, v8=False)
    _prefill(att, reqs, lens)
    r, n = reqs[0], lens[0]
    # make the new key the one every query row attends to
    u = r[0][n : n + 1].float().mean(1, keepdim=True)
    k = 40.0 * u / u.norm()
    ref = _attend(r[0][n : n + 1], torch.cat([r[1][:n], k.bfloat16()]), r[2][: n + 1])[0]
    missed = att.attend(r[0][n : n + 1])[0]
    att.append(k.bfloat16(), r[2][n : n + 1])
    seen = att.attend(r[0][n : n + 1])[0]
    assert _l2(seen, ref) < 4e-3
    assert _l2(missed, ref) > 0.1


@pytest.mark.parametrize("v8", [False, True])
def test_the_range_check_recovers_a_reference_out_of_range(v8):
    """A reference far enough under a row's peak overflows its weights. The
    denominator flags that row and only that row, and the rerun at the row's
    log-sum-exp gives the softmax back. A reference far over the mass is
    flagged too: it would make every gate shallower."""
    lens, H, HKV, D = [900, 1500], 8, 1, 128
    reqs = _batch(lens, H, HKV, D, 1, seed=5)
    att = FoldKVCache(len(lens), H, HKV, D, 2048, v8=v8, split=2, check_range=True)
    _prefill(att, reqs, lens)
    o = _step(att, reqs, lens, 0)
    assert not bool(att._out_of_range(o, att.den).any())
    bad = torch.zeros_like(att.z, dtype=torch.bool)
    bad[0] = True
    z = torch.where(bad, att.z - (70.0 if v8 else 140.0), att.z)
    far = att._prepare_decode(2, None, True, 0, z=z, reference="given")
    of, den, _ = far.run()
    flags = att._out_of_range(of, den)
    assert bool(flags[0].all()) and not bool(flags[1].any())
    fixed, _ = att._rerun(flags, den, 2, torch.float32, z=z)
    r, n = reqs[0], lens[0]
    ref = _attend(r[0][n : n + 1], r[1][: n + 1], r[2][: n + 1])[0]
    assert _l2(fixed.view(len(lens), H, D)[0], ref) < 4e-3
    assert _l2(o[0], ref) < 4e-3
    high = torch.where(bad, att.z + 8.0, att.z)
    of, den, _ = att._prepare_decode(2, None, True, 0, z=high, reference="given").run()
    flags = att._out_of_range(of, den)
    assert bool(flags[0].all()) and not bool(flags[1].any())
    fixed, _ = att._rerun(flags, den, 2, torch.float32, z=high)
    assert _l2(fixed.view(len(lens), H, D)[0], ref) < 4e-3


@pytest.mark.parametrize("depth", [16.0, 12.0])
def test_a_truncated_decode_stays_near_the_dense_one(depth):
    lens, H, HKV, D, steps = [2000, 1200], 16, 2, 128, 4
    reqs = _batch(lens, H, HKV, D, steps, seed=3)
    dense = FoldKVCache(len(lens), H, HKV, D, 4096)
    cut = FoldKVCache(len(lens), H, HKV, D, 4096, depth=depth)
    _prefill(dense, reqs, lens)
    _prefill(cut, reqs, lens)
    for t in range(steps):
        a = _step(dense, reqs, lens, t)
        c = _step(cut, reqs, lens, t)
        for b, n in enumerate(lens):
            r = reqs[b]
            ref = _attend(r[0][n + t : n + t + 1], r[1][: n + t + 1], r[2][: n + t + 1])[0]
            assert _l2(c[b], ref) < 2.0 ** (-depth / 3.0), (t, b, _l2(c[b], ref), _l2(a[b], ref))


def _alone_and_together(
    lens, steps, split, seed=11, H=16, HKV=2, D=128, depth=None, max_len=2048, chunk=None
):
    """Each step's output for every request, batched and run alone."""
    reqs = _batch(lens, H, HKV, D, steps, seed=seed)
    kw = dict(v8=False, split=split, depth=depth, chunk=chunk)
    together = FoldKVCache(len(lens), H, HKV, D, max_len, **kw)
    _prefill(together, reqs, lens)
    alone = []
    for b in range(len(lens)):
        a = FoldKVCache(1, H, HKV, D, max_len, **kw)
        _prefill(a, [reqs[b]], [lens[b]])
        alone.append(a)
    out = []
    for t in range(steps):
        o = _step(together, reqs, lens, t)
        out.append([(o[b], _step(alone[b], [reqs[b]], [lens[b]], t)[0]) for b in range(len(lens))])
    return out, together, alone


def test_a_request_does_not_see_its_batch():
    """Batch invariance end to end under a fixed split: the same request alone
    and beside others gives the same bits at every step, through the mass
    reference each step estimates. A bf16 V keeps the cache itself per
    request, since the 8-bit V's scale is the layer's."""
    out, _, _ = _alone_and_together([900, 1700, 300], 5, split=3)
    for t, row in enumerate(out):
        for b, (x, y) in enumerate(row):
            assert torch.equal(x, y), (t, b, (x.float() - y.float()).abs().max())


def test_a_request_does_not_see_its_batch_across_refine_bands():
    """The batch's longest request sits in a later refine band than request 0,
    whose gates, and so its bits, are still its own."""
    out, _, _ = _alone_and_together([4000, 7000], 3, split=3, H=8, HKV=1, max_len=8192)
    for t, row in enumerate(out):
        for b, (x, y) in enumerate(row):
            assert torch.equal(x, y), (t, b, (x.float() - y.float()).abs().max())


@pytest.mark.parametrize("v8", [False, True])
def test_a_second_cache_of_another_size_gets_its_own_kernels(v8):
    """Two layers of one process with the same shapes but different pools. A
    kernel is built for its tensors' view shapes, so each pool gets its own,
    and each answers as the other does, to the bit."""
    lens, H, HKV, D = [700, 1300], 16, 2, 128
    reqs = _batch(lens, H, HKV, D, 3)
    big = FoldKVCache(2, H, HKV, D, 8192, v8=v8, depth=6.0, split=3)
    small = FoldKVCache(2, H, HKV, D, 1536, v8=v8, depth=6.0, split=3)
    _prefill(big, reqs, lens)
    _prefill(small, reqs, lens)
    for t in range(3):
        a = _step(big, reqs, lens, t)
        b = _step(small, reqs, lens, t)
        torch.cuda.synchronize()
        assert torch.equal(a, b), t


def test_a_request_s_tail_does_not_see_its_batch():
    """The tail's map is fitted over the whole batch at once, and a request's
    map, block rows and outputs are still its own bits."""
    lens = [900, 1700, 300]
    out, together, alone = _alone_and_together(lens, 4, split=3, depth=10.0)
    assert together.tail_rank == 16
    for t, row in enumerate(out):
        for b, (x, y) in enumerate(row):
            assert torch.equal(x, y), (t, b)
    HKV = together.n_kv_heads
    for b, a in enumerate(alone):
        for key in ("u", "vr"):
            assert torch.equal(
                getattr(a.tail, key), getattr(together.tail, key)[b * HKV : (b + 1) * HKV]
            ), key


@pytest.mark.parametrize("depth", [None, 10.0])
def test_a_fixed_chunk_needs_no_fixed_split(depth):
    """With a fixed number of keys per split, the batch's longest request sets
    how many splits run and a shorter request's extra splits add zeros, so
    the same request alone and batched is the same bits with no split fixed."""
    lens = [800, 1300, 300]
    out, together, alone = _alone_and_together(lens, 4, None, depth=depth, chunk=256)
    assert together._launch().config.split != alone[2]._launch().config.split
    for t, row in enumerate(out):
        for b, (x, y) in enumerate(row):
            assert torch.equal(x, y), (t, b, (x.float() - y.float()).abs().max())


def test_control_the_batch_sized_split_moves_the_bits():
    """The control, and the reason `split` exists: `pick_split` gives request 0
    a different SPLIT alone than in the batch, so its partials are summed in a
    different grouping. The difference is rounding, and it is not zero. The
    longest request, 1, is cut to the same SPLIT either way. The decode's own
    f32 output is compared: a regrouped f32 sum moves a bf16 output only where
    it sits near a rounding boundary, which a small case can miss entirely."""
    _, together, alone = _alone_and_together([800, 1300, 300], 5, split=None)
    assert together._launch().config.split != alone[0]._launch().config.split
    HKV = together.n_kv_heads
    x = together.prepare_replay_decode()()[0][:HKV]
    y = alone[0].prepare_replay_decode()()[0]
    diff = (x - y).abs().max().item()
    scale = y.abs().max().item()
    assert 0.0 < diff < 1e-2 * scale, (diff, scale)


def test_the_first_step_after_a_prefill_has_a_reference_near_the_peak():
    """The sink's and the newest keys' max sits 20 binades under the peak on
    this cache, where f16 weights overflow; the mass reference's strata
    sample every few keys of a request this short, so it sees the peak's
    mass from the first step on."""
    lens, H, HKV, D = [300, 900], 16, 2, 128
    reqs = _batch(lens, H, HKV, D, 1)
    att = FoldKVCache(2, H, HKV, D, 1024, v8=True)
    _prefill(att, reqs, lens)
    o = _step(att, reqs, lens, 0)
    peak = []
    for b, n in enumerate(lens):
        q = reqs[b][0][n].float().view(HKV, H // HKV, D)
        s = torch.einsum("hgd,nhd->hgn", q, reqs[b][1][: n + 1].float())
        peak.append(s.max(-1).values * (1.4426950408889634 / math.sqrt(D)))
    under = (torch.stack(peak).reshape(att.z.shape) - att.z).max().item()
    assert under < 8.0, under
    assert torch.isfinite(o).all()


@pytest.mark.parametrize(
    "tail,v8,H,HKV,D",
    [
        ("auto", False, 16, 2, 128),
        (None, False, 16, 2, 128),
        (None, True, 16, 2, 128),
        # the tile's own Q split: 16, 4 and 2 elements a thread, and a key row
        # of 2-byte stores at D = 64
        ("auto", False, 32, 2, 128),
        (None, False, 8, 2, 128),
        ("auto", False, 16, 2, 64),
        (None, False, 8, 8, 64),
    ],
)
def test_the_front_is_the_step_and_the_prepass_to_the_bit(tail, v8, H, HKV, D):
    """One launch for Q, the token and the mass reference writes what the step
    kernel and the separate prepass write, and the decode behind it reads it
    the same, whichever CTA owns Q and the new key."""
    lens = [300, 900]
    reqs = _batch(lens, H, HKV, D, 3)
    atts = [FoldKVCache(2, H, HKV, D, 1024, depth=14.0, v8=v8, tail=tail) for _ in range(4)]
    atts[2]._front_qk = 1
    atts[3]._front_qk = 2
    for att in atts:
        _prefill(att, reqs, lens)
    for t in range(3):
        o = [_step(att, reqs, lens, t) for att in atts[:1] + atts[2:]]
        o.insert(1, _two_launch_step(atts[1], reqs, lens, t))
        assert all(torch.equal(o[0], x) for x in o[1:])
        for name in ("z", "qa", "qb", "eq", "ka", "kb", "va", "vsum", "vmean", "seq_lens"):
            assert all(
                torch.equal(getattr(atts[0], name), getattr(att, name)) for att in atts[1:]
            ), name


def _two_launch_step(att, reqs, lens, t):
    """`_step` as two launches: the step kernel writes Q and the token, and
    the decode's own mass prepass writes the reference."""
    D = att.head_dim
    q = torch.stack([r[0][n + t] for r, n in zip(reqs, lens)]).reshape(-1, D)
    k = torch.stack([r[1][n + t] for r, n in zip(reqs, lens)])
    v = torch.stack([r[2][n + t] for r, n in zip(reqs, lens)])
    write.prepare_step(
        att.qa.view(-1, D),
        att.qb.view(-1, D),
        att.eq.view(-1),
        att.ek,
        att.page_table,
        att.page_size,
        att.seq_lens,
        att.ka,
        att.kb,
        att.va,
        att.vb,
        att.v_scale,
        att.vsum,
        att.vmean,
        att.seq_lens,
        heads=att.n_heads,
        n_kv_heads=att.n_kv_heads,
        dtype=q.dtype,
        tail=att.tail,
    )(q, k, v)
    att.lens = [x + 1 for x in att.lens]
    tl = None
    if att.tail is not None:
        tl = att.tail.decode_tail()
    f = _prepare(
        att.qa,
        att.qb,
        att.eq,
        att.ka,
        att.kb,
        att.ek,
        att.va,
        att.z,
        att.cut,
        split=att._resolve_split(None),
        truncate=True,
        refine_k=att._refine()[0],
        refine_v=att._refine()[1],
        v8=att.v8,
        v2=att.vb,
        v_scale=att.v_scale,
        vmean=att.vmean if tl is None else None,
        tail=tl,
        page_table=att.page_table,
        page_size=att.page_size,
        seq_lens=att.seq_lens,
        n_kv_heads=att.n_kv_heads,
        reference="mass",
        group_cut="auto",
        weight_terms=att.weight_terms,
        order=att.order,
    )
    return f()[0].view(att.batch, att.n_heads, D).to(q.dtype)


def test_front_replay_graph_and_split_tuning_publish_ready_words():
    lens, H, HKV, D = [300], 8, 1, 64
    reqs = _batch(lens, H, HKV, D, 1)
    att = FoldKVCache(1, H, HKV, D, 512, depth=14.0, tail=None)
    _prefill(att, reqs, lens)
    _step(att, reqs, lens, 0)
    kernel = att.prepare_replay_decode()
    kernel_want = tuple(x.clone() for x in kernel())
    kernel_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(kernel_graph):
        kernel_got = kernel()
    for x in kernel_got:
        x.zero_()
    kernel_graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(x, y) for x, y in zip(kernel_got, kernel_want))
    assert not att.ready.any()
    att._active_front_qk = 2
    assert torch.all((att._front_seed()[:, 0] & 3) == 2)
    att._active_front_qk = 0
    q = reqs[0][0][lens[0] : lens[0] + 1]
    k = reqs[0][1][lens[0] : lens[0] + 1]
    v = reqs[0][2][lens[0] : lens[0] + 1]
    replays = {}
    wants = {}
    for mode in (0, 1, 2):
        att._front_qk = mode
        replays[mode] = att.prepare_replay_step(q, k, v, at=att.seq_lens - 1)
        wants[mode] = tuple(x.clone() for x in replays[mode]())
    assert all(torch.equal(x, y) for x, y in zip(wants[0], wants[1]))
    assert all(torch.equal(x, y) for x, y in zip(wants[0], wants[2]))
    att._front_qk = None
    replay = replays[0]
    torch.cuda.synchronize()
    assert not att.ready.any()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = replay()
    for x in got:
        x.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert got[0].abs().any()
    assert torch.isfinite(got[0]).all()
    assert not replay.ready.any()
    times = att.tune_split(candidates=(1,), rounds=1, reps=1)
    assert att.split == min(times, key=times.get)
    assert not att.ready.any()


@pytest.mark.parametrize(
    "B,S,H,HKV,D,v8,tail,depth,want",
    [
        (8, 1024, 8, 1, 128, False, None, 14.0, 0),
        (8, 1024, 8, 1, 128, True, None, 14.0, 0),
        (64, 8192, 8, 1, 128, True, None, 14.0, 2),
        (8, 30000, 8, 1, 128, True, None, 14.0, 2),
        (32, 16384, 16, 1, 128, True, None, 14.0, 0),
        (64, 8192, 16, 1, 128, True, None, 14.0, 2),
        (32, 16384, 16, 1, 64, True, None, 14.0, 1),
        (16, 30000, 32, 1, 128, True, None, 14.0, 1),
        (8, 1024, 8, 1, 128, False, "auto", 14.0, 0),
        (8, 24576, 8, 1, 128, False, "auto", 14.0, 0),
    ],
)
def test_front_quantization_ownership_dispatch(B, S, H, HKV, D, v8, tail, depth, want):
    att = FoldKVCache(B, H, HKV, D, 64, depth=depth, v8=v8, tail=tail)
    att.lens = [S] * B
    assert att._front_ownership() == want


def test_a_wide_group_decodes_under_its_mass_reference():
    lens, H, HKV, D = [700], 32, 2, 128
    reqs = _batch(lens, H, HKV, D, 2)
    att = FoldKVCache(1, H, HKV, D, 1024)
    _prefill(att, reqs, lens)
    for t in range(2):
        o = _step(att, reqs, lens, t)
        r, n = reqs[0], lens[0]
        ref = _attend(r[0][n + t : n + t + 1], r[1][: n + t + 1], r[2][: n + t + 1])[0]
        assert _l2(o[0], ref) < 4e-3


def test_the_cache_refuses_what_it_cannot_hold():
    att = FoldKVCache(2, 8, 1, 128, 256)
    q, k, v = _layer(300, 8, 1, 128)
    with pytest.raises(ValueError, match="not in 1..256"):
        att.prefill(q, k, v, torch.tensor([0, 10, 300], device="cuda", dtype=torch.int32))
    with pytest.raises(ValueError, match="batch of 2"):
        att.prefill(q, k, v, torch.tensor([0, 300], device="cuda", dtype=torch.int32))
    att.prefill(
        q[:512], k[:512], v[:512], torch.tensor([0, 256, 300], device="cuda", dtype=torch.int32)
    )
    with pytest.raises(ValueError, match="max_len=256"):
        att.append(k[:2], v[:2])


def _linear_v(reqs, HKV, D, seed=5):
    """V made linear in the rotated K through a rank-16 map plus a per-block
    offset, which is what the tail's rank term predicts exactly."""
    from fold_attention.utils import hadamard

    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.randn(HKV, D, 16, device="cuda", generator=g) / math.sqrt(D)
    Bm = torch.randn(HKV, 16, D, device="cuda", generator=g)
    out = []
    for q, k, v in reqs:
        kh = hadamard(k.float())
        lin = torch.einsum("nhd,hdr,hre->nhe", kh, A, Bm)
        n = k.shape[0]
        off = torch.randn(-(-n // 64), HKV, D, device="cuda", generator=g)
        off = off.repeat_interleave(64, 0)[:n]
        out.append((q, k, (lin + off).bfloat16()))
    return out


def test_the_serving_tail_keeps_every_block_row_current():
    """After a prefill and appends across block boundaries, every block's sums
    and row are the prompt's map applied to every key written so far. A key's
    projection is its plane-A integers against the map, exact, times its own
    `ek / 256`; the prompt sums a block's in f64 and the append adds each new
    one in f32, so the sums agree to rounding rather than to the bit."""
    from fold_attention.decode.cache import KBR, key_planes

    H, HKV, D, lens, extra = 16, 2, 128, [300, 700], 90
    reqs = _batch(lens, H, HKV, D, extra, seed=3)
    att = FoldKVCache(len(lens), H, HKV, D, 1024, depth=12.0)
    assert att.tail_rank == 16
    _prefill(att, reqs, lens)
    for t in range(extra):
        _step(att, reqs, lens, t)
    t_ = att.tail
    assert t_ is not None and t_.u is not None and t_.vr is not None and t_.ysum is not None
    for b, n in enumerate(lens):
        n += extra
        k = reqs[b][1][:n].float().transpose(0, 1)
        v = reqs[b][2][:n].float().transpose(0, 1)
        ka_, _, ek_ = key_planes(k)
        u = t_.u[b * HKV : (b + 1) * HKV].float()
        vr = t_.vr[b * HKV : (b + 1) * HKV]
        y = (ka_.float() @ u.transpose(1, 2)) * (ek_ * (1.0 / KBR))[..., None]
        for j in range(-(-n // 64)):
            pos0 = 64 * j
            pg = int(att.page_table[b, pos0 // att.page_size])
            for h in range(HKV):
                blk = ((pg * HKV + h) * att.page_size + pos0 % att.page_size) // 64
                sl = slice(pos0, min(n, pos0 + 64))
                cnt = sl.stop - sl.start
                vs, ys = v[h, sl].sum(0), y[h, sl].sum(0)
                # a block's terms can cancel, so the tolerance is their size's
                ys64 = y[h, sl].double().sum(0)
                mag = y[h, sl].double().abs().sum(0)
                err = (t_.ysum[blk].double() - ys64).abs()
                assert bool((err <= 1e-6 * mag + 1e-6).all()), (b, j, h)
                assert torch.allclose(t_.vsum[blk], vs, rtol=1e-5, atol=1e-4), (b, j, h)
                row = vs / cnt - (ys / cnt) @ vr[h].T
                got = t_.rows[blk].float()
                assert float((got - row).abs().max()) <= 2**-7 * float(row.abs().max()), (b, j, h)


def test_the_serving_tail_recovers_what_its_map_predicts():
    """On a stream whose V is its K's image plus a block offset, a truncated
    step with the tail lands on the full softmax; with one V mean it does not."""
    H, HKV, D, lens, extra = 16, 2, 128, [1500, 1100], 40
    reqs = _linear_v(_batch(lens, H, HKV, D, extra, seed=9), HKV, D)
    errs = {}
    for name, tail in (("tail", "auto"), ("mean", None)):
        att = FoldKVCache(len(lens), H, HKV, D, 2048, depth=6.0, tail=tail)
        _prefill(att, reqs, lens)
        e = []
        for t in range(extra):
            o = _step(att, reqs, lens, t)
            ref = torch.stack(
                [
                    _attend(r[0][n + t : n + t + 1], r[1][: n + t + 1], r[2][: n + t + 1])[0]
                    for r, n in zip(reqs, lens)
                ]
            )
            e.append(_l2(o, ref))
        errs[name] = sum(e) / len(e)
    assert errs["tail"] < 0.3 * errs["mean"], errs


def test_the_serving_tail_is_refused_where_it_cannot_run():
    with pytest.raises(ValueError, match="bf16 V cache"):
        FoldKVCache(1, 8, 1, 128, 512, depth=12.0, v8=True, tail=16)
    with pytest.raises(ValueError, match="truncates nothing"):
        FoldKVCache(1, 8, 1, 128, 512, depth=None, tail=16)
    assert FoldKVCache(1, 8, 1, 128, 512, depth=12.0, v8=True).tail_rank == -1
    assert FoldKVCache(1, 8, 1, 128, 512).tail_rank == -1


def test_a_prepared_step_owns_the_temporaries_it_was_bound_to():
    """The step kernel takes raw addresses; a scratch tensor passed inline
    must stay alive through the allocator's churn, and take the step's sums."""
    import gc

    from fold_attention.decode import write as fstep

    H, HKV, D, lens = 16, 2, 128, [700, 900]
    reqs = _batch(lens, H, HKV, D, 1, seed=4)
    att = FoldKVCache(len(lens), H, HKV, D, 1024, depth=12.0)
    _prefill(att, reqs, lens)
    q = torch.stack([r[0][n] for r, n in zip(reqs, lens)])
    k = torch.stack([r[1][n] for r, n in zip(reqs, lens)])
    v = torch.stack([r[2][n] for r, n in zip(reqs, lens)])
    before = att.vsum.clone()
    assert att.tail is not None
    st = fstep.prepare_step(
        att.qa.view(-1, D),
        att.qb.view(-1, D),
        att.eq.view(-1),
        att.ek,
        att.page_table,
        att.page_size,
        att.seq_lens.clone(),
        att.ka,
        att.kb,
        att.va,
        att.vb,
        att.v_scale,
        att.vsum.clone(),
        att.vmean.clone(),
        torch.empty_like(att.seq_lens),
        heads=H,
        n_kv_heads=HKV,
        dtype=q.dtype,
        tail=dataclasses.replace(
            att.tail, **{n: x.clone() for n, x in vars(att.tail).items() if x is not None}
        ),
    )
    gc.collect()
    junk = [torch.full((1 << 20,), 7, device="cuda", dtype=torch.int32) for _ in range(64)]
    st(q.reshape(-1, D), k, v)
    torch.cuda.synchronize()
    del junk
    vsum = st.held[10]
    assert torch.allclose(vsum, before + v.float().reshape(-1, D), rtol=1e-5, atol=1e-4)
    assert torch.equal(att.vsum, before)


def test_the_functional_entry_point_is_the_cache_method():
    """`fold_attn_with_kvcache` is `FoldKVCache.decode` with a token and
    `attend` without one, in FlashAttention's `(batch, 1, heads, dim)` layout
    as well as without the sequence axis."""
    lens, H, HKV, D = [300, 500], 16, 2, 128
    reqs = _batch(lens, H, HKV, D, 2)
    a = FoldKVCache(2, H, HKV, D, 1024, depth=14.0, split=2)
    b = FoldKVCache(2, H, HKV, D, 1024, depth=14.0, split=2)
    _prefill(a, reqs, lens)
    _prefill(b, reqs, lens)
    for t in range(2):
        q = torch.stack([r[0][n + t] for r, n in zip(reqs, lens)])
        k = torch.stack([r[1][n + t] for r, n in zip(reqs, lens)])
        v = torch.stack([r[2][n + t] for r, n in zip(reqs, lens)])
        want = a.decode(q, k, v)
        got = fold_attn_with_kvcache(q[:, None], b, k[:, None], v[:, None])
        assert got.shape == (2, 1, H, D)
        assert torch.equal(got[:, 0], want)
    assert torch.equal(fold_attn_with_kvcache(q, b), a.attend(q))


@pytest.mark.parametrize("lens", [[900, 1500], [4000, 7000]])
def test_the_coarse_logits_give_the_kernels_refine_count(lens):
    """`coarse_logits` and `refine_gates` are the kernel's screen: the keys
    whose group-largest coarse logit clears each request's own gate are the
    keys the step refined, as the kernel counted them."""
    H, HKV, D = 8, 2, 128
    reqs = _batch(lens, H, HKV, D, 2, seed=3)
    att = FoldKVCache(len(lens), H, HKV, D, 8192, v8=False, split=2)
    _prefill(att, reqs, lens)
    for t in range(2):
        _step(att, reqs, lens, t)
    s = att.coarse_logits()
    B, G = len(lens), H // HKV
    rel = s - att.z.view(B, HKV, G, 1)
    rk, _ = att.refine_gates()
    refined = (rel.amax(2) >= -rk.to(rel.device)[:, None, None]).sum()
    counted = att.counts[:, 1].sum()
    assert abs(int(refined) - int(counted)) <= 2e-4 * sum(att.lens) * HKV, (refined, counted)
