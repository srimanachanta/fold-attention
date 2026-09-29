"""The two-plane decode kernel: does it compute the function it claims to.

Three properties carry the whole design, and each is testable independently of
performance:

* the truncated softmax **is** the function, so a declined key contributes
  exactly zero and the live count is a property of the data, not the schedule;
* the answer must not depend on the schedule at all -- tile width, warp count
  and the split over keys are free choices, and each is a place a kernel can
  quietly go wrong;
* the second cache plane is a precision gate, so loosening it may only reduce
  error.

Each positive claim is paired with a control that can fail, because a positive
that cannot fail is a measurement rather than a test.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

cute_available = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
if cute_available:
    from fold_attention.decode import (
        SharedPrefix,
        capture_decode,
        fold_decode,
        pick_config,
        pick_split,
        prefix_image,
        prepare_fold_decode,
    )
    from fold_attention.decode.cache import (
        KBR,
        Tail,
        key_planes,
        pad_scales,
        paged_cache,
        paged_scale,
        quantize_k,
        quantize_kq,
        quantize_v,
        swizzle_of,
        swizzle_rows,
        tail_model,
        write_kv,
    )
    from fold_attention.decode.config import smem_bytes
    from fold_attention.decode.heuristics import resident, v8_live_fraction, weight_terms_for
    from fold_attention.decode.launch import _prepare
    from fold_attention.decode.rows import (
        cascade_degree,
        cascade_rows,
        draft_mask,
        pack_rows,
        pack_tree,
        unpack_rows,
    )
    from fold_attention.utils import hadamard, rotation

pytestmark = pytest.mark.skipif(not cute_available, reason="needs an SM90 GPU")

LOG2E = 1.4426950408889634


def _case(NBH=8, S=1024, D=128, G=8, seed=0, alpha=30.0, vmu=3.6, device="cuda", tchk=8.0):
    """A cache whose logits are heavy-tailed the way a real one's are.

    `randn` alone gives a one-binade spread and the truncation never fires.
    The spread has to come from a few keys aligning with the query, which is
    what a real cache does: inflating |k| instead widens the logits while
    leaving every dot product just as ill-conditioned, and then the coarse
    plane's error in units of the logit is large enough to flip liveness near
    the cut.

    V needs the same care. `randn` has zero mean, so the dropped-mass
    correction would place the mass it recovers at the origin and read as a
    no-op. A real V is far from centred: `||vbar|| / ||v||` is 0.18-0.40 per
    head on real layers, and `vmu` sets the generator at 0.30.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(NBH, G, D, device=device, generator=g)
    k = torch.randn(NBH, S, D, device=device, generator=g)
    v = torch.randn(NBH, S, D, device=device, generator=g)
    m = torch.randn(NBH, 1, D, device=device, generator=g)
    v = (v + vmu * m / m.norm(dim=-1, keepdim=True)).contiguous()
    u = q.mean(1)
    u = u / u.norm(dim=-1, keepdim=True)
    w = torch.rand(NBH, S, 1, device=device, generator=g) ** 6
    k = (k + alpha * w * u[:, None, :]).contiguous()
    qs = q * LOG2E / math.sqrt(D)
    s = torch.einsum("bgd,bsd->bgs", qs, k)
    z = torch.ceil(s.max(-1).values * 1024) / 1024
    # the guard is against the two degenerate ends, not a tight target: with
    # few keys the row max is smaller, the cut sits lower, and more survives
    if tchk is not None:
        live = (s >= (z - tchk)[..., None]).any(1).float().mean().item()
        assert 0.02 < live < 0.95, f"case has no usable live set: {live}"
    return qs, k, v, s, z


Planes = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _pack(
    qs, k, v
) -> tuple[Planes, Planes, tuple[torch.Tensor, torch.Tensor, float], torch.Tensor]:
    qa, qb, eq = quantize_kq(qs)
    ka, kb, ek = quantize_k(k)
    va, vbp, evs = quantize_v(v)
    return (qa, qb, eq), (ka, kb, ek), (va, vbp, evs), v.bfloat16().contiguous()


def _slack(qq, kk):
    """The bound the kernel shifts its cut by.

    The screen sees plane A alone, which omits K's second plane -- up to
    `ek / 2` a channel -- so the cut it compares against is the caller's less
    `(ek / 2) sum_d |q_d|`, per key. Written the way the kernel accumulates
    it, from the quantised Q the matmul actually uses.
    """
    qa, qb, eq = qq
    ek = kk[2].float()
    a1 = qa.float().abs().sum(-1)
    b1 = qb.float().abs().sum(-1)
    return (eq[..., None] * (ek[:, None, :] / KBR)) * (128.0 * a1 + 0.5 * b1)[..., None]


def _s16(qs, k):
    """The 16-bit logit `256 c1 + c2 + c3 + c4 / 256` the kernel's liveness is
    defined on.

    Not the f32 one: the certificate is that no key the full plane-A/B path
    calls live is dropped, and that path is this.
    """
    qa, qb, eq = quantize_kq(qs)
    ka, kb, ek = key_planes(k)
    i = (
        KBR * torch.einsum("bgd,bsd->bgs", qa.float(), ka.float())
        + torch.einsum("bgd,bsd->bgs", qb.float(), ka.float())
        + torch.einsum("bgd,bsd->bgs", qa.float(), kb.float())
        + torch.einsum("bgd,bsd->bgs", qb.float(), kb.float()) / KBR
    )
    return i * (eq[..., None] * (ek[:, None, :] / KBR))


def _ref(s, z, v, cut):
    p = torch.where(s < cut[..., None], torch.zeros_like(s), torch.exp2(s - z[..., None]))
    return (p @ v) / p.sum(-1, keepdim=True), p


def _run(qq: Planes, kk: Planes, vv, vb, z, cut, *, v8=False, **kw):
    """One decode over the packed case, with the build knobs tests reach for."""
    va, vbp, evs = vv
    return _prepare(
        *qq,
        *kk,
        va if v8 else vb,
        z,
        cut,
        v2=vbp if v8 else None,
        v_scale=evs if v8 else 1.0,
        v8=v8,
        **kw,
    )()


@pytest.mark.parametrize("v8", [False, True])
@pytest.mark.parametrize("T", [1e4, 8.0])
def test_matches_the_truncated_softmax(v8, T):
    qs, k, v, s, z = _case()
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - float(T)
    ref, _ = _ref(s, z, v, cut)
    o, _, _ = _run(qq, kk, vv, vb, z, cut, v8=v8, truncate=0 if T > 1e3 else 1, refine_k=16.0)
    err = (o - ref).abs().max().item() / ref.abs().max().item()
    assert err < 2e-2, err


def test_an_fp16_v_is_refused():
    """The weight buffer takes V's element type, and a weight is 2^(s - Z). An
    f16 one overflows 16 binades above the reference, which the mass reference
    has been measured missing a peak by 15.5 of, so the cache is bf16 only."""
    qs, k, v, _, z = _case()
    qq, kk, _, vb = _pack(qs, k, v)
    with pytest.raises(ValueError, match="overflows"):
        fold_decode(*qq, *kk, vb.to(torch.float16), z, z - 1e4, truncate=0, refine_k=16.0)


def test_unquantized_v_refuses_a_wider_cache():
    qs, k, v, _, z = _case(S=256)
    qq, kk, _, _ = _pack(qs, k, v)
    with pytest.raises(ValueError, match="must be bf16 or fp16"):
        prepare_fold_decode(*qq, *kk, v.float(), z, z - 8.0)


def test_control_a_wrong_reference_fails():
    """The pairing control: the same kernel against a reference it does not
    compute must fail, or the tolerance above is meaningless."""
    qs, k, v, s, z = _case()
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    ref_full, _ = _ref(s, z, v, z - 1e4)
    o, _, _ = _run(qq, kk, vv, vb, z, cut, truncate=1, refine_k=16.0)
    err = (o - ref_full).abs().max().item() / ref_full.abs().max().item()
    assert err > 2e-2, (
        "truncation at T=8 changed nothing, so the case has no "
        f"live-set spread and the accuracy tests are vacuous: {err}"
    )


@pytest.mark.parametrize("sound", [0, 1])
def test_declined_keys_are_exactly_zero(sound):
    """A key below the cut -- and below its slack, where the screen is sound --
    must contribute nothing, so perturbing V there may not move a single bit
    of the output."""
    qs, k, v, s, z = _case()
    cut = z - 8.0
    qq, kk, vv, vb = _pack(qs, k, v)
    lim = cut[..., None] - _slack(qq, kk) if sound else cut[..., None]
    dead = (s < lim).all(1)
    assert 0.05 < dead.float().mean().item() < 0.99, dead.float().mean().item()
    # bounded, because `quantize_v` takes one amax over the whole tensor: a
    # perturbation that moved it would rescale the live rows too
    v2 = v.clone()
    v2[dead] = -v2[dead]
    qq2, kk2, vv2, vb2 = _pack(qs, k, v2)
    a = _run(qq, kk, vv, vb, z, cut, truncate=1, refine_k=16.0, sound=sound)[0]
    b = _run(qq2, kk2, vv2, vb2, z, cut, truncate=1, refine_k=16.0, sound=sound)[0]
    assert torch.equal(a, b), (a - b).abs().max().item()


def test_the_screen_is_sound_only_when_asked():
    """Under `sound=1` no key the full two-plane path calls live is dropped.

    The default screen compares a plane-A estimate to the cut, and that
    estimate differs from the 16-bit logit by up to `ek / 2` a channel, so its
    error is two-sided: on this case it drops up to 2 live keys a head and
    keeps a different 1-2 that are not. `sound=1` shifts the cut by that
    bound, which makes the error one-sided -- at the cost of the keys the
    shift buys, worth 0.8-1.2 binades on the captures, which is why it is not
    the default. The second assert is the control: without it this test would
    pass on a kernel where the shift did nothing.
    """
    qs, k, v, _, z = _case()
    cut = z - 8.0
    qq, kk, vv, vb = _pack(qs, k, v)
    want = (_s16(qs, k) >= cut[..., None]).any(1).sum(-1).to(torch.int32)
    kw = dict(truncate=1, refine_k=16.0)
    got = _run(qq, kk, vv, vb, z, cut, sound=1, **kw)[2][:, 0]
    assert (got >= want).all(), (got - want).min().item()
    bare = _run(qq, kk, vv, vb, z, cut, **kw)[2][:, 0]
    assert not (bare >= want).all(), "the default screen is already sound"


@pytest.mark.parametrize("v8", [False, True])
@pytest.mark.parametrize("split", [1, 2, 3, 4, 5, 6, 7])
def test_split_is_invariant_including_partial_chunks(v8, split):
    """The key split is a schedule, so it may not move the answer.

    A split whose last chunk is partial leaves the final tiles empty, where an
    mbarrier wait with no arrive would hang where `cp_async_wait_group` on
    nothing returns.
    """
    qs, k, v, _s, z = _case(S=2048)
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    base = _run(qq, kk, vv, vb, z, cut, v8=v8, truncate=1, split=1, refine_k=16.0)[0]
    got = _run(qq, kk, vv, vb, z, cut, v8=v8, truncate=1, split=split, refine_k=16.0)[0]
    rel = (got - base).abs().max().item() / base.abs().max().item()
    assert rel < 1e-5, rel


def test_both_key_counts_are_the_data_not_the_schedule():
    qs, k, v, s, z = _case(S=2048)
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    refc = 16.0
    counts = []
    for sp in (1, 4, 3):
        counts.append(_run(qq, kk, vv, vb, z, cut, truncate=1, split=sp, refine_k=refc)[2])
    for c in counts[1:]:
        assert torch.equal(c, counts[0]), "a key set moved with the schedule"
    # and they are the right sets: the coarse plane decides both, so either may
    # differ from the exact logit's only by keys within a quantisation step of
    # its threshold. Plane B is gathered only for a key some row keeps, so the
    # refined set is the live keys the refine gate also passes.
    live = (s >= cut[..., None]).any(1)
    for col, want in ((0, live), (1, live & (s >= (z - refc)[..., None]).any(1))):
        want = want.sum(-1).float()
        rel = (counts[0][:, col].float() - want).abs().max().item() / want.mean().item()
        assert rel < 0.05, (col, rel)
    assert (counts[0][:, 1] <= counts[0][:, 0]).all()


@pytest.mark.parametrize("v8", [False, True])
def test_a_looser_refine_gate_only_helps(v8):
    """`refine_k`/`refine_v` gate precision, never membership, so relaxing them
    may not make the answer worse."""
    qs, k, v, s, z = _case()
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 1e4
    ref, _ = _ref(s, z, v, cut)
    rn = ref.abs().max().item()
    errs = []
    for refc in (2.0, 8.0, 16.0, 32.0):
        # the terms the rule picks for this depth: at one term a bf16 weight's
        # rounding is the floor and it hides what the gate does
        o = _run(
            qq,
            kk,
            vv,
            vb,
            z,
            cut,
            v8=v8,
            truncate=0,
            refine_k=refc,
            weight_terms=weight_terms_for(None),
        )[0]
        errs.append((o - ref).abs().max().item() / rn)
    assert errs[-1] <= errs[0] * 1.05, errs
    # The bf16 V is limited by the logit, so refining K cuts its error 1.8x
    # here even though a key's own scale leaves plane A alone close. The 8-bit
    # V floors on its own e4m3 rounding (measured 2.2e-3 here against the bf16
    # arm's 1.5e-3) and cannot reach that however exact the logit is, so it
    # only has to improve clearly (1.24x measured).
    want = 1.15 if v8 else 1.5
    assert errs[-1] < errs[0] / want, (
        f"the second plane changed almost nothing, so the gate is not being exercised: {errs}"
    )


def test_swizzle_is_an_involution():
    x = torch.randint(-100, 100, (4, 64, 128), device="cuda", dtype=torch.int8).view(
        torch.float8_e4m3fn
    )
    once = swizzle_rows(x)
    assert not torch.equal(once.view(torch.int8), x.view(torch.int8))
    twice = swizzle_rows(once)
    assert torch.equal(twice.view(torch.int8), x.view(torch.int8))


@pytest.mark.parametrize("nbh,s", [(1, 512), (3, 640), (8, 1088)])
def test_odd_shapes(nbh, s):
    """S that is not a multiple of the tile, and a batch of one."""
    qs, k, v, sc, z = _case(NBH=nbh, S=s)
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    ref, _ = _ref(sc, z, v, cut)
    o = _run(qq, kk, vv, vb, z, cut, truncate=1, split=2, refine_k=16.0)[0]
    err = (o - ref).abs().max().item() / ref.abs().max().item()
    assert err < 2e-2, err


def test_bad_arguments_are_rejected():
    qs, k, v, _s, z = _case(NBH=1, S=512)
    qq, kk, vv, _vb = _pack(qs, k, v)
    with pytest.raises(ValueError):
        fold_decode(*qq, *kk, vv[0], z, z - 8.0, v8=True, v2=None)


def test_pick_config_follows_the_floor_rule():
    assert pick_config(1.0)[0] is True, "dense must take the 8-bit V"
    assert pick_config(0.02)[0] is False, "a sparse live set must not"
    # monotone in the live fraction, and total
    seen = [pick_config(x / 20.0)[0] for x in range(21)]
    assert seen == sorted(seen), seen
    # a narrow group crosses far earlier than the wide-group fit predicts, which
    # is worth 7-11% at depths 14-16
    for D, cross in ((128, 0.11), (64, 0.28)):
        assert pick_config(cross, D=D, G=8)[0] is True, D
        assert pick_config(cross, D=D, G=16)[0] is False, D


def _vmean(v, vv, v8):
    """V's mean in the units the kernel's accumulator holds."""
    return (v.mean(1).float() / vv[2]) if v8 else v.mean(1).float()


@pytest.mark.parametrize("v8", [False, True])
def test_the_correction_restores_the_dropped_mass(v8):
    """Truncation loses the direction of the mass below the cut, not the mass:
    the kernel reads plane A for every key, so it holds every term of that sum.
    Putting it back at the block mean must move the answer towards the *full*
    softmax, which is the thing the truncation is an approximation of.
    """
    qs, k, v, s, z = _case()
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    full, _ = _ref(s, z, v, z - 1e4)
    rn = full.abs().max().item()

    def err(o):
        return (o - full).abs().max().item() / rn

    a = err(_run(qq, kk, vv, vb, z, cut, v8=v8, truncate=1, refine_k=16.0)[0])
    b = err(
        _run(qq, kk, vv, vb, z, cut, v8=v8, truncate=1, refine_k=16.0, vmean=_vmean(v, vv, v8))[0]
    )
    # the control: without it the truncation is visibly wrong, so the
    # comparison below is not two numbers that are both already zero
    assert a > 5e-3, f"nothing to correct: {a}"
    assert b < a / 3.0, (a, b)


@pytest.mark.parametrize("v8", [False, True])
def test_dropped_mass_accepts_a_direction_per_query_row(v8):
    qs, k, v, _, z = _case()
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    vm = _vmean(v, vv, v8)
    shared = _run(qq, kk, vv, vb, z, cut, truncate=1, v8=v8, vmean=vm)[0]
    rows = _run(
        qq,
        kk,
        vv,
        vb,
        z,
        cut,
        truncate=1,
        v8=v8,
        vmean=vm[:, None, :].expand(-1, qs.shape[1], -1).contiguous(),
    )[0]
    assert torch.equal(shared, rows)


@pytest.mark.parametrize("v8", [False, True])
def test_the_corrected_denominator_is_the_total_mass(v8):
    """The correction's first half is exact: the returned denominator stops
    being the live mass and becomes the true total, which is checkable against
    the full softmax rather than against the kernel itself."""
    qs, k, v, s, z = _case()
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    total = torch.exp2(s - z[..., None]).sum(-1)
    lo = _run(qq, kk, vv, vb, z, cut, v8=v8, truncate=1, refine_k=16.0)[1]
    lc = _run(qq, kk, vv, vb, z, cut, v8=v8, truncate=1, refine_k=16.0, vmean=_vmean(v, vv, v8))[1]
    assert ((lc - total).abs() / total).max().item() < 1e-2
    # and the control: the uncorrected one is the live mass, and the two must
    # actually differ, or the assertion above is vacuous
    assert ((lo - total).abs() / total).max().item() > 2e-3


@pytest.mark.parametrize("split", [1, 3, 4])
def test_the_correction_survives_the_split(split):
    """Both the dropped mass and its image under the mean are additive, so a
    split may sum them the way it sums everything else."""
    qs, k, v, _s, z = _case(S=2048)
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    vm = _vmean(v, vv, False)
    base = _run(qq, kk, vv, vb, z, cut, truncate=1, split=1, refine_k=16.0, vmean=vm)[0]
    got = _run(qq, kk, vv, vb, z, cut, truncate=1, split=split, refine_k=16.0, vmean=vm)[0]
    assert (got - base).abs().max().item() / base.abs().max().item() < 1e-5


def test_the_correction_is_refused_where_there_is_nothing_to_correct():
    qs, k, v, _s, z = _case()
    qq, kk, vv, vb = _pack(qs, k, v)
    with pytest.raises(ValueError, match="nothing to correct"):
        _run(qq, kk, vv, vb, z, z - 1e4, truncate=0, vmean=v.mean(1).float())


@pytest.mark.parametrize("split", [1, 2, 3, 4])
def test_the_warpgroup_path_is_split_invariant(split):
    """Inside the warpgroup path the answer is schedule again: only the matmul
    instruction moves it, and the split does not change that."""
    qs, k, v, _s, z = _case(S=2048)
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    vm = _vmean(v, vv, False)
    kw = dict(truncate=1, refine_k=16.0, vmean=vm)
    base = _run(qq, kk, vv, vb, z, cut, split=1, **kw)[0]
    got = _run(qq, kk, vv, vb, z, cut, split=split, **kw)[0]
    assert (got - base).abs().max().item() / base.abs().max().item() < 1e-5


def test_the_warpgroup_tile_is_visible_to_the_descriptor():
    """A warpgroup matmul reads its operands through the *async* proxy.

    The value tile and the weight buffer are written by ordinary shared stores
    and by `cp.async`, which are the generic proxy, and `bar.sync` orders the
    two threads rather than the two proxies: without
    `fence.proxy.async.shared::cta` a descriptor can read a stale tile. It does
    so rarely and only when CTAs share an SM -- about one block in a hundred at
    four per SM and never at one per SM -- so the grid here has to be big enough
    for that to happen, and the check that bites is determinism, not accuracy.
    """
    qs, k, v, s, z = _case(NBH=256, S=1024)
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    ref, _ = _ref(s, z, v, cut)
    kw = dict(truncate=1, refine_k=12.0, split=1)
    a = _run(qq, kk, vv, vb, z, cut, **kw)[0]
    b = _run(qq, kk, vv, vb, z, cut, **kw)[0]
    assert torch.equal(a, b), (a - b).abs().max().item()
    assert (a - ref).abs().max().item() / ref.abs().max().item() < 2e-2


def test_the_value_matmul_accumulates_over_tiles():
    """A warpgroup matmul *writes* its accumulator unless it is told otherwise,
    and this one carries the whole KV loop. One tile cannot see the difference;
    the control is a case with many."""
    err = {}
    for S in (64, 2048):
        qs, k, v, s, z = _case(NBH=8, S=S)
        qq, kk, vv, vb = _pack(qs, k, v)
        cut = z - 1e4
        ref, _ = _ref(s, z, v, cut)
        got = _run(qq, kk, vv, vb, z, cut, truncate=0, split=1)[0]
        err[S] = (got - ref).abs().max().item() / ref.abs().max().item()
    # a matmul that overwrote would keep only the last tile: 1.5e-1 at two
    # tiles, 1.0 at thirty-two
    assert err[64] < 2e-2, err
    assert err[2048] < 2e-2, err
    assert err[2048] < 4.0 * err[64], err


def test_captured_prepared_decode_replays_real_work():
    qs, k, v, _, z = _case(NBH=8, S=1024)
    qq, kk, _, vb = _pack(qs, k, v)
    args = (*qq, *kk, vb, z, z - 8.0)
    eager = prepare_fold_decode(*args, truncate=1, refine_k=16.0, split=2)
    want = tuple(x.clone() for x in eager())
    captured = capture_decode(eager)
    got = captured()
    torch.cuda.synchronize()
    assert all(torch.equal(x, y) for x, y in zip(got, want))
    for x in got:
        x.zero_()
    captured()
    torch.cuda.synchronize()
    assert all(torch.equal(x, y) for x, y in zip(got, want))


def _paged(qs, k, v, S, page_size, hkv, perm=True):
    """The same cache, scattered into pages in a shuffled order."""
    NBH = k.shape[0]
    B = NBH // hkv
    npages = (S + page_size - 1) // page_size
    tot = B * npages
    order = torch.randperm(tot, device=k.device) if perm else torch.arange(tot, device=k.device)
    pt = order.reshape(B, npages).int()
    qq, kk, vv, vb = _pack(qs, k, v)
    pka, pkb, pva, pvb = paged_cache(kk[0], kk[1], vv[0], vv[1], pt, page_size, hkv)
    pek = paged_scale(kk[2], pt, page_size, hkv)
    return qq, kk, vv, vb, pt, (pka, pkb, pva, pvb, pek)


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("group_cut", [0, 1])
@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
def test_a_paged_cache_is_the_same_bits(page_size, group_cut, D):
    """Paging moves an address, not a layout.

    A tile is still `BN` contiguous rows of one page, the shared tile it lands
    in is the same object, and the swizzle survives because its period is eight
    and a page holds a multiple of eight slots. So the paged kernel must agree
    with the contiguous one *bitwise*, not merely to a tolerance -- there is no
    instruction between them that differs.

    At D=64 the swizzle's period is still eight rows -- a row *pair* shares one
    XOR -- so a page still holds a multiple of eight slots and the argument is
    unchanged. It is covered at both head dims rather than at the one the
    cache was captured at.
    """
    S, hkv = 512, 2
    qs, k, v, _s, z = _case(S=S, NBH=8, D=D)
    qq, kk, vv, vb, pt, (pka, pkb, pva, pvb, pek) = _paged(qs, k, v, S, page_size, hkv)
    cut = z - 8.0
    sl = torch.full((pt.shape[0],), S, device=k.device, dtype=torch.int32)
    kw = dict(truncate=1, refine_k=16.0, split=2, v8=True, group_cut=group_cut)
    a = _run(qq, kk, vv, vb, z, cut, **kw)[0]
    b = fold_decode(
        qq[0],
        qq[1],
        qq[2],
        pka,
        pkb,
        pek,
        pva,
        z,
        cut,
        v2=pvb,
        v_scale=vv[2],
        page_table=pt,
        page_size=page_size,
        seq_lens=sl,
        n_kv_heads=hkv,
        **kw,
    )[0]
    assert torch.equal(a, b), (a - b).abs().max().item()


def test_the_page_order_is_what_the_table_says():
    """The control: a shuffled page table and an identity one cannot agree, or
    the test above is only checking that both kernels read *something*."""
    S, hkv, page_size = 512, 2, 64
    qs, k, v, _s, z = _case(S=S, NBH=8)
    qq, _kk, vv, _vb, pt, pc = _paged(qs, k, v, S, page_size, hkv, perm=True)
    cut = z - 8.0
    sl = torch.full((pt.shape[0],), S, device=k.device, dtype=torch.int32)
    kw = dict(
        truncate=1,
        refine_k=16.0,
        split=2,
        v8=True,
        page_size=page_size,
        seq_lens=sl,
        n_kv_heads=hkv,
    )
    good = fold_decode(
        qq[0],
        qq[1],
        qq[2],
        pc[0],
        pc[1],
        pc[4],
        pc[2],
        z,
        cut,
        v2=pc[3],
        v_scale=vv[2],
        page_table=pt,
        **kw,
    )[0]
    bad = fold_decode(
        qq[0],
        qq[1],
        qq[2],
        pc[0],
        pc[1],
        pc[4],
        pc[2],
        z,
        cut,
        v2=pc[3],
        v_scale=vv[2],
        page_table=torch.arange(pt.numel(), device=k.device, dtype=torch.int32).reshape(pt.shape),
        **kw,
    )[0]
    assert not torch.equal(good, bad)


@pytest.mark.parametrize("D", [64, 128])
def test_a_ragged_batch_matches_one_request_at_a_time(D):
    """Each request's own length, not the longest in the batch."""
    S, hkv, page_size = 512, 2, 64
    qs, k, v, _s, z = _case(S=S, NBH=8, D=D)
    B = k.shape[0] // hkv
    lens = torch.tensor(
        [S, S - 3 * page_size, S - page_size, 2 * page_size][:B], device=k.device, dtype=torch.int32
    )
    qq, _kk, vv, _vb, pt, pc = _paged(qs, k, v, S, page_size, hkv)
    cut = z - 8.0
    kw = dict(truncate=1, refine_k=16.0, split=2, v8=True, page_size=page_size, n_kv_heads=hkv)
    o = fold_decode(
        qq[0],
        qq[1],
        qq[2],
        pc[0],
        pc[1],
        pc[4],
        pc[2],
        z,
        cut,
        v2=pc[3],
        v_scale=vv[2],
        page_table=pt,
        seq_lens=lens,
        **kw,
    )[0]
    for b in range(B):
        n = int(lens[b])
        uni = torch.full((B,), n, device=k.device, dtype=torch.int32)
        ref = fold_decode(
            qq[0],
            qq[1],
            qq[2],
            pc[0],
            pc[1],
            pc[4],
            pc[2],
            z,
            cut,
            v2=pc[3],
            v_scale=vv[2],
            page_table=pt,
            seq_lens=uni,
            **kw,
        )[0]
        rows = slice(b * hkv, (b + 1) * hkv)
        assert torch.equal(o[rows], ref[rows]), b


@pytest.mark.parametrize("D", [64, 128])
def test_the_cache_writer_round_trips(D):
    """`write_kv` must put a new token exactly where `paged_cache` would have.

    The rotation is off on both sides: it is one D x D matmul whose fp result
    depends on the batch shape, so a tie in `round` can land either way and the
    placement -- page, slot and the sixteen-byte swizzle -- is what this is
    testing. The rotated path is checked by reconstruction below.
    """
    S, hkv, page_size = 256, 2, 64
    _qs, k, v, _s, _z = _case(S=S, NBH=4, D=D)
    B = k.shape[0] // hkv
    npages = S // page_size
    pt = torch.randperm(B * npages, device=k.device).reshape(B, npages).int()
    ka, kb, ek = quantize_k(k, rot=False)
    va, vbp, evs = quantize_v(v)
    pka, pkb, pva, pvb = paged_cache(ka, kb, va, vbp, pt, page_size, hkv)
    pek = paged_scale(ek, pt, page_size, hkv)
    zk = [torch.zeros_like(x) for x in (pka, pkb, pva, pvb)]
    zek = torch.zeros_like(pek)
    for pos in range(S):
        p = torch.full((B,), pos, device=k.device, dtype=torch.int64)
        kn = k.reshape(B, hkv, S, -1)[:, :, pos]
        vn = v.reshape(B, hkv, S, -1)[:, :, pos]
        write_kv(zk[0], zk[1], zek, zk[2], zk[3], kn, vn, evs, pt, p, page_size, hkv, rot=False)
    for a, b in zip(zk, (pka, pkb, pva, pvb)):
        assert torch.equal(a.view(torch.int8), b.view(torch.int8))
    assert torch.equal(zek, pek)


def test_the_cache_writer_reconstructs_the_rotated_key():
    """With the rotation on, the two planes the writer lays down must be the
    key itself to the grid's own resolution."""
    S, hkv, page_size = 128, 2, 64
    _qs, k, v, _s, _z = _case(S=S, NBH=4)
    B, D = k.shape[0] // hkv, k.shape[-1]
    pt = torch.randperm(B * (S // page_size), device=k.device).reshape(B, S // page_size).int()
    ka, kb, ek = quantize_k(k)
    va, vbp, evs = quantize_v(v)
    zk = [torch.zeros_like(x) for x in paged_cache(ka, kb, va, vbp, pt, page_size, hkv)]
    zek = torch.zeros_like(paged_scale(ek, pt, page_size, hkv))
    for pos in range(S):
        p = torch.full((B,), pos, device=k.device, dtype=torch.int64)
        write_kv(
            zk[0],
            zk[1],
            zek,
            zk[2],
            zk[3],
            k.reshape(B, hkv, S, -1)[:, :, pos],
            v.reshape(B, hkv, S, -1)[:, :, pos],
            evs,
            pt,
            p,
            page_size,
            hkv,
        )
    # undo the page map and the swizzle, then the two planes are the key
    rows = (pt[:, :, None] * hkv + torch.arange(hkv, device=k.device)[None, None, :])[
        ..., None
    ] * page_size + torch.arange(page_size, device=k.device)
    rows = rows.permute(0, 2, 1, 3).reshape(B * hkv, S)
    u = (
        torch.arange(D // 16, device=k.device)[None, :]
        ^ (torch.arange(S, device=k.device) & 7)[:, None]
    )
    idx = u.view(1, S, D // 16, 1).expand(B * hkv, S, D // 16, 16)
    got = 0.0
    for plane, sc in ((zk[0], 1.0), (zk[1], 1.0 / 256.0)):
        g = plane.view(torch.int8)[rows.reshape(-1)].reshape(B * hkv, S, D)
        g = torch.gather(g.reshape(B * hkv, S, D // 16, 16), 2, idx)
        got = got + g.reshape(B * hkv, S, D).float() * sc
    got = got * ek.float()[..., None]
    want = k.float() @ rotation(D, k.device).T
    assert (got - want).abs().max().item() < 2e-3 * want.abs().max().item()


@pytest.mark.parametrize("G", [8, 16, 24, 32])
def test_the_query_group_rides_one_warpgroup_matmul(G):
    """N is the query group rounded up to eight, carried by one instruction.

    A warpgroup matmul's C at N = 8k is k blocks of the m64n8 pattern in order,
    so register `ng * 4 + i` is the pair the N = 8 form put at `[ng][i]` and the
    verdict, the refine and the epilogue all index it the same way. What must
    hold is that it computes the function, at every group the shape allows.
    """
    T = 8.0
    # the guard is on the *union* over query rows, which saturates as the group
    # widens even where each row's own live set is small
    qs, k, v, s, z = _case(S=4096, NBH=8, G=G, alpha=60.0, tchk=8.0 if G <= 16 else 4.0)
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - T
    ref, _ = _ref(s, z, v, cut)
    rn = ref.abs().max().item()
    kw = dict(truncate=1, refine_k=16.0, split=2)
    a = _run(qq, kk, vv, vb, z, cut, v8=True, **kw)[0]
    b = _run(qq, kk, vv, vb, z, cut, v8=False, **kw)[0]
    for o in (a, b):
        assert (o - ref).abs().max().item() / rn < 2e-2


@pytest.mark.parametrize("G", [16, 32, 48, 64])
@pytest.mark.parametrize("gate", ["no refine", "refine everything"])
def test_a_wide_group_agrees_with_a_narrow_one_on_its_first_rows(G, gate):
    """The pairing control for one accumulator per query group.

    At N = 8k the value matmul's accumulator is 4k registers wide and carries
    the whole KV loop, so a half left uninitialised is invisible to any test
    that only checks the rows it did initialise -- it was, and the error it hid
    was 1e30. With no gate live, query rows 0-7 of a wide group are **bitwise**
    the narrow group's answer: the same keys, the same order, one instruction.

    A gate has to be pinned, and this is not a weakening. Every verdict in this
    kernel is one bit per *key*, ORed over the query rows: a key some row keeps
    is refined for all of them, and a key some row keeps has its *refined*
    logit tested against the cut for all of them. So a wider group makes the
    rows it shares with a narrow one **more accurate**, not equal -- 1.2e-3 at
    G=16 here. That is the graded-precision mechanism, and the two arms below
    pin it from either side.

    Q's planes are sliced rather than requantised, because the rotation is a
    D x D matmul whose fp result depends on the batch shape and a tie in
    `round` then lands either way.
    """
    qs, k, v, _s, z = _case(S=4096, NBH=8, G=G, alpha=60.0, tchk=8.0 if G <= 16 else 4.0)
    kw = dict(split=2, v8=True, refine_v=1e4)
    if gate == "no refine":
        # a K gate no key can clear
        cut, kw = z - 8.0, dict(kw, truncate=1, refine_k=-1e30)
    else:
        cut, kw = z - 1e4, dict(kw, truncate=0, refine_k=1e4)
    qq, kk, vv, vb = _pack(qs, k, v)
    wide = _run(qq, kk, vv, vb, z, cut, **kw)[0]
    qa, qb, eq = qq
    q8 = (qa[:, :8].contiguous(), qb[:, :8].contiguous(), eq[:, :8].contiguous())
    narrow = _run(q8, kk, vv, vb, z[:, :8].contiguous(), cut[:, :8].contiguous(), **kw)[0]
    assert torch.equal(wide[:, :8], narrow), (wide[:, :8] - narrow).abs().max().item()


def _prep(qq: Planes, kk: Planes, vv, vb, z, cut, *, v8=False, **kw):
    """`_run` without the launch."""
    va, vbp, evs = vv
    return _prepare(
        *qq,
        *kk,
        va if v8 else vb,
        z,
        cut,
        v2=vbp if v8 else None,
        v_scale=evs if v8 else 1.0,
        v8=v8,
        **kw,
    )


@pytest.mark.parametrize("G,D", [(1, 64), (8, 128)])
def test_combine_groups_independent_rows_without_changing_outputs(G, D):
    qs, k, v, _, z = _case(NBH=16, S=1024, G=G, D=D)
    qq, kk, vv, vb = _pack(qs, k, v)
    kw = {"split": 3, "truncate": 1, "refine_k": 16.0}
    one = _prep(qq, kk, vv, vb, z, z - 8.0, combine_warps=1, **kw)
    four = _prep(qq, kk, vv, vb, z, z - 8.0, combine_warps=4, **kw)
    assert all(torch.equal(a, b) for a, b in zip(one(), four()))


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_combine_writes_the_same_rounded_output_as_a_cast(D, dtype):
    G = 8
    qs, k, v, _, z = _case(NBH=16, S=1024, G=G, D=D)
    qq, kk, vv, vb = _pack(qs, k, v)
    kw = {"split": 3, "truncate": 1, "refine_k": 16.0}
    base = _prep(qq, kk, vv, vb, z, z - 8.0, **kw)
    fused = _prep(qq, kk, vv, vb, z, z - 8.0, out_dtype=dtype, **kw)
    out = torch.empty((16, G, D), device="cuda", dtype=dtype)
    got, _, _ = fused(out=out)
    assert got is out
    assert torch.equal(got, base()[0].to(dtype))


@pytest.mark.parametrize("T", [None, 8.0])
@pytest.mark.parametrize("split", [1, 3])
def test_the_register_value_operand_is_the_same_bits(T, split):
    """Taking the value matmul's A from registers relabels, it does not
    approximate: the fragment's rows stand for channels in another order,
    which the epilogue undoes, the conversion is the widening pass's, and the
    second plane is the same single f16 fma. So the answer is bitwise the
    tile path's. `refine_v=4` makes the second plane move some keys."""
    qs, k, v, _s, z = _case(S=2048)
    qq, kk, vv, vb = _pack(qs, k, v)
    kw = dict(refine_k=12.0, refine_v=4.0, split=split, v8=True)
    if T is None:
        cut, kw = z - 1e4, dict(kw, truncate=0)
    else:
        cut, kw = z - T, dict(kw, truncate=1, vmean=_vmean(v, vv, True))
    a = _run(qq, kk, vv, vb, z, cut, v_regs=0, **kw)
    b = _run(qq, kk, vv, vb, z, cut, v_regs=1, **kw)
    for x, y in zip(a, b):
        assert torch.equal(x, y), (x.float() - y.float()).abs().max().item()


def _fma(a, b, c):
    """`a * b + c` with one rounding, as an FFMA has. The product of two fp32
    values needs 48 bits and its sum with a third needs fewer than 53 at these
    magnitudes, so the fp64 expression is exact and rounding it once is the
    same number the hardware produces."""
    return (a.double() * b.double() + c.double()).float()


def _coarse(qq, kk, S, z=None):
    """The kernel's coarse logit `(256 c1 + c2) . eq . (ek / 256)`, each key's
    own `ek`, exactly:
    the integer dot products are exact in fp64 and the sum is rounded to fp32
    once, as the kernel's convert does.

    With `z` it returns the logit relative to that reference, which is the
    currency the tile loop carries: the scale and the subtraction are one
    fused multiply-add there, so it is one rounding and not two."""
    qa, qb, eq = qq
    ka, _kb, ek = kk
    A = swizzle_rows(ka)[:, :S].double()
    c1 = torch.einsum("ngd,nsd->ngs", qa.double(), A)
    c2 = torch.einsum("ngd,nsd->ngs", qb.double(), A)
    e = eq.float()[..., None] * (ek.float()[:, None, :S] * (1.0 / KBR))
    cc = (c1 * 256 + c2).float()
    if z is None:
        return cc * e
    return _fma(cc, e, -z[..., None].expand_as(cc))


@pytest.mark.parametrize("D", [64, 128])
def test_the_butterfly_is_the_rotation_and_a_row_owns_its_bits(D):
    """`hadamard` is `x @ R.T` to fp32 rounding, and a row rotated alone or
    among thousands gets the same bits. The control is the equivalent matmul,
    whose rounding follows the number of rows it is handed and so would make
    a cached key's planes depend on the batch it was written with."""
    x = torch.randn(4096, D, device="cuda")
    R = rotation(D, x.device)
    full = hadamard(x)
    assert (full - x @ R.T).abs().max().item() < 1e-5 * x.abs().max().item()
    assert torch.equal(full[:1], hadamard(x[:1]))
    assert torch.equal(full[123:130], hadamard(x[123:130]))
    mm = x @ R.T
    assert any(not torch.equal(mm[i : i + 1], x[i : i + 1] @ R.T) for i in range(64))


@pytest.mark.parametrize("page_size", [16, 64])
def test_the_mass_prepass_and_register_operand_page_like_a_contiguous_cache(page_size):
    """The mass prepass and the register value operand read the cache through
    the same page table: the paged call is the contiguous one's bits, the
    reference included."""
    S, hkv = 512, 2
    qs, k, v, _s, z = _case(S=S, NBH=8)
    qq, kk, vv, vb, pt, (pka, pkb, pva, pvb, pek) = _paged(qs, k, v, S, page_size, hkv)
    sl = torch.full((pt.shape[0],), S, device=k.device, dtype=torch.int32)
    kw = dict(truncate=1, refine_k=16.0, split=2, v8=True, v_regs=1, reference="mass")
    a = _prep(qq, kk, vv, vb, torch.empty_like(z), torch.full_like(z, 8.0), **kw)
    oa = [x.clone() for x in a()]
    b = _prepare(
        qq[0],
        qq[1],
        qq[2],
        pka,
        pkb,
        pek,
        pva,
        torch.empty_like(z),
        torch.full_like(z, 8.0),
        v2=pvb,
        v_scale=vv[2],
        page_table=pt,
        page_size=page_size,
        seq_lens=sl,
        n_kv_heads=hkv,
        **kw,
    )
    ob = b()
    assert torch.equal(a.z, b.z)
    for x, y in zip(oa, ob):
        assert torch.equal(x, y), (x.float() - y.float()).abs().max().item()


def _mass_model(s, n, vis=None, NS=128, TRIM=4):
    """The mass prepass's estimate for one row group in float64, on the
    kernel's own coarse logits `s` (G, >= n): the sink and the window exact,
    the stratum centres the window does not hold as samples, the TRIM largest
    of those counted once and the rest extrapolated over the keys nothing
    scored. `vis` (G, n) masks a draft."""
    npr = -(-(1 + NS + 31) // 64) * 64
    W = npr - 1 - NS
    lo = n - W
    s = s[:, :n].double()
    exact = [0] + [n - r for r in range(1, W + 1) if n - r >= 1]
    strat, prev = [], None
    for j in range(NS):
        k = ((2 * j + 1) * n) // (2 * NS)
        if 1 <= k < lo and k != prev:
            strat.append(k)
        prev = k
    se = s[:, exact]
    if vis is not None:
        se = torch.where(vis[:, exact], se, torch.full_like(se, -math.inf))
    ss = s[:, strat]
    M = torch.maximum(se.amax(-1), ss.amax(-1)) if strat else se.amax(-1)
    d = torch.exp2(ss - M[:, None])
    mass = torch.exp2(se - M[:, None]).sum(-1) + d.sum(-1)
    uns = n - len(exact) - len(strat)
    if len(strat) > TRIM and uns > 0:
        rest = d.sum(-1) - d.sort(-1, descending=True).values[:, :TRIM].sum(-1)
        mass = mass + rest * uns / (len(strat) - TRIM)
    return M + torch.log2(mass)


@pytest.mark.parametrize("S,G,D", [(8192, 8, 128), (4096, 16, 64), (8192, 1, 128)])
def test_the_mass_reference_is_the_estimate_it_documents(S, G, D):
    """`reference="mass"` writes each row's log-sum-exp as the prepass estimates
    it, to fp32 rounding of the float64 model."""
    NBH = 8
    qs, k, v, _s, z = _case(NBH=NBH, S=S, D=D, G=G, tchk=8.0 if G <= 8 else 4.0)
    qq, kk, vv, vb = _pack(qs, k, v)
    f = _prep(
        qq,
        kk,
        vv,
        vb,
        torch.empty_like(z),
        torch.full_like(z, 12.0),
        truncate=1,
        refine_k=12.0,
        split=3,
        reference="mass",
    )
    f()
    sc = _coarse(qq, kk, S)
    want = torch.stack([_mass_model(sc[b], S) for b in range(NBH)])
    assert float((f.z.double() - want).abs().max()) < 1e-4


@pytest.mark.parametrize("page_size", [16, 64])
def test_the_mass_prepass_pages_like_the_contiguous_call(page_size):
    """Paged, the estimate is the contiguous call's bits; ragged, each
    request's is its own length's, down to requests shorter than twice the
    strata, whose centres collide and whose window covers most of the row."""
    S, hkv = 4096, 2
    qs, k, v, _s, z = _case(S=S, NBH=8)
    qq, kk, vv, vb, pt, (pka, pkb, pva, pvb, pek) = _paged(qs, k, v, S, page_size, hkv)
    NBH = z.shape[0]
    kw = dict(truncate=1, refine_k=12.0, split=2, v8=True, reference="mass")
    dep = torch.full_like(z, 12.0)
    a = _prep(qq, kk, vv, vb, torch.empty_like(z), dep, **kw)
    a()

    def paged(lens):
        f = prepare_fold_decode(
            qq[0],
            qq[1],
            qq[2],
            pka,
            pkb,
            pek,
            pva,
            torch.empty_like(z),
            dep,
            v2=pvb,
            v_scale=vv[2],
            page_table=pt,
            page_size=page_size,
            seq_lens=lens,
            n_kv_heads=hkv,
            **kw,
        )
        f()
        return f

    B = NBH // hkv
    b = paged(torch.full((B,), S, device="cuda", dtype=torch.int32))
    assert torch.equal(a.z, b.z)
    lens = torch.tensor([S, 97, 300, S - 777], device="cuda", dtype=torch.int32)
    r = paged(lens)
    sc = _coarse(qq, kk, S)
    for bh in range(NBH):
        want = _mass_model(sc[bh], int(lens[bh // hkv]))
        assert float((r.z[bh].double() - want).abs().max()) < 1e-4, bh


def test_a_draft_row_estimates_only_the_mass_it_sees():
    """Under a draft the window holds the draft's keys, and a row's estimate
    counts only those its mask admits. The control is aimed: the newest key
    points at row 0, which does not attend it."""
    Gq, q_len, S, NBH = 2, 4, 2048, 4
    G = Gq * q_len
    qs, k, v, s, z = _case(NBH=NBH, S=S, G=G)
    k = k.clone()
    r0 = qs[:, 0]
    k[:, S - 1] = r0 / r0.norm(dim=-1, keepdim=True) * (4.0 * s.max())
    s = torch.einsum("bgd,bsd->bgs", qs, k)
    qq, kk, vv, vb = _pack(qs, k, v)
    f = _prep(
        qq,
        kk,
        vv,
        vb,
        torch.empty_like(z),
        torch.full_like(z, 8.0),
        truncate=1,
        refine_k=12.0,
        split=2,
        reference="mass",
        draft_len=q_len,
        causal=True,
    )
    f()
    sc = _coarse(qq, kk, S)
    vis = torch.ones_like(sc, dtype=torch.bool)
    for t in range(q_len):
        for j in range(t + 1, q_len):
            vis[:, t::q_len, S - q_len + j] = False
    want = torch.stack([_mass_model(sc[b], S, vis[b]) for b in range(NBH)])
    assert float((f.z.double() - want).abs().max()) < 1e-4
    blind = torch.stack([_mass_model(sc[b], S) for b in range(NBH)])
    assert float((blind[:, 0] - f.z[:, 0].double()).min()) > 1.0


@pytest.mark.parametrize("v8", [False, True])
def test_a_given_reference_is_the_prepass_one_to_the_bit(v8):
    """`reference="given"` is the decode of `reference="mass"` without its prepass:
    handed the Z and cut the prepass wrote, it returns the same bits."""
    qs, k, v, _s, z = _case(S=2048)
    qq, kk, vv, vb = _pack(qs, k, v)
    kw = dict(truncate=1, refine_k=12.0, split=3, v8=v8)
    a = _prep(qq, kk, vv, vb, torch.empty_like(z), torch.full_like(z, 10.0), reference="mass", **kw)
    oa = [x.clone() for x in a()]
    b = _prep(qq, kk, vv, vb, a.z.clone(), a.z - 10.0, reference="given", **kw)
    ob = b()
    for x, y in zip(oa, ob):
        assert torch.equal(x, y)


@pytest.mark.parametrize("v8", [False, True])
def test_two_term_weights_make_the_reference_irrelevant_to_precision(v8):
    """A bf16 weight's rounding is 2^-9 of it, and which weights round badly
    depends on where Z sits: the heaviest key's weight is exact at Z = the
    peak and a lottery anywhere else. Carried as two bf16 terms, a weight is
    good to 2^-17 at bf16's range, so with every key refined the answer does
    not move with Z, and it is closer to the exact one than one term at the
    peak."""
    qs, k, v, s, z = _case(S=2048)
    qq, kk, vv, vb = _pack(qs, k, v)
    full = _ref(s, z, v, z - 1e4)[0]
    kw = dict(truncate=0, refine_k=40.0, refine_v=40.0, split=3, v8=v8)
    one = _l2(_prep(qq, kk, vv, vb, z, z - 1e4, **kw)()[0], full)
    errs = []
    for off in (0.0, 0.37, 3.0, -2.5, 7.3):
        f = _prep(qq, kk, vv, vb, z + off, z - 1e4, weight_terms=2, **kw)
        errs.append(_l2(f()[0], full))
    assert max(errs) < 1.02 * min(errs), errs
    assert max(errs) < one, (errs, one)


@pytest.mark.parametrize("v8", [False, True])
def test_keep_all_spends_the_gather_the_group_already_paid_for(v8):
    """The live verdict is one bit per key, ORed over the query group, so a key
    any row keeps is fetched for all of them and held in registers for all of
    them. Applying the cut a second time per (key, row) buys no bytes: it
    replaces an exact weight the kernel is holding with `vmean`'s surrogate,
    which exists for mass that was never read. Spending it instead must move
    the answer towards the full softmax.
    """
    qs, k, v, s, z = _case()
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    full, _ = _ref(s, z, v, z - 1e4)
    rn = full.abs().max().item()

    def err(o):
        return (o - full).abs().max().item() / rn

    kw = dict(v8=v8, truncate=1, refine_k=16.0, vmean=_vmean(v, vv, v8))
    a = err(_run(qq, kk, vv, vb, z, cut, **kw)[0])
    b = err(_run(qq, kk, vv, vb, z, cut, group_cut=1, **kw)[0])
    # the control: with the cut this shallow there has to be something to win,
    # or the comparison below is two numbers that are both already zero
    assert a > 1e-3, f"nothing to win: {a}"
    assert b < a, (a, b)


def test_keep_all_widens_no_gather():
    """It spends the union, it does not enlarge it. The live count is what the
    kernel reads, and it must come back bit for bit."""
    qs, k, v, _s, z = _case()
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    kw = dict(truncate=1, refine_k=16.0, vmean=_vmean(v, vv, False))
    o0, _, c0 = _run(qq, kk, vv, vb, z, cut, **kw)
    o1, _, c1 = _run(qq, kk, vv, vb, z, cut, group_cut=1, **kw)
    assert torch.equal(c0, c1), "group_cut changed the live set"
    # and the control: the output did move, so the equality above is not
    # asserting that the flag did nothing at all
    assert not torch.equal(o0, o1)


def test_keep_all_keeps_the_denominator_the_total_mass():
    """Every key still lands in exactly one of the two accumulators, so the
    corrected denominator is the true total either way."""
    qs, k, v, s, z = _case()
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    total = torch.exp2(s - z[..., None]).sum(-1)
    lc = _run(
        qq, kk, vv, vb, z, cut, truncate=1, refine_k=16.0, group_cut=1, vmean=_vmean(v, vv, False)
    )[1]
    assert ((lc - total).abs() / total).max().item() < 1e-2


def test_keep_all_needs_a_truncation():
    qs, k, v, _s, z = _case(NBH=1, S=512)
    qq, kk, vv, vb = _pack(qs, k, v)
    with pytest.raises(ValueError):
        _run(qq, kk, vv, vb, z, z - 8.0, truncate=0, group_cut=1)


def test_keep_all_auto_is_the_single_level_policy():
    qs, k, v, _, z = _case()
    qq, kk, vv, vb = _pack(qs, k, v)
    cut = z - 8.0
    auto = _prep(qq, kk, vv, vb, z, cut, truncate=1, group_cut="auto")
    explicit = _prep(qq, kk, vv, vb, z, cut, truncate=1, group_cut=1)
    assert auto.config.keep_all
    assert torch.equal(auto()[0], explicit()[0])

    qs, k, v, _, z = _case(D=64)
    qq, kk, vv, vb = _pack(qs, k, v)
    auto = _prep(qq, kk, vv, vb, z, z - 8.0, truncate=1, group_cut="auto")
    assert auto.config.keep_all
    dense = _prep(qq, kk, vv, vb, z, z - 8.0, truncate=0, group_cut="auto")
    assert not dense.config.keep_all


@pytest.mark.parametrize("lever", ["plain", "v8", "v8 v_regs", "bf16 V split 3", "mass"])
def test_keep_all_composes_with_every_lever(lever):
    """`group_cut` spends the group's gather and touches nothing else, so on every
    path the live count has to come back bit for bit and the answer has to move
    towards the full softmax. The levers that change *what* the verdict reads
    (an estimated Z) is the one that could have broken it.
    """
    S = 2048
    qs, k, v, s, z = _case(S=S)
    qq, kk, vv, vb = _pack(qs, k, v)
    full, _ = _ref(s, z, v, z - 1e4)
    rn = full.abs().max().item()
    kw = dict(truncate=1, refine_k=12.0, split=2, vmean=_vmean(v, vv, "v8" in lever))
    if lever == "v8 v_regs":
        kw.update(v8=True, v_regs=1)
    elif lever == "v8":
        kw.update(v8=True)
    elif lever == "bf16 V split 3":
        kw.update(split=3)
    elif lever == "mass":
        kw.update(reference="mass")

    def call(group_cut):
        kws = dict(kw, group_cut=group_cut)
        if lever == "mass":
            f = _prep(qq, kk, vv, vb, torch.empty_like(z), torch.full_like(z, 8.0), **kws)
            return [x.clone() for x in f()]
        return _run(qq, kk, vv, vb, z, z - 8.0, **kws)

    o0, _, c0 = call(0)
    o1, _, c1 = call(1)
    assert torch.equal(c0, c1), f"{lever}: group_cut changed the live set"
    e0 = (o0 - full).abs().max().item() / rn
    e1 = (o1 - full).abs().max().item() / rn
    # the control: the cut has to be biting, or both errors are already zero
    assert e0 > 1e-3, f"{lever}: nothing to win ({e0})"
    assert e1 < e0, (lever, e0, e1)


# --------------------------------------------------------------------------
# Head dimension: the cache format, the descriptor atom and every lane map at
# D=64 as well as D=128, and what is refused, so a shape cannot quietly start
# returning wrong numbers.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("D", [32, 64, 128])
def test_the_swizzle_is_the_relabelling_the_descriptor_reads(D):
    """`swizzle_of` is the cache format, and the format is the hardware's.

    wgmma takes `S<3,4,3>`, `S<2,4,3>` and `S<1,4,3>` and nothing else -- the
    shift is 3 in all three, so a 64-byte row XORs by `(r >> 1) & 3` and not
    by `r & 3`. This holds the host side to it: the relabelling, element for
    element, and that applying it twice is the identity.
    """
    nu, sh, aln = swizzle_of(D)
    assert (nu, aln) == (D // 16, D * 8)
    assert sh == 3 - (nu.bit_length() - 1)
    x = torch.randint(-100, 100, (3, 64, D), device="cuda", dtype=torch.int8)
    y = swizzle_rows(x)
    assert not torch.equal(y, x)
    assert torch.equal(swizzle_rows(y), x)
    xv = x.reshape(3, 64, nu, 16)
    yv = y.reshape(3, 64, nu, 16)
    for r in range(0, 64, 7):
        for u in range(nu):
            assert torch.equal(yv[:, r, u], xv[:, r, u ^ ((r >> sh) & (nu - 1))])


def test_a_row_wider_than_one_swizzle_atom_is_refused():
    """D=256 is two atoms per row, so the cache would have to be stored in
    128-byte channel blocks rather than relabelled in place. It says so."""
    with pytest.raises(ValueError, match="128-byte"):
        swizzle_rows(torch.zeros(1, 8, 256, device="cuda", dtype=torch.int8))


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("v8", [False, True])
def test_a_narrow_head_is_the_same_function(D, v8):
    qs, k, v, s, z = _case(D=D)
    qq, kk, vv, vb = _pack(qs, k, v)
    for T in (1e4, 8.0):
        cut = z - T
        o, _, _ = _run(qq, kk, vv, vb, z, cut, v8=v8, truncate=int(T < 1e3), split=2, refine_k=12.0)
        r = _ref(s, z, v, cut)[0]
        err = float((o - r).abs().max() / r.abs().max())
        assert err < 2e-2, (D, v8, T, err)
    # the control: scored against the *dense* reference the truncated arm has
    # to be wrong, or the tolerance above is measuring nothing
    o, _, _ = _run(qq, kk, vv, vb, z, z - 8.0, v8=v8, truncate=1, split=2, refine_k=12.0)
    rd = _ref(s, z, v, z - 1e4)[0]
    assert float((o - rd).abs().max() / rd.abs().max()) > 2e-2


def test_the_head_dim_contract_is_stated_at_the_call():
    """Every shape the kernel cannot serve says which property it fails."""
    # `_case`'s own live-fraction guard is shape sensitive, so this takes the
    # default batch it is tuned for rather than a one-row one
    qs, k, v, _s, z = _case(D=64)
    qq, kk, vv, vb = _pack(qs, k, v)
    # D=96 is not a power of two: the Sylvester rotation cannot be built
    with pytest.raises(ValueError, match="power of two"):
        rotation(96, device="cuda")
    for bad, msg in ((256, "128-byte"),):
        with pytest.raises(ValueError, match=msg):
            swizzle_rows(torch.zeros(1, 8, bad, device="cuda", dtype=torch.int8))
    # and D=64 itself is served, on every V format, which is the other half
    # of the claim
    for v8 in (False, True):
        o, _, _ = _run(qq, kk, vv, vb, z, z - 1e4, v8=v8, split=1)
        assert torch.isfinite(o).all()


def test_the_residency_model_reproduces_its_measured_footprints():
    """`pick_split` chooses a split for a residency it cannot measure, so the
    model has to be right. These three are the numbers the tuned D=128 build
    was measured at, and they are why the model is trusted at other shapes."""
    assert resident(False, False) == 6  # bf16 V, 36.5 KB
    assert resident(True, True) == 6  # 8-bit V under v_regs, 28.3 KB, registers cap it
    assert resident(True, False) == 5  # 8-bit V without v_regs, 44.7 KB
    # `v_regs` is only the 8-bit V's operand, so asking for it on a bf16 V must
    # not shrink the tile the model thinks that build holds
    assert smem_bytes(128, 8, False, True) == smem_bytes(128, 8, False, False)
    # the headline's split: at 256 row groups and 16K keys, dense, within 0.5%
    # of the swept best on the 8-bit V and the swept best on bf16
    assert pick_split(256, 16384, True, sms=132, truncate=False, front=True, weight_terms=2) == 6
    assert pick_split(256, 16384, False, v_regs=False, sms=132) == 3
    # The arms are a whole CTA apart at a wide group and the split follows:
    # at G=24 the bf16 V holds four and the register operand three, which is
    # split 2 (139.7 us measured) against split 3 (170.0), so each arm's split
    # has to read its own registers.
    assert resident(False, False, G=24) == 4
    assert resident(True, True, G=24) == 3
    assert pick_split(256, 8192, False, G=24, v_regs=False, sms=132) == 2
    assert pick_split(256, 8192, True, G=24, sms=132) == 3
    # a narrower head holds less and fits more
    assert resident(True, True, D=64) > resident(True, True, D=128)
    # what a wider group costs in shared memory is the weight buffer and the
    # Q tile and nothing else
    for G in (24, 32):
        grew = smem_bytes(128, G, True, True) - smem_bytes(128, 8, True, True)
        assert grew < 12 * 1024, (G, grew)
    # what a wide group does cost is registers, which is why the residency
    # falls
    assert resident(True, True, G=32) < resident(True, True, G=8)


# (D, G, depth, row groups, mean length, longest, splits within 1% of the best
# swept one), cold L2, the serving build; D = 64 at G <= 4 is the packed tile.
_SWEPT_NEAR = [
    (128, 8, 14, 512, 25015, 32768, (3, 6)),
    (128, 8, 14, 8, 3264, 4096, (24, 32)),
    (128, 8, None, 64, 26632, 32768, (8, 10, 12, 20)),
    (128, 8, None, 32, 13760, 16384, (16, 20)),
    (128, 1, 14, 64, 32768, 32768, (10, 12)),
    (128, 1, 14, 8, 3264, 4096, (32,)),
    (128, 1, None, 256, 4096, 4096, (2, 3)),
    (128, 1, None, 32, 27296, 32768, (16, 20, 24, 48)),
    (128, 16, 14, 256, 3134, 4096, (2,)),
    (128, 16, 14, 4, 16384, 16384, (64,)),
    (128, 16, None, 256, 32768, 32768, (2, 5)),
    (128, 16, None, 4, 16384, 16384, (64,)),
    (64, 1, 14, 64, 16384, 16384, (12,)),
    (64, 1, 14, 8, 4096, 4096, (32,)),
    (64, 1, None, 256, 26596, 32768, (5, 6, 12, 16)),
    (64, 1, None, 32, 28992, 32768, (12, 20, 24)),
    (64, 8, 14, 256, 4096, 4096, (3, 4)),
    (64, 8, 14, 4, 32768, 32768, (64,)),
    (64, 8, None, 64, 27296, 32768, (10, 12, 16, 20)),
    (64, 8, None, 16, 28224, 32768, (48, 64)),
]


@pytest.mark.parametrize("D, G, depth, nbh, mean, longest, near", _SWEPT_NEAR)
def test_the_bf16_split_lands_near_the_swept_best(D, G, depth, nbh, mean, longest, near):
    from fold_attention.decode.heuristics import tail_rank_for, weight_terms_for

    skip = depth is not None
    sp = pick_split(
        nbh,
        mean,
        False,
        sms=132,
        max_len=longest,
        D=D,
        G=G,
        truncate=skip,
        front=True,
        tail=tail_rank_for(D, G, False) if skip else -1,
        weight_terms=weight_terms_for(depth),
    )
    assert sp in near


def test_d128_v8_long_split_and_residency():
    def choose(nbh, mean, longest, *, G, truncate):
        return pick_split(
            nbh,
            mean,
            True,
            sms=132,
            max_len=longest,
            D=128,
            G=G,
            truncate=truncate,
            front=True,
            tail=-1,
            weight_terms=1 if truncate else 2,
        )

    # the front asks every register-V build up to G = 8 for the deep cap's six
    for G, truncate in ((8, True), (8, False), (1, False)):
        assert resident(True, True, D=128, G=G, truncate=truncate, front=True, weight_terms=2) == 6
    # the wave model's pick on swept cells (depth 13 when truncated), from 8
    # row groups, which need 64 CTAs apiece to fill the machine, to 256, which
    # fill it at 6; each within 7% of the best swept split, on curves flat
    # near their minimum with 5-7% spikes
    assert choose(8, 15744, 16384, G=8, truncate=False) == 64
    assert choose(8, 16384, 16384, G=1, truncate=True) == 64
    assert choose(64, 16384, 16384, G=8, truncate=False) == 12
    assert choose(64, 32768, 32768, G=8, truncate=True) == 12
    assert choose(64, 12744, 16384, G=1, truncate=True) == 20
    assert choose(256, 16384, 16384, G=8, truncate=False) == 6
    assert choose(256, 12684, 16384, G=8, truncate=True) == 6
    assert isinstance(choose(128, 12288, torch.tensor(16384), G=8, truncate=True), int)


def test_a_ragged_batch_cuts_its_longest_request_to_a_slots_load():
    """A ragged batch ends when its longest request's CTAs do, so the wave
    model prices the longest request's tiles per CTA as well as the waves.

    Measured with the longest-first grid order. The losses below are what the
    rule pays against the best swept split on these batches, not a bound it is
    held to.
    """
    # B=8, 2K-32K, dense: swept 28:79 35:77 40:74 48:76 us, best 40; 48 gives
    # up 2.7%
    assert pick_split(32, 19280, True, sms=132, max_len=29889, truncate=False) == 48
    # B=4, 512-8K: the measured best is 27, and both formats take the nearest
    # candidate
    assert pick_split(16, 5002, True, sms=132, max_len=6758, truncate=False) == 32
    assert pick_split(16, 5002, False, sms=132, max_len=6758, truncate=True) == 32


def _shape_case(NBH, G, S=512, D=128, seed=0):
    """Operands of the right shape and nothing else. `_case`'s live-fraction
    guard is a property of the data, and a group wide enough to test a bound
    saturates it: the verdict is ORed over the rows, so at 128 rows every key
    is live and the guard fires on a case that was only ever there to carry a
    shape past `prepare_fold_decode`."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(NBH, G, D, device="cuda", generator=g)
    k = torch.randn(NBH, S, D, device="cuda", generator=g)
    v = torch.randn(NBH, S, D, device="cuda", generator=g)
    z = torch.zeros(NBH, G, device="cuda")
    return _pack(q * LOG2E / math.sqrt(D), k, v), z


def test_a_query_group_with_no_wgmma_n_is_refused_by_name():
    """N is `8.ceil(G/8)`, and the instruction takes N = 8i for i in 1..4 and
    N = 16i above that. So 33..40 rows and 49..56 rows have no instruction at
    all, which is a gap and not a ceiling -- 48 and 64 both run, and
    `test_a_wide_group_agrees_with_a_narrow_one_on_its_first_rows` pins them.
    The message has to say which widths are next, because a caller who lands
    in a gap has no way to read it off the shape."""
    for G in (40, 56):
        (qq, kk, vv, vb), z = _shape_case(2, G)
        with pytest.raises(ValueError, match="not a query group width wgmma can issue"):
            _prep(qq, kk, vv, vb, z, z - 8.0, truncate=1, split=1)


def test_a_query_group_past_64_rows_runs_wide_or_is_refused_with_the_remedy():
    """The decode kernel holds every row's accumulators in registers, so past
    64 rows it would spill (resident 1: measured 5.3x per row at G=128). A
    single level that wide runs on the wide kernel instead, rows along M; a
    call the wide kernel cannot take, such as an 8-bit V, is refused with the
    split into calls of at most 64 rows."""
    (qq, kk, vv, vb), z = _shape_case(2, 128)
    run = _prep(qq, kk, vv, vb, z, z - 1e4, split=2)
    o = run()[0]
    assert torch.isfinite(o).all()
    with pytest.raises(ValueError, match="split the rows across calls"):
        _prep(qq, kk, vv, vb, z, z - 8.0, truncate=1, split=1, v8=True)
    # the model the routing reads is the measured one, on both head dims
    assert resident(False, False, G=64) == 2
    assert resident(False, False, G=128) == 1
    assert resident(True, True, D=64, G=64) == 2
    assert resident(True, True, D=64, G=128) == 1


def test_the_v_format_crossover_moves_with_the_head_dim_and_the_group():
    """A lane owns `D/32` channels of the value operand, so the 8-bit V's
    shared load carries four bytes at D=128 and two at D=64 for the same
    instruction. That puts a fixed per-tile cost under an arm whose bytes
    shrink with the cut, so the crossover is real and it moves with `D`.

    A wider query group adds the same kind of cost and moves it the same way.
    On a layer whose G=16 is real query heads the 8-bit V loses at 39.2% live
    where it wins at 33.1% with half the group. The band and the `v_regs`
    default both have to know it. At D=64, on gpt-oss layers through the
    ragged serving path, the 8-bit V wins from a live fraction of 0.28 up, by
    0.9% to 14%, with equal or better error at every point."""
    assert pick_config(0.42, D=128)[0] is True
    assert pick_config(0.42, D=64)[0] is True
    assert pick_config(0.20, D=64)[0] is False
    assert pick_config(0.70, D=64)[0] is True
    # a group past one wgmma block pushes it up, at either head dim
    assert pick_config(0.42, D=128, G=16)[0] is False
    assert pick_config(0.50, D=128, G=16)[0] is True
    assert pick_config(0.90, D=64, G=16)[0] is False
    assert v8_live_fraction(128, 16) > v8_live_fraction(128, 8)
    assert v8_live_fraction(64, 8) > v8_live_fraction(128, 8)
    # `v_regs` is the D=128 operand and defaults off below it, but stays reachable
    # and computes the same thing
    qs, k, v, _, z = _case(D=64)
    qq, kk, vv, vb = _pack(qs, k, v)
    kw = dict(truncate=1, v8=True, split=2, refine_k=12.0)
    auto = _run(qq, kk, vv, vb, z, z - 8.0, **kw)[0]
    asked = _run(qq, kk, vv, vb, z, z - 8.0, v_regs=1, **kw)[0]
    assert torch.equal(auto, asked)


# --- speculative / tree verification -----------------------------------------
#
# A draft is a tree in every implementation that matters, and node `i` attends
# its ancestor path rather than everything before it. Three things make that
# one conjunct here rather than a mechanism: the rows of a row group are
# already independent (per-row Z, per-row cut, per-row output), the kernel
# already seeds every pair with `-3e38` and overwrites it only where the pair
# exists, and `2^(s - Z)` is a pure function of one logit, so a row's
# contribution does not depend on how many rows shared the pass.


def _draft_case(Gq=4, draft_len=2, S=1024, NBH=4, D=128, seed=3, tchk=8.0):
    """A group of `Gq` query heads verifying a `draft_len`-node draft.

    The draft's own keys are the last `draft_len` of the cache, which is where
    the caller's `write_kv` puts them. `tchk=None` skips `_case`'s live-set
    check, which a group past 64 rows fails: some row keeps every key.
    """
    qs, k, v, s, z = _case(NBH=NBH, S=S, D=D, G=Gq * draft_len, seed=seed, tchk=tchk)
    qq, kk, vv, vb = _pack(qs, k, v)
    return qs, k, v, s, z, qq, kk, vv, vb


def _tree_ref(s, z, v, cut, tree, q_len):
    """The masked truncated softmax in fp64: row `t` sees every key before the
    draft and the draft keys its bitmask names."""
    S = s.shape[-1]
    vis = torch.ones_like(s, dtype=torch.bool)
    base = S - q_len
    for t in range(q_len):
        for j in range(q_len):
            if not bool(tree[t, j]):
                vis[:, t::q_len, base + j] = False
    p = torch.where(
        vis & (s >= cut[..., None]),
        torch.exp2(s.double() - z[..., None].double()),
        torch.zeros_like(s, dtype=torch.float64),
    )
    return ((p @ v.double()) / p.sum(-1, keepdim=True)).float()


@pytest.mark.parametrize("group_cut", [0, 1])
def test_a_chain_draft_is_the_causal_mask(group_cut):
    """A chain's ancestor bitmask is lower-triangular, which is exactly the
    causal mask over the draft's own keys. So `causal=True` and the explicit
    mask must be the same bits -- this is where a wrong row coordinate or
    wrong column arithmetic shows up and a closeness check would not."""
    q_len = 4
    _, _, _, _, z, qq, kk, vv, vb = _draft_case(Gq=2, draft_len=q_len)
    kw = dict(truncate=1, split=2, refine_k=10.0, group_cut=group_cut, draft_len=q_len)
    a = _run(qq, kk, vv, vb, z, z - 8.0, causal=True, **kw)
    chain = draft_mask([-1, 0, 1, 2])
    assert torch.equal(chain, torch.tril(torch.ones(q_len, q_len, dtype=torch.bool)))
    b = _run(qq, kk, vv, vb, z, z - 8.0, mask=chain, **kw)
    for x, y in zip(a, b):
        assert torch.equal(x, y), (x.float() - y.float()).abs().max().item()
    # the control: an unmasked call over the same cache is a different function
    c = _run(qq, kk, vv, vb, z, z - 8.0, **kw)
    assert not torch.equal(a[0], c[0])


def _sequential(qq, kk, vb, z, t, q_len, n, **kw):
    """Step `t` of a sequential decode: the `Gq` rows at that draft position,
    over exactly the keys that step can see. It is **not told it is one of
    `q_len`** -- it pins the key length it actually has."""
    return fold_decode(
        qq[0][:, t::q_len].contiguous(),
        qq[1][:, t::q_len].contiguous(),
        qq[2][:, t::q_len].contiguous(),
        kk[0],
        kk[1],
        kk[2],
        vb[:, :n].contiguous(),
        z[:, t::q_len].contiguous(),
        (z - 8.0)[:, t::q_len].contiguous(),
        **kw,
    )


def test_a_draft_row_is_the_step_it_verifies():
    """Row `t` of a `q_len`-row verification is step `t` of `q_len` sequential
    single-token decodes, **bitwise**.

    Speculative decoding is proved lossless on the assumption that the target
    distribution used while verifying `K` tokens at once is the one that would
    have been used generating them one at a time. Batching changes the tile,
    the reduction order and the split, so in floating point that assumption is
    false in every implementation and the error is bounded nowhere. Under a
    static reference it holds: `2^(s - Z)` is a pure function of one logit, a
    masked pair's weight is exactly `+0.0`, and adding zero to an accumulator
    is exact -- so the row sees the same addends in the same order whether it
    arrived in a `q_len`-row pass or as the `t`-th of `q_len` steps.

    The sequential arm is not told it is one of `q_len`. The two agree because
    they compute the same quantity, not because they were configured to.

    Two of the kernel's own gates have to be held for this to be the claim it
    looks like, and both for the same reason: every gate here is one bit per
    *key* **ORed over the query group**, so a `Gq * q_len` verification
    refines and gathers a superset of what any one `Gq` step does. Refining
    every key (`refine_k` past the logit range) and truncating nothing (`truncate=0`)
    make both verdicts group-independent, which leaves exactly the mask's own
    claim under test: a masked pair is a key that is not there, and the tiles
    past the shorter cache add exactly nothing.

    With the default gates the two arms are *not* the same bits, and the
    direction is itself a result --
    `test_a_draft_is_at_least_as_accurate_as_the_steps_it_replaces`.
    """
    Gq, q_len, S = 4, 2, 1024
    _qs, _k, _v, _s, z, qq, kk, vv, vb = _draft_case(Gq=Gq, draft_len=q_len, S=S)
    kw = dict(truncate=0, refine_k=1e4, split=1)
    ver, dv, _ = _run(qq, kk, vv, vb, z, z - 8.0, causal=True, draft_len=q_len, **kw)
    for t in range(q_len):
        n = S - q_len + 1 + t
        seq, ds, _ = _sequential(qq, kk, vb, z, t, q_len, n, **kw)
        assert torch.equal(ds, dv[:, t::q_len]), t
        assert torch.equal(seq, ver[:, t::q_len]), (t, (seq - ver[:, t::q_len]).abs().max().item())


def test_a_draft_is_at_least_as_accurate_as_the_steps_it_replaces():
    """With the default refine gate the verification is not the same bits as
    the sequential steps, and the direction is the point.

    Every gate in this kernel is one bit per key **ORed over the query group**
    -- the refine's and the live verdict's alike -- so a draft's `Gq * q_len`
    rows refine and gather a **superset** of what any one step's `Gq` rows
    would. Neither gate decides membership in the function: `refine_k` gates
    precision, and a key the group's coarse verdict keeps still needs the
    row's own cut to carry weight. A superset can therefore only help, and
    verifying `q_len` tokens at once comes out at least as accurate as
    generating them one at a time -- the opposite of what batching a
    verification usually costs.
    """
    Gq, q_len, S = 4, 4, 1024
    _qs, _k, v, s, z, qq, kk, vv, vb = _draft_case(Gq=Gq, draft_len=q_len, S=S)
    cut = z - 8.0
    kw = dict(truncate=1, refine_k=10.0, split=1)
    ver = _run(qq, kk, vv, vb, z, cut, causal=True, draft_len=q_len, **kw)[0]
    tree = torch.tril(torch.ones(q_len, q_len, dtype=torch.bool))
    ref = _tree_ref(s, z, v, cut, tree, q_len).double()
    worse = 0
    for t in range(q_len):
        n = S - q_len + 1 + t
        seq = _sequential(qq, kk, vb, z, t, q_len, n, **kw)[0]
        rt = ref[:, t::q_len]
        ev = float((ver[:, t::q_len].double() - rt).norm() / rt.norm())
        es = float((seq.double() - rt).norm() / rt.norm())
        assert ev <= es * 1.02, (t, ev, es)
        worse += int(es > ev * 1.02)
    assert worse, "the wider group gathered nothing extra: the case cannot show it"


def test_a_branching_tree_is_its_masked_softmax():
    """A tree that is not a chain, against the masked truncated softmax in
    fp64. The chain is the control: both arms must land on the same floor, so
    a tree costs no accuracy over the chain it generalises."""
    q_len = 4
    parents = [-1, 0, 0, 1]
    tree = draft_mask(parents)
    assert not torch.equal(tree, torch.tril(torch.ones(q_len, q_len, dtype=torch.bool)))
    _qs, _k, v, s, z, qq, kk, vv, vb = _draft_case(Gq=2, draft_len=q_len)
    cut = z - 8.0
    kw = dict(truncate=1, split=2, refine_k=16.0, draft_len=q_len)
    o = _run(qq, kk, vv, vb, z, cut, mask=tree, **kw)[0]
    ref = _tree_ref(s, z, v, cut, tree, q_len)
    err = (o - ref).abs().max().item() / ref.abs().max().item()
    chain = draft_mask([-1, 0, 1, 2])
    oc = _run(qq, kk, vv, vb, z, cut, mask=chain, **kw)[0]
    refc_ = _tree_ref(s, z, v, cut, chain, q_len)
    errc = (oc - refc_).abs().max().item() / refc_.abs().max().item()
    assert err < 2e-2, err
    assert err < 4 * errc, (err, errc)


def test_a_draft_key_a_row_declines_moves_no_byte_and_no_mass():
    """A masked pair keeps the `-3e38` sentinel, so it is not a dropped key
    the correction re-enters: the denominator is the masked sum exactly.

    The gather is still the group's, so the key counts do not move -- a draft
    key one row descends from is read for the whole group, which is the same
    union every other gate is ORed over."""
    q_len = 4
    _qs, _k, v, s, z, qq, kk, vv, vb = _draft_case(Gq=2, draft_len=q_len)
    cut = z - 8.0
    vm = v.mean(1).float().contiguous()
    kw = dict(truncate=1, split=2, refine_k=16.0, draft_len=q_len, vmean=vm)
    _o, den, cnt = _run(qq, kk, vv, vb, z, cut, causal=True, **kw)
    S = s.shape[-1]
    vis = torch.ones_like(s, dtype=torch.bool)
    for t in range(q_len):
        for j in range(t + 1, q_len):
            vis[:, t::q_len, S - q_len + j] = False
    p = torch.where(
        vis,
        torch.exp2(s.double() - z[..., None].double()),
        torch.zeros_like(s, dtype=torch.float64),
    )
    # the whole mass of the keys the row sees, kept and dropped alike, which is
    # what `vmean` makes the denominator
    assert torch.allclose(den.double(), p.sum(-1), rtol=2e-2)
    # and the mask never adds a byte: a key is gathered when some row of the
    # group descends from it and keeps it, which is a subset of the rows that
    # would have kept it unmasked
    un = _run(qq, kk, vv, vb, z, cut, **dict(kw, draft_len=1))[2]
    assert bool((cnt <= un).all())
    assert bool((cnt < un).any()), "the case cannot show the mask saving a byte"


def test_a_draft_widens_the_group_the_config_rules_read():
    """A draft of `K` puts `K` rows per query head in one group, so the group
    the byte rules read is `Gq * K`. They already price it: a wide enough
    draft moves the V-format crossover past the live fraction that would have
    taken the 8-bit V without one."""
    live = 0.40
    assert pick_config(live, D=128, G=8)[0] is True
    assert pick_config(live, D=128, G=8 * 4)[0] is False
    assert v8_live_fraction(128, 8 * 4) > v8_live_fraction(128, 8)


def test_a_paged_draft_is_the_same_bits_as_a_contiguous_one():
    """The draft's keys are the last `q_len` of the **request**, which under a
    ragged batch is `seq_lens[b]` and not `S`. Paging moves an address and not
    a layout, so the two must still agree bitwise -- and a request shorter
    than the table's extent is where a mask anchored to the wrong length would
    show up."""
    S, hkv, page_size, q_len = 512, 2, 64, 4
    qs, k, v, _s, z = _case(S=S, NBH=8, G=2 * q_len)
    qq, kk, vv, vb, pt, (pka, pkb, pva, pvb, pek) = _paged(qs, k, v, S, page_size, hkv)
    cut = z - 8.0
    sl = torch.full((pt.shape[0],), S, device=k.device, dtype=torch.int32)
    kw = dict(truncate=1, refine_k=16.0, split=2, v8=True, draft_len=q_len, causal=True)
    a = _run(qq, kk, vv, vb, z, cut, **kw)[0]
    b = fold_decode(
        qq[0],
        qq[1],
        qq[2],
        pka,
        pkb,
        pek,
        pva,
        z,
        cut,
        v2=pvb,
        v_scale=vv[2],
        page_table=pt,
        page_size=page_size,
        seq_lens=sl,
        n_kv_heads=hkv,
        **kw,
    )[0]
    assert torch.equal(a, b), (a - b).abs().max().item()
    # the control: a shorter request puts its draft somewhere else entirely
    sl2 = sl.clone()
    sl2[0] = S - 64
    c = fold_decode(
        qq[0],
        qq[1],
        qq[2],
        pka,
        pkb,
        pek,
        pva,
        z,
        cut,
        v2=pvb,
        v_scale=vv[2],
        page_table=pt,
        page_size=page_size,
        seq_lens=sl2,
        n_kv_heads=hkv,
        **kw,
    )[0]
    assert not torch.equal(b[:hkv], c[:hkv])
    assert torch.equal(b[hkv:], c[hkv:])


def test_the_conventional_draft_layout_round_trips():
    """`(B, draft_len, H_q, D)` is FlashAttention's speculative shape; the kernel's
    is the transposed form. The pair that maps between them is tested rather
    than documented, because a caller that gets the row order wrong gets a
    plausible wrong answer."""
    B, T, Hq, D, hkv = 3, 4, 8, 16, 2
    x = torch.arange(B * T * Hq * D, dtype=torch.float32).reshape(B, T, Hq, D)
    p = pack_rows(x, hkv, T)
    assert p.shape == (B * hkv, (Hq // hkv) * T, D)
    assert torch.equal(unpack_rows(p, hkv, T), x)
    # row `g * q_len + t` of group `b * H_KV + h` is query head `h * G0 + g`
    g0 = Hq // hkv
    for b, h, g, t in ((0, 1, 2, 3), (2, 0, 0, 1)):
        assert torch.equal(p[b * hkv + h, g * T + t], x[b, t, h * g0 + g])
    # z and the cut go through the same pair, with no channel axis
    zz = torch.arange(B * T * Hq, dtype=torch.float32).reshape(B, T, Hq)
    assert torch.equal(unpack_rows(pack_rows(zz, hkv, T), hkv, T), zz)


def test_draft_mask_builds_the_ancestor_sets():
    assert torch.equal(draft_mask([-1, 0, 1, 2]), torch.tril(torch.ones(4, 4, dtype=torch.bool)))
    m = draft_mask([-1, 0, 0, 1, 1, 2, 2, 3])
    assert bool(m[3, 0]) and bool(m[3, 1]) and not bool(m[3, 2])
    assert bool(m.diagonal().all())
    with pytest.raises(ValueError, match="lower index"):
        draft_mask([-1, 2, 0])


def test_an_unsupported_draft_is_refused_by_name():
    q_len = 4
    _, _, _, _, z, qq, kk, vv, vb = _draft_case(Gq=2, draft_len=q_len)
    kw = dict(truncate=1, split=2, refine_k=10.0)
    chain = draft_mask([-1, 0, 1, 2])
    with pytest.raises(ValueError, match="two answers to one question"):
        _run(qq, kk, vv, vb, z, z - 8.0, draft_len=q_len, causal=True, mask=chain, **kw)
    with pytest.raises(ValueError, match="not a multiple of draft_len"):
        _run(qq, kk, vv, vb, z, z - 8.0, draft_len=3, causal=True, **kw)
    with pytest.raises(ValueError, match="at most 32"):
        _run(qq, kk, vv, vb, z, z - 8.0, draft_len=64, causal=True, **kw)
    bad = chain.clone()
    bad[2, 2] = False
    with pytest.raises(ValueError, match="own key"):
        _run(qq, kk, vv, vb, z, z - 8.0, draft_len=q_len, mask=bad, **kw)
    with pytest.raises(ValueError, match="leading axis is the row group"):
        _run(
            qq,
            kk,
            vv,
            vb,
            z,
            z - 8.0,
            draft_len=q_len,
            mask=chain[None].expand(3, q_len, q_len),
            **kw,
        )


def test_a_per_group_tree_is_not_one_tree_for_every_group():
    """The mask's leading axis is the row group, so two groups can verify
    different trees in one call."""
    q_len, NBH = 4, 4
    _, _, _, _, z, qq, kk, vv, vb = _draft_case(Gq=2, draft_len=q_len, NBH=NBH)
    kw = dict(truncate=1, split=2, refine_k=10.0, draft_len=q_len)
    chain = draft_mask([-1, 0, 1, 2])
    tree = draft_mask([-1, 0, 0, 1])
    per = torch.stack([chain, tree, chain, tree]).cuda()
    mix = _run(qq, kk, vv, vb, z, z - 8.0, mask=per, **kw)[0]
    a = _run(qq, kk, vv, vb, z, z - 8.0, mask=chain, **kw)[0]
    b = _run(qq, kk, vv, vb, z, z - 8.0, mask=tree, **kw)[0]
    assert torch.equal(mix[0::2], a[0::2])
    assert torch.equal(mix[1::2], b[1::2])
    assert not torch.equal(a[1::2], b[1::2])


def _cascade_case(NBH=4, S=1024, L=512, G=8, D=128, seed=3):
    """One cache, split at `L` into a prefix every row group shares and a
    suffix of its own.

    Every row group holds the same cache and a query of its own, which is the
    shape a shared prefix has. It is also what makes the comparison exact:
    a key's planes and scale are its own, so a cache that is equal across
    them quantises to equal bytes, and the prefix the shared level reads is
    then byte for byte the prefix the one-level call reads. Slicing
    the swizzled plane at a multiple of eight is the other half of that -- the
    swizzle's period -- and it is the rule `prepare_fold_decode` enforces.
    """
    qs, k, v, s, z = _case(NBH=NBH, S=S, D=D, G=G, seed=seed)
    k = k[0:1].expand(NBH, -1, -1).contiguous()
    v = v[0:1].expand(NBH, -1, -1).contiguous()
    s = torch.einsum("bgd,bsd->bgs", qs, k)
    z = torch.ceil(s.max(-1).values * 1024) / 1024
    return qs, k, v, s, z


def _cascade_split(qs, k, v, L, G, NBH):
    """The two levels' arguments, from one quantised cache."""
    ka, kb, ek = quantize_k(k)
    vb = v.bfloat16().contiguous()
    rows, G_s = cascade_rows(NBH, G, 1, None, k.device)
    shared = SharedPrefix(
        rows=rows,
        ka=ka[0:1, :L].contiguous(),
        kb=kb[0:1, :L].contiguous(),
        ek=pad_scales(ek[0:1, :L]),
        v=vb[0:1, :L].contiguous(),
        length=L,
    )
    uniq = dict(
        ka=ka[:, L:].contiguous(),
        kb=kb[:, L:].contiguous(),
        ek=pad_scales(ek[:, L:]),
        v=vb[:, L:].contiguous(),
    )
    return (ka, kb, ek, vb), uniq, shared, G_s


@pytest.mark.parametrize("T", [1e4, 8.0])
def test_a_cascade_is_the_cache_its_two_levels_concatenate(T):
    """The property the whole mechanism rests on.

    `Z` is a per-row reference known before either level runs, so a key's
    weight `2^(s - Z)` does not depend on which level read it and the two
    levels' partials are addends of one sum. The combine already sums partial
    slots, so the merge *is* that sum: there is no log-sum-exp to carry and no
    output to rescale. This asserts the consequence -- a shared prefix plus a
    unique suffix is the concatenated cache -- against the same bytes, so the
    only difference left is which slots the addends were grouped into.
    """
    NBH, S, L, G = 4, 1024, 512, 8
    qs, k, v, _s, z = _cascade_case(NBH=NBH, S=S, L=L, G=G)
    qq = quantize_kq(qs)
    (ka, kb, ek, vb), uniq, shared, _G_s = _cascade_split(qs, k, v, L, G, NBH)
    cut = z - float(T)
    skip = 0 if T > 1e3 else 1

    one = prepare_fold_decode(*qq, ka, kb, ek, vb, z, cut, truncate=skip, refine_k=16.0)
    o1, l1, c1 = one()
    two = prepare_fold_decode(
        *qq,
        uniq["ka"],
        uniq["kb"],
        uniq["ek"],
        uniq["v"],
        z,
        cut,
        truncate=skip,
        refine_k=16.0,
        shared=shared,
    )
    o2, l2, c2 = two()

    # the summation order of the partials is all that is left to differ
    scale = o1.abs().max().item()
    assert (o2 - o1).abs().max().item() / scale < 1e-5, (o2 - o1).abs().max().item() / scale
    assert (l2 - l1).abs().max().item() / l1.abs().max().item() < 1e-6

    # and the bytes went down: the prefix was read once for the whole batch
    # rather than once per row group. The shared level's live set is the
    # group's, which is the union over the rows stacked into it, so what must
    # fall is the total and not any one row group's count.
    cs = two.counts_shared.sum(0)
    assert int(c2[:, 0].sum() + cs[0]) < int(c1[:, 0].sum()), (
        int(c2[:, 0].sum() + cs[0]),
        int(c1[:, 0].sum()),
    )


def test_parallel_cascade_is_serial_to_the_bit_under_graph_replay():
    NBH, S, L, G = 4, 1024, 768, 8
    qs, k, v, _, z = _cascade_case(NBH=NBH, S=S, L=L, G=G)
    qq = quantize_kq(qs)
    _, uniq, shared, _ = _cascade_split(qs, k, v, L, G, NBH)
    cut = z - 8.0
    args = (*qq, uniq["ka"], uniq["kb"], uniq["ek"], uniq["v"], z, cut)
    kw = dict(truncate=1, refine_k=16.0, shared=shared)
    serial = prepare_fold_decode(*args, **kw)
    parallel = prepare_fold_decode(*args, parallel_shared=True, **kw)
    want = tuple(x.clone() for x in serial())
    got = parallel()
    torch.cuda.synchronize()
    assert all(torch.equal(x, y) for x, y in zip(got, want))

    for _ in range(3):
        parallel()
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        parallel()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        parallel()
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(x, y) for x, y in zip(got, want))


def test_a_cascade_carries_no_log_sum_exp():
    """The control for the claim above: the merge is an add, so the shared
    level's denominator lands in the same sum the splits do.

    A cascade that needed a rescale would have to know each level's own
    normaliser, and the denominator this call returns would be one level's
    rather than the whole cache's. Read it against the reference the two
    levels together define.
    """
    NBH, S, L, G = 4, 1024, 512, 8
    qs, k, v, s, z = _cascade_case(NBH=NBH, S=S, L=L, G=G)
    qq = quantize_kq(qs)
    _, uniq, shared, _ = _cascade_split(qs, k, v, L, G, NBH)
    cut = z - 8.0
    two = prepare_fold_decode(
        *qq,
        uniq["ka"],
        uniq["kb"],
        uniq["ek"],
        uniq["v"],
        z,
        cut,
        truncate=1,
        refine_k=16.0,
        shared=shared,
    )
    _o2, l2, _ = two()
    _, p = _ref(s, z, v, cut)
    den = p.sum(-1)
    assert (l2 - den).abs().max().item() / den.abs().max().item() < 2e-2


def test_a_cascade_gives_each_range_of_requests_its_own_prefix():
    """Two prefixes, two ranges of requests, one launch.

    Inside one range the map is a relabelling -- a stacked row reads the Q the
    map names and stores where the map says, so permuting it permutes
    nothing. What is not a relabelling is which range a request is in, and the
    control below swaps the two prefixes: a cascade that read the wrong one,
    or scattered a range's partials into the other's row groups, would still
    return plausible numbers.
    """
    NBH, S, L, G, D = 4, 1024, 512, 8, 128
    qs, k, v, _, _ = _case(NBH=NBH, S=S, D=D, G=G, seed=5)
    k = k.clone()
    v = v.clone()
    for lo, src in ((0, 0), (2, 3)):
        k[lo : lo + 2, :L] = k[src, :L]
        v[lo : lo + 2, :L] = v[src, :L]
    sf = torch.einsum("bgd,bsd->bgs", qs, k)
    z = torch.ceil(sf.max(-1).values * 1024) / 1024
    cut = z - 8.0
    ref, _ = _ref(sf, z, v, cut)

    qq = quantize_kq(qs)
    kau, kbu, eku = quantize_k(k[:, L:].contiguous())
    pre_k = torch.stack([k[0, :L], k[3, :L]]).contiguous()
    pre_v = torch.stack([v[0, :L], v[3, :L]]).contiguous()
    kap, kbp, ekp = quantize_k(pre_k)
    rows, G_s = cascade_rows(NBH, G, 1, [0, 2, 4], k.device)
    assert (rows.shape, G_s) == ((2, 16), 16)

    def run(order):
        shared = SharedPrefix(
            rows=rows,
            ka=kap[order].contiguous(),
            kb=kbp[order].contiguous(),
            ek=pad_scales(ekp[order]),
            v=pre_v[order].bfloat16().contiguous(),
            length=L,
        )
        o, _, _ = prepare_fold_decode(
            *qq,
            kau,
            kbu,
            eku,
            v[:, L:].bfloat16().contiguous(),
            z,
            cut,
            truncate=1,
            refine_k=16.0,
            shared=shared,
        )()
        return (o - ref).abs().max().item() / ref.abs().max().item()

    assert run([0, 1]) < 2e-2, run([0, 1])
    # the control: the same two prefixes, handed to the other range
    assert run([1, 0]) > 2e-2, run([1, 0])


def test_a_cascade_pads_a_sharing_range_it_cannot_fill():
    """Ranges of different widths leave padding rows, and a padding row is
    exactly a row outside the group: it stores nothing and moves no bytes.

    Three requests over two prefixes is the smallest case that has one, and
    the answer must not depend on the padding at all.
    """
    NBH, S, L, G = 4, 1024, 512, 8
    qs, k, v, _s, z = _cascade_case(NBH=NBH, S=S, L=L, G=G)
    qq = quantize_kq(qs)
    (ka, kb, ek, vb), uniq, _, _ = _cascade_split(qs, k, v, L, G, NBH)
    cut = z - 8.0
    # [0, 3, 4]: a range of three requests and one of a single request, so the
    # stacked width is three rows' worth and the second range pads two thirds
    rows, G_s = cascade_rows(NBH, G, 1, [0, 3, 4], k.device)
    assert G_s == 24
    shared = SharedPrefix(
        rows=rows,
        ka=ka[0:1, :L].expand(2, -1, -1)[:, :L].contiguous(),
        kb=kb[0:1, :L].expand(2, -1, -1)[:, :L].contiguous(),
        ek=pad_scales(ek[0:1, :L].expand(2, -1)),
        v=vb[0:1, :L].expand(2, -1, -1)[:, :L].contiguous(),
        length=L,
    )
    o2, l2, _ = prepare_fold_decode(
        *qq,
        uniq["ka"],
        uniq["kb"],
        uniq["ek"],
        uniq["v"],
        z,
        cut,
        truncate=1,
        refine_k=16.0,
        shared=shared,
    )()
    one = prepare_fold_decode(*qq, ka, kb, ek, vb, z, cut, truncate=1, refine_k=16.0)
    o1, l1, _ = one()
    assert (o2 - o1).abs().max().item() / o1.abs().max().item() < 1e-5
    assert (l2 - l1).abs().max().item() / l1.abs().max().item() < 1e-6


def _cascade_arm(qs, k, v, L, G, NBH, arm):
    """Both levels' operands for one cache format, from one quantisation.

    Slicing the quantised planes rather than quantising each level is what
    makes the comparison exact: `quantize_v` takes one scale for the whole
    tensor and a key's scale is its own, so the prefix the shared level reads
    is byte for byte the prefix the one-level call reads. The slice is
    at a multiple of eight, the swizzle's period, which is the rule
    `prepare_fold_decode` states.
    """
    v8 = arm.get("v8", False)
    ka, kb, ek = quantize_k(k)
    va, vbp, evs = quantize_v(v)
    vb = v.bfloat16().contiguous()
    kaf = ka
    vf = va if v8 else vb
    rows, _ = cascade_rows(NBH, G, 1, None, k.device)
    # Every gate open and no truncation, which is what leaves this testing
    # the cache path alone. Every gate in this kernel is one bit per key ORed
    # over the query group, so a stacked group of `degree * G` rows gathers
    # and refines a superset of what a group of `G` does -- a real difference
    # with its own test below, and one that only a complete live set empties.
    kw = dict(
        truncate=0,
        refine_k=1e4,
        refine_v=1e4,
        v8=v8,
        v_scale=evs if v8 else 1.0,
        v2=vbp if v8 else None,
    )
    if arm.get("v_regs") is not None:
        kw["v_regs"] = arm["v_regs"]
    one = dict(kw, ka=kaf, kb=kb, ek=ek, v=vf)
    sh = SharedPrefix(
        rows=rows,
        ka=kaf[0:1, :L].contiguous(),
        kb=kb[0:1, :L].contiguous(),
        ek=pad_scales(ek[0:1, :L]),
        v=vf[0:1, :L].contiguous(),
        length=L,
        v2=vbp[0:1, :L].contiguous() if v8 else None,
    )
    two = dict(
        kw,
        ka=kaf[:, L:].contiguous(),
        kb=kb[:, L:].contiguous(),
        ek=pad_scales(ek[:, L:]),
        v=vf[:, L:].contiguous(),
        shared=sh,
    )
    if v8:
        # every plane of a level is that level's: a sliced V beside a whole
        # second plane reads the second plane at the wrong key
        two["v2"] = vbp[:, L:].contiguous()
    return one, two


@pytest.mark.parametrize(
    "arm",
    [
        dict(v8=False),
        dict(v8=True),
        dict(v8=True, v_regs=1),
    ],
    ids=["bf16 V", "8-bit V", "8-bit V v_regs"],
)
def test_a_cascade_holds_under_every_cache_format(arm):
    """A cascade is an address, not a format.

    Each level reads the cache through the same tile path, so every arm has to
    come out where the one-level call over the concatenation does. The ones
    that could plausibly not: the register V operand, which converts
    its own bytes.
    """
    NBH, S, L, G = 4, 1024, 512, 8
    qs, k, v, _s, z = _cascade_case(NBH=NBH, S=S, L=L, G=G)
    qq = quantize_kq(qs)
    one, two = _cascade_arm(qs, k, v, L, G, NBH, arm)
    cut = z - 1e4

    def run(d):
        d = dict(d)
        args = (*qq, d.pop("ka"), d.pop("kb"), d.pop("ek"), d.pop("v"), z, cut)
        return _prepare(*args, **d)()

    o1, l1, _ = run(one)
    o2, l2, _ = run(two)
    assert (o2 - o1).abs().max().item() / o1.abs().max().item() < 2e-5
    assert (l2 - l1).abs().max().item() / l1.abs().max().item() < 1e-6


@pytest.mark.parametrize(
    "arm",
    [
        dict(v8=False),
        dict(v8=True),
    ],
    ids=["bf16 V", "8-bit V"],
)
def test_a_cascade_is_at_least_as_accurate_as_the_pass_it_replaces(arm):
    """Truncated, the two arms are *not* equal, and the difference has a sign.

    Every gate in this kernel is one bit per key ORed over the query group, so
    a stacked level's `degree * G` rows gather and refine a **superset** of
    what any one row group's `G` rows would, and none of those gates decides
    membership in the function -- the cut does, per (key, row), on the refined
    logit, and both arms apply the same one. So a shared prefix read once is at
    least as accurate as the same prefix read per request, for the reason a
    draft is at least as accurate as the steps it verifies (section 15).

    Where it shows is the **live** gate, not the refine gates: liveness is
    decided on the *coarse* logit, so a key whose coarse logit sits near the
    cut can clear it for the wider group and not for the narrow one, and its
    refined logit then enters one sum and not the other, which is why the
    equality test above needs the live set complete.
    """
    NBH, S, L, G = 4, 1024, 512, 8
    qs, k, v, s, z = _cascade_case(NBH=NBH, S=S, L=L, G=G)
    qq = quantize_kq(qs)
    one, two = _cascade_arm(qs, k, v, L, G, NBH, arm)
    cut = z - 8.0
    ref, _ = _ref(s, z, v, cut)

    def run(d):
        d = dict(d, refine_k=12.0, refine_v=8.0)
        args = (*qq, d.pop("ka"), d.pop("kb"), d.pop("ek"), d.pop("v"), z, cut)
        o = prepare_fold_decode(*args, **d)()[0]
        return (o - ref).abs().max().item(), (o - ref).square().sum().item()

    _m1, l1 = run(one)
    _m2, l2 = run(two)
    # Either the wider group admitted the same set, in which case the arms
    # differ only in which slots their addends were grouped into and the
    # difference has no sign, or it admitted more and the cascade is the more
    # accurate one. The two are not close: a flipped key moves L2 by a quarter
    # here and a regrouped sum moves it by 2e-5.
    assert abs(l2 - l1) <= 1e-3 * l1 or l2 < l1, (l2, l1)


def test_a_cascade_verifies_a_draft_over_a_shared_prefix():
    """A draft and a cascade compose: the draft's keys are the tail of the
    unique level, so the shared level carries no mask at all.

    They also compete, and the test is where: the stacked width is
    `degree * Gq * q_len`, so a draft of two halves the degree the same
    budget allows.
    """
    NBH, S, L, G, QL = 4, 1024, 512, 8, 2
    qs, k, v, _s, z = _cascade_case(NBH=NBH, S=S, L=L, G=G)
    qq = quantize_kq(qs)
    (ka, kb, ek, vb), uniq, shared, _ = _cascade_split(qs, k, v, L, G, NBH)
    cut = z - 8.0
    kw = dict(truncate=1, refine_k=16.0, draft_len=QL, causal=True)
    o1, l1, _ = prepare_fold_decode(*qq, ka, kb, ek, vb, z, cut, **kw)()
    o2, l2, _ = prepare_fold_decode(
        *qq, uniq["ka"], uniq["kb"], uniq["ek"], uniq["v"], z, cut, shared=shared, **kw
    )()
    assert (o2 - o1).abs().max().item() / o1.abs().max().item() < 2e-5
    assert (l2 - l1).abs().max().item() / l1.abs().max().item() < 1e-6


def test_a_cascade_takes_the_reference_the_prepass_computed():
    """The mass reference is the one serving uses, so a cascade has to run
    under it.

    The prepass scores the unique level's keys and writes one Z per (row
    group, row); both levels then read that same Z. The prefix's mass is not
    in the estimate, so a weight can exceed one there, which the weight's
    range allows; what this checks is the whole cache against the reference
    the kernel chose, not against a Z of its own.
    """
    NBH, S, L, G = 4, 1024, 512, 8
    qs, k, v, s, _ = _cascade_case(NBH=NBH, S=S, L=L, G=G)
    qq = quantize_kq(qs)
    _, uniq, shared, _ = _cascade_split(qs, k, v, L, G, NBH)
    zo = torch.empty((NBH, G), device=k.device)
    dep = torch.full((NBH, G), 8.0, device=k.device)
    lau = prepare_fold_decode(
        *qq,
        uniq["ka"],
        uniq["kb"],
        uniq["ek"],
        uniq["v"],
        zo,
        dep,
        truncate=1,
        reference="mass",
        refine_k=16.0,
        shared=shared,
        parallel_shared=True,
    )
    o, _den, _ = lau()
    zk = lau.z.clone()
    ref, _ = _ref(s, zk, v, zk - 8.0)
    err = (o - ref).abs().max().item() / ref.abs().max().item()
    assert err < 2e-2, err


def test_a_cascade_takes_v_s_format_per_piece():
    """A bf16 prefix beside an e4m3 suffix, which is the configuration the
    measurement wants: the shared level is out of issue slots rather than
    waiting on bytes, so halving V there costs instructions and buys nothing.

    The combine carries one `EVS` for the whole sum, so the bf16 piece's V is
    divided by it. That is exact rather than nearly so -- `quantize_v` takes a
    power-of-two scale, so the division only moves an exponent -- and the test
    is that the mixed call lands on the reference as squarely as the uniform
    one does.
    """
    NBH, S, L, G = 4, 1024, 512, 8
    qs, k, v, s, z = _cascade_case(NBH=NBH, S=S, L=L, G=G)
    qq = quantize_kq(qs)
    ka, kb, ek = quantize_k(k)
    va, vbp, evs = quantize_v(v)
    vb = v.bfloat16().contiguous()
    cut = z - 8.0
    ref, _ = _ref(s, z, v, cut)
    rows, _ = cascade_rows(NBH, G, 1, None, k.device)
    base = dict(truncate=1, refine_k=16.0, v8=True, v_scale=evs, v2=vbp[:, L:].contiguous())
    shr = SharedPrefix(
        rows=rows,
        ka=ka[0:1, :L].contiguous(),
        kb=kb[0:1, :L].contiguous(),
        ek=pad_scales(ek[0:1, :L]),
        v=vb[0:1, :L].contiguous(),
        length=L,
    )
    o, _den, _ = prepare_fold_decode(
        *qq,
        ka[:, L:].contiguous(),
        kb[:, L:].contiguous(),
        pad_scales(ek[:, L:]),
        va[:, L:].contiguous(),
        z,
        cut,
        shared=shr,
        **base,
    )()
    sc = ref.abs().max().item()
    assert (o - ref).abs().max().item() / sc < 2e-2
    # the control: the scale really is on the prefix's V. Undo the division
    # the host does and the prefix contributes at `evs` times its weight.
    bad = dataclasses.replace(shr, v=(vb[0:1, :L].float() * evs).bfloat16().contiguous())
    o2 = prepare_fold_decode(
        *qq,
        ka[:, L:].contiguous(),
        kb[:, L:].contiguous(),
        pad_scales(ek[:, L:]),
        va[:, L:].contiguous(),
        z,
        cut,
        shared=bad,
        **base,
    )()[0]
    assert (o2 - ref).abs().max().item() / sc > 2e-2
    # and the other direction is refused by name rather than computed
    with pytest.raises(ValueError, match="quantised prefix beside a bf16"):
        prepare_fold_decode(
            *qq,
            ka[:, L:].contiguous(),
            kb[:, L:].contiguous(),
            pad_scales(ek[:, L:]),
            vb[:, L:].contiguous(),
            z,
            cut,
            truncate=1,
            shared=dataclasses.replace(
                shr, v=va[0:1, :L].contiguous(), v2=vbp[0:1, :L].contiguous()
            ),
        )


def test_a_cascade_of_two_pieces_has_no_levels():
    """A prefix the batch shares and a prefix half of it shares, in one call.

    With the merge free there is no hierarchy to express: each piece takes its
    own partial slots and the combine sums all of them, so the pieces are
    addends and their order is nothing. FlashInfer's cascade needs a
    `merge_state_in_place` launch per level, which is why levels are a concept
    there.

    Three key ranges, so this also pins the alignment rule twice: every piece
    boundary is a multiple of eight because a key's swizzle phase comes from
    its own logical position.
    """
    NBH, S, G = 4, 1024, 8
    A, Bk = 256, 512
    qs, k, v, _s, z = _cascade_case(NBH=NBH, S=S, L=Bk, G=G)
    qq = quantize_kq(qs)
    ka, kb, ek = quantize_k(k)
    vb = v.bfloat16().contiguous()
    cut = z - 8.0
    kw = dict(truncate=1, refine_k=1e4)

    def piece(lo, hi, groups):
        rows, _ = cascade_rows(NBH, G, 1, groups, k.device)
        return SharedPrefix(
            rows=rows,
            ka=ka[0:1, lo:hi].contiguous(),
            kb=kb[0:1, lo:hi].contiguous(),
            ek=pad_scales(ek[0:1, lo:hi]),
            v=vb[0:1, lo:hi].contiguous(),
            length=hi - lo,
        )

    # keys [0, A) for every request, [A, B) for requests 0 and 1 only, and
    # [B, S) each request's own
    p0 = piece(0, A, [0, NBH])
    p1 = piece(A, Bk, [0, 2])
    o2, l2, _ = prepare_fold_decode(
        *qq,
        ka[:, Bk:].contiguous(),
        kb[:, Bk:].contiguous(),
        pad_scales(ek[:, Bk:]),
        vb[:, Bk:].contiguous(),
        z,
        cut,
        shared=[p0, p1],
        **kw,
    )()
    op, lp, _ = prepare_fold_decode(
        *qq,
        ka[:, Bk:].contiguous(),
        kb[:, Bk:].contiguous(),
        pad_scales(ek[:, Bk:]),
        vb[:, Bk:].contiguous(),
        z,
        cut,
        shared=[p0, p1],
        parallel_shared=True,
        **kw,
    )()
    torch.cuda.synchronize()
    assert torch.equal(op, o2)
    assert torch.equal(lp, l2)
    # requests 0 and 1 hold the whole cache; 2 and 3 hold it with [A, B) gone
    o1, _l1, _ = prepare_fold_decode(*qq, ka, kb, ek, vb, z, cut, **kw)()
    kx = torch.cat([k[:, :A], k[:, Bk:]], 1).contiguous()
    vx = torch.cat([v[:, :A], v[:, Bk:]], 1).contiguous()
    kax, kbx, ekx = quantize_k(kx)
    o3, _l3, _ = prepare_fold_decode(*qq, kax, kbx, ekx, vx.bfloat16().contiguous(), z, cut, **kw)()
    sc = o1.abs().max().item()
    assert (o2[:2] - o1[:2]).abs().max().item() / sc < 2e-5
    assert (o2[2:] - o3[2:]).abs().max().item() / sc < 2e-5
    # the control: the two halves of the batch got different key sets, so the
    # piece that covers only half of it is doing something
    assert (o1[2:] - o3[2:]).abs().max().item() / sc > 1e-3
    assert (
        len(
            prepare_fold_decode(
                *qq,
                ka[:, Bk:].contiguous(),
                kb[:, Bk:].contiguous(),
                pad_scales(ek[:, Bk:]),
                vb[:, Bk:].contiguous(),
                z,
                cut,
                shared=[p0, p1],
                **kw,
            ).counts_shared
        )
        == 2
    )


def test_cascade_rows_is_the_map_it_documents():
    """Host only: the packing, the padding and the widths."""
    rows, G_s = cascade_rows(4, 8, 1, None, "cpu")
    assert (rows.shape, G_s) == ((1, 32), 32)
    w = rows[0].tolist()
    assert all(x & 1 for x in w)
    assert [((x >> 1) >> 6, (x >> 1) & 63) for x in w[:9]] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (1, 0),
    ]
    # two kv heads: the row group is `request * n_kv_heads + head`, which is
    # what the kernel's own `bh` is
    rows, G_s = cascade_rows(2, 8, 2, None, "cpu")
    assert rows.shape == (2, 16)
    assert [(x >> 7) for x in rows[1].tolist()] == [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]
    # a range narrower than the widest pads, and a padding row points at a
    # readable row with its bit clear
    rows, G_s = cascade_rows(3, 8, 1, [0, 2, 3], "cpu")
    assert G_s == 16
    assert [x & 1 for x in rows[1].tolist()] == [1] * 8 + [0] * 8
    assert all((x >> 1) == (rows[1, 0].item() >> 1) for x in rows[1, 8:].tolist())
    # past 64 rows a range is a wide level, a multiple of eight rows
    rows, G_s = cascade_rows(16, 8, 1, [0, 16], "cpu")
    assert (rows.shape, G_s) == ((1, 128), 128)
    rows, G_s = cascade_rows(9, 8, 1, [0, 9], "cpu")
    assert G_s == 72
    with pytest.raises(ValueError, match="indptr"):
        cascade_rows(4, 8, 1, [0, 5], "cpu")
    # a piece may cover a run of the batch rather than all of it
    rows, G_s = cascade_rows(4, 8, 1, [2, 4], "cpu")
    assert (rows.shape, G_s) == ((1, 16), 16)
    assert [(x >> 7) for x in rows[0].tolist()][:1] == [2]
    # left to itself the map chunks a batch of up to 64 rows at the measured
    # degree, 32 rows' worth (`cascade_degree`), and stacks a wider batch whole
    # for the wide kernel
    assert [cascade_degree(g) for g in (1, 2, 4, 8, 16, 32, 64)] == [32, 16, 8, 4, 2, 1, 1]
    rows, G_s = cascade_rows(8, 8, 1, None, "cpu")
    assert (rows.shape, G_s) == ((2, 32), 32)
    assert [(x >> 7) for x in rows[1].tolist()][:2] == [4, 4]
    rows, G_s = cascade_rows(16, 8, 1, None, "cpu")
    assert (rows.shape, G_s) == ((1, 128), 128)


def test_a_cascade_states_its_prefix_alignment_and_its_reference():
    """Two rules a caller has to meet, stated where they are met."""
    NBH, S, L, G = 4, 1024, 512, 8
    qs, k, v, _s, z = _cascade_case(NBH=NBH, S=S, L=L, G=G)
    qq = quantize_kq(qs)
    _, uniq, shared, _ = _cascade_split(qs, k, v, L, G, NBH)
    cut = z - 8.0
    kw = dict(truncate=1)
    args = (*qq, uniq["ka"], uniq["kb"], uniq["ek"], uniq["v"], z, cut)
    with pytest.raises(ValueError, match="multiple of eight"):
        prepare_fold_decode(*args, shared=dataclasses.replace(shared, length=L - 4), **kw)
    with pytest.raises(TypeError, match="SharedPrefix"):
        prepare_fold_decode(*args, shared=dataclasses.asdict(shared), **kw)
    with pytest.raises(ValueError, match="vmean for both levels or neither"):
        prepare_fold_decode(
            *args, shared=shared, **kw, vmean=torch.zeros((NBH, 128), device=k.device)
        )


def _tail_case(mode, rank=16, NBH=8, S=2048, D=128, G=8, seed=5):
    """A cache on which a tail is exact. `blocks`: V constant inside every
    64-key block, so each block's mean is every dropped key's V. `linear`: V
    is its block's mean plus a rank-`rank` linear image of K's deviation from
    its own block mean, which is the whole of the rank's model. K and Q are
    `_case`'s, so the truncation has a real live set to cut."""
    qs, k, _, s, z = _case(NBH=NBH, S=S, D=D, G=G, seed=seed)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    nb = -(-S // 64)
    vb = torch.randn(NBH, nb, D, device="cuda", generator=g) + 1.5
    v = vb.repeat_interleave(64, 1)[:, :S]
    if mode == "linear":
        a = torch.randn(NBH, D, rank, device="cuda", generator=g) / math.sqrt(D)
        b = torch.randn(NBH, rank, D, device="cuda", generator=g)
        kh = hadamard(k)
        km = kh.reshape(NBH, nb, 64, D).mean(2).repeat_interleave(64, 1)[:, :S]
        v = v + (kh - km) @ a @ b
    return qs, k, v.contiguous(), s, z


def _tail_run(qs, k, v, z, T, *, rank=None, vmean=False, **kw):
    """Decode with the tail at `rank` (None for none), or the V mean."""
    qq, kk, vv, vb = _pack(qs, k, v)
    ka, ek = kk[0], kk[2]
    if rank is not None:
        kw["tail"] = tail_model(ka, ek, v, rank=rank)
    elif vmean:
        kw["vmean"] = _vmean(v, vv, False)
        kw.setdefault("group_cut", 1)
    return _run(qq, kk, vv, vb, z, z - T, truncate=1, **kw)


def _l2(o, ref):
    return ((o - ref).norm() / ref.norm()).item()


def test_the_tail_is_exact_where_v_is_its_block_mean():
    """Every dropped key re-enters at its own 64-key block's row. Where V is
    constant inside each block that row is each dropped key's V, so the
    truncated answer is the full softmax's, as close as the dense kernel
    gets."""
    qs, k, v, s, z = _tail_case("blocks")
    full = _ref(s, z, v, z - 1e4)[0]
    dense = _l2(_tail_run(qs, k, v, z, 1e4, rank=0)[0], full)
    for T in (4.0, 6.0):
        tail = _l2(_tail_run(qs, k, v, z, T, rank=0)[0], full)
        mean = _l2(_tail_run(qs, k, v, z, T, vmean=True)[0], full)
        assert tail < 1.1 * dense, (T, tail, dense)
        # the control: one mean per head is far off on the same cache
        assert mean > 3 * dense, (T, mean, dense)


@pytest.mark.parametrize("rank", [16, 32])
def test_the_rank_is_exact_where_v_is_linear_in_k(rank):
    """The rank adds the part of a dropped key's V that its K predicts. Where
    V is exactly its block mean plus a rank-r image of K's deviation, the
    truncated answer is the full one again; the block rows alone are not."""
    qs, k, v, s, z = _tail_case("linear", rank=rank)
    full = _ref(s, z, v, z - 1e4)[0]
    dense = _l2(_tail_run(qs, k, v, z, 1e4, rank=rank)[0], full)
    for T in (6.0, 8.0):
        got = _l2(_tail_run(qs, k, v, z, T, rank=rank)[0], full)
        blk = _l2(_tail_run(qs, k, v, z, T, rank=0)[0], full)
        assert got < 1.15 * dense, (T, got, dense)
        assert blk > 3 * dense, (T, blk, dense)


@pytest.mark.parametrize("rank", [0, 16])
def test_the_tail_denominator_is_the_total_mass(rank):
    """Each tile's dropped mass re-enters as the weight of its block row, so
    the denominator is the whole softmax's, to the weights' own rounding."""
    qs, k, v, s, z = _case()
    total = torch.exp2(s - z[..., None]).sum(-1)
    den = _tail_run(qs, k, v, z, 8.0, rank=rank)[1]
    assert ((den - total).abs() / total).max().item() < 1e-2
    # the control: without a correction the denominator is the kept mass
    live = _tail_run(qs, k, v, z, 8.0, group_cut=1)[1]
    assert ((live - total).abs() / total).max().item() > 2e-3


@pytest.mark.parametrize("split", [1, 3, 4])
def test_the_tail_survives_the_split(split):
    qs, k, v, _s, z = _case(S=2048)
    base = _tail_run(qs, k, v, z, 8.0, rank=16, split=1)[0]
    got = _tail_run(qs, k, v, z, 8.0, rank=16, split=split)[0]
    assert (got - base).abs().max().item() / base.abs().max().item() < 1e-5


def test_the_tail_is_ordered_by_what_it_knows():
    """On a cache with no exact structure the block rows still beat one mean,
    and the rank beats the block rows: each knows strictly more of the
    dropped keys' V."""
    qs, k, v, s, z = _case(S=2048)
    full = _ref(s, z, v, z - 1e4)[0]
    mean = _l2(_tail_run(qs, k, v, z, 8.0, vmean=True)[0], full)
    blk = _l2(_tail_run(qs, k, v, z, 8.0, rank=0)[0], full)
    assert blk < mean, (blk, mean)
    rk = _l2(_tail_run(qs, k, v, z, 8.0, rank=16)[0], full)
    assert rk <= blk * 1.02, (rk, blk)


@pytest.mark.parametrize("S", [1024, 5000])
def test_the_tail_pages_like_the_contiguous_call(S):
    """Past 64 tiles a CTA's ring of tail entries re-enters in more than one
    batch, so the long case pins the flush too."""
    qs, k, v, _s, z = _case(S=S)
    qq, kk, vv, vb = _pack(qs, k, v)
    hkv = 2
    NBH, S, _D = v.shape
    tail = tail_model(kk[0], kk[2], v, rank=16)
    flat = _run(qq, kk, vv, vb, z, z - 8.0, truncate=1, tail=tail, split=1)
    ps = 256
    B = NBH // hkv
    pages = -(-S // ps)
    table = torch.arange(B * pages, device="cuda", dtype=torch.int32).flip(0).reshape(B, pages)
    pka, pkb, pva, _pvb = paged_cache(kk[0], kk[1], vb, vb, table, ps, hkv)
    pek = paged_scale(kk[2], table, ps, hkv)
    seq = torch.full((B,), S, device="cuda", dtype=torch.int32)
    got = fold_decode(
        qq[0],
        qq[1],
        qq[2],
        pka,
        pkb,
        pek,
        pva,
        z,
        z - 8.0,
        truncate=1,
        split=1,
        tail=tail.paged(table, ps, hkv),
        page_table=table,
        page_size=ps,
        seq_lens=seq,
        n_kv_heads=hkv,
    )
    assert torch.equal(flat[0], got[0])
    assert torch.equal(flat[1], got[1])


def test_scales_past_the_end_can_be_anything():
    """A short last tile copies whole 16-byte units of scales, so up to seven
    pool rows past the sequence reach shared memory, and the tail multiplies
    their projections before a zero weight meets them. NaN there changes no
    bit."""
    qs, k, v, _s, z = _case(S=1001)
    qq, kk, _vv, vb = _pack(qs, k, v)
    hkv, ps = 2, 256
    NBH, S, _D = v.shape
    B = NBH // hkv
    pages = -(-S // ps)
    table = torch.arange(B * pages, device="cuda", dtype=torch.int32).reshape(B, pages)
    pka, pkb, pva, _pvb = paged_cache(kk[0], kk[1], vb, vb, table, ps, hkv)
    tail = tail_model(kk[0], kk[2], v, rank=16).paged(table, ps, hkv)
    seq = torch.full((B,), S, device="cuda", dtype=torch.int32)
    outs = []
    for fill in (0.0, float("nan")):
        ek = torch.full((NBH, pages * ps), fill, device="cuda", dtype=torch.bfloat16)
        ek[:, :S] = kk[2][:, :S]
        outs.append(
            fold_decode(
                qq[0],
                qq[1],
                qq[2],
                pka,
                pkb,
                paged_scale(ek, table, ps, hkv),
                pva,
                z,
                z - 8.0,
                truncate=1,
                split=1,
                tail=tail,
                page_table=table,
                page_size=ps,
                seq_lens=seq,
                n_kv_heads=hkv,
            )
        )
    assert torch.isfinite(outs[0][0]).all()
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])


def test_the_tail_composes_with_the_mass_reference_and_a_draft():
    """Under the mass reference the tail still beats the mean, and a draft's
    masked pairs carry no mass; both stay close to the reference."""
    qs, k, v, s, z = _case(S=2048)
    full = _ref(s, z, v, z - 1e4)[0]
    qq, kk, vv, vb = _pack(qs, k, v)
    tail = tail_model(kk[0], kk[2], v, rank=16)
    o = _run(
        qq,
        kk,
        vv,
        vb,
        torch.empty_like(z),
        torch.full_like(z, 8.0),
        truncate=1,
        tail=tail,
        reference="mass",
    )[0]
    m = _run(
        qq,
        kk,
        vv,
        vb,
        torch.empty_like(z),
        torch.full_like(z, 8.0),
        truncate=1,
        vmean=_vmean(v, vv, False),
        group_cut=1,
        reference="mass",
    )[0]
    assert _l2(o, full) < _l2(m, full)
    # a two-position chain draft over the same cache: row g*2 + t sees the
    # draft's own keys by the mask, and its masked pairs are exact zeros
    chain = torch.tril(torch.ones(2, 2, dtype=torch.bool))
    dfull = _tree_ref(s, z, v, z - 1e4, chain, 2)
    t = _l2(
        _run(qq, kk, vv, vb, z, z - 8.0, truncate=1, tail=tail, draft_len=2, causal=True)[0], dfull
    )
    m = _l2(
        _run(
            qq,
            kk,
            vv,
            vb,
            z,
            z - 8.0,
            truncate=1,
            vmean=_vmean(v, vv, False),
            group_cut=1,
            draft_len=2,
            causal=True,
        )[0],
        dfull,
    )
    assert t < m, (t, m)


def test_the_tail_is_refused_where_it_cannot_run():
    qs, k, v, _s, z = _case()
    qq, kk, vv, vb = _pack(qs, k, v)
    tail = tail_model(kk[0], kk[2], v, rank=16)
    with pytest.raises(ValueError, match="truncates nothing"):
        _run(qq, kk, vv, vb, z, z - 1e4, truncate=0, tail=tail)
    with pytest.raises(ValueError, match="one or the other"):
        _run(qq, kk, vv, vb, z, z - 8.0, truncate=1, tail=tail, vmean=_vmean(v, vv, False))
    with pytest.raises(ValueError, match="one per 64"):
        bad = Tail(tail.vblk[:, :-1].contiguous(), tail.u, tail.vr)
        _run(qq, kk, vv, vb, z, z - 8.0, truncate=1, tail=bad)
    with pytest.raises(ValueError, match="rank"):
        tail_model(kk[0], kk[2], v, rank=12)
    with pytest.raises(ValueError, match="whole number of blocks"):
        table = torch.zeros((4, 8), device="cuda", dtype=torch.int32)
        tail.paged(table, 32, 2)
    with pytest.raises(ValueError, match="bf16 V"):
        _run(qq, kk, vv, vb, z, z - 8.0, v8=True, truncate=1, tail=tail)


def test_the_tail_rank_follows_the_measured_policy():
    from fold_attention.decode.heuristics import tail_rank_for

    assert tail_rank_for(128, 8, False) == 16
    assert tail_rank_for(64, 16, False) == 16
    assert tail_rank_for(64, 32, False) == 0
    assert tail_rank_for(128, 32, False) == -1
    assert tail_rank_for(128, 8, True) == -1


def test_the_mass_gates_follow_the_measured_rule():
    from fold_attention.decode.heuristics import refine_for, weight_terms_for

    def mass(depth, v8=False, S=4096, D=128, G=8, tail=True):
        return refine_for(depth, v8, seq_len=S, head_dim=D, group=G, tail=tail)

    assert mass(None) == (10.0, 8.0) and mass(None, S=32768) == (12.0, 8.0)
    assert mass(None, G=1)[0] == 12.0 and mass(None, G=1, S=32768)[0] == 12.0
    assert mass(None, D=64, G=8, S=8192)[0] == 12.0 and mass(None, D=64, G=8, S=32768)[0] == 10.0
    assert mass(None, v8=True) == (10.0, 10.0) and mass(None, v8=True, S=32768) == (10.0, 12.0)
    assert mass(12.0) == (8.0, 8.0) and mass(12.0, tail=False)[0] == 10.0
    assert mass(14.0)[0] == 8.0 and mass(16.0)[0] == 10.0
    assert mass(18.0)[0] == 12.0 and mass(20.0)[0] == 12.0
    assert mass(14.0, True) == (6.0, 8.0) and mass(16.0, True) == (10.0, 12.0)
    assert mass(18.0, True) == (10.0, 10.0) and mass(20.0, True) == (12.0, 10.0)
    assert mass(14.0, True, D=64, G=1) == (10.0, 10.0)
    assert mass(18.0, True, D=64, G=4) == (8.0, 8.0) and mass(None, True, D=64, G=4) == (10.0, 8.0)
    assert weight_terms_for(14.0) == 1 and weight_terms_for(16.0) == 2
    assert weight_terms_for(None) == 2
    # a group between the measured ones takes the next wider class, and D = 64
    # has no class past 8
    assert mass(None, G=2) == mass(None, G=1) and mass(None, G=6) == mass(None, G=8)
    assert mass(None, D=64, G=16) == mass(None, D=64, G=8)


def test_the_front_register_rule_follows_the_measurements():
    from fold_attention.decode.heuristics import front_min_blocks, resident

    # a D128 register-V build up to G = 8 always asks for the deep cap's six
    assert front_min_blocks(128, 8, True, True, True, -1, 2) == 6
    assert resident(True, True, D=128, G=8, truncate=True, front=True, weight_terms=2) == 6
    # the D128/G8 tail builds hold six uncapped, their weights in plane B's
    # tile, at one split too
    assert front_min_blocks(128, 8, False, False, True, 16, 1) == 0
    assert front_min_blocks(128, 8, False, False, True, 16, 2) == 0
    assert front_min_blocks(128, 8, False, False, True, 16, 2, one=True) == 0
    for terms in (1, 2):
        assert (
            resident(
                False, False, D=128, G=8, truncate=True, front=True, tail=16, weight_terms=terms
            )
            == 6
        )
    # D64 never squeezes
    assert front_min_blocks(64, 8, True, True, True, -1, 1) == 0
    assert front_min_blocks(64, 8, False, False, True, 16, 1) == 0


def _wide_levels(NBH, S, L, G, seed=5, T=1e4, draft=0):
    """A prefix every request shares, stacked whole into one wide level, with
    the one-level call over the concatenation beside it: `(one, cascade,
    reference)` launches and the fp32 answer. The cascade takes `image` or
    `paged` through `kw`."""
    qs, k, v, s, z = _cascade_case(NBH=NBH, S=S, L=L, G=G, seed=seed)
    qq = quantize_kq(qs)
    ka, kb, ek = quantize_k(k)
    vb = v.bfloat16().contiguous()
    cut = z - float(T)
    skip = int(T < 1e3)
    kw = dict(truncate=skip, refine_k=16.0)
    if draft:
        kw.update(draft_len=draft, causal=True)
    one = prepare_fold_decode(*qq, ka, kb, ek, vb, z, cut, **kw)
    rows, G_s = cascade_rows(NBH, G, 1, None, k.device)
    assert G_s > 64
    planes = (ka[0:1, :L].contiguous(), kb[0:1, :L].contiguous(), pad_scales(ek[0:1, :L]))

    def cascade(image=False, page=0):
        pk: dict = dict(
            ka=planes[0], kb=planes[1], ek=planes[2], v=vb[0:1, :L].contiguous(), length=L
        )
        if image:
            pk["image"] = prefix_image(*planes, pk["v"])
        if page:
            table = torch.randperm(L // page, device=k.device, dtype=torch.int32)[None]
            pka, pkb, pv = paged_cache(planes[0], planes[1], pk["v"], pk["v"], table, page, 1)[:3]
            pk.update(
                ka=pka,
                kb=pkb,
                v=pv,
                ek=paged_scale(planes[2], table, page, 1),
                page_table=table,
                seq_lens=torch.tensor([L], device=k.device, dtype=torch.int32),
            )
        uk = (ka[:, L:].contiguous(), kb[:, L:].contiguous(), pad_scales(ek[:, L:]))
        return prepare_fold_decode(
            *qq,
            *uk,
            vb[:, L:].contiguous(),
            z,
            cut,
            shared=SharedPrefix(rows=rows, **pk),
            page_size=page,
            **kw,
        )

    return one, cascade, s, z, v, cut


@pytest.mark.parametrize("T", [1e4, 8.0])
def test_a_wide_cascade_reads_the_prefix_once_for_the_whole_batch(T):
    """Past 64 stacked rows a shared level runs on the wide kernel, rows along
    M, and one range holds the whole batch. Its logits are fp16 (Q and K each
    rounded once from their planes), so it is not the one-level call to the
    bit; it is the same attention to within that rounding, against fp32."""
    NBH, S, L, G = 16, 1024, 512, 8
    one, cascade, s, z, v, cut = _wide_levels(NBH, S, L, G, T=T)
    ref, _ = _ref(s, z, v, cut)
    o1 = one()[0]
    o2 = cascade(image=True)()[0]
    scale = ref.abs().max().item()
    e1 = (o1 - ref).abs().max().item() / scale
    e2 = (o2 - ref).abs().max().item() / scale
    assert e2 < 3 * e1 + 2e-3, (e2, e1)


def test_a_prefix_image_is_the_conversion_the_kernel_would_do():
    """The image holds the fp16 K the kernel rounds from the planes on every
    step when it has none, so the two paths give the same bits, from a paged
    prefix too, whose tiles span four pages."""
    NBH, S, L, G = 16, 1024, 512, 8
    _one, cascade, *_ = _wide_levels(NBH, S, L, G, T=8.0)
    a = cascade(image=True)()
    b = tuple(x.clone() for x in cascade()())
    c = cascade(page=16)()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert torch.equal(a[0], c[0]) and torch.equal(a[1], c[1])


def test_a_draft_verifies_over_a_wide_prefix():
    """A draft's rows stack like requests do: 8 positions of 2 heads, eight
    requests, 128 rows on one read of the prefix, the mask in the unique level."""
    NBH, S, L, G = 8, 1024, 512, 16
    one, cascade, *_ = _wide_levels(NBH, S, L, G, draft=8)
    o1, l1, _ = one()
    o2, l2, _ = cascade(image=True)()
    assert (o2 - o1).abs().max().item() / o1.abs().max().item() < 5e-3
    assert (l2 - l1).abs().max().item() / l1.abs().max().item() < 5e-3


@pytest.mark.parametrize("G", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["dense", "vmean paged", "tail", "tail two terms"])
def test_the_packed_tile_is_the_64_key_tile_to_rounding(G, mode):
    """Two keys a row of the logit's M form the products the 64-key tile
    forms, so every verdict is the same bit: only the value matmul's grouping
    and, under the tail, the virtual key's span (32 keys, not 16) move an
    output."""
    qs, k, v, s, z = _case(D=64, G=G, S=3000)
    qq, kk, vv, vb = _pack(qs, k, v)
    NBH, S, _D = v.shape
    kw: dict = dict(split=3)
    cut = z - 1e4 if mode == "dense" else z - 8.0
    kw["truncate"] = int(mode != "dense")
    if mode.startswith("tail"):
        kw["tail"] = tail_model(kk[0], kk[2], v, rank=16)
        kw["weight_terms"] = 2 if mode == "tail two terms" else 1
    if mode == "vmean paged":
        # pages narrower than a tile: two segments of 64, the last one short
        hkv, ps = 2, 64
        B = NBH // hkv
        pages = -(-S // ps)
        table = torch.randperm(B * pages, device="cuda", dtype=torch.int32).reshape(B, pages)
        pka, pkb, pva, _pvb = paged_cache(kk[0], kk[1], vb, vb, table, ps, hkv)
        kk = (pka, pkb, paged_scale(kk[2], table, ps, hkv))
        vb = pva
        seq = torch.full((B,), S, device="cuda", dtype=torch.int32)
        kw.update(
            vmean=v.float().mean(1),
            page_table=table,
            page_size=ps,
            seq_lens=seq,
            n_kv_heads=hkv,
        )
    outs = {p: _run(qq, kk, vv, vb, z, cut, pack=p, **kw) for p in (1, 2)}
    assert torch.equal(outs[1][2], outs[2][2])
    o1, o2 = outs[1][0], outs[2][0]
    if mode.startswith("tail"):
        # both against the full softmax, which the tail approximates
        want, _ = _ref(s, z, v.float(), z - 1e4)
        e1 = float((o1 - want).abs().max())
        e2 = float((o2 - want).abs().max())
        assert e2 <= 1.25 * e1 + 1e-4, (e1, e2)
    else:
        assert float((o2 - o1).abs().max() / o1.abs().max()) < 1e-5


def test_the_packed_tile_is_refused_where_it_does_not_apply():
    qs, k, v, _s, z = _case(D=128, G=4, S=512)
    qq, kk, vv, vb = _pack(qs, k, v)
    with pytest.raises(ValueError, match="pack=2"):
        _prep(qq, kk, vv, vb, z, z - 8.0, truncate=1, pack=2)


def test_a_draft_past_64_rows_runs_wide_and_is_its_masked_softmax():
    """Eight query heads verifying a 16-node tree are 128 rows, which a single
    level runs on the wide kernel. Its logits are fp16, and the tree's mask and
    the cut still hold: the output is the masked truncated softmax."""
    q_len, Gq = 16, 8
    tree = draft_mask([-1] + [i // 2 for i in range(q_len - 1)])
    _qs, _k, v, s, z, qq, kk, vv, vb = _draft_case(Gq=Gq, draft_len=q_len, S=2048, tchk=None)
    kw = dict(truncate=1, split=3, draft_len=q_len, mask=tree)
    o = _prep(qq, kk, vv, vb, z, z - 8.0, **kw)()[0]
    ref = _tree_ref(s, z, v, z - 8.0, tree, q_len)
    err = (o - ref).abs().max().item() / ref.abs().max().item()
    assert err < 2e-2, err


def test_a_wide_draft_takes_the_mass_reference_it_estimates():
    """At 128 rows the mass prepass still estimates each row's reference, and
    the wide kernel decodes against it exactly as against the same reference
    given: the two calls are the same bits."""
    q_len, Gq = 16, 8
    _qs, _k, _v, _s, z, qq, kk, vv, vb = _draft_case(Gq=Gq, draft_len=q_len, S=2048, tchk=None)
    kw = dict(truncate=1, split=3, draft_len=q_len, causal=True)
    mass = _prep(qq, kk, vv, vb, z.clone(), torch.full_like(z, 8.0), reference="mass", **kw)
    om, lm, _ = (x.clone() for x in mass())
    zm = mass.z.clone()
    og, lg, _ = _prep(qq, kk, vv, vb, zm, zm - 8.0, **kw)()
    assert torch.equal(om, og) and torch.equal(lm, lg)


def test_a_draft_past_64_rows_refuses_what_the_wide_kernel_lacks():
    q_len, Gq = 16, 8
    _qs, _k, _v, _s, z, qq, kk, vv, vb = _draft_case(Gq=Gq, draft_len=q_len, S=1024, tchk=None)
    with pytest.raises(ValueError, match="wide kernel"):
        _prep(qq, kk, vv, vb, z, z - 8.0, truncate=1, v8=True, draft_len=q_len, causal=True)
    with pytest.raises(ValueError, match="wide kernel"):
        _prep(qq, kk, vv, vb, z, z - 8.0, truncate=1, sound=True, draft_len=q_len, causal=True)


def test_a_32_node_chain_is_its_causal_mask():
    """Bit 31 of a 32-node draft's mask is the int32 sign: the chain spelled
    as `causal=True` and as its ancestor mask pack to the same words."""
    chain = pack_tree(None, True, 32, 64, 2, "cpu")
    tree = pack_tree(draft_mask(list(range(-1, 31))), False, 32, 64, 2, "cpu")
    assert torch.equal(chain, tree)
    assert int(chain[0, 31]) == -1
