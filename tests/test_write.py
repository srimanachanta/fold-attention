"""Every cache write is `decode.cache`'s torch quantiser to the bit: the step's
Q and appended token, and the prompt's.

Bitwise because a plane that differs by one unit moves a key across a gate,
and then the live set, and with it every output.
"""

from __future__ import annotations

import math

import pytest
import torch

cuda = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
if cuda:
    from exact_fold_attn.decode.cache import quantize_kq, write_kv, write_rows
    from exact_fold_attn.decode.write import step, write_prompt

pytestmark = pytest.mark.skipif(not cuda, reason="needs an SM90 GPU")

LOG2E = 1.4426950408889634


def _state(B, H, HKV, D, PS, ppr, v8):
    def z8(n):
        return torch.zeros(n, D, device="cuda", dtype=torch.int8)

    rows = B * ppr * HKV * PS
    vdt = torch.float8_e4m3fn if v8 else torch.bfloat16
    return dict(
        qa=z8(B * H),
        qb=z8(B * H),
        eq=torch.zeros(B * H, device="cuda"),
        ka=z8(rows),
        kb=z8(rows),
        ek=torch.zeros(rows, device="cuda", dtype=torch.bfloat16),
        va=torch.zeros(rows, D, device="cuda", dtype=vdt),
        vb=torch.zeros(rows, D, device="cuda", dtype=vdt) if v8 else None,
        vsum=torch.zeros(B * HKV, D, device="cuda"),
        vmean=torch.zeros(B * HKV, D, device="cuda"),
    )


def _run(st, q, k, v, pt, PS, lens, evs, lens_out):
    step(
        q,
        k,
        v,
        st["qa"],
        st["qb"],
        st["eq"],
        st["ek"],
        pt,
        PS,
        lens,
        st["ka"],
        st["kb"],
        st["va"],
        st["vb"],
        evs,
        st["vsum"],
        st["vmean"],
        lens_out,
    )


def _case(B=6, H=32, HKV=4, D=128, PS=64, ppr=8, seed=3, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(B * H, D, device="cuda", generator=g) * 3).to(dtype)
    k = (torch.randn(B, HKV, D, device="cuda", generator=g) * 2).to(dtype)
    v = (torch.randn(B, HKV, D, device="cuda", generator=g) + 1).to(dtype)
    lens = torch.randint(1, ppr * PS - 1, (B,), device="cuda", generator=g).int()
    pt = torch.randperm(B * ppr, device="cuda", generator=g).int().view(B, ppr)
    return q, k, v, lens, pt


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("G", [1, 8, 16])
def test_q_planes_are_quantize_kq(D, G, dtype):
    B, HKV = 5, 2
    q, _, _, lens, pt = _case(B=B, H=G * HKV, HKV=HKV, D=D, dtype=dtype)
    q[3] = 0
    st = _state(B, G * HKV, HKV, D, 64, 8, True)
    _run(st, q, None, None, pt, 64, lens, 1.0, lens)
    qa, qb, eq = quantize_kq(q.float() * (LOG2E / math.sqrt(D)))
    assert torch.equal(st["qa"], qa)
    assert torch.equal(st["qb"], qb)
    assert torch.equal(st["eq"], eq.squeeze(-1))


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("v8", [False, True])
@pytest.mark.parametrize("HKV", [1, 4, 8])
def test_the_append_is_write_kv(D, v8, HKV):
    """K's planes and scale, V's planes, the V sum and mean, the length
    advanced."""
    B, PS = 6, 64
    _, k, v, lens, pt = _case(B=B, HKV=HKV, D=D, PS=PS)
    got, want = _state(B, 8, HKV, D, PS, 8, v8), _state(B, 8, HKV, D, PS, 8, v8)
    sums = torch.randn(B * HKV, D, device="cuda")
    got["vsum"].copy_(sums)
    evs = 0.25 if v8 else 1.0
    out = torch.empty_like(lens)
    _run(got, None, k, v, pt, PS, lens, evs, out)
    assert torch.equal(out, lens + 1)
    if v8:
        write_kv(
            want["ka"],
            want["kb"],
            want["ek"],
            want["va"],
            want["vb"],
            k,
            v,
            evs,
            pt,
            lens.long(),
            PS,
            HKV,
        )
    else:
        pos = lens.long()
        row = (
            pt[torch.arange(B), pos // PS].long()[:, None] * HKV + torch.arange(HKV, device="cuda")
        ) * PS + (pos % PS)[:, None]
        write_rows(want["ka"], want["kb"], want["ek"], want["va"], None, k, v, evs, row, pos % PS)
    for n in ("ka", "kb", "ek", "va", "vb"):
        if got[n] is not None:
            assert torch.equal(got[n].view(torch.int8), want[n].view(torch.int8)), n
    want_sum = sums + v.float().reshape(B * HKV, D)
    assert torch.equal(got["vsum"], want_sum)
    n = (lens + 1).repeat_interleave(HKV).float()[:, None]
    assert torch.equal(got["vmean"], torch.div(want_sum, n * evs))


def test_the_lengths_advance_in_place():
    """`lens_out` is `lens`: a request spanning two warps (H_KV = 8 at D =
    128) still reads its length before it moves, once."""
    B, HKV, D, PS = 9, 8, 128, 64
    _, k, v, lens, pt = _case(B=B, HKV=HKV, D=D, PS=PS)
    st = _state(B, 8, HKV, D, PS, 8, True)
    want = _state(B, 8, HKV, D, PS, 8, True)
    write_kv(
        want["ka"],
        want["kb"],
        want["ek"],
        want["va"],
        want["vb"],
        k,
        v,
        0.25,
        pt,
        lens.long(),
        PS,
        HKV,
    )
    start = lens.clone()
    _run(st, None, k, v, pt, PS, lens, 0.25, lens)
    assert torch.equal(lens, start + 1)
    assert torch.equal(st["ka"], want["ka"])


def test_the_step_is_its_two_halves():
    B, H, HKV, D, PS = 6, 32, 4, 128, 64
    q, k, v, lens, pt = _case(B=B, H=H, HKV=HKV, D=D, PS=PS)
    a, b = _state(B, H, HKV, D, PS, 8, True), _state(B, H, HKV, D, PS, 8, True)
    oa, ob = torch.empty_like(lens), torch.empty_like(lens)
    _run(a, q, None, None, pt, PS, lens, 0.5, oa)
    _run(a, None, k, v, pt, PS, lens, 0.5, oa)
    _run(b, q, k, v, pt, PS, lens, 0.5, ob)
    for n in a:
        if a[n] is not None:
            assert torch.equal(a[n].view(torch.int8), b[n].view(torch.int8)), n
    assert torch.equal(oa, ob)


def test_a_row_s_bits_are_its_own():
    """A request stepped alone and inside a batch writes the same bits."""
    B, H, HKV, D, PS = 7, 32, 4, 128, 64
    q, k, v, lens, pt = _case(B=B, H=H, HKV=HKV, D=D, PS=PS)
    full = _state(B, H, HKV, D, PS, 8, True)
    _run(full, q, k, v, pt, PS, lens, 0.5, torch.empty_like(lens))
    for b in (0, 4, 6):
        one = _state(B, H, HKV, D, PS, 8, True)
        _run(
            one,
            q[b * H : (b + 1) * H],
            k[b : b + 1],
            v[b : b + 1],
            pt[b : b + 1],
            PS,
            lens[b : b + 1],
            0.5,
            torch.empty_like(lens[:1]),
        )
        assert torch.equal(one["qa"][:H], full["qa"][b * H : (b + 1) * H])
        assert torch.equal(one["qb"][:H], full["qb"][b * H : (b + 1) * H])
        rows = one["ka"].abs().sum(-1).nonzero().flatten()
        assert rows.numel() > 0
        assert torch.equal(one["ka"][rows], full["ka"][rows])
        assert torch.equal(one["kb"][rows], full["kb"][rows])
        assert torch.equal(one["ek"][rows], full["ek"][rows])


def test_every_e4m3_value_and_tie_is_torch_s():
    """V's planes over every e4m3 value, every midpoint between neighbours
    (the ties the conversion rounds to even), and both zeros."""
    allv = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    allv = allv[torch.isfinite(allv)].unique()
    vals = torch.cat([allv, (allv[1:] + allv[:-1]) / 2, torch.tensor([-0.0, 0.0])]).cuda()
    D, HKV, PS, evs = 128, 1, 64, 0.5
    n = -(-vals.numel() // D) * D
    # a bf16 V that is `vals` in the scale's units wherever bf16 holds it
    v = torch.cat([vals, torch.zeros(n - vals.numel(), device="cuda")]) * evs
    v = v.bfloat16().view(-1, HKV, D)
    B = v.shape[0]
    k = torch.randn(B, HKV, D, device="cuda").bfloat16()
    lens = torch.arange(B, device="cuda", dtype=torch.int32)
    pt = torch.arange(B, device="cuda", dtype=torch.int32).view(B, 1)
    got, want = _state(B, 8, HKV, D, PS, 1, True), _state(B, 8, HKV, D, PS, 1, True)
    _run(got, None, k, v, pt, PS, lens, evs, torch.empty_like(lens))
    write_kv(
        want["ka"],
        want["kb"],
        want["ek"],
        want["va"],
        want["vb"],
        k,
        v,
        evs,
        pt,
        lens.long(),
        PS,
        HKV,
    )
    for n in ("va", "vb"):
        assert torch.equal(got[n].view(torch.int8), want[n].view(torch.int8)), n


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("v8", [False, True])
def test_the_prompt_writer_is_the_torch_path(D, v8):
    """Each key's scale from its own rotated amax, planes and scale laid down
    at each token's own page slot, the V sum in order."""
    g = torch.Generator(device="cuda").manual_seed(4)
    lens, H, PS = [700, 1, 256, 513], 4, 64
    N, B, ppr = sum(lens), len(lens), 16
    k = (torch.randn(N, H, D, device="cuda", generator=g) * 3).bfloat16()
    v = (torch.randn(N, H, D, device="cuda", generator=g) + 1).bfloat16()
    pt = torch.randperm(B * ppr, device="cuda", generator=g).int().view(B, ppr)
    got = _state(B, H, H, D, PS, ppr, v8)
    evs = float(
        write_prompt(
            k,
            v,
            lens,
            pt,
            PS,
            got["ka"],
            got["kb"],
            got["va"],
            got["vb"],
            got["ek"],
            got["vsum"],
            got["vmean"],
            v_headroom=2.0,
            v_scale=None,
        )
    )
    req = torch.repeat_interleave(torch.arange(B, device="cuda"), torch.tensor(lens, device="cuda"))
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda")
    pos = torch.arange(N, device="cuda") - cu[req]
    if v8:
        assert evs == 2.0 ** math.ceil(math.log2(float(v.abs().amax()) * 2.0 / 448.0))
    else:
        assert evs == 1.0
    want = _state(B, H, H, D, PS, ppr, v8)
    row = (pt[req, pos // PS].long()[:, None] * H + torch.arange(H, device="cuda")) * PS + (
        pos % PS
    )[:, None]
    write_rows(want["ka"], want["kb"], want["ek"], want["va"], want["vb"], k, v, evs, row, pos % PS)
    for n in ("ka", "kb", "ek", "va", "vb"):
        if got[n] is not None:
            assert torch.equal(got[n].view(torch.int8), want[n].view(torch.int8)), n
    ref = torch.stack([v[cu[b] : cu[b + 1]].double().sum(0) for b in range(B)]).view(B * H, D)
    assert (got["vsum"].double() - ref).abs().max().item() < 1e-3
    n = torch.tensor(lens, device="cuda").repeat_interleave(H).float()[:, None]
    assert torch.equal(got["vmean"], torch.div(got["vsum"], n * evs))


def test_a_prompt_s_scales_and_sums_are_its_own():
    """The chunks are counted from each request's start, so a request written
    alone and inside a batch gets the same scale, V sum and planes."""
    g = torch.Generator(device="cuda").manual_seed(6)
    lens, H, D, PS = [300, 1100, 57], 2, 128, 64
    N, B, ppr = sum(lens), len(lens), 32
    k = (torch.randn(N, H, D, device="cuda", generator=g) * 3).bfloat16()
    v = (torch.randn(N, H, D, device="cuda", generator=g) + 1).bfloat16()
    pt = torch.arange(B * ppr, device="cuda", dtype=torch.int32).view(B, ppr)
    full = _state(B, H, H, D, PS, ppr, False)
    write_prompt(
        k,
        v,
        lens,
        pt,
        PS,
        full["ka"],
        full["kb"],
        full["va"],
        None,
        full["ek"],
        full["vsum"],
        full["vmean"],
        v_headroom=2.0,
        v_scale=None,
    )
    cu = [0] + list(torch.tensor(lens).cumsum(0).tolist())
    for b in range(B):
        one = _state(B, H, H, D, PS, ppr, False)
        write_prompt(
            k[cu[b] : cu[b + 1]],
            v[cu[b] : cu[b + 1]],
            [lens[b]],
            pt[b : b + 1],
            PS,
            one["ka"],
            one["kb"],
            one["va"],
            None,
            one["ek"],
            one["vsum"],
            one["vmean"],
            v_headroom=2.0,
            v_scale=None,
        )
        assert torch.equal(one["vsum"][:H], full["vsum"][b * H : (b + 1) * H])
        rows = slice(int(pt[b, 0]) * H * PS, (int(pt[b, -1]) + 1) * H * PS)
        assert torch.equal(one["ka"][rows], full["ka"][rows])
        assert torch.equal(one["ek"][rows], full["ek"][rows])
