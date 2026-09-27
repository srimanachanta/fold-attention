"""The decode cache format, and torch quantisers that define it.

K and Q are int8 fixed point under a Hadamard rotation, two planes each with
`x = e (a + b / 256)` and one scale per row: a query row's, or a key's own,
stored as bf16 rounded up so it is exact wherever it is read. A whole logit
is `256 c1 + c2 + c3 + c4 / 256` over the plane products (`c4` is plane B
against plane B), and `eq ek / 256` is its only scale. The rotation changes
no logit (`q.k == Rq.Rk`) and lets a row's scale fit its rms instead of an
outlier channel. V is two e4m3 planes under one power-of-two scale, or bf16.

K's planes are stored XOR-swizzled (`swizzle_rows`), which is the layout a
`wgmma` descriptor reads, so a tile is one contiguous bulk copy. A paged cache
is laid out `(page, kv head, slot, channel)`; one page of one head is
contiguous, so paging moves an address and not a layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..utils import hadamard
from .config import BN

FMAX = 448.0
# `1 / 127` rounded once to fp32, as torch computes it: the step of an int8
# plane under a unit amax
INV127 = float(np.float32(1.0) / np.float32(127.0))
# K's second plane is 1/256 of the first, so the planes are one 16-bit fixed
# point number; 256 keeps `256 c1 + c2 + c3` under 2^31 at D = 128.
KBR = 256.0


def swizzle_of(row_bytes: int):
    """`(units, shift, atom bytes)` of the swizzle a row of `row_bytes` takes.

    Unit `u` of row `r` lives at `u ^ ((r >> shift) & (units - 1))`: `r & 7` at
    128-byte rows and `(r >> 1) & 3` at 64-byte ones. The descriptor modes
    (`S<3,4,3>`, `S<2,4,3>`, `S<1,4,3>`) all shift by 3, so the period is eight
    rows at every width, and a narrower row pair shares one XOR.
    """
    nu = min(row_bytes // 16, 8)
    if nu < 1 or nu & (nu - 1):
        raise ValueError(
            f"a {row_bytes}-byte row is {row_bytes / 16} sixteen-byte units; "
            "a descriptor swizzle needs a power of two, at least one"
        )
    return nu, 3 - (nu.bit_length() - 1), nu * 128


def swizzle_rows(x):
    """Store 16-byte unit `u` of row `r` at unit `u ^ ((r >> s) & (n - 1))`."""
    b, S, D = x.shape
    if D % 16:
        raise ValueError(f"a row is 16-byte units; D={D} is not a multiple of 16")
    nu, sh, _ = swizzle_of(D)
    if nu * 16 != D:
        raise ValueError(
            f"D={D} is {D // 16} units, more than the eight a descriptor "
            "swizzles; a wider row has to be stored in 128-byte blocks"
        )
    dt = x.dtype
    v = x.view(torch.int8).reshape(b, S, nu, 16)
    r = (torch.arange(S, device=x.device) >> sh) & (nu - 1)
    idx = torch.arange(nu, device=x.device)[None, :] ^ r[:, None]
    idx = idx.view(1, S, nu, 1).expand(b, S, nu, 16)
    return torch.gather(v, 2, idx).reshape(b, S, D).view(dt).contiguous()


def _int8_planes(x, e):
    # one correctly rounded reciprocal and a multiply, which is what the write
    # kernels do: a division per element costs them half their bandwidth
    pa = torch.round(x * (1.0 / e)).clamp(-128, 127)
    r = x - pa * e
    pb = torch.round(r * (KBR / e)).clamp(-128, 127)
    return pa.to(torch.int8), pb.to(torch.int8)


def _e4m3_planes(v, e):
    pa = torch.clamp(v / e, -FMAX, FMAX).to(torch.float8_e4m3fn)
    r = v / e - pa.float()
    pb = torch.clamp(r * 8.0, -FMAX, FMAX).to(torch.float8_e4m3fn)
    return pa, pb


def quantize_kq(x, rot=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q's two int8 planes and one scale per row."""
    if rot:
        x = hadamard(x)
    e = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127.0
    pa, pb = _int8_planes(x, e)
    return pa.contiguous(), pb.contiguous(), e.squeeze(-1).contiguous()


def key_scale(a):
    """A key's scale from its rotated row's largest magnitude `a`: `a / 127`
    as the writers compute it, rounded up to the next bf16, which the cache
    stores. Rounding up keeps every element within plane A's range, and a
    bf16 scale is half the bytes of an f32 one at no measurable cost in
    precision; a power of two would cost most of what a key's own scale buys
    over one shared by its row group."""
    e = a.float().clamp_min(1e-30) * INV127
    b = e.view(torch.int32)
    return ((b + 0xFFFF) & -65536).view(torch.float32)


def key_planes(k, rot=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keys' two int8 planes, unswizzled, and each key's scale as f32.

    A key's own scale fits its own range, so no key ever overflows plane A,
    a key appended later needs no reserve in an earlier scale, and a key's
    bits depend on the key alone."""
    x = hadamard(k) if rot else k.float()
    e = key_scale(x.abs().amax(-1, keepdim=True))
    pa, pb = _int8_planes(x, e)
    return pa.contiguous(), pb.contiguous(), e.squeeze(-1).contiguous()


def quantize_q(q, rot=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q's two int8 planes `(NBH, G, D)` and one scale per query row. `q` is
    already scaled by `LOG2E / sqrt(D)`: every weight is `exp2(s - Z)`."""
    return quantize_kq(q, rot=rot)


def quantize_k(k, rot=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K `(NBH, S, D)` as two int8 planes, both swizzled, and one bf16 scale
    per key `(NBH, S)`. The decode copies a tile's scales in whole 16-byte
    units, so each row of scales starts 16 bytes apart and has room for
    eight past its last key: the scales are a view of rows padded to a
    multiple of eight."""
    ka, kb, ek = key_planes(k, rot)
    return swizzle_rows(ka), swizzle_rows(kb), pad_scales(ek)


def pad_scales(ek) -> torch.Tensor:
    """Per-key scales `(NBH, S)` as bf16 in the layout a contiguous cache's
    decode reads: rows a multiple of eight scales apart, returned as the
    `(NBH, S)` view. Scales selected or reordered by row come out unpadded;
    this pads them again."""
    NBH, S = ek.shape
    ep = torch.zeros((NBH, -(-S // 8) * 8), device=ek.device, dtype=torch.bfloat16)
    ep[:, :S] = ek
    return ep[:, :S]


def quantize_v(v) -> tuple[torch.Tensor, torch.Tensor, float]:
    """V's two e4m3 planes and their one power-of-two scale.

    The second plane is the residual at 2^-3, which the kernel gathers only
    for keys whose weight clears `refine_v`. A per-tensor scale that is a
    power of two makes dividing a bf16 cascade level by it exact.
    """
    a = v.abs().amax().clamp_min(1e-30)
    e = torch.exp2(torch.ceil(torch.log2(a / FMAX)))
    pa, pb = _e4m3_planes(v, e)
    return pa.contiguous(), pb.contiguous(), float(e)


def _page_plane(x, page_table, page_size, n_kv_heads):
    NBH, S, w = x.shape
    B = NBH // n_kv_heads
    npg = int(page_table.max().item()) + 1
    wb = w * x.element_size()
    t = torch.zeros((npg, n_kv_heads, page_size, w), device=x.device, dtype=x.dtype)
    xv = x.reshape(NBH, S, w).view(torch.int8).reshape(B, n_kv_heads, S, wb)
    tv = t.view(torch.int8).reshape(npg, n_kv_heads, page_size, wb)
    for i in range(page_table.shape[1]):
        lo = i * page_size
        n = min(page_size, S - lo)
        if n <= 0:
            break
        pgi = page_table[:, i]
        tv[pgi, :, :n] = xv[:, :, lo : lo + n]
    return t.reshape(npg * n_kv_heads * page_size, w)


def paged_scale(ek, page_table, page_size, n_kv_heads) -> torch.Tensor:
    """Per-key scales `(NBH, S)` paged as `paged_cache` pages their keys, one
    per pool row."""
    return _page_plane(ek[..., None], page_table, page_size, n_kv_heads).view(-1)


def paged_cache(ka, kb, va, vb, page_table, page_size, n_kv_heads, *extra) -> list[torch.Tensor]:
    """Scatter contiguous `(NBH, S, W)` planes into the paged table the kernel
    reads, `(n_pages * n_kv_heads * page_size, W)`. `extra` pages further
    planes the same way, such as a bf16 V beside the e4m3 pair.

    The swizzle survives because its period is eight and a page holds a
    multiple of eight slots.
    """
    return [
        _page_plane(x, page_table, page_size, n_kv_heads) for x in (ka, kb, va, vb) + tuple(extra)
    ]


# keys per block of the tail's rows, which is the kernel's tile
TAIL_BLOCK = 64


@dataclass
class Tail:
    """Where a truncated step's dropped mass re-enters (`tail_model`).

    `vblk` holds one bf16 row per 64 cache rows: the block's mean V less the
    part of it the block's mean K predicts. At rank r, `u` `(NBH, r, D)` int8
    projects plane A onto the rank's basis and `vr` `(NBH, D, r)` f32 maps
    that projection back to V; both are None at rank 0. The tail runs on a
    bf16 V only: on an 8-bit V it loses time at matched error
    (`heuristics.tail_rank_for`).
    """

    vblk: torch.Tensor
    u: torch.Tensor | None = None
    vr: torch.Tensor | None = None

    @property
    def rank(self) -> int:
        return 0 if self.u is None else int(self.u.shape[1])

    def paged(self, page_table, page_size, n_kv_heads):
        """This tail for a cache paged by `paged_cache`: a block row lives at
        its first key's pool row over 64, so a page must hold whole blocks."""
        if page_size % TAIL_BLOCK:
            raise ValueError(
                f"page_size={page_size}: the tail keeps one row per {TAIL_BLOCK} "
                "keys, so a page must hold a whole number of blocks"
            )
        pb = page_size // TAIL_BLOCK

        def pg(x):
            return None if x is None else _page_plane(x, page_table, pb, n_kv_heads)

        return Tail(pg(self.vblk), self.u, self.vr)


@dataclass
class TailState:
    """The serving cache's tail, which the step writer keeps current as
    tokens append: every 64-key block's V row (`rows`, V's dtype), V sum
    (`vsum`) and, at a rank, projection sum (`ysum`), and each row group's
    rank map `u` and `vr` (`tail_basis`)."""

    rows: torch.Tensor
    vsum: torch.Tensor
    ysum: torch.Tensor | None = None
    u: torch.Tensor | None = None
    vr: torch.Tensor | None = None

    @property
    def rank(self) -> int:
        return 0 if self.u is None else int(self.u.shape[1])

    def decode_tail(self) -> Tail:
        """What the decode reads of it."""
        return Tail(self.rows, self.u, self.vr)


def _block_mean(x):
    """Means over consecutive 64-row blocks of `(N, S, W)`, the last block
    over the rows it has."""
    N, S, W = x.shape
    nb = -(-S // TAIL_BLOCK)
    xp = torch.nn.functional.pad(x, (0, 0, 0, nb * TAIL_BLOCK - S))
    cnt = torch.clamp(S - torch.arange(nb, device=x.device) * TAIL_BLOCK, max=TAIL_BLOCK)
    return xp.reshape(N, nb, TAIL_BLOCK, W).sum(2) / cnt[None, :, None].to(x.dtype)


def tail_basis(kint, ek, vs, rank, lam=1e-3, fit=None):
    """The rank's projection for `(N, S, D)` plane-A integers `kint` under
    per-key scales `ek` `(N, S)` and values in the cache's stored units `vs`.

    `v - vblk ~ (k - kblk) M` is a least-squares fit over the first `fit` keys
    (all of them by default), with K the kernel's own view of it (plane A times
    its scale), reduced to rank r on its fitted values. Serving fits on the
    prompt and applies the map to every key generated after it. Returns U as
    int8 `(N, r, D)` with one scale per rank, the map back to V `(N, D, r)`
    f32 with those scales and 256 folded in, and every key's projection
    `(N, S, r)`: the integer the kernel's logit matmul computes from plane A,
    times the key's `ek / 256`, rounded once as the kernel rounds it.
    """
    _, S, D = kint.shape
    kall = kint
    if fit is not None:
        kint, vs = kint[:, :fit], vs[:, :fit]
        S = kint.shape[1]
    ekf = ek.float()
    kq = kint * ekf[:, :S, None]
    kc = kq - _block_mean(kq).repeat_interleave(TAIL_BLOCK, 1)[:, :S]
    vc = vs - _block_mean(vs).repeat_interleave(TAIL_BLOCK, 1)[:, :S]
    A = (kc.transpose(1, 2) @ kc).double()
    reg = lam * A.diagonal(dim1=1, dim2=2).sum(-1) / D
    A = A + reg[:, None, None] * torch.eye(D, device=kint.device, dtype=A.dtype)
    M = torch.linalg.solve(A, (kc.transpose(1, 2) @ vc).double()).float()
    fv = kc @ M
    _, evec = torch.linalg.eigh((fv.transpose(1, 2) @ fv).double())
    basis = evec[:, :, -rank:].flip(-1).float()
    uu = M @ basis
    eu = uu.abs().amax(1).clamp_min(1e-30) / 127.0
    q8 = torch.round(uu / eu[:, None, :]).clamp(-127, 127)
    y = (kall.double() @ q8.double()).float() * (ekf * (1.0 / KBR))[:, :, None]
    vr = (basis.transpose(1, 2) * (eu * KBR)[:, :, None]).transpose(1, 2)
    return q8.transpose(1, 2).to(torch.int8).contiguous(), vr.contiguous(), y


def tail_model(ka, ek, v, *, rank=0, lam=1e-3, chunk=16, fit=None):
    """The dropped mass's re-entry for a cache: its `Tail`.

    A dropped key re-enters at its block's mean V plus the part of its own V
    that its K predicts (`tail_basis`), and each block's row absorbs the
    projection of the block's mean K. `ka` and `ek` are plane A and the
    scales as `quantize_k` stores them, `v` the values `(NBH, S, D)` of the
    bf16 cache. `fit` fits the rank's
    map on the first `fit` keys only, as serving fits it on the prompt; the
    block rows always cover every key.
    """
    NBH, _, D = v.shape
    if rank and (rank % 8 or rank > D):
        raise ValueError(f"rank={rank}: the projection is 8i rows of the logit's matmul, at most D")
    vs = v.float()
    w = _block_mean(vs)
    u8 = vr = None
    if rank:
        us, vrs, ws = [], [], []
        for c0 in range(0, NBH, chunk):
            c1 = min(NBH, c0 + chunk)
            q8, vrc, y = tail_basis(
                swizzle_rows(ka[c0:c1]).float(), ek[c0:c1], vs[c0:c1], rank, lam, fit
            )
            ws.append(w[c0:c1] - _block_mean(y) @ vrc.transpose(1, 2))
            us.append(q8)
            vrs.append(vrc)
        w = torch.cat(ws)
        u8 = torch.cat(us)
        vr = torch.cat(vrs)
    return Tail(w.to(torch.bfloat16).contiguous(), u8, vr)


def write_kv(
    ka,
    kb,
    ek,
    va,
    vb,
    k_new,
    v_new,
    ev,
    page_table,
    pos,
    page_size,
    n_kv_heads,
    rot=True,
):
    """Append one decode step's K and V to a paged cache in its own format.

    `k_new` and `v_new` are `(B, n_kv_heads, D)`, `pos` is `(B,)`. `ek` is
    the pool of per-key scales, which the new keys' own scales go into; `ev`
    is V's scale. `rot` states the bases K and V were quantised in
    (`quantize_k` rotates by default, `quantize_v` does not); a row appended
    in the wrong basis is undetectable downstream.
    """
    B, H, _ = k_new.shape
    pg = page_table[torch.arange(B, device=pos.device), pos // page_size]
    off = pos % page_size
    row = (pg[:, None] * H + torch.arange(H, device=pos.device)[None, :]) * page_size + off[:, None]
    write_rows(ka, kb, ek, va, vb, k_new, v_new, ev, row, off, rot=rot)


def write_rows(ka, kb, ek, va, vb, k, v, ev, row, off, rot=True):
    """Quantise `(N, H, D)` keys and values into pool rows `row` `(N, H)`.

    `off` `(N,)` is each token's slot in its page, which sets K's swizzle
    phase. Each key's scale goes into the pool `ek` at its row. `vb=None`
    says the V pool `va` is 16-bit and takes the value as it is, so `ev` is
    unused.
    """
    N, H, D = k.shape
    kq, kq2, e = key_planes(k, rot)
    ek[row.reshape(-1)] = e.reshape(-1).to(ek.dtype)
    if vb is None:
        planes = ((ka, kq, 1), (kb, kq2, 1))
        va[row.reshape(-1)] = v.reshape(N * H, D).to(va.dtype)
    else:
        vq, vq2 = _e4m3_planes(v.float(), float(ev))
        planes = ((ka, kq, 1), (kb, kq2, 1), (va, vq, 0), (vb, vq2, 0))
    # K's planes are stored swizzled by their slot; V is gathered a row at a
    # time and is not
    nu, sh, _ = swizzle_of(D)
    u = torch.arange(nu, device=row.device)[None, :] ^ ((off >> sh) & (nu - 1))[:, None]
    idx = u.view(N, 1, nu, 1).expand(N, H, nu, 16)
    for dst, src, sw in planes:
        sv = src.view(torch.int8).reshape(N, H, nu, 16)
        if sw:
            sv = torch.gather(sv, 2, idx)
        dst.view(torch.int8)[row.reshape(-1)] = sv.reshape(N * H, D)


@dataclass
class PrefixImage:
    """A prefix's K and V as the wide kernel's tiles: `k` is
    `fp16(Ka + Kb / 256)` and `v` bf16, each `(NBH, tiles, 64 D)` with a
    tile's 64 keys in 64-channel blocks of the 128-byte swizzle a `wgmma`
    descriptor reads, so a tile is one bulk copy; `e` is each key's scale as
    f32, `(NBH, tiles * 64)`. `length` is the prefix's key count; rows past it
    are zero."""

    k: torch.Tensor
    v: torch.Tensor
    e: torch.Tensor
    length: int


def _tile_image(x, tiles):
    """`(NBH, L, D)` 16-bit rows into `(NBH, tiles, BN * D)` tile images."""
    NBH, L, D = x.shape
    y = torch.zeros((NBH, tiles * BN, D), device=x.device, dtype=x.dtype)
    y[:, :L] = x
    # (bh, tile, row, block, unit, 8) -> (bh, tile, block, row, unit ^ (row & 7), 8)
    y = y.view(NBH, tiles, BN, D // 64, 8, 8).permute(0, 1, 3, 2, 4, 5)
    r = torch.arange(BN, device=x.device)[:, None]
    u = torch.arange(8, device=x.device)[None, :]
    perm = (u ^ (r & 7)).expand(BN, 8)
    out = torch.empty_like(y)
    out.scatter_(4, perm[None, None, None, :, :, None].expand_as(y), y)
    return out.reshape(NBH, tiles, BN * D).contiguous()


def prefix_image(ka, kb, ek, v, length=None) -> PrefixImage:
    """A contiguous prefix cache (`quantize_k`'s swizzled planes and scales,
    bf16 V) as the wide kernel's tiles. A shared prefix is read by every step
    of every request that holds it, so its K is rounded to fp16 once here
    rather than per tile in every step; the rounding is the kernel's own."""
    L = ka.shape[1] if length is None else int(length)
    a = swizzle_rows(ka[:, :L]).float()
    b = swizzle_rows(kb[:, :L]).float()
    k16 = (a + b * (1.0 / KBR)).half()
    tiles = -(-L // BN)
    e = torch.zeros((ka.shape[0], tiles * BN), device=ka.device, dtype=torch.float32)
    e[:, :L] = ek[:, :L].float()
    return PrefixImage(
        _tile_image(k16, tiles), _tile_image(v[:, :L].bfloat16(), tiles), e.contiguous(), L
    )
